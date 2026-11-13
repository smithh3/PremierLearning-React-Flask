"""Masked PPO. There is no second implementation.

Install requirements-ppo.txt into an environment that already runs this
project. Do not add those packages to the TensorFlow requirement files.

Discounting is the library's. The episode puts the terminal reward on the
last gameweek only, and the weekly rank on that gameweek. With gamma below
1, earlier weeks weigh more than later ones, which is why 0.995 and 1.0 are
worth comparing with the 0.99 start.

Checkpoints are written after each policy update and again when training
returns, including after an interrupt. The rollout buffer is not stored.
Stable-Baselines3 does not restore one, so a resume starts a fresh rollout
from the saved weights and optimiser. A new run refuses an existing
checkpoint directory. It can still read the optimiser cache when the
prediction snapshot matches; that cache is not part of this file.
"""
from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

import numpy as np

from fpl_manager.actions import ACTIONS
from fpl_manager.observation import (
    MONEY_SCALE,
    OBSERVATION_NAMES,
    POINT_SCALE,
    version_for_features,
)
from fpl_manager.reward import (
    PERCENTILE_DIVISOR,
    TERMINAL_DIVISOR,
    WEEKLY_COEFFICIENT,
    WEEKLY_POINTS_WEIGHT,
)

# Safety stop for an evaluation run that never reports its end.
_EVALUATION_STEP_LIMIT = 10000


class CheckpointMismatch(RuntimeError):
    """The saved policy was trained against a different observation or reward."""


def reward_settings(features=None) -> dict:
    """What the manifest records about the reward. v3 adds its own fields."""
    settings = {
        'terminal_divisor': TERMINAL_DIVISOR,
        'weekly_coefficient': WEEKLY_COEFFICIENT,
    }
    if features is not None and version_for_features(features) == 'v3':
        settings.update({
            'version': 'v3',
            'weekly_points_weight': WEEKLY_POINTS_WEIGHT,
            'percentile_divisor': PERCENTILE_DIVISOR,
            'per_season_episodes': True,
        })
    return settings


def _same(stored, expected) -> bool:
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return stored == expected
    try:
        return float(stored) == float(expected)
    except (TypeError, ValueError):
        return False


def require_ppo():
    try:
        import gymnasium  # noqa: F401
        import sb3_contrib  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            'Masked PPO needs the optional requirements-ppo.txt libraries '
            '(gymnasium, stable-baselines3, sb3-contrib, torch). They are '
            'not part of the TensorFlow training environment, and a second '
            'PPO implementation is not included. Import failed: %s' % exc
        ) from exc


def _manifest(model, gamma: float, features=None) -> dict:
    return {
        'features': list(OBSERVATION_NAMES if features is None else features),
        'actions': list(ACTIONS),
        'reward': reward_settings(features),
        'scales': {'point': POINT_SCALE, 'money': MONEY_SCALE},
        'gamma': float(gamma),
        'num_timesteps': int(model.num_timesteps),
    }


def _check_manifest(manifest: dict, features=None):
    expected = list(OBSERVATION_NAMES if features is None else features)
    if list(manifest.get('features') or ()) != expected:
        raise CheckpointMismatch('Observation fields do not match this code')
    if list(manifest.get('actions') or ()) != list(ACTIONS):
        raise CheckpointMismatch('Action order does not match this code')
    reward = manifest.get('reward') or {}
    for key, value in reward_settings(features).items():
        if not _same(reward.get(key), value):
            raise CheckpointMismatch('Reward settings do not match this code')
    scales = manifest.get('scales') or {}
    if (
        float(scales.get('point', -1)) != float(POINT_SCALE)
        or float(scales.get('money', -1)) != float(MONEY_SCALE)
    ):
        raise CheckpointMismatch('Feature scales do not match this code')


def save_checkpoint(model, directory, gamma: float, features=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(str(directory / 'policy'))
    if features is None:
        features = getattr(model, 'checkpoint_features', None)
    manifest = _manifest(model, gamma, features)
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    import torch
    with (directory / 'rng.pkl').open('wb') as handle:
        pickle.dump({
            'numpy': np.random.get_state(),
            'python': random.getstate(),
            'torch': torch.get_rng_state(),
        }, handle)
    return manifest


def load_checkpoint(directory, env=None, features=None):
    require_ppo()
    import torch

    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('No checkpoint manifest at %s' % directory)
    manifest = json.loads(manifest_path.read_text())
    _check_manifest(manifest, features)
    model = _checkpoint_class().load(str(directory / 'policy'), env=env, device='cpu')
    if int(model.num_timesteps) != int(manifest['num_timesteps']):
        raise CheckpointMismatch(
            'Saved step counter %s does not match the policy file %s'
            % (manifest['num_timesteps'], model.num_timesteps)
        )
    with (directory / 'rng.pkl').open('rb') as handle:
        state = pickle.load(handle)
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    return model, manifest


_PPO_CLASS = None


def _checkpoint_class():
    """A MaskablePPO that writes a checkpoint after each update.

    The class is built on first use so this module imports without PyTorch,
    then stored on the module so a saved policy can be unpickled by name.
    """
    global _PPO_CLASS
    if _PPO_CLASS is not None:
        return _PPO_CLASS
    from sb3_contrib import MaskablePPO

    class CheckpointPPO(MaskablePPO):
        def train(self):
            super().train()
            every = int(getattr(self, 'evaluate_every', 0) or 0)
            updates = int(self._n_updates) // max(1, int(self.n_epochs))
            if every and updates % every == 0:
                evaluate_once(self)
            directory = getattr(self, 'checkpoint_dir', None)
            if directory:
                save_checkpoint(self, directory, float(getattr(self, 'checkpoint_gamma', 0.99)))

    globals()['CheckpointPPO'] = CheckpointPPO
    _PPO_CLASS = CheckpointPPO
    return CheckpointPPO


def evaluate_once(model) -> bool:
    """Play one whole run in the first worker with the policy's top choice.

    Training runs sample their actions, so their scores include exploration.
    This run does not: it shows what the policy itself would score. It goes
    to eval.csv and is not used for learning. It only starts when the first
    worker is at the very start of a run, which it is after each update when
    every update covers whole runs. Returns whether it ran.
    """
    env = model.get_env()
    try:
        started = env.env_method('mark_evaluation', indices=[0])[0]
    except (AttributeError, BrokenPipeError, EOFError, OSError) as error:
        print('Evaluation skipped: %s' % error, flush=True)
        return False
    if not started:
        print('Evaluation skipped: the first worker is part-way through a run.', flush=True)
        return False
    # Read again: an evaluation on synthetic copies starts on different seasons.
    observation = np.asarray(env.env_method('current_observation', indices=[0])[0])
    mask = np.asarray(env.env_method('action_masks', indices=[0])[0], dtype=bool)
    finished = False
    steps = 0
    while not finished and steps < _EVALUATION_STEP_LIMIT:
        action, _state = model.predict(
            observation[None], deterministic=True, action_masks=mask[None],
        )
        observation, mask, finished = env.env_method(
            'evaluation_step', int(np.asarray(action).reshape(-1)[0]), indices=[0],
        )[0]
        observation = np.asarray(observation)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
        steps += 1
    model._last_obs[0] = np.asarray(env.env_method('restart', indices=[0])[0])
    model._last_episode_starts[0] = True
    return finished


def _process_envs(env_fns):
    """One environment per process. A single function stays in this process."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = list(env_fns)
    if len(fns) == 1:
        return DummyVecEnv(fns)

    class _VecEnv(SubprocVecEnv):
        def get_attr(self, attr_name, indices=None):
            # MaskablePPO fetches the action_masks method to see that it exists.
            # The worker environment cannot be pickled, so answer that here.
            # The masks themselves are still read by calling the method.
            if attr_name == 'action_masks':
                return [True for _remote in self._get_target_remotes(indices)]
            return super().get_attr(attr_name, indices)

        def close(self):
            if self.closed:
                return
            try:
                super().close()
            except (BrokenPipeError, EOFError, OSError):
                self.closed = True
                for process in self.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)

    return _VecEnv(fns, start_method='spawn')


def _training_env(episode_factory, features=None):
    """Vector env whose monitor owns action_masks.

    Stable-Baselines looks the mask up on the monitor wrapper. Defining it
    there keeps Gymnasium from warning about the wrapper attribute lookup.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    from fpl_manager.env import ManagerEnv, masked_monitor

    names = features
    return DummyVecEnv([
        lambda: masked_monitor(ManagerEnv(episode_factory, names)),
    ])


def train_policy(
    episode_factory,
    directory,
    total_timesteps: int,
    resume: bool = False,
    gamma: float = 0.99,
    seed: int = 0,
    n_steps: int = 38,
    batch_size: int = 38,
    n_epochs: int = 10,
    env_fns=None,
    features=None,
    gae_lambda: float = 0.95,
    evaluate_every: int = 0,
):
    """Train or continue a MaskablePPO. gamma 0.995 and 1.0 are the later comparisons.

    env_fns runs each environment in its own process. They sample the one
    policy, and the weight update uses every process's finished run.
    `gae_lambda` applies to a new policy; a resumed one keeps its own.
    `evaluate_every` plays one top-choice run after every that many updates.
    """
    require_ppo()

    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    feature_names = tuple(OBSERVATION_NAMES if features is None else features)
    env = (
        _process_envs(env_fns)
        if env_fns else _training_env(episode_factory, feature_names)
    )
    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError('No checkpoint to resume at %s' % directory)
        model, _manifest = load_checkpoint(directory, env=env, features=feature_names)
        gamma = float(_manifest['gamma'])
        _keep_runs_per_update(model, n_steps, batch_size)
    else:
        if manifest_path.is_file():
            raise CheckpointMismatch(
                'A checkpoint already exists at %s. Resume it, or use a new directory.'
                % directory
            )
        model = _checkpoint_class()(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            n_steps=int(n_steps),
            batch_size=int(batch_size),
            n_epochs=int(n_epochs),
            gamma=float(gamma),
            gae_lambda=float(gae_lambda),
            clip_range=0.2,
            ent_coef=0.01,
            seed=int(seed),
            policy_kwargs={'net_arch': [64, 64]},
            device='cpu',
            verbose=0,
        )
    model.checkpoint_dir = str(directory)
    model.checkpoint_gamma = float(gamma)
    model.checkpoint_features = feature_names
    model.evaluate_every = int(evaluate_every)
    processes = len(env_fns) if env_fns else 1
    try:
        model.learn(
            total_timesteps=int(total_timesteps),
            reset_num_timesteps=not resume,
        )
    except (BrokenPipeError, EOFError, OSError) as error:
        print(
            'A worker process died (%s). The last finished weight update is '
            'saved at %s. Resume that checkpoint.'
            % (error, directory),
            flush=True,
        )
    finally:
        save_checkpoint(model, directory, gamma)
        if processes > 1:
            try:
                env.close()
            except (BrokenPipeError, EOFError, OSError):
                pass
    return model


def _keep_runs_per_update(model, n_steps, batch_size):
    """Use this call's steps per process on a resumed policy.

    The saved policy keeps the steps per process it was started with. Resumed
    with more processes, each update would then hold more runs (4 per process
    x 4 processes = 16 instead of 8). Rebuilding the rollout buffer with this
    call's n_steps keeps the runs per update the same, so more processes only
    make each update come sooner.
    """
    n_steps, batch_size = int(n_steps), int(batch_size)
    if int(model.n_steps) == n_steps and int(model.batch_size) == batch_size:
        return
    model.n_steps = n_steps
    model.batch_size = batch_size
    model.rollout_buffer = model.rollout_buffer_class(
        n_steps,
        model.observation_space,
        model.action_space,
        model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
        **(model.rollout_buffer_kwargs or {}),
    )


def policy_weight_sum(model) -> float:
    total = 0.0
    for parameter in model.policy.parameters():
        total += float(parameter.detach().sum().item())
    return total


def _one_point(_player_id, _gameweek):
    return 1.0


def tiny_hold_factory():
    """One-week hold episode used to check that a checkpoint reloads."""
    from squadtools.simulation_strategy import OwnedPlayer, PlayerProjection, PredictionPool, SquadState

    from fpl_manager.episode import SeasonEpisode

    counts = {1: 2, 2: 5, 3: 5, 4: 3}
    players = {}
    owned = {}
    player_id = 1
    for position, count in counts.items():
        for _index in range(count):
            player = PlayerProjection(
                player_id=player_id,
                player_code=1000 + player_id,
                name='p%s' % player_id,
                position=position,
                team_id=player_id,
                price=45,
                selectable=True,
            )
            player.points_by_gameweek[1] = 2.0
            players[player_id] = player
            owned[player_id] = OwnedPlayer(player_id, 45)
            player_id += 1
    # Five spare midfielders so every transfer action has someone to buy.
    for extra in range(5):
        spare_id = 100 + extra
        spare = PlayerProjection(
            player_id=spare_id,
            player_code=1000 + spare_id,
            name='s%s' % spare_id,
            position=3,
            team_id=spare_id,
            price=45,
            selectable=True,
        )
        spare.points_by_gameweek[1] = 3.0
        players[spare_id] = spare
    return SeasonEpisode(
        PredictionPool(players),
        SquadState(owned=owned, bank=50, free_transfers=1),
        _one_point,
        last_gameweek=1,
    )


def tiny_vec_env():
    """Picklable one-week env for a process worker."""
    from stable_baselines3.common.monitor import Monitor

    from fpl_manager.env import ManagerEnv

    class _MaskedMonitor(Monitor):
        def action_masks(self):
            return self.env.action_masks()

    return _MaskedMonitor(ManagerEnv(tiny_hold_factory))
