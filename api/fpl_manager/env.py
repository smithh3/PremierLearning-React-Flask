"""Masked season environment.

The observation is already scaled. Do not wrap this in a second normaliser:
training and evaluation have to see the same fields.
"""
from __future__ import annotations

import numpy as np
from gymnasium import Env, spaces

from fpl_manager.actions import ACTIONS
from fpl_manager.observation import OBSERVATION_NAMES


class ManagerEnv(Env):
    def __init__(self, factory, observation_names=None):
        super().__init__()
        self.factory = factory
        self.episode = None
        self.observation_names = tuple(observation_names or OBSERVATION_NAMES)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.observation_names),),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode = self.factory()
        return self._observation(), {}

    def step(self, action):
        observation, reward, done, info = self.episode.step(int(action))
        if done:
            observation = np.zeros(len(self.observation_names), dtype=np.float32)
        else:
            observation = np.asarray(observation, dtype=np.float32)
        return observation, float(reward), bool(done), False, info

    def action_masks(self):
        return np.asarray(self.episode.mask(), dtype=bool)

    def __getstate__(self):
        state = dict(self.__dict__)
        # The live episode holds the outcome callable for this rollout.
        # A checkpoint only needs the factory, which builds a new one.
        state['episode'] = None
        return state

    def _observation(self):
        return np.asarray(self.episode.observation(), dtype=np.float32)


def masked_monitor(env):
    """Wrap a ManagerEnv so Stable-Baselines finds the masks and the evaluation helpers."""
    return _monitor_class()(env)


_MONITOR_CLASS = None


def _monitor_class():
    global _MONITOR_CLASS
    if _MONITOR_CLASS is not None:
        return _MONITOR_CLASS
    from stable_baselines3.common.monitor import Monitor

    class MaskedMonitor(Monitor):
        """The monitor owns action_masks, so Gymnasium does not warn about the lookup.

        The other methods let the training process play one evaluation run
        in this worker: the policy's top choice every week, logged apart.
        """

        def action_masks(self):
            return self.env.action_masks()

        def mark_evaluation(self) -> bool:
            run = getattr(self.env, 'episode', None)
            if run is None or not getattr(run, 'at_start', False):
                return False
            run.evaluation = True
            print(
                'Run %s is an evaluation run: the policy plays its top choice '
                'every week and the result goes to eval.csv.' % run.run_id,
                flush=True,
            )
            # A run on synthetic copies swaps in the held-out versions here.
            if hasattr(run, 'begin_evaluation'):
                run.begin_evaluation()
            return True

        def current_observation(self):
            return self.env._observation()

        def evaluation_step(self, action):
            observation, _reward, terminated, _truncated, info = self.step(int(action))
            finished = bool(terminated and info.get('run_finished', True))
            if terminated and not finished:
                observation, _info = self.reset()
            mask = None if finished else self.action_masks()
            return observation, mask, finished

        def restart(self):
            observation, _info = self.reset()
            return observation

    globals()['MaskedMonitor'] = MaskedMonitor
    _MONITOR_CLASS = MaskedMonitor
    return MaskedMonitor
