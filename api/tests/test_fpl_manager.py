"""Episode rules, solver, cache, scenarios and a reloadable policy."""
import io
import json
import re
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from squadtools.simulation_strategy import (
    ChipCalendar,
    OwnedPlayer,
    PlayerProjection,
    PredictionPool,
    SquadState,
)

from fpl_manager.actions import ACTIONS, FREE_HIT, HOLD, TRANSFER_1, TRANSFER_2, WILDCARD, legal_mask
from fpl_manager.baselines import play_existing, play_hold, ExistingManager
from fpl_manager.cache import (
    RULES_VERSION_FAST,
    RULES_VERSION_V2,
    OptimiserCache,
    approximate_key,
    candidate_is_legal,
    exact_key,
    purchase_pairs,
    squad_codes,
    team_value,
    team_values_close,
)
from fpl_manager.episode import SeasonEpisode, hold_action_index
from fpl_manager.forecasts import (
    MissingPredictionSnapshot,
    fixed_payload,
    load_forecast_pool,
)
from fpl_manager.mip import MipOptimiser
from fpl_manager.observation import (
    OBSERVATION_NAMES,
    OBSERVATION_NAMES_FAST,
    OBSERVATION_NAMES_V2,
    build_observation,
    build_observation_fast,
    build_observation_v2,
    version_for_features,
)
from fpl_manager.reward import terminal_reward, weekly_reward
from fpl_manager.scenarios import (
    FixtureSpec,
    MissingMatchXG,
    PlayerFixture,
    ScenarioSeason,
    adjust_points,
)


def _player(player_id, position, price, points, team_id=None):
    player = PlayerProjection(
        player_id=player_id,
        player_code=1000 + player_id,
        name='p%s' % player_id,
        position=position,
        team_id=player_id if team_id is None else team_id,
        price=price,
        selectable=True,
    )
    for gameweek in range(1, 6):
        player.points_by_gameweek[gameweek] = float(points)
        player.fixture_ids_by_gameweek[gameweek] = {gameweek}
    return player


def _pool_and_squad():
    counts = {1: 2, 2: 5, 3: 5, 4: 3}
    players = {}
    owned = {}
    player_id = 1
    for position, count in counts.items():
        for _index in range(count):
            players[player_id] = _player(player_id, position, 45, 2.0)
            owned[player_id] = OwnedPlayer(player_id, 45)
            player_id += 1
    # A clearly better midfielder the one-transfer action can buy.
    players[100] = _player(100, 3, 45, 12.0)
    pool = PredictionPool(players)
    squad = SquadState(owned=owned, bank=50, free_transfers=1)
    return pool, squad


class EpisodeTests(unittest.TestCase):
    def test_hold_is_always_legal_and_a_used_chip_is_not(self):
        pool, squad = _pool_and_squad()
        episode = SeasonEpisode(pool, squad, lambda _player_id, _gw: 1.0, last_gameweek=3)
        mask = episode.mask()
        self.assertTrue(mask[ACTIONS.index(HOLD)])
        self.assertTrue(mask[ACTIONS.index(TRANSFER_2)])
        self.assertTrue(mask[ACTIONS.index(WILDCARD)])
        episode.step(ACTIONS.index(WILDCARD))
        self.assertFalse(episode.mask()[ACTIONS.index(WILDCARD)])
        self.assertTrue(episode.mask()[ACTIONS.index(HOLD)])

    def test_a_hit_costs_four_and_free_hit_puts_the_squad_back(self):
        pool, squad = _pool_and_squad()

        def points(player_id, _gameweek):
            return 10.0 if player_id == 100 else 1.0

        squad.free_transfers = 0
        episode = SeasonEpisode(pool, squad, points, last_gameweek=2)
        before = episode.squad.player_ids
        _obs, _reward, _done, info = episode.step(ACTIONS.index('TRANSFER_1'))
        self.assertEqual(4, info['hits'])
        self.assertNotEqual(before, episode.squad.player_ids)
        self.assertIn(100, episode.squad.player_ids)

        fresh = SeasonEpisode(pool, squad, points, last_gameweek=2)
        original = fresh.squad.player_ids
        fresh.step(ACTIONS.index(FREE_HIT))
        self.assertEqual(original, fresh.squad.player_ids)

    def test_lineup_is_picked_from_the_forecast_and_scored_on_actual_points(self):
        pool, squad = _pool_and_squad()
        pool[1].points_by_gameweek[1] = 10.0
        pool[2].points_by_gameweek[1] = 1.0

        def points(player_id, _gameweek):
            if player_id == 1:
                return 0.0
            if player_id == 2:
                return 20.0
            return 1.0

        episode = SeasonEpisode(pool, squad, points, last_gameweek=1)
        _obs, _reward, _done, info = episode.step(hold_action_index())
        self.assertEqual(10.0, info['points'])

    def test_terminal_reward_is_added_once_and_weekly_reward_stays_small(self):
        pool, squad = _pool_and_squad()
        episode = SeasonEpisode(
            pool, squad,
            lambda _player_id, _gw: 4.0,
            last_gameweek=2,
            reference_scores=lambda _gw: (0.0, 1.0),
        )
        done = False
        rewards = []
        while not done:
            _obs, reward, done, info = episode.step(hold_action_index())
            rewards.append((reward, info['terminal_reward'], info['weekly_reward']))
        self.assertEqual(0.0, rewards[0][1])
        self.assertGreater(rewards[-1][1], 0.0)
        self.assertEqual(terminal_reward(episode.points_total), rewards[-1][1])
        self.assertLess(abs(weekly_reward(10.0, (0.0, 1.0))), 0.01)
        self.assertTrue(episode.done)

    def test_observation_uses_forecasts_and_ignores_the_realised_outcome(self):
        pool, squad = _pool_and_squad()
        first = build_observation(25, 1, squad, ChipCalendar(), pool, {})
        pool[1].points_by_gameweek[1] = 50.0
        changed_forecast = build_observation(25, 1, squad, ChipCalendar(), pool, {})
        self.assertEqual(len(OBSERVATION_NAMES), len(first))
        self.assertNotEqual(first, changed_forecast)
        self.assertIn('best_5_transfer_gain_6gw', OBSERVATION_NAMES)

    def test_each_gameweek_loads_that_weeks_snapshot(self):
        pool, squad = _pool_and_squad()
        later_players = {}
        for player_id, player in pool.players.items():
            copy = _player(player_id, player.position, player.price, 2.0, player.team_id)
            if player_id == 1:
                for gameweek in range(1, 6):
                    copy.points_by_gameweek[gameweek] = 50.0
            later_players[player_id] = copy
        later = PredictionPool(later_players)
        seen = []

        def forecast_at(gameweek):
            seen.append(int(gameweek))
            if int(gameweek) == 1:
                return pool, 11
            if int(gameweek) == 2:
                return later, 22
            raise MissingPredictionSnapshot('GW%s' % gameweek)

        episode = SeasonEpisode(
            pool, squad, lambda *_args: 0.0, last_gameweek=3, forecast_at=forecast_at,
        )
        first = episode.observation()
        self.assertEqual([1], seen)
        self.assertEqual(11, episode.optimiser.snapshot_id)
        episode.step(hold_action_index())
        second = episode.observation()
        self.assertEqual([1, 2], seen)
        self.assertEqual(22, episode.optimiser.snapshot_id)
        self.assertNotEqual(first, second)
        with self.assertRaises(MissingPredictionSnapshot):
            episode.step(hold_action_index())

    def test_an_owned_player_who_leaves_the_snapshot_stays_and_scores_nothing(self):
        pool, squad = _pool_and_squad()
        later = PredictionPool({
            player_id: _player(player_id, player.position, player.price, 2.0, player.team_id)
            for player_id, player in pool.players.items()
            if player_id != 1
        })

        def forecast_at(gameweek):
            return (pool, 11) if int(gameweek) == 1 else (later, 22)

        episode = SeasonEpisode(
            pool,
            squad,
            lambda *_args: 1.0,
            last_gameweek=3,
            forecast_at=forecast_at,
            optimiser=MipOptimiser(horizon_gameweeks=5, observation_version='v2'),
            observation_version='v2',
        )
        episode.step(hold_action_index())
        observation = episode.observation()
        self.assertEqual(len(OBSERVATION_NAMES_V2), len(observation))
        self.assertIn(1, episode.squad.player_ids)
        self.assertFalse(episode.pool[1].selectable)
        self.assertEqual(0.0, episode.pool[1].points(2))
        # The solver still plans around him; here it buys the better midfielder instead.
        episode.step(ACTIONS.index(TRANSFER_1))
        self.assertIn(100, episode.squad.player_ids)

    def test_season_log_prints_each_gameweek_and_the_cache_totals(self):
        from fpl_manager.optimiser import GreedyOptimiser

        class _Cache:
            def __init__(self):
                self.hits = 4
                self.misses = 7

        class _Counting(GreedyOptimiser):
            def __init__(self):
                super().__init__()
                self.cache = _Cache()

            def execute(self, action, state, pool, gameweek):
                self.cache.misses += 1
                return super().execute(action, state, pool, gameweek)

            def transfer_gains(self, state, pool, gameweek):
                self.cache.hits += 1
                return {}

        pool, squad = _pool_and_squad()
        output = io.StringIO()
        with redirect_stdout(output):
            episode = SeasonEpisode(
                pool, squad, lambda *_args: 4.0, last_gameweek=2, optimiser=_Counting(),
            )
            episode.step(hold_action_index())
            episode.step(hold_action_index())
        lines = [line for line in output.getvalue().splitlines() if line]
        self.assertEqual(1, len(lines))
        self.assertNotIn('GW1', output.getvalue())
        total = episode.points_total
        score = str(int(total)) if total == int(total) else ('%.1f' % total)
        from fpl_manager.percentiles import describe_season
        finish = describe_season(25, total)
        if finish:
            score = '%s, %s' % (score, finish)
        self.assertRegex(
            lines[0],
            r'^Season score %s\. Squad optimisations run 2, from cache 1\. \d'
            % re.escape(score),
        )

    def test_a_multi_season_run_pays_the_sum_once_at_the_end(self):
        import fpl_manager.multi as multi
        from fpl_manager.multi import MultiSeasonRun
        from fpl_manager.reward import multi_season_reward

        multi._RUN_IDS = 0

        class _Slot:
            def __init__(self, year):
                self.season = year
                self.pool, self.squad = _pool_and_squad()

            def episode(self):
                return SeasonEpisode(
                    self.pool, self.squad, lambda *_args: 4.0,
                    season_start_year=self.season, last_gameweek=1,
                )

        log_path = Path(tempfile.mkdtemp()) / 'runs.csv'
        output = io.StringIO()
        with redirect_stdout(output):
            run = MultiSeasonRun([_Slot(23), _Slot(24)], log_path=log_path)
            _observation, first, done, _info = run.step(hold_action_index())
            _observation, second, finished, _info = run.step(hold_action_index())
        self.assertFalse(done)
        self.assertAlmostEqual(0.0, first)
        self.assertTrue(finished)
        self.assertAlmostEqual(
            multi_season_reward(run.points, run.percentiles), second,
        )
        self.assertAlmostEqual(4.5, multi_season_reward([1000, 2000], [50, 100]))
        text = output.getvalue()
        self.assertIn('Run 1, season 1/2, 2023/24', text)
        self.assertIn('Run 1, season 2/2, 2024/25', text)
        self.assertIn('Run 1 finished.', text)
        self.assertNotIn('GW1', text)
        logged = log_path.read_text(encoding='utf-8').splitlines()
        self.assertIn('23_points', logged[0])
        self.assertIn('24_percentile', logged[0])
        self.assertTrue(logged[1].startswith('1,'))
        self.assertIn(str(int(sum(run.points))), logged[1])

    def test_run_numbers_continue_after_the_finished_runs(self):
        from fpl_manager.multi import _allocate_run_id, note_completed_runs

        directory = Path(tempfile.mkdtemp())
        note_completed_runs(directory, 10)
        self.assertEqual(
            [11, 12, 13, 14, 15],
            [_allocate_run_id(directory) for _index in range(5)],
        )

    def test_mask_helper_matches_the_episode(self):
        pool, squad = _pool_and_squad()
        episode = SeasonEpisode(pool, squad, lambda *_args: 0.0, last_gameweek=1)
        self.assertEqual(
            legal_mask(25, 1, episode.played, episode.squad),
            episode.mask(),
        )


def _with_minutes(pool):
    for player in pool.players.values():
        for gameweek in range(1, 7):
            player.minutes_by_gameweek[gameweek] = 90.0
    return pool


class BaselineTests(unittest.TestCase):
    def test_hold_logs_terminal_and_weekly_rewards_separately(self):
        pool, squad = _pool_and_squad()
        episode = SeasonEpisode(
            pool, squad,
            lambda _player_id, _gw: 4.0,
            last_gameweek=2,
            reference_scores=lambda _gw: (0.0, 1.0),
        )
        result = play_hold(episode)
        self.assertEqual(terminal_reward(result['points']), result['terminal_reward'])
        self.assertEqual(episode.weekly_reward_total, result['weekly_reward'])
        self.assertLess(abs(result['weekly_reward']), 0.02)
        self.assertGreater(result['terminal_reward'], abs(result['weekly_reward']))

    def test_existing_manager_finishes_a_week_under_the_same_rules(self):
        pool, squad = _pool_and_squad()
        _with_minutes(pool)
        episode = SeasonEpisode(pool, squad, lambda _player_id, _gw: 1.0, last_gameweek=1)
        manager = ExistingManager(
            episode.season_start_year,
            horizon_gameweeks=1,
            min_horizon_gain=0.0,
            earliest_chip_gameweek=99,
            one_optimise_per_kind=True,
            forecast_future_halves=False,
        )
        result = play_existing(episode, manager)
        self.assertTrue(episode.done)
        self.assertEqual(terminal_reward(result['points']), result['terminal_reward'])
        self.assertEqual(0.0, result['weekly_reward'])


class SolverTests(unittest.TestCase):
    def test_a_hit_inside_the_objective_blocks_a_pointless_transfer(self):
        pool, squad = _pool_and_squad()
        _with_minutes(pool)
        for gameweek in range(1, 7):
            pool[100].points_by_gameweek[gameweek] = 2.2
        squad.free_transfers = 0
        plan, chip = MipOptimiser(horizon_gameweeks=1).execute(TRANSFER_1, squad, pool, 1)
        self.assertIsNone(chip)
        self.assertEqual((), plan.transfers)
        self.assertEqual(squad.player_ids, plan.selected_player_ids)

    def test_exact_cache_matches_a_fresh_solve_and_rejects_a_bad_candidate(self):
        pool, squad = _pool_and_squad()
        _with_minutes(pool)
        for gameweek in range(1, 7):
            pool[100].points_by_gameweek[gameweek] = 30.0
        pool.players[101] = _player(101, 3, 500, 1.0)
        _with_minutes(pool)
        squad.free_transfers = 0
        cache = _memory_cache()
        try:
            codes = squad_codes(pool, squad.player_ids)
            illegal = [code for code in codes if code != 1012] + [1101]
            cache.put_candidate(
                approximate_key(9, 1, 1, 1, TRANSFER_1, codes),
                {
                    'snapshot_id': 9,
                    'gameweek': 1,
                    'horizon': 1,
                    'transfer_count': 1,
                    'squad_codes': codes,
                },
                {'selected_codes': illegal, 'transfers': []},
            )
            optimiser = MipOptimiser(horizon_gameweeks=1, cache=cache, snapshot_id=9)
            plan, _chip = optimiser.execute(TRANSFER_1, squad, pool, 1)
            self.assertEqual(1, cache.rejected)
            self.assertEqual(1, cache.misses)
            self.assertIn(100, plan.selected_player_ids)
            self.assertNotIn(101, plan.selected_player_ids)
            again, _chip = optimiser.execute(TRANSFER_1, squad, pool, 1)
            self.assertEqual(plan.selected_player_ids, again.selected_player_ids)
            self.assertEqual(1, cache.hits)
            fresh, _chip = MipOptimiser(horizon_gameweeks=1, snapshot_id=9).execute(
                TRANSFER_1, squad, pool, 1,
            )
            self.assertEqual(fresh.selected_player_ids, plan.selected_player_ids)
        finally:
            _close_cache(cache)

    def test_exact_key_changes_with_snapshot_bank_and_transfer_count(self):
        pool, squad = _pool_and_squad()
        codes = squad_codes(pool, squad.player_ids)
        purchases = purchase_pairs(squad, pool)
        base = exact_key(1, 1, 6, 1, 50, 1, 25, TRANSFER_1, codes, purchases, True)
        self.assertNotEqual(
            base,
            exact_key(2, 1, 6, 1, 50, 1, 25, TRANSFER_1, codes, purchases, True),
        )
        self.assertNotEqual(
            base,
            exact_key(1, 1, 6, 1, 10, 1, 25, TRANSFER_1, codes, purchases, True),
        )
        self.assertNotEqual(
            base,
            exact_key(1, 1, 6, 2, 50, 1, 25, TRANSFER_1, codes, purchases, True),
        )

    def test_an_unaffordable_candidate_is_rejected(self):
        pool, squad = _pool_and_squad()
        pool.players[101] = _player(101, 3, 500, 1.0)
        squad.bank = 0
        selected = [player_id for player_id in squad.player_ids if player_id != 12] + [101]
        self.assertFalse(candidate_is_legal(squad, pool, selected))
        self.assertTrue(candidate_is_legal(squad, pool, squad.player_ids))


class ForecastTests(unittest.TestCase):
    def test_missing_snapshot_is_an_error(self):
        class _Empty:
            def get_historical_predictions(self, *_args):
                return []

        with self.assertRaises(MissingPredictionSnapshot):
            load_forecast_pool(_Empty(), 1, 25, 1)

    def test_a_saved_row_becomes_the_forecast(self):
        class _Rows:
            def get_historical_predictions(self, *_args):
                return [{
                    'player_id': 1,
                    'player_code': 10,
                    'player_name': 'A',
                    'position': 3,
                    'team_id': 1,
                    'price': 45,
                    'selectable': True,
                    'target_gameweek': 1,
                    'predicted_points': 4.5,
                    'predicted_minutes': 90,
                    'fixture_id': 7,
                }]

        pool = load_forecast_pool(_Rows(), 1, 25, 1)
        self.assertAlmostEqual(4.5, pool[1].points(1))


class CacheSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_path = os.environ.get('DATABASE_PATH')
        self.previous_url = os.environ.pop('DATABASE_URL', None)
        os.environ['DATABASE_PATH'] = str(Path(self.temp_dir.name) / 'test.db')
        from flask import Flask
        self.app = Flask(__name__)
        self.context = self.app.app_context()
        self.context.push()
        from persistence import DB_Handler
        self.db = DB_Handler()

    def tearDown(self):
        self.db.close_connection()
        self.context.pop()
        if self.previous_path is None:
            os.environ.pop('DATABASE_PATH', None)
        else:
            os.environ['DATABASE_PATH'] = self.previous_path
        if self.previous_url is None:
            os.environ.pop('DATABASE_URL', None)
        else:
            os.environ['DATABASE_URL'] = self.previous_url
        self.temp_dir.cleanup()

    def test_migration_keeps_an_existing_future_match(self):
        self.db.cursor.execute(
            '''create table future_matches (
                   id integer primary key autoincrement not null,
                   player_id integer not null,
                   minutes real not null,
                   points real not null,
                   gameweek integer,
                   opponent varchar(255)
               )'''
        )
        self.db.cursor.execute(
            'insert into future_matches (player_id, minutes, points, gameweek, opponent) values (1, 90, 5, 1, ?)',
            ('ABC',),
        )
        self.db.connection.commit()
        self.db.ensure_simulation_schema()
        self.db.cursor.execute('select points from future_matches where player_id = 1')
        self.assertEqual(5, self.db.cursor.fetchone()[0])
        self.db.cursor.execute(
            "select name from sqlite_master where type='table' and name='manager_exact_cache'"
        )
        self.assertEqual('manager_exact_cache', self.db.cursor.fetchone()[0])

    def test_fixed_cache_is_scoped_to_the_prediction_snapshot(self):
        pool, _squad = _pool_and_squad()
        cache = OptimiserCache(self.db)
        payload = fixed_payload(pool, 1)
        cache.put_fixed(4, 1, payload)
        self.assertEqual(payload['points'], cache.get_fixed(4, 1)['points'])
        self.assertIsNone(cache.get_fixed(5, 1))


class ScenarioTests(unittest.TestCase):
    def test_zero_expected_goals_replaces_only_the_score_components(self):
        home = PlayerFixture(
            player_id=1, team_id=1, position=3, minutes=90, total_points=7,
            goals=1, expected_goals=0.0, expected_assists=0.0,
        )
        away = PlayerFixture(
            player_id=2, team_id=2, position=1, minutes=90, total_points=2,
            expected_goals=0.0, expected_assists=0.0,
        )
        season = ScenarioSeason([
            FixtureSpec(1, 1, 1, 2, (home, away)),
        ], seed=1)
        self.assertAlmostEqual(adjust_points(home, 0, 0, 0), season.outcomes(1, 1))
        self.assertEqual((0, 0), season.scores[1])

    def test_the_same_seed_replays_goals_and_another_seed_does_not(self):
        home = PlayerFixture(
            player_id=1, team_id=1, position=4, minutes=90, total_points=2,
            expected_goals=2.5, expected_assists=0.2,
        )
        away = PlayerFixture(
            player_id=2, team_id=2, position=1, minutes=90, total_points=2,
            expected_goals=0.0, expected_assists=0.0,
        )
        fixture = FixtureSpec(1, 1, 1, 2, (home, away))
        first = ScenarioSeason([fixture], seed=1)
        replay = ScenarioSeason([fixture], seed=1)
        other = ScenarioSeason([fixture], seed=2)
        first.outcomes(1, 1)
        replay.outcomes(1, 1)
        other.outcomes(1, 1)
        self.assertEqual(first.scores, replay.scores)
        self.assertEqual((2, 0), first.scores[1])
        self.assertEqual(1, other.scores[1][0])

    def test_a_missing_expected_goals_value_fails_the_fixture(self):
        home = PlayerFixture(
            player_id=1, team_id=1, position=4, minutes=90, total_points=2,
            expected_goals=None,
        )
        away = PlayerFixture(
            player_id=2, team_id=2, position=1, minutes=90, total_points=2,
            expected_goals=0.0,
        )
        season = ScenarioSeason([FixtureSpec(1, 1, 1, 2, (home, away))], seed=1)
        with self.assertRaises(MissingMatchXG):
            season.outcomes(1, 1)

    def test_two_episodes_on_one_scenario_see_the_same_points(self):
        home = PlayerFixture(
            player_id=1, team_id=1, position=3, minutes=90, total_points=2,
            expected_goals=0.0, expected_assists=0.0,
        )
        away = PlayerFixture(
            player_id=2, team_id=2, position=1, minutes=90, total_points=2,
            expected_goals=0.0, expected_assists=0.0,
        )
        shared = ScenarioSeason([FixtureSpec(1, 1, 1, 2, (home, away))], seed=4)
        pool, squad = _pool_and_squad()

        def play():
            episode = SeasonEpisode(pool, squad, shared.outcomes, last_gameweek=1)
            episode.step(hold_action_index())
            return episode.points_total

        self.assertEqual(play(), play())


class CheckpointTests(unittest.TestCase):
    def test_a_new_process_keeps_the_weights_and_the_step_counter(self):
        from fpl_manager.evaluate import play_policy
        from fpl_manager.train import (
            CheckpointMismatch,
            load_checkpoint,
            policy_weight_sum,
            tiny_hold_factory,
            train_policy,
        )

        directory = tempfile.mkdtemp()
        model = train_policy(
            tiny_hold_factory,
            directory,
            total_timesteps=2,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            seed=1,
        )
        steps = int(model.num_timesteps)
        weight = policy_weight_sum(model)
        played = play_policy(tiny_hold_factory(), model)
        self.assertGreater(played['terminal_reward'], 0.0)
        script = (
            'import json\n'
            'from fpl_manager.train import load_checkpoint, policy_weight_sum\n'
            'model, manifest = load_checkpoint(%r)\n'
            'print(json.dumps({"steps": int(model.num_timesteps), '
            '"weight": policy_weight_sum(model), '
            '"manifest_steps": int(manifest["num_timesteps"])}))\n'
        ) % directory
        completed = subprocess.run(
            [sys.executable, '-c', script],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(steps, payload['steps'])
        self.assertEqual(steps, payload['manifest_steps'])
        self.assertAlmostEqual(weight, payload['weight'], places=4)

        with self.assertRaises(CheckpointMismatch):
            train_policy(tiny_hold_factory, directory, total_timesteps=2)

        continued = train_policy(
            tiny_hold_factory,
            directory,
            total_timesteps=2,
            resume=True,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
        )
        self.assertGreater(int(continued.num_timesteps), steps)

        reloaded, _manifest = load_checkpoint(directory)
        first = __import__('numpy').random.random()
        load_checkpoint(directory)
        second = __import__('numpy').random.random()
        self.assertEqual(first, second)
        self.assertEqual(int(reloaded.num_timesteps), int(continued.num_timesteps))

        manifest_path = Path(directory) / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['features'] = ['not-a-real-field']
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(CheckpointMismatch):
            load_checkpoint(directory)

    def test_two_processes_step_one_shared_environment_api(self):
        from sb3_contrib.common.maskable.utils import is_masking_supported

        from fpl_manager.train import _process_envs, tiny_vec_env

        env = _process_envs([tiny_vec_env, tiny_vec_env])
        try:
            env.reset()
            self.assertTrue(is_masking_supported(env))
            masks = env.env_method('action_masks')
            self.assertEqual(2, len(masks))
            self.assertEqual(2, len(env.step([0, 0])[0]))
        finally:
            env.close()


class ObservationV2Tests(unittest.TestCase):
    def test_v2_columns_are_a_separate_config_from_v1(self):
        self.assertEqual('v1', version_for_features(OBSERVATION_NAMES))
        self.assertEqual('v2', version_for_features(OBSERVATION_NAMES_V2))
        self.assertEqual('', version_for_features(['not-a-real-field']))
        self.assertEqual(33, len(OBSERVATION_NAMES_V2))
        self.assertNotIn('best_5_transfer_gain_6gw', OBSERVATION_NAMES_V2)
        self.assertIn('wildcard_in_5_gain_10', OBSERVATION_NAMES_V2)

    def test_v2_observation_scales_the_plan_gains(self):
        pool, squad = _pool_and_squad()
        observed = build_observation_v2(
            25, 1, squad, ChipCalendar(), pool,
            {'roll_gain_5': 50.0, 'wildcard_in_5_gain_10': -25.0},
        )
        self.assertEqual(len(OBSERVATION_NAMES_V2), len(observed))
        names = list(OBSERVATION_NAMES_V2)
        self.assertAlmostEqual(0.5, observed[names.index('roll_gain_5')])
        self.assertAlmostEqual(-0.25, observed[names.index('wildcard_in_5_gain_10')])
        self.assertEqual(0.0, observed[names.index('free_hit_gain_5')])

    def test_v2_masks_a_transfer_that_would_take_a_hit(self):
        pool, squad = _pool_and_squad()
        episode = SeasonEpisode(
            pool, squad, lambda _player_id, _gw: 1.0, observation_version='v2',
        )
        mask = episode.mask()
        self.assertTrue(mask[ACTIONS.index(TRANSFER_1)])
        self.assertFalse(mask[ACTIONS.index(TRANSFER_2)])
        self.assertTrue(mask[ACTIONS.index(HOLD)])

    def test_v2_cache_key_does_not_match_a_v1_squad(self):
        pool, squad = _pool_and_squad()
        codes = squad_codes(pool, squad.player_ids)
        purchases = purchase_pairs(squad, pool)
        v1_exact = exact_key(1, 1, 5, 1, 50, 1, 25, TRANSFER_1, codes, purchases, False)
        v2_exact = exact_key(
            1, 1, 5, 1, 50, 1, 25, TRANSFER_1, codes, purchases, False,
            rules_version=RULES_VERSION_V2,
        )
        self.assertNotEqual(v1_exact, v2_exact)
        v1_approx = approximate_key(1, 1, 5, 1, TRANSFER_1, codes)
        v2_approx = approximate_key(
            1, 1, 5, 1, TRANSFER_1, codes, rules_version=RULES_VERSION_V2,
        )
        self.assertNotEqual(v1_approx, v2_approx)
        self.assertEqual(v1_approx, approximate_key(1, 1, 5, 1, TRANSFER_1, codes))

    def test_roll_ladder_stops_at_four_and_at_the_2023_bank(self):
        from squadtools.simulation_strategy import FreeTransferRules
        from fpl_manager.plan_v2 import roll_steps

        modern = FreeTransferRules(25)
        steps = roll_steps(modern, 10, 1, (10, 11, 12, 13, 14))
        self.assertEqual((10, 11, 12, 13), tuple(step[0] for step in steps))
        self.assertEqual((1, 2, 3, 4), tuple(step[1] for step in steps))

        early = FreeTransferRules(23)
        early_steps = roll_steps(early, 10, 1, (10, 11, 12, 13, 14))
        self.assertEqual((1, 2), tuple(step[1] for step in early_steps))

        banked = roll_steps(modern, 16, 1, (16, 17, 18, 19, 20))
        self.assertEqual(((16, 5),), tuple((step[0], step[1]) for step in banked))

    def test_v2_plans_do_not_charge_hits_and_the_ten_week_wildcard_is_separate(self):
        from fpl_manager.mip import MipOptimiser, _plan_from_selected

        class _Recording(MipOptimiser):
            def __init__(self):
                super().__init__(observation_version='v2', season_start_year=25)
                self.calls = []

            def _solve(
                self, action, state, pool, gameweek, max_transfers, horizon,
                charge_hits, chip, gameweeks=None, rules_version=None, value_band=False,
            ):
                weeks = tuple(gameweeks or ())
                self.calls.append({
                    'action': action,
                    'gameweek': int(gameweek),
                    'weeks': weeks,
                    'hits': bool(charge_hits),
                    'rules': rules_version,
                })
                return _plan_from_selected(
                    state, pool, gameweek, len(weeks) or 1, state.player_ids, chip,
                )

        pool, squad = _pool_and_squad()
        optimiser = _Recording()
        short = optimiser.transfer_gains(squad, pool, 34, ChipCalendar())
        self.assertEqual(0.0, short['wildcard_in_5_gain_10'])
        self.assertFalse(any(call['gameweek'] >= 39 for call in optimiser.calls))
        self.assertTrue(optimiser.calls)
        self.assertTrue(all(call['rules'] == RULES_VERSION_V2 for call in optimiser.calls))
        self.assertTrue(all(call['hits'] is False for call in optimiser.calls))

        optimiser.calls.clear()
        long = optimiser.transfer_gains(squad, pool, 30, ChipCalendar())
        self.assertIn('roll_gain_5', long)
        self.assertIn('wildcard_gain_5', long)
        self.assertIn('free_hit_gain_5', long)
        long_wildcards = [
            call for call in optimiser.calls
            if call['action'] == 'WILDCARD'
            and call['gameweek'] == 35
            and call['weeks'] == (35, 36, 37, 38)
        ]
        self.assertEqual(1, len(long_wildcards))

    def test_a_v1_checkpoint_refuses_a_v2_launch(self):
        from fpl_manager.__main__ import _observation_version
        from fpl_manager.train import CheckpointMismatch

        directory = tempfile.mkdtemp()
        (Path(directory) / 'manifest.json').write_text(json.dumps({
            'features': list(OBSERVATION_NAMES),
        }))
        self.assertEqual('v1', _observation_version(directory, None))
        self.assertEqual('v1', _observation_version(directory, 'v1'))
        with self.assertRaises(CheckpointMismatch):
            _observation_version(directory, 'v2')
        self.assertEqual('v2', _observation_version(tempfile.mkdtemp(), 'v2'))

    def test_a_roll_scores_the_new_squad_from_the_spend_week(self):
        from fpl_manager.plan_v2 import _ids_with_chip, _rolled_points

        pool, squad = _pool_and_squad()
        base = squad.player_ids
        better = tuple(100 if player_id == 8 else player_id for player_id in base)
        self.assertGreater(
            _rolled_points(base, better, pool, (1, 2), 2),
            _rolled_points(base, base, pool, (1, 2), 2),
        )
        chosen = _ids_with_chip(
            (1, 2, 3), base, 2, better, None, None, 3, (100,),
        )
        self.assertEqual(base, chosen[1])
        self.assertEqual(better, chosen[2])
        self.assertEqual((100,), chosen[3])


class ObservationFastTests(unittest.TestCase):
    def test_fast_columns_keep_the_v2_context_and_two_gains(self):
        self.assertEqual('fast', version_for_features(OBSERVATION_NAMES_FAST))
        self.assertEqual(OBSERVATION_NAMES_V2[:29], OBSERVATION_NAMES_FAST[:-2])
        self.assertEqual(
            ('roll_1_gain_5', 'wildcard_now_gain_5'),
            OBSERVATION_NAMES_FAST[-2:],
        )
        pool, squad = _pool_and_squad()
        observed = build_observation_fast(
            25, 1, squad, ChipCalendar(), pool,
            {'roll_1_gain_5': 50.0, 'wildcard_now_gain_5': -25.0},
        )
        names = list(OBSERVATION_NAMES_FAST)
        self.assertEqual(len(names), len(observed))
        self.assertAlmostEqual(0.5, observed[names.index('roll_1_gain_5')])
        self.assertAlmostEqual(-0.25, observed[names.index('wildcard_now_gain_5')])

    def test_fast_masks_a_transfer_that_would_take_a_hit(self):
        pool, squad = _pool_and_squad()
        episode = SeasonEpisode(
            pool, squad, lambda _player_id, _gw: 1.0, observation_version='fast',
        )
        mask = episode.mask()
        self.assertTrue(mask[ACTIONS.index(TRANSFER_1)])
        self.assertFalse(mask[ACTIONS.index(TRANSFER_2)])

    def test_a_saved_v2_config_refuses_the_fast_launch(self):
        from fpl_manager.__main__ import _observation_version
        from fpl_manager.train import CheckpointMismatch

        directory = tempfile.mkdtemp()
        (Path(directory) / 'manifest.json').write_text(json.dumps({
            'features': list(OBSERVATION_NAMES_V2),
        }))
        with self.assertRaises(CheckpointMismatch):
            _observation_version(directory, 'fast')
        self.assertEqual('fast', _observation_version(tempfile.mkdtemp(), 'fast'))

    def test_half_a_million_is_close_enough_to_reuse_a_squad(self):
        owned = OwnedPlayer(1, 50)
        state = SquadState(owned={1: owned}, bank=10, free_transfers=1)
        pool = PredictionPool({1: _player(1, 3, 60, 2.0)})
        self.assertEqual(65, team_value(state, pool))
        self.assertTrue(team_values_close(team_value(state, pool), 70))
        self.assertFalse(team_values_close(team_value(state, pool), 71))
        self.assertFalse(team_values_close(team_value(state, pool), None))

    def test_fast_prices_one_roll_and_a_wildcard_played_now(self):
        from fpl_manager.mip import MipOptimiser, _plan_from_selected

        class _Recording(MipOptimiser):
            def __init__(self):
                super().__init__(observation_version='fast', season_start_year=25)
                self.calls = []

            def _solve(
                self, action, state, pool, gameweek, max_transfers, horizon,
                charge_hits, chip, gameweeks=None, rules_version=None, value_band=False,
            ):
                weeks = tuple(gameweeks or ())
                self.calls.append({
                    'action': action,
                    'gameweek': int(gameweek),
                    'weeks': weeks,
                    'transfers': int(max_transfers),
                    'hits': bool(charge_hits),
                    'rules': rules_version,
                    'band': bool(value_band),
                })
                return _plan_from_selected(
                    state, pool, gameweek, len(weeks) or 1, state.player_ids, chip,
                )

        pool, squad = _pool_and_squad()
        optimiser = _Recording()
        gains = optimiser.transfer_gains(squad, pool, 10, ChipCalendar())
        self.assertEqual({'roll_1_gain_5', 'wildcard_now_gain_5'}, set(gains))
        rolls = [call for call in optimiser.calls if call['action'].startswith('TRANSFER')]
        wildcards = [call for call in optimiser.calls if call['action'] == 'WILDCARD']
        self.assertEqual(1, len(rolls))
        self.assertEqual(1, len(wildcards))
        self.assertEqual(11, rolls[0]['gameweek'])
        self.assertEqual(2, rolls[0]['transfers'])
        self.assertEqual((11, 12, 13, 14), rolls[0]['weeks'])
        self.assertEqual(10, wildcards[0]['gameweek'])
        self.assertEqual((10, 11, 12, 13, 14), wildcards[0]['weeks'])
        self.assertTrue(all(call['rules'] == RULES_VERSION_FAST for call in optimiser.calls))
        self.assertTrue(all(call['hits'] is False for call in optimiser.calls))
        self.assertTrue(all(call['band'] is True for call in optimiser.calls))

        optimiser.calls.clear()
        final = optimiser.transfer_gains(squad, pool, 38, ChipCalendar())
        self.assertEqual(0.0, final['roll_1_gain_5'])
        self.assertFalse(any(call['action'].startswith('TRANSFER') for call in optimiser.calls))
        self.assertFalse(any(call['gameweek'] > 38 for call in optimiser.calls))

    def test_execute_holds_when_the_solver_fails(self):
        class _Failing(MipOptimiser):
            def _solve(self, *args, **kwargs):
                raise ValueError('CBC failed')

        pool, squad = _pool_and_squad()
        plan, chip = _Failing(observation_version='fast').execute(
            TRANSFER_1, squad, pool, 1,
        )
        self.assertEqual((), plan.transfers)
        self.assertEqual(squad.player_ids, plan.selected_player_ids)
        self.assertIsNone(chip)


def _memory_cache():
    """SQLite cache on a throwaway file, without the live database."""
    from flask import Flask
    from persistence import DB_Handler

    directory = tempfile.TemporaryDirectory()
    previous_path = os.environ.get('DATABASE_PATH')
    previous_url = os.environ.pop('DATABASE_URL', None)
    os.environ['DATABASE_PATH'] = str(Path(directory.name) / 'cache.db')
    app = Flask(__name__)
    context = app.app_context()
    context.push()
    db = DB_Handler()
    cache = OptimiserCache(db)
    cache._cleanup = (directory, context, db, previous_path, previous_url)
    return cache


def _close_cache(cache):
    directory, context, db, previous_path, previous_url = cache._cleanup
    db.close_connection()
    context.pop()
    if previous_path is None:
        os.environ.pop('DATABASE_PATH', None)
    else:
        os.environ['DATABASE_PATH'] = previous_path
    if previous_url is None:
        os.environ.pop('DATABASE_URL', None)
    else:
        os.environ['DATABASE_URL'] = previous_url
    directory.cleanup()


if __name__ == '__main__':
    unittest.main()
