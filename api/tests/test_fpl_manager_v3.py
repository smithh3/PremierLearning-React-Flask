"""v3 manager: in-process solver, range cache, inputs, autosubs, per-season reward."""
import csv
import tempfile
import unittest
from pathlib import Path

from squadtools.simulation_strategy import (
    CHIP_WILDCARD,
    ChipCalendar,
    OwnedPlayer,
    PlayerProjection,
    PredictionPool,
    SquadState,
)

from fpl_manager import highs_solver
from fpl_manager.actions import ACTIONS, HOLD, TRANSFER_1, WILDCARD
from fpl_manager.cache import RULES_VERSION_V3
from fpl_manager.episode import SeasonEpisode
from fpl_manager.lineup import score_with_autosubs
from fpl_manager.mip import MipOptimiser, _horizon_points, _solve_squad
from fpl_manager.multi import EVALUATION_LOG, MultiSeasonRun, SeasonByRun
from fpl_manager.observation import (
    OBSERVATION_NAMES_V2,
    OBSERVATION_NAMES_V3,
    POINT_SCALE,
    build_observation_v3,
    features_for,
    version_for_features,
)
from fpl_manager.optimiser import GreedyOptimiser
from fpl_manager.reward import (
    WEEKLY_POINTS_WEIGHT,
    season_end_reward,
    weekly_points_reward,
)
from tests.test_fpl_manager import _close_cache, _memory_cache


def _player(player_id, position, price, points, team_id=None, weeks=range(1, 11)):
    player = PlayerProjection(
        player_id=player_id,
        player_code=1000 + player_id,
        name='p%s' % player_id,
        position=position,
        team_id=player_id if team_id is None else team_id,
        price=price,
        selectable=True,
    )
    for gameweek in weeks:
        player.points_by_gameweek[gameweek] = float(points)
        player.fixture_ids_by_gameweek[gameweek] = {gameweek}
    return player


def _pool(extra=()):
    """Fifteen owned 2-point players at £4.5m plus cheap and strong spares."""
    counts = {1: 2, 2: 5, 3: 5, 4: 3}
    players = {}
    owned = {}
    player_id = 1
    for position, count in counts.items():
        for _index in range(count):
            players[player_id] = _player(player_id, position, 45, 2.0)
            owned[player_id] = OwnedPlayer(player_id, 45)
            player_id += 1
    # Spares: a strong midfielder and forward, and cheap fillers per position.
    players[100] = _player(100, 3, 50, 12.0)
    players[101] = _player(101, 4, 50, 9.0)
    spare = 200
    for position, count in counts.items():
        for _index in range(count):
            players[spare] = _player(spare, position, 40, 1.0)
            spare += 1
    for player in extra:
        players[player.player_id] = player
    return PredictionPool(players), SquadState(owned=owned, bank=20, free_transfers=1)


class HighsSolverTests(unittest.TestCase):
    def setUp(self):
        if not highs_solver.available():
            self.skipTest('highspy is not installed')

    def _both(self, state, pool, max_transfers, chip=None, weeks=(1, 2, 3, 4, 5)):
        from squadtools.simulation_strategy import optimiser_candidate_ids, selling_price

        cbc = _solve_squad(state, pool, weeks[0], len(weeks), max_transfers, False, chip, 25, gameweeks=weeks)
        current = set(state.player_ids)
        candidates = optimiser_candidate_ids(pool, current, weeks)
        costs = {
            player_id: (
                selling_price(state.owned[player_id].purchase_price, pool[player_id].price)
                if player_id in current else int(pool[player_id].price)
            )
            for player_id in candidates
        }
        budget = state.bank + sum(costs[player_id] for player_id in current)
        highs = highs_solver.solve_squad(
            pool, weeks, candidates, costs, budget, keep_ids=current, max_transfers=max_transfers,
        )
        return cbc, highs

    def test_highs_matches_cbc_points_for_transfers_and_a_wildcard(self):
        pool, state = _pool()
        for max_transfers, chip in ((1, None), (2, None), (15, 'wildcard')):
            cbc, highs = self._both(state, pool, max_transfers, chip)
            self.assertAlmostEqual(
                _horizon_points(cbc, pool, 1, 5, None),
                _horizon_points(highs, pool, 1, 5, None),
                places=6,
            )
        _cbc, one = self._both(state, pool, 1)
        self.assertIn(100, one)

    def test_highs_respects_the_bank(self):
        pool, state = _pool()
        poor = SquadState(owned=state.owned, bank=0, free_transfers=1)
        _cbc, highs = self._both(poor, pool, 1)
        # 100 costs 5.0 and every owned player sells for 4.5: unaffordable.
        self.assertNotIn(100, highs)

    def test_a_dominated_player_is_dropped_only_with_enough_clubs(self):
        pool, _state = _pool()
        weeks = (1, 2)
        ids = list(pool.players)
        costs = {player_id: int(pool[player_id].price) for player_id in ids}
        kept = highs_solver.prune_dominated(pool, ids, weeks, costs, keep_ids=())
        # Forwards: 3 places, so 7 dominating clubs are needed. There are only
        # a handful of forwards, so nobody can be dropped.
        forwards = [player_id for player_id in ids if pool[player_id].position == 4]
        self.assertTrue(set(forwards) <= set(kept))

    def test_of_identical_players_one_always_stays(self):
        clones = [_player(500 + index, 2, 40, 0.0, team_id=50 + index) for index in range(12)]
        pool, _state = _pool(clones)
        ids = [player.player_id for player in clones]
        costs = {player_id: 40 for player_id in ids}
        kept = highs_solver.prune_dominated(pool, ids, (1,), costs)
        self.assertIn(500, kept)
        self.assertLess(len(kept), len(ids))

    def test_owned_players_are_never_dropped(self):
        clones = [_player(500 + index, 2, 40, 0.0, team_id=50 + index) for index in range(12)]
        pool, _state = _pool(clones)
        ids = [player.player_id for player in clones]
        costs = {player_id: 40 for player_id in ids}
        kept = highs_solver.prune_dominated(pool, ids, (1,), costs, keep_ids=(511,))
        self.assertIn(511, kept)


class RangeCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = _memory_cache()

    def tearDown(self):
        _close_cache(self.cache)

    def test_a_stored_answer_covers_its_range_only(self):
        self.cache.put_range('k', 990, 1000, {'selected_codes': [1]}, 0.1)
        self.assertEqual({'selected_codes': [1]}, self.cache.get_range('k', 995))
        self.assertIsNone(self.cache.get_range('k', 1001))
        self.assertIsNone(self.cache.get_range('k', 989))
        self.assertIsNone(self.cache.get_range('other', 995))

    def test_answers_reach_the_database_when_flushed(self):
        self.cache.put_range('k', 10, 20, {'selected_codes': [2]}, 0.1)
        self.cache.flush()
        self.cache._ranges = {}
        self.assertEqual({'selected_codes': [2]}, self.cache.get_range('k', 15))

    def _optimiser(self, solver='cbc'):
        return MipOptimiser(
            horizon_gameweeks=5, cache=self.cache, snapshot_id=7, season_start_year=25,
            observation_version='v3', solver=solver, ranged=True,
        )

    def test_a_wildcard_squad_is_reused_for_another_squad_with_the_same_budget(self):
        pool, state = _pool()
        optimiser = self._optimiser()
        weeks = (1, 2, 3, 4, 5)
        first = optimiser._solve(
            WILDCARD, state, pool, 1, 15, 5, False, CHIP_WILDCARD,
            gameweeks=weeks, rules_version=RULES_VERSION_V3,
        )
        self.assertEqual((1, 0), (self.cache.misses, self.cache.hits))
        # Swap one owned 4.5 defender for a different 4.5 defender: same money.
        other = dict(state.owned)
        other.pop(3)
        swapped = _player(600, 2, 45, 2.0)
        pool.players[600] = swapped
        other[600] = OwnedPlayer(600, 45)
        second_state = SquadState(owned=other, bank=state.bank, free_transfers=1)
        second = optimiser._solve(
            WILDCARD, second_state, pool, 1, 15, 5, False, CHIP_WILDCARD,
            gameweeks=weeks, rules_version=RULES_VERSION_V3,
        )
        self.assertEqual((1, 1), (self.cache.misses, self.cache.hits))
        self.assertEqual(first.selected_player_ids, second.selected_player_ids)

    def test_a_player_who_rose_since_purchase_is_part_of_the_squad_key(self):
        pool, state = _pool()
        optimiser = self._optimiser()
        weeks = (1, 2, 3, 4, 5)
        optimiser._solve(
            WILDCARD, state, pool, 1, 15, 5, False, CHIP_WILDCARD,
            gameweeks=weeks, rules_version=RULES_VERSION_V3,
        )
        risen = dict(state.owned)
        risen[1] = OwnedPlayer(1, 40)  # bought at 4.0, now 4.5: sells for 4.2
        cheaper = SquadState(owned=risen, bank=state.bank + 3, free_transfers=1)
        optimiser._solve(
            WILDCARD, cheaper, pool, 1, 15, 5, False, CHIP_WILDCARD,
            gameweeks=weeks, rules_version=RULES_VERSION_V3,
        )
        self.assertEqual((2, 0), (self.cache.misses, self.cache.hits))

    def test_a_transfer_plan_covers_banks_between_need_and_the_solved_bank(self):
        pool, state = _pool()
        optimiser = self._optimiser()
        weeks = (1, 2, 3, 4, 5)
        plan = optimiser._solve(
            TRANSFER_1, state, pool, 1, 1, 5, False, None,
            gameweeks=weeks, rules_version=RULES_VERSION_V3,
        )
        self.assertIn(100, plan.selected_player_ids)
        # Buying 100 (5.0) after selling a 4.5 needs 0.5 in the bank.
        enough = SquadState(owned=state.owned, bank=5, free_transfers=1)
        optimiser._solve(
            TRANSFER_1, enough, pool, 1, 1, 5, False, None,
            gameweeks=weeks, rules_version=RULES_VERSION_V3,
        )
        self.assertEqual((1, 1), (self.cache.misses, self.cache.hits))
        short = SquadState(owned=state.owned, bank=4, free_transfers=1)
        optimiser._solve(
            TRANSFER_1, short, pool, 1, 1, 5, False, None,
            gameweeks=weeks, rules_version=RULES_VERSION_V3,
        )
        self.assertEqual((2, 1), (self.cache.misses, self.cache.hits))

    def test_highs_and_cbc_give_the_same_ranged_points(self):
        if not highs_solver.available():
            self.skipTest('highspy is not installed')
        pool, state = _pool()
        weeks = (1, 2, 3, 4, 5)
        answers = []
        for solver in ('cbc', 'highs'):
            optimiser = self._optimiser(solver)
            optimiser.snapshot_id = 8 if solver == 'cbc' else 9
            plan = optimiser._solve(
                WILDCARD, state, pool, 1, 15, 5, False, CHIP_WILDCARD,
                gameweeks=weeks, rules_version=RULES_VERSION_V3,
            )
            answers.append(_horizon_points(plan.selected_player_ids, pool, 1, 5, None))
        self.assertAlmostEqual(answers[0], answers[1], places=6)


class AutosubTests(unittest.TestCase):
    def setUp(self):
        self.position = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4,
                         12: 1, 13: 2, 14: 3, 15: 4}
        self.starters = tuple(range(1, 12))
        self.bench = (14, 13, 15, 12)
        self.points = {player_id: 2.0 for player_id in range(1, 16)}

    def _score(self, played, captain=5, vice=6, bench_boost=False, triple=False, points=None):
        points = points or self.points
        return score_with_autosubs(
            lambda player_id: points[player_id],
            lambda player_id: player_id in played,
            lambda player_id: self.position[player_id],
            self.starters, self.bench, captain, vice, bench_boost, triple,
        )

    def test_everyone_played_is_the_xi_plus_captain(self):
        self.assertEqual(24.0, self._score(set(range(1, 16))))

    def test_a_missing_midfielder_is_replaced_by_the_first_bench_player(self):
        played = set(range(1, 16)) - {9}
        points = {**self.points, 14: 7.0, 9: 0.0}
        self.assertEqual(2.0 * 10 + 7.0 + 2.0, self._score(played, points=points))

    def test_a_keeper_is_only_replaced_by_the_bench_keeper(self):
        played = set(range(1, 16)) - {1}
        points = {**self.points, 1: 0.0, 12: 3.0}
        self.assertEqual(2.0 * 10 + 3.0 + 2.0, self._score(played, points=points))

    def test_a_sub_that_breaks_the_formation_is_skipped(self):
        # Three defenders: losing one needs a defender, so the midfielder on
        # the bench is skipped for the defender behind him.
        played = set(range(1, 16)) - {2}
        points = {**self.points, 2: 0.0, 14: 9.0, 13: 4.0}
        self.assertEqual(2.0 * 10 + 4.0 + 2.0, self._score(played, points=points))

    def test_the_vice_takes_the_armband_and_the_triple(self):
        played = set(range(1, 16)) - {5}
        points = {**self.points, 5: 0.0, 6: 5.0, 14: 1.0}
        # XI: nine 2s, the vice's 5, the sub's 1; vice counted three times on TC.
        self.assertEqual(2.0 * 9 + 5.0 + 1.0 + 2 * 5.0, self._score(played, triple=True, points=points))

    def test_bench_boost_counts_everyone_and_makes_no_subs(self):
        played = set(range(1, 16)) - {9}
        points = {**self.points, 9: 0.0}
        self.assertEqual(2.0 * 14 + 2.0, self._score(played, bench_boost=True, points=points))


class RewardTests(unittest.TestCase):
    def test_weekly_points_are_a_quarter_of_the_season_scale(self):
        self.assertAlmostEqual(WEEKLY_POINTS_WEIGHT * 60 / 1000.0, weekly_points_reward(60))

    def test_the_season_end_outweighs_the_weeks(self):
        season_points = 2100.0
        weekly_total = weekly_points_reward(season_points)
        end = season_end_reward(season_points, 70.0)
        self.assertGreater(end, 4 * weekly_total)
        self.assertAlmostEqual(2.1 + 0.7, end)


class ObservationV3Tests(unittest.TestCase):
    def test_v3_is_its_own_config_and_keeps_the_v2_context(self):
        self.assertEqual('v3', version_for_features(OBSERVATION_NAMES_V3))
        self.assertEqual(42, len(OBSERVATION_NAMES_V3))
        self.assertEqual(OBSERVATION_NAMES_V2[:29], OBSERVATION_NAMES_V3[:29])
        self.assertEqual(OBSERVATION_NAMES_V3, features_for('v3'))

    def test_chip_values_compare_this_week_with_the_best_week(self):
        pool, squad = _pool()
        # Make GW3 the big bench week: the four bench players score 6 there.
        for player_id in (1, 2, 3, 6):
            pool.players[player_id].points_by_gameweek[3] = 6.0
        observed = build_observation_v3(25, 1, squad, ChipCalendar(), pool, {})
        values = dict(zip(OBSERVATION_NAMES_V3, observed))
        self.assertEqual(len(OBSERVATION_NAMES_V3), len(observed))
        self.assertLess(values['bench_boost_now'], values['bench_boost_best'])
        self.assertAlmostEqual(2.0 / POINT_SCALE, values['triple_captain_now'])
        self.assertEqual(0.0, values['xi_minutes_shortfall'])

    def test_minutes_shortfall_counts_starters_not_expected_to_play(self):
        pool, squad = _pool()
        for player in pool.players.values():
            player.minutes_by_gameweek[1] = 90.0
        pool.players[7].minutes_by_gameweek[1] = 0.0
        observed = build_observation_v3(25, 1, squad, ChipCalendar(), pool, {})
        values = dict(zip(OBSERVATION_NAMES_V3, observed))
        self.assertAlmostEqual(1.0 / 11.0, values['xi_minutes_shortfall'])

    def test_v3_gains_come_from_the_real_optimiser(self):
        cache = _memory_cache()
        try:
            pool, squad = _pool()
            optimiser = MipOptimiser(
                horizon_gameweeks=5, cache=cache, snapshot_id=3, season_start_year=25,
                observation_version='v3', solver='cbc', ranged=True,
            )
            from fpl_manager.plan_v2 import _chip_legal

            gains = optimiser.transfer_gains(squad, pool, 2, ChipCalendar())
            self.assertGreater(gains['transfer_1_gain_5'], 0.0)
            self.assertGreaterEqual(gains['spend_now_gain_5'], gains['transfer_1_gain_5'])
            self.assertGreaterEqual(gains['roll_gain_5'], gains['spend_now_gain_5'])
            self.assertGreaterEqual(gains['wildcard_gain_5'], gains['wildcard_now_gain_5'])
            if _chip_legal(25, ChipCalendar(), CHIP_WILDCARD, 2):
                self.assertGreater(gains['wildcard_now_gain_5'], 0.0)
            solved = cache.misses
            again = optimiser.transfer_gains(squad, pool, 2, ChipCalendar())
            self.assertEqual(solved, cache.misses)
            self.assertEqual(gains, again)
        finally:
            _close_cache(cache)


class _FakeSeason:
    """Two-gameweek season on the toy pool, for run and evaluation plumbing."""

    def __init__(self, year):
        self.season = year
        self.first = 1
        self.last = 2
        self.pool, self.squad = _pool()

    def episode(self):
        return SeasonEpisode(
            self.pool, self.squad, lambda _player_id, _gw: 2.0,
            season_start_year=self.season, first_gameweek=1, last_gameweek=2,
            optimiser=GreedyOptimiser(), observation_version='v3',
            appearances=lambda _player_id, _gw: True,
        )


class PerSeasonRunTests(unittest.TestCase):
    def test_each_season_ends_its_own_episode_and_the_run_logs_once(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / 'runs.csv'
            factory = SeasonByRun([_FakeSeason(23), _FakeSeason(24)], log_path=log_path)
            run = factory()
            hold = ACTIONS.index(HOLD)
            run.step(hold)
            _obs, reward, done, info = run.step(hold)
            self.assertTrue(done)
            self.assertFalse(info['run_finished'])
            self.assertGreater(reward, weekly_points_reward(24))
            self.assertIs(run, factory())
            run.step(hold)
            _obs, _reward, done, info = run.step(hold)
            self.assertTrue(done and info['run_finished'])
            with log_path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(1, len(rows))
            self.assertIsNot(run, factory())

    def test_an_evaluation_run_logs_to_its_own_file(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / 'runs.csv'
            run = MultiSeasonRun([_FakeSeason(23)], log_path=log_path, per_season=True)
            self.assertTrue(run.at_start)
            run.evaluation = True
            run.step(ACTIONS.index(HOLD))
            run.step(ACTIONS.index(HOLD))
            self.assertFalse(log_path.is_file())
            self.assertTrue((Path(directory) / EVALUATION_LOG).is_file())

    def test_the_monitor_plays_an_evaluation_run_to_the_end(self):
        from fpl_manager.env import ManagerEnv, masked_monitor

        with tempfile.TemporaryDirectory() as directory:
            factory = SeasonByRun([_FakeSeason(23), _FakeSeason(24)], log_path=Path(directory) / 'runs.csv')
            env = masked_monitor(ManagerEnv(factory, OBSERVATION_NAMES_V3))
            env.reset()
            self.assertTrue(env.mark_evaluation())
            finished = False
            steps = 0
            while not finished:
                _observation, _mask, finished = env.evaluation_step(ACTIONS.index(HOLD))
                steps += 1
            self.assertEqual(4, steps)
            env.restart()
            self.assertTrue((Path(directory) / EVALUATION_LOG).is_file())


class EvaluationDuringTrainingTests(unittest.TestCase):
    def test_an_update_is_followed_by_a_top_choice_run(self):
        from fpl_manager.train import train_policy

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'v3_tiny'
            checkpoint.mkdir()
            factory = SeasonByRun(
                [_FakeSeason(23), _FakeSeason(24)], log_path=checkpoint / 'runs.csv',
            )
            model = train_policy(
                factory, checkpoint, total_timesteps=8, n_steps=4, batch_size=4,
                n_epochs=1, features=OBSERVATION_NAMES_V3, gamma=0.995,
                gae_lambda=0.98, evaluate_every=1,
            )
            with (checkpoint / EVALUATION_LOG).open() as handle:
                evaluations = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(evaluations), 1)
            self.assertAlmostEqual(0.98, model.gae_lambda)
            manifest = (checkpoint / 'manifest.json').read_text()
            self.assertIn('per_season_episodes', manifest)


class RuleBaselineTests(unittest.TestCase):
    def _observation(self, **points):
        values = [0.0] * len(OBSERVATION_NAMES_V3)
        for name, value in points.items():
            values[OBSERVATION_NAMES_V3.index(name)] = value / POINT_SCALE
        return values

    def test_a_good_wildcard_week_is_played(self):
        from fpl_manager.rule_baseline import DEFAULT, choose

        mask = [True] * len(ACTIONS)
        observation = self._observation(
            wildcard_now_gain_5=25.0, wildcard_gain_5=25.5,
            gameweeks_until_wildcard_deadline=10.0,
        )
        self.assertEqual(ACTIONS.index(WILDCARD), choose(observation, mask, DEFAULT))

    def test_nothing_worth_doing_holds(self):
        from fpl_manager.rule_baseline import DEFAULT, choose

        mask = [True] + [False] * (len(ACTIONS) - 1)
        self.assertEqual(ACTIONS.index(HOLD), choose(self._observation(), mask, DEFAULT))

    def test_spending_later_beats_spending_now_so_it_holds(self):
        from fpl_manager.rule_baseline import DEFAULT, choose

        mask = [True, True, False, False, False, False, False, False, False, False]
        observation = self._observation(spend_now_gain_5=3.0, roll_gain_5=6.0, transfer_1_gain_5=0.5)
        self.assertEqual(ACTIONS.index(HOLD), choose(observation, mask, DEFAULT))


class ManifestTests(unittest.TestCase):
    def test_v3_records_its_reward_and_v2_is_unchanged(self):
        from fpl_manager.train import reward_settings

        v2 = reward_settings(OBSERVATION_NAMES_V2)
        v3 = reward_settings(OBSERVATION_NAMES_V3)
        self.assertEqual({'terminal_divisor', 'weekly_coefficient'}, set(v2))
        self.assertEqual('v3', v3['version'])
        self.assertTrue(v3['per_season_episodes'])


if __name__ == '__main__':
    unittest.main()
