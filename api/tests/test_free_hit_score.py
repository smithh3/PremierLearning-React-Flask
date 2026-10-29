import unittest

from simulation.actual_scoring import score_lineup
from simulation.free_hit_score import (
    actual_free_hit_points,
    printed_free_hit_points,
    score_selected_squad,
)
from squadtools.simulation_strategy import (
    ChipCalendar,
    OwnedPlayer,
    SquadState,
    TransferPlan,
    pick_lineup,
)
from tests.test_simulation_strategy import make_pool


class Actual:
    def __init__(self, points, minutes):
        self.points = points
        self.minutes = minutes


class FreeHitScoreTests(unittest.TestCase):
    def test_free_hit_rebuild_scores_the_spiked_player(self):
        extras = [{
            'position': 2,
            'team_id': 20,
            'points_by_gameweek': {1: 40.0},
        }]
        pool, current_ids, _ = make_pool(gameweeks=(1,), extra_players=extras)
        spiked_id = extras[0]['player_id']
        state = SquadState(
            owned={
                player_id: OwnedPlayer(player_id, pool[player_id].price)
                for player_id in current_ids
            },
            bank=0,
            free_transfers=1,
        )
        actuals = {
            player_id: Actual(points=2, minutes=90)
            for player_id in pool.players
        }
        actuals[spiked_id] = Actual(points=30, minutes=90)
        fh_points = actual_free_hit_points(state, pool, 1, actuals)
        hold_points = score_lineup(
            pick_lineup(current_ids, pool, 1), pool, actuals
        ).points
        self.assertGreater(fh_points, hold_points)
        self.assertGreaterEqual(fh_points, 60)

    def test_free_hit_rebuild_uses_the_same_horizon_as_wildcard(self):
        calendar = ChipCalendar(free_hits=(6,))
        wildcard = ChipCalendar(wildcards=(6,))
        self.assertEqual((6, 7, 8, 9, 10), calendar.planning_gameweeks(6, 5))
        self.assertEqual(
            wildcard.planning_gameweeks(6, 5),
            calendar.planning_gameweeks(6, 5),
        )
        self.assertEqual((4, 5, 7, 8, 9), calendar.planning_gameweeks(4, 5))

    def test_print_reuses_stored_plan_without_a_new_rebuild(self):
        extras = [{
            'position': 2,
            'team_id': 20,
            'points_by_gameweek': {1: 1.0},
        }]
        pool, current_ids, _ = make_pool(gameweeks=(1,), extra_players=extras)
        spiked_id = extras[0]['player_id']
        selected = tuple(sorted(current_ids[:14] + (spiked_id,)))
        plan = TransferPlan(
            transfers=(),
            selected_player_ids=selected,
            expected_hold_current=0.0,
            expected_selected_current=10.0,
            expected_hold_horizon=0.0,
            expected_selected_horizon=10.0,
        )
        actuals = {
            player_id: Actual(points=2, minutes=90)
            for player_id in pool.players
        }
        actuals[spiked_id] = Actual(points=30, minutes=90)
        printed = printed_free_hit_points(
            chip=None,
            played_points=22.0,
            gameweek=1,
            pool=pool,
            actuals=actuals,
            stored_plan=plan,
            state=None,
        )
        self.assertAlmostEqual(
            score_selected_squad(selected, pool, 1, actuals),
            printed,
        )

    def test_print_uses_played_points_when_the_chip_is_used(self):
        extras = [{
            'position': 2,
            'team_id': 20,
            'points_by_gameweek': {1: 1.0},
        }]
        pool, current_ids, _ = make_pool(gameweeks=(1,), extra_players=extras)
        plan = TransferPlan(
            transfers=(),
            selected_player_ids=tuple(sorted(current_ids)),
            expected_hold_current=0.0,
            expected_selected_current=1.0,
            expected_hold_horizon=0.0,
            expected_selected_horizon=1.0,
        )
        actuals = {
            player_id: Actual(points=2, minutes=90)
            for player_id in pool.players
        }
        self.assertEqual(
            88.0,
            printed_free_hit_points(
                chip='free_hit',
                played_points=88.0,
                gameweek=1,
                pool=pool,
                actuals=actuals,
                stored_plan=plan,
                state=None,
            ),
        )
