import unittest
from types import SimpleNamespace

from squadtools.availability import (
    MINUTES_SCALE_RAW,
    MINUTES_SCALE_STEP_60,
    MINUTES_SCALE_STEP_70,
    MINUTES_SCALE_STEP_75,
    MINUTES_SCALE_STEP_80,
    chance_multiplier,
    match_optimiser_points,
    minutes_for_points,
    optimiser_points,
    row_optimiser_points,
    slot_bench_weights,
)
from squadtools.simulation_strategy import PredictionPool


class AvailabilityTests(unittest.TestCase):
    def test_none_chance_leaves_points_unchanged(self):
        self.assertEqual(1.0, chance_multiplier(None))
        self.assertAlmostEqual(6.0, optimiser_points(6.0, 90, None))

    def test_chance_scales_after_minutes_mapping(self):
        # 6 pp90 at 90 minutes → 6 points, then 50% chance → 3
        self.assertAlmostEqual(3.0, optimiser_points(6.0, 90, 50))
        self.assertAlmostEqual(0.0, optimiser_points(6.0, 90, 0))
        # Live/Fargate uses predicted minutes as-is: 6 × 80/90 × 50% chance
        self.assertAlmostEqual(6.0 * 80.0 / 90.0 * 0.5, optimiser_points(6.0, 80, 50))

    def test_raw_minutes_scale_skips_the_full_if_likely_mapping(self):
        # 6 pp90 × 80/90 × 50% chance, with no ≥75→90 bump
        self.assertAlmostEqual(
            6.0 * 80.0 / 90.0 * 0.5,
            optimiser_points(6.0, 80, 50, minutes_scale=MINUTES_SCALE_RAW),
        )
        self.assertAlmostEqual(
            6.0 * 80.0 / 90.0 * 0.5,
            optimiser_points(6.0, 80, 50),
        )

    def test_step_80_needs_more_minutes_to_count_as_a_starter(self):
        self.assertEqual(90.0, minutes_for_points(80, MINUTES_SCALE_STEP_80))
        self.assertLess(minutes_for_points(79, MINUTES_SCALE_STEP_80), 90.0)
        self.assertEqual(75.0, minutes_for_points(75, None))
        self.assertLess(minutes_for_points(75, MINUTES_SCALE_STEP_80), 90.0)
        self.assertAlmostEqual(70.0, minutes_for_points(55, MINUTES_SCALE_STEP_80))
        self.assertAlmostEqual(54.0, minutes_for_points(54, MINUTES_SCALE_STEP_80))

    def test_step_75_bumps_starters_and_scales_the_65_to_75_band(self):
        self.assertEqual(90.0, minutes_for_points(75, MINUTES_SCALE_STEP_75))
        self.assertEqual(90.0, minutes_for_points(80, MINUTES_SCALE_STEP_75))
        self.assertAlmostEqual(65.0, minutes_for_points(65, MINUTES_SCALE_STEP_75))
        self.assertAlmostEqual(77.5, minutes_for_points(70, MINUTES_SCALE_STEP_75))
        self.assertAlmostEqual(89.75, minutes_for_points(74.9, MINUTES_SCALE_STEP_75))
        self.assertAlmostEqual(64.0, minutes_for_points(64, MINUTES_SCALE_STEP_75))
        self.assertAlmostEqual(60.0, minutes_for_points(60, MINUTES_SCALE_STEP_75))

    def test_step_60_treats_rotation_minutes_as_a_full_match(self):
        self.assertEqual(90.0, minutes_for_points(60, MINUTES_SCALE_STEP_60))
        self.assertAlmostEqual(60.0, minutes_for_points(60, None))
        self.assertAlmostEqual(60.0, minutes_for_points(45, MINUTES_SCALE_STEP_60))
        self.assertAlmostEqual(44.0, minutes_for_points(44, MINUTES_SCALE_STEP_60))

    def test_step_70_is_a_cliff_with_no_mid_band(self):
        self.assertEqual(90.0, minutes_for_points(70, MINUTES_SCALE_STEP_70))
        self.assertAlmostEqual(69.0, minutes_for_points(69, MINUTES_SCALE_STEP_70))
        self.assertAlmostEqual(60.0, minutes_for_points(60, MINUTES_SCALE_STEP_70))

    def test_raw_minutes_still_apply_chance(self):
        self.assertAlmostEqual(
            6.0 * 60.0 / 90.0,
            optimiser_points(6.0, 60, None, minutes_scale=MINUTES_SCALE_RAW),
        )
        self.assertAlmostEqual(
            6.0 * 60.0 / 90.0 * 0.25,
            optimiser_points(6.0, 60, 25, minutes_scale=MINUTES_SCALE_RAW),
        )

    def test_old_rows_keep_baked_predicted_points(self):
        row = {
            'predicted_points': 4.2,
            'predicted_minutes': 80,
        }
        self.assertAlmostEqual(4.2, row_optimiser_points(row))

    def test_new_rows_recompute_from_pp90_minutes_and_chance(self):
        row = {
            'predicted_pp90': 6.0,
            'predicted_minutes': 90.0,
            'predicted_points': 99.0,
            'chance_of_playing': 50,
        }
        self.assertAlmostEqual(3.0, row_optimiser_points(row))

    def test_prediction_pool_applies_chance_for_new_rows(self):
        rows = [
            {
                'player_id': 1,
                'player_code': 10,
                'player_name': 'A',
                'position': 3,
                'team_id': 1,
                'price': 50,
                'selectable': True,
                'target_gameweek': 2,
                'predicted_pp90': 6.0,
                'predicted_minutes': 90.0,
                'predicted_points': 6.0,
                'chance_of_playing': 50,
            },
            {
                'player_id': 2,
                'player_code': 11,
                'player_name': 'B',
                'position': 3,
                'team_id': 1,
                'price': 50,
                'selectable': True,
                'target_gameweek': 2,
                'predicted_minutes': 90.0,
                'predicted_points': 6.0,
            },
        ]
        pool = PredictionPool.from_rows(rows)
        self.assertAlmostEqual(3.0, pool[1].points(2))
        self.assertAlmostEqual(6.0, pool[2].points(2))

    def test_prediction_pool_can_use_raw_minutes(self):
        rows = [
            {
                'player_id': 1,
                'player_code': 10,
                'player_name': 'A',
                'position': 3,
                'team_id': 1,
                'price': 50,
                'selectable': True,
                'target_gameweek': 2,
                'predicted_pp90': 6.0,
                'predicted_minutes': 80.0,
                'predicted_points': 6.0,
                'chance_of_playing': 50,
            },
        ]
        mapped = PredictionPool.from_rows(rows)
        raw = PredictionPool.from_rows(rows, minutes_scale=MINUTES_SCALE_RAW)
        self.assertAlmostEqual(6.0 * 80.0 / 90.0 * 0.5, mapped[1].points(2))
        self.assertAlmostEqual(6.0 * 80.0 / 90.0 * 0.5, raw[1].points(2))

    def test_slot_bench_weights_keeps_keeper_separate(self):
        self.assertEqual((), slot_bench_weights())
        self.assertEqual(
            (0.6, 0.21, 0.075, 0.075),
            slot_bench_weights((0.6, 0.21, 0.075), 0.075),
        )
        self.assertEqual(
            (0.6, 0.21, 0.075, 0.0),
            slot_bench_weights((0.6, 0.21, 0.075)),
        )
        self.assertEqual(
            (0.5, 0.0, 0.0, 0.0),
            slot_bench_weights((0.5,)),
        )
        self.assertEqual(
            (0.25, 0.25, 0.0, 0.0),
            slot_bench_weights((0.25, 0.25)),
        )

    def test_match_optimiser_points_uses_stored_pp90_and_chance(self):
        match = SimpleNamespace(
            predicted_pp90=6.0,
            minutes=90.0,
            game_stats={'minutes': 90.0},
            points=99.0,
            chance_of_playing=50,
        )
        self.assertAlmostEqual(3.0, match_optimiser_points(match))

    def test_match_optimiser_points_defaults_to_step_75(self):
        match = SimpleNamespace(
            predicted_pp90=6.0,
            minutes=80.0,
            game_stats={'minutes': 80.0},
            points=99.0,
            chance_of_playing=None,
        )
        self.assertAlmostEqual(6.0, match_optimiser_points(match))
        self.assertAlmostEqual(
            6.0 * 80.0 / 90.0,
            match_optimiser_points(match, minutes_scale=MINUTES_SCALE_RAW),
        )
        rotation = SimpleNamespace(
            predicted_pp90=6.0,
            minutes=70.0,
            game_stats={'minutes': 70.0},
            points=99.0,
            chance_of_playing=None,
        )
        self.assertAlmostEqual(6.0 * 77.5 / 90.0, match_optimiser_points(rotation))

    def test_75_percent_chance_scales_before_the_lineup_score(self):
        match = SimpleNamespace(
            predicted_pp90=4.1,
            minutes=90.0,
            game_stats={'minutes': 90.0},
            points=4.1,
            chance_of_playing=75,
        )
        self.assertAlmostEqual(4.1 * 0.75, match_optimiser_points(match))

    def test_match_without_pp90_implies_it_from_unadjusted_points(self):
        match = SimpleNamespace(
            predicted_pp90=None,
            minutes=90.0,
            game_stats={'minutes': 90.0},
            points=6.0,
            chance_of_playing=50,
        )
        self.assertAlmostEqual(3.0, match_optimiser_points(match))


if __name__ == '__main__':
    unittest.main()
