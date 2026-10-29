import unittest
from unittest import mock

from premierlearning.learning import (
    HORIZON_REPS_DEFAULT,
    MINUTES_EARLY_STOPPING_MIN_DELTA,
    MINUTES_EARLY_STOPPING_PATIENCE,
    MINUTES_REPS_DEFAULT,
    MLData,
    PlayerNetHParams,
    parse_horizon_reps,
    parse_minutes_reps,
    select_reps_for_average,
)
from premierlearning.runner import Runner


class _FakeMatch:
    def __init__(self, minutes=0, points=0):
        self.minutes = minutes
        self.points = points
        self.game_stats = {'minutes': minutes, 'total_points': points}

    def set_minutes_from_stats(self):
        self.minutes = self.game_stats['minutes']


class _FakePlayer:
    def __init__(self, last_5_avg=12):
        self.past_stats = {'minutes': {'last_5_avg': last_5_avg}}

    def get_name(self):
        return 'Test Player'


class HorizonOnlyMinutesTests(unittest.TestCase):
    def test_horizon_only_assumes_90_per_fixture(self):
        runner = Runner(update_data=False)
        runner._horizon_only = True
        futs = [_FakeMatch(minutes=0) for _ in range(5)]
        self.assertEqual(
            450.0,
            runner._expected_minutes_over_horizon(_FakePlayer(), futs, 5),
        )

    def test_full_learn_uses_predicted_minutes(self):
        runner = Runner(update_data=False)
        runner._horizon_only = False
        futs = [_FakeMatch(minutes=80) for _ in range(2)]
        # Live/Fargate uses predicted minutes as-is
        self.assertEqual(
            160.0,
            runner._expected_minutes_over_horizon(_FakePlayer(last_5_avg=0), futs, 5),
        )

    def test_horizon_only_redistribute_uses_pp90_as_fixture_points(self):
        runner = Runner(update_data=False)
        runner._horizon_only = True
        futs = [_FakeMatch(minutes=0) for _ in range(5)]
        runner.redistribute_horizon_points(
            [_FakePlayer(last_5_avg=0)],
            [futs],
            [[6.0]],
            5,
        )
        # Horizon-only: predicted pp90 is used as each fixture's points at 90 mins.
        for match in futs:
            self.assertAlmostEqual(6.0, match.points)
            self.assertAlmostEqual(90.0, match.game_stats['minutes'])
            self.assertAlmostEqual(90.0, match.minutes)

    def test_redistribute_keeps_the_single_week_points_and_its_scale(self):
        runner = Runner(update_data=False)
        runner._horizon_only = False
        futs = [_FakeMatch(minutes=90, points=2.0), _FakeMatch(minutes=90, points=6.0)]
        runner.redistribute_horizon_points([_FakePlayer()], [futs], [[6.0]], 2)
        # pp90 6 over 180 predicted minutes is a 12-point window, against an h1
        # sum of 8, so every fixture is stretched by the same 1.5.
        for match, raw in zip(futs, (2.0, 6.0)):
            self.assertAlmostEqual(raw, match.h1_points)
            self.assertAlmostEqual(1.5, match.horizon_scale)
            self.assertAlmostEqual(raw * 1.5, match.points)

    def test_even_split_quotes_no_scale_factor(self):
        runner = Runner(update_data=False)
        runner._horizon_only = False
        futs = [_FakeMatch(minutes=90, points=0.0) for _ in range(2)]
        runner.redistribute_horizon_points([_FakePlayer()], [futs], [[6.0]], 2)
        for match in futs:
            self.assertAlmostEqual(0.0, match.h1_points)
            self.assertIsNone(match.horizon_scale)
            self.assertAlmostEqual(6.0, match.points)

    def test_train_labels_are_raw_fpl_pp90(self):
        runner = Runner(update_data=False)
        self.assertAlmostEqual(
            6.0,
            runner._clean_match_points_per_90(
                {'game_stats': {'total_points': 6, 'minutes': 90}}
            ),
        )
        self.assertAlmostEqual(
            1.0 * 90.0 / 8.0,
            runner._clean_match_points_per_90(
                {'game_stats': {'total_points': 1, 'minutes': 8}}
            ),
        )

    def test_pooled_horizon_labels_use_raw_fpl_points(self):
        runner = Runner(update_data=False)
        window = [
            _FakeMatch(minutes=90, points=6),
            _FakeMatch(minutes=8, points=1),
            _FakeMatch(minutes=20, points=5),
        ]
        self.assertAlmostEqual(12.0 * 90.0 / 118.0, runner._pooled_points_per_90(window))


    def test_horizon_reps_default_is_five_and_flag_overrides(self):
        with mock.patch('sys.argv', ['premierlearning']):
            self.assertEqual(HORIZON_REPS_DEFAULT, parse_horizon_reps())
            self.assertEqual(5, parse_horizon_reps())
        with mock.patch('sys.argv', ['premierlearning', '--horizon-reps', '1']):
            self.assertEqual(1, parse_horizon_reps())

    def test_minutes_reps_default_is_five_and_flag_overrides(self):
        with mock.patch('sys.argv', ['premierlearning']):
            self.assertEqual(MINUTES_REPS_DEFAULT, parse_minutes_reps())
            self.assertEqual(5, parse_minutes_reps())
        with mock.patch('sys.argv', ['premierlearning', '--minutes-reps', '1']):
            self.assertEqual(1, parse_minutes_reps())

    def test_h1_poisson_recipe_flags(self):
        from premierlearning.learning import (
            parse_h1_poisson_recipe,
            parse_no_horizon_model,
            parse_points_loss,
            parse_points_minutes_weight,
            parse_points_reps,
        )
        with mock.patch('sys.argv', ['premierlearning']):
            self.assertFalse(parse_h1_poisson_recipe())
            self.assertFalse(parse_no_horizon_model())
            self.assertFalse(parse_points_minutes_weight())
            self.assertEqual('huber', parse_points_loss())
            self.assertEqual(1, parse_points_reps())
        with mock.patch('sys.argv', ['premierlearning', '--h1-poisson']):
            self.assertTrue(parse_h1_poisson_recipe())
            self.assertTrue(parse_no_horizon_model())
            self.assertTrue(parse_points_minutes_weight())
            self.assertEqual('poisson', parse_points_loss())
            self.assertEqual(5, parse_points_reps())
        with mock.patch(
            'sys.argv',
            ['premierlearning', '--h1-poisson', '--points-reps', '2'],
        ):
            self.assertEqual(2, parse_points_reps())
        with mock.patch(
            'sys.argv',
            ['premierlearning', '--points-loss', 'poisson', '--no-horizon-model'],
        ):
            self.assertEqual('poisson', parse_points_loss())
            self.assertTrue(parse_no_horizon_model())
            self.assertEqual(5, parse_points_reps())

    def test_learning_rep_flags_are_written_onto_argv(self):
        from premierlearning.learning import with_learning_rep_flags
        self.assertEqual(
            ['sim', '--horizon-reps', '5', '--minutes-reps', '5'],
            with_learning_rep_flags(['sim'], horizon_reps=5, minutes_reps=5),
        )
        self.assertEqual(
            ['sim', '--horizon-reps', '5', '--minutes-reps', '5'],
            with_learning_rep_flags(
                ['sim', '--horizon-reps', '3'],
                horizon_reps=5,
                minutes_reps=5,
            ),
        )

    def test_select_reps_drops_mae_more_than_eight_percent_over_best(self):
        hp = PlayerNetHParams()
        kept, threshold, _rule = select_reps_for_average(
            [
                {'best_val_mae': 10.0},
                {'best_val_mae': 10.5},
                {'best_val_mae': 12.0},
            ],
            hp,
        )
        self.assertEqual([0, 1], kept)
        self.assertAlmostEqual(10.8, threshold)

    def test_minutes_early_stopping_requires_a_quarter_minute_gain(self):
        minutes = type('Net', (), {
            'output_key': 'minutes',
            'mae_name': 'mean_absolute_error',
            'hparams': PlayerNetHParams(),
        })()
        points = type('Net', (), {
            'output_key': 'total_points',
            'mae_name': 'mean_absolute_error',
            'hparams': PlayerNetHParams(),
        })()
        minutes_kwargs = MLData.early_stopping_kwargs(minutes)
        points_kwargs = MLData.early_stopping_kwargs(points)
        self.assertEqual(MINUTES_EARLY_STOPPING_PATIENCE, minutes_kwargs['patience'])
        self.assertAlmostEqual(
            MINUTES_EARLY_STOPPING_MIN_DELTA, minutes_kwargs['min_delta']
        )
        self.assertEqual(25, points_kwargs['patience'])
        self.assertAlmostEqual(0.0, points_kwargs['min_delta'])


if __name__ == '__main__':
    unittest.main()
