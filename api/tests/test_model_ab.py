import unittest
from types import SimpleNamespace

import numpy

from premierlearning.learning import (
    SMEAR_CONDITIONAL,
    SMEAR_GLOBAL,
    SMEAR_OFF,
    SMEAR_RELATIVE,
    TweedieDeviance,
    apply_smearing,
    smearing_factor,
    uses_log_targets,
)
from premierlearning.model_ab import (
    ARMS,
    FIT_BY_KEY,
    FITS,
    LIVE_ARM_NAMES,
    ORIGINAL_ARM_NAMES,
    _fit_cost,
    _selected,
    _worker_command,
    appearance_points,
    combine_hurdle,
    expected_points,
    fixture_pp90,
    partition_fits,
    pooled_pp90,
    resolve_args,
    score,
    slice_calibration,
    to_fixture_and_window,
)


def _match(minutes, points):
    return SimpleNamespace(
        minutes=minutes,
        points=points,
        game_stats={'minutes': minutes, 'total_points': points},
    )


class SmearingTests(unittest.TestCase):
    def test_factor_matches_the_lognormal_form_on_normal_residuals(self):
        rng = numpy.random.default_rng(0)
        pred = rng.uniform(0.5, 2.5, 40000)
        actual = pred + rng.normal(0.0, 0.5, 40000)
        stats = smearing_factor(actual, pred)
        self.assertAlmostEqual(stats['S'], stats['S_if_lognormal'], places=2)
        self.assertAlmostEqual(0.5, stats['residual_sd_log'], places=2)

    def test_factor_is_one_when_the_fit_is_perfect(self):
        pred = numpy.linspace(0.1, 2.0, 500)
        self.assertAlmostEqual(1.0, smearing_factor(pred, pred)['S'], places=6)

    def test_global_correction_undoes_the_log_shrink(self):
        # E[y|x] = exp(pred) * S - 1, so the corrected value is (y+1)*S - 1.
        stats = {'S': 1.5, 'by_pred_level': []}
        corrected = apply_smearing(numpy.array([3.0]), stats, SMEAR_GLOBAL)
        self.assertAlmostEqual(5.0, float(corrected[0]))

    def test_off_leaves_predictions_alone(self):
        preds = numpy.array([1.0, 4.0])
        numpy.testing.assert_allclose(
            preds, apply_smearing(preds, {'S': 2.0}, SMEAR_OFF)
        )

    def test_conditional_uses_the_bin_the_prediction_falls_in(self):
        stats = {
            'S': 1.4,
            'by_pred_level': [
                {'lo': 0.0, 'hi': 3.0, 'S': 1.2, 'n': 10},
                {'lo': 3.0, 'hi': 9.0, 'S': 1.8, 'n': 10},
            ],
        }
        low, high = apply_smearing(numpy.array([1.0, 5.0]), stats, SMEAR_CONDITIONAL)
        self.assertAlmostEqual(1.0 * 1.2 + 1.2 - 1.0, float(low))
        self.assertAlmostEqual(5.0 * 1.8 + 1.8 - 1.0, float(high))

    def test_relative_keeps_the_level_and_only_moves_ranks(self):
        # Bins average 1.5, so relative factors are 1.2/1.5 and 1.8/1.5: one bin is
        # pushed down and the other up, while the pair leaves the level alone.
        stats = {
            'S': 1.5,
            'by_pred_level': [
                {'lo': 0.0, 'hi': 3.0, 'S': 1.2, 'n': 10},
                {'lo': 3.0, 'hi': 9.0, 'S': 1.8, 'n': 10},
            ],
        }
        low, high = apply_smearing(numpy.array([1.0, 5.0]), stats, SMEAR_RELATIVE)
        self.assertLess(float(low), 1.0)
        self.assertGreater(float(high), 5.0)

    def test_correction_never_returns_a_negative_prediction(self):
        stats = {'S': 0.1, 'by_pred_level': []}
        corrected = apply_smearing(numpy.array([0.0]), stats, SMEAR_GLOBAL)
        self.assertGreaterEqual(float(corrected[0]), 0.0)


class LabelTests(unittest.TestCase):
    def test_appearance_points_follow_the_sixty_minute_rule(self):
        self.assertEqual(0.0, appearance_points(0))
        self.assertEqual(1.0, appearance_points(30))
        self.assertEqual(2.0, appearance_points(60))

    def test_cameo_pp90_is_the_outlier_the_weighting_exists_for(self):
        # One point in six minutes reads as a 15 pp90 superstar.
        self.assertAlmostEqual(15.0, fixture_pp90(_match(6, 1)))
        self.assertAlmostEqual(1.0, fixture_pp90(_match(90, 1)))

    def test_stripping_appearance_points_zeroes_a_blank_start(self):
        self.assertAlmostEqual(0.0, fixture_pp90(_match(90, 2), strip_appearance=True))
        self.assertAlmostEqual(2.0, fixture_pp90(_match(90, 2)))

    def test_pooled_pp90_skips_dnps_and_pools_over_real_minutes(self):
        window = [_match(90, 6), _match(0, 0), _match(45, 3)]
        self.assertAlmostEqual(9.0 * 90.0 / 135.0, pooled_pp90(window))

    def test_dnp_scores_zero_rather_than_dividing_by_zero(self):
        self.assertEqual(0.0, fixture_pp90(_match(0, 0)))
        self.assertEqual(0.0, pooled_pp90([_match(0, 0)]))


def _windows():
    """One player, one window of two played fixtures at 90 and 45 minutes."""
    return [{
        'player_id': 1,
        'anchor': 21,
        'window_row': {},
        'fixtures': [
            {'index': 0, 'gameweek': 21, 'minutes': 90.0, 'actual_pp90': 4.0},
            {'index': 1, 'gameweek': 22, 'minutes': 45.0, 'actual_pp90': 2.0},
        ],
        'actual_window_pp90': (4.0 * 90.0 + 2.0 * 45.0) / 135.0,
    }]


class ProjectionTests(unittest.TestCase):
    def test_horizon_rate_applies_to_every_fixture_in_its_window(self):
        fit = FIT_BY_KEY['hz_base']
        fixture_pred, window_pred = to_fixture_and_window(fit, _windows(), [3.5])
        numpy.testing.assert_allclose([3.5, 3.5], fixture_pred)
        numpy.testing.assert_allclose([3.5], window_pred)

    def test_h1_window_value_is_weighted_by_realised_minutes(self):
        fit = FIT_BY_KEY['h1_base']
        fixture_pred, window_pred = to_fixture_and_window(fit, _windows(), [6.0, 3.0])
        numpy.testing.assert_allclose([6.0, 3.0], fixture_pred)
        # 90 minutes at 6 and 45 at 3 pools to 5, not the unweighted 4.5.
        self.assertAlmostEqual(5.0, float(window_pred[0]))

    def test_appearance_arm_adds_participation_points_back_before_scoring(self):
        fit = FIT_BY_KEY['h1_appearance']
        fixture_pred, _window = to_fixture_and_window(fit, _windows(), [4.0, 2.0])
        # 2 appearance points over 90 minutes is 2 pp90; over 45 it is 2 as well
        # (1 point, since it is under 60) scaled to 90.
        self.assertAlmostEqual(4.0 + 2.0, float(fixture_pred[0]))
        self.assertAlmostEqual(2.0 + 1.0 * 90.0 / 45.0, float(fixture_pred[1]))


class ScoreTests(unittest.TestCase):
    def test_calibration_is_predicted_points_over_actual_points(self):
        windows = _windows()
        # Predict double the realised rate on every fixture.
        metrics = score(windows, [8.0, 4.0], [8.0])
        self.assertAlmostEqual(2.0, metrics['points_calibration'])

    def test_perfect_predictions_score_zero_error(self):
        windows = _windows()
        metrics = score(windows, [4.0, 2.0], [windows[0]['actual_window_pp90']])
        self.assertAlmostEqual(0.0, metrics['weighted_mae'])
        self.assertAlmostEqual(0.0, metrics['window_mae'])
        self.assertAlmostEqual(1.0, metrics['points_calibration'])

    def test_weighting_follows_minutes_not_fixture_count(self):
        windows = _windows()
        # 1.0 off on the 90-minute fixture, 3.0 off on the 45-minute one. Unweighted
        # that averages 2.0; weighted by minutes it is (90*1 + 45*3) / 135 = 1.667.
        metrics = score(windows, [5.0, 5.0], [5.0])
        self.assertAlmostEqual(2.0, metrics['fixture_mae'])
        self.assertAlmostEqual(225.0 / 135.0, metrics['weighted_mae'], places=4)

    def test_predicted_minutes_turn_pp90_into_expected_points(self):
        windows = _windows()
        # Same minutes as realised, so cal matches the pp90-weighted ratio.
        metrics = score(windows, [8.0, 4.0], [8.0], pred_minutes=[90.0, 45.0])
        self.assertAlmostEqual(2.0, metrics['points_calibration'])

    def test_overpredicted_minutes_on_a_blank_raise_cal(self):
        windows = [{
            'player_id': 1,
            'anchor': 21,
            'window_row': {},
            'fixtures': [
                {
                    'index': 0, 'gameweek': 21, 'player_id': 1,
                    'minutes': 0.0, 'actual_pp90': 0.0, 'actual_points': 0.0,
                },
                {
                    'index': 1, 'gameweek': 21, 'player_id': 2,
                    'minutes': 90.0, 'actual_pp90': 4.0, 'actual_points': 4.0,
                },
            ],
            'actual_window_pp90': 4.0,
        }]
        metrics = score(windows, [4.0, 4.0], [4.0], pred_minutes=[90.0, 90.0])
        self.assertAlmostEqual(2.0, metrics['points_calibration'])


class HurdleAndSliceTests(unittest.TestCase):
    def test_hurdle_multiplies_play_probability_by_rate_if_played(self):
        numpy.testing.assert_allclose(combine_hurdle([0.5, 1.0], [8.0, 3.0]), [4.0, 3.0])

    def test_expected_points_clip_minutes_at_ninety(self):
        numpy.testing.assert_allclose(expected_points([6.0], [180.0]), [6.0])

    def test_slice_cal_is_by_predicted_points_not_actual_scorers(self):
        pred = [1.0] * 20 + [10.0] * 20
        actual = [2.0] * 20 + [10.0] * 20
        slices = slice_calibration(pred, actual, n_slices=2, min_rows=5)
        self.assertAlmostEqual(0.5, slices[0]['cal'])
        self.assertAlmostEqual(1.0, slices[1]['cal'])


class LossTargetTests(unittest.TestCase):
    def test_raw_mean_losses_do_not_use_log1p(self):
        for loss in ('poisson', 'tweedie', 'mse', 'huber_raw', 'hurdle'):
            self.assertFalse(uses_log_targets(loss), loss)
        self.assertTrue(uses_log_targets('huber'))
        self.assertTrue(uses_log_targets(None))

    def test_tweedie_is_finite_at_zero_and_zero_when_the_mean_is_right(self):
        loss = TweedieDeviance(1.5)
        zero = float(loss(numpy.array([[0.0]]), numpy.array([[2.0]])))
        self.assertTrue(numpy.isfinite(zero))
        perfect = float(loss(numpy.array([[4.0]]), numpy.array([[4.0]])))
        self.assertAlmostEqual(0.0, perfect, places=5)


class GridTests(unittest.TestCase):
    def test_every_arm_points_at_a_real_fit(self):
        for arm in ARMS:
            self.assertIn(arm.fit, FIT_BY_KEY, arm.name)

    def test_arm_names_and_fit_keys_are_unique(self):
        self.assertEqual(len({a.name for a in ARMS}), len(ARMS))
        self.assertEqual(len({f.key for f in FITS}), len(FITS))

    def test_poisson_arms_never_ask_for_a_smearing_correction(self):
        # Poisson fits the mean on the raw scale, so there is no log shrink to undo
        # and a correction would double-count.
        for arm in ARMS:
            if not uses_log_targets(FIT_BY_KEY[arm.fit].loss):
                self.assertEqual(SMEAR_OFF, arm.smear, arm.name)

    def test_full_pack_keeps_the_original_nineteen_arms(self):
        _fits, arms = _selected(SimpleNamespace(pack='full', only=None))
        self.assertEqual(set(ORIGINAL_ARM_NAMES), {arm.name for arm in arms})
        self.assertEqual(19, len(arms))

    def test_live_pack_is_the_minutes_question_not_the_full_grid(self):
        fits, arms = _selected(SimpleNamespace(pack='live', only=None))
        self.assertEqual(set(LIVE_ARM_NAMES), {arm.name for arm in arms})
        self.assertEqual({arm.fit for arm in arms}, {fit.key for fit in fits})

    def test_live_pack_defaults_to_predicted_minutes(self):
        args = resolve_args(SimpleNamespace(pack='live', score_minutes=None))
        self.assertEqual('predicted', args.score_minutes)
        full = resolve_args(SimpleNamespace(pack='full', score_minutes=None))
        self.assertEqual('realised', full.score_minutes)

    def test_deleaked_arm_has_a_row_count_control(self):
        deleak = FIT_BY_KEY['hz_deleak']
        control = FIT_BY_KEY['hz_subsample']
        self.assertEqual(1, control.stride)
        self.assertEqual(deleak.stride, control.subsample)


class PartitionTests(unittest.TestCase):
    def test_every_fit_is_assigned_exactly_once(self):
        for workers in (1, 2, 3, 4, 8, 32):
            groups = partition_fits(FITS, workers)
            assigned = [fit.key for group in groups for fit in group]
            self.assertEqual(sorted(f.key for f in FITS), sorted(assigned), workers)

    def test_no_more_groups_than_workers_and_none_are_empty(self):
        for workers in (1, 2, 4, 8):
            groups = partition_fits(FITS, workers)
            self.assertLessEqual(len(groups), workers)
            for group in groups:
                self.assertTrue(group)

    def test_load_is_shared_rather_than_piled_onto_one_worker(self):
        # All five h1 fits on one worker would serialise the expensive half of
        # the grid behind a single process.
        groups = partition_fits(FITS, 4)
        loads = [sum(_fit_cost(fit) for fit in group) for group in groups]
        total = sum(_fit_cost(fit) for fit in FITS)
        self.assertLess(max(loads), 0.75 * total / 2)

    def test_overlapping_horizon_fits_cost_more_than_strided_ones(self):
        self.assertGreater(
            _fit_cost(FIT_BY_KEY['hz_base']), _fit_cost(FIT_BY_KEY['hz_deleak'])
        )
        self.assertGreater(
            _fit_cost(FIT_BY_KEY['h1_base']), _fit_cost(FIT_BY_KEY['hz_subsample'])
        )


class WorkerCommandTests(unittest.TestCase):
    def _args(self, **kwargs):
        base = dict(reps=5, split_gw=20, horizon=5, holdout_season=None, quick=False)
        base.update(kwargs)
        return SimpleNamespace(**base)

    def test_command_carries_the_group_and_the_run_settings(self):
        group = [FIT_BY_KEY['hz_base'], FIT_BY_KEY['hz_deleak']]
        cmd = _worker_command(group, ARMS, 'out', 2, self._args(reps=3))
        self.assertIn('--internal-worker', cmd)
        self.assertEqual('hz_base,hz_deleak', cmd[cmd.index('--fits') + 1])
        self.assertEqual('2', cmd[cmd.index('--worker-index') + 1])
        self.assertEqual('3', cmd[cmd.index('--reps') + 1])
        self.assertNotIn('--quick', cmd)
        self.assertNotIn('--holdout-season', cmd)

    def test_optional_settings_are_only_passed_when_set(self):
        cmd = _worker_command(
            [FIT_BY_KEY['h1_base']], ARMS, 'out', 0,
            self._args(quick=True, holdout_season=24),
        )
        self.assertIn('--quick', cmd)
        self.assertEqual('24', cmd[cmd.index('--holdout-season') + 1])

    def test_workers_are_told_which_arms_the_parent_selected(self):
        # Otherwise a --only run would report every arm sharing the chosen fit.
        arms = [arm for arm in ARMS if arm.name == '01_h1_base']
        cmd = _worker_command([FIT_BY_KEY['h1_base']], arms, 'out', 0, self._args())
        self.assertEqual('01_h1_base', cmd[cmd.index('--arms') + 1])

    def test_predicted_minutes_path_is_passed_to_workers(self):
        cmd = _worker_command(
            [FIT_BY_KEY['h1_poisson_mw']], ARMS, 'out', 0,
            self._args(score_minutes='predicted', minutes_pred='out/minutes_pred.npy'),
        )
        self.assertEqual('predicted', cmd[cmd.index('--score-minutes') + 1])
        self.assertEqual('out/minutes_pred.npy', cmd[cmd.index('--minutes-pred') + 1])


if __name__ == '__main__':
    unittest.main()
