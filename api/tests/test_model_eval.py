"""Evaluation harness: buckets, scores, rolling calibration, panel rows, cache hashes."""
import copy
import unittest

import numpy

from simulation.historical_data import HistoricalActual
from simulation.model_eval import scoring
from simulation.model_eval.calibrate import rolling_linear
from simulation.model_eval.config import DEFAULT_CONFIG, load_config, validate_config
from simulation.model_eval.metrics import RankCache, bucket_rows, compute
from simulation.model_eval.panel import (
    SCORE_OPTIMISER,
    SCORE_RAW_POINTS,
    SCORE_RAW_POINTS_CHANCE,
    FixtureParts,
    SeasonFacts,
    Unit,
    attach_predictions,
    row_points,
    scorer_for_set,
)


def _config(**overrides):
    config = load_config()
    config['min_rank_players'] = 3
    config['bootstrap']['resamples'] = 20
    config.update(overrides)
    return config


def _unit(origin, actual, preds, season=25, has_fixture=None, valid_leads=None):
    actual = numpy.asarray(actual, dtype=float)
    if has_fixture is None:
        has_fixture = numpy.ones(actual.shape, dtype=bool)
    unit = Unit(
        season, origin, numpy.arange(actual.shape[0]) + 1, has_fixture, actual,
        actual.shape[1] if valid_leads is None else valid_leads,
    )
    unit.parts = {}
    for name, values in preds.items():
        values = numpy.asarray(values, dtype=float)
        unit.preds[name] = values
        # One unflagged fixture per scheduled player-gameweek unless a test says otherwise.
        parts = FixtureParts(values.shape, keep_flagged=False)
        parts.base[..., 0] = values
        parts.count[...] = numpy.asarray(has_fixture, dtype=int)
        unit.parts[name] = parts
    return unit


def _single_lead(values, horizon=5):
    grid = numpy.zeros((len(values), horizon))
    grid[:, 0] = values
    return grid


class BucketTests(unittest.TestCase):
    def test_rank_positions_split_ties_by_player_id(self):
        buckets = DEFAULT_CONFIG['buckets']
        # Nine clear leaders, three tied on 50 across the 50th-percentile line, eight trailing.
        values = numpy.array(list(range(100, 91, -1)) + [50.0, 50.0, 50.0] + list(range(8)), dtype=float)
        ids = numpy.array(list(range(100, 109)) + [205, 201, 203] + list(range(300, 308)))
        chosen = bucket_rows(values, ids, buckets)
        # 20 players: top 5% is 1 player, 80-95th the next 3, 50-80th the next 6, bottom 10.
        self.assertEqual([0], chosen['top_5pct'].tolist())
        self.assertEqual([1, 2, 3], chosen['p80_95'].tolist())
        self.assertEqual(6, len(chosen['p50_80']))
        self.assertEqual(10, len(chosen['bottom_50']))
        self.assertEqual([0, 1, 2, 3, 4], chosen['top_5'].tolist())
        # The tie is split, lowest id first: 201 makes the 50-80th bucket, 203 and 205 do not.
        self.assertIn(10, chosen['p50_80'].tolist())
        self.assertIn(11, chosen['bottom_50'].tolist())
        self.assertIn(9, chosen['bottom_50'].tolist())


class MetricTests(unittest.TestCase):
    def test_perfect_predictions_score_zero_error_and_full_rank(self):
        actual = _single_lead(numpy.arange(40) / 4.0)
        unit = _unit(1, actual, {'m': actual})
        metrics = compute([unit], 'm', _config())
        self.assertAlmostEqual(0.0, metrics['overall_mse'])
        self.assertAlmostEqual(0.0, metrics['2A_actual'])
        self.assertAlmostEqual(0.0, metrics['2B_predicted'])
        self.assertAlmostEqual(0.0, metrics['calibration_gap'])
        self.assertAlmostEqual(1.0, metrics['spearman_single_gw'])
        self.assertAlmostEqual(1.0, metrics['spearman_five_gw'])

    def test_actual_side_buckets_make_an_honest_model_look_low_and_mirror_catches_inflation(self):
        expected = numpy.arange(40) / 4.0
        noise = numpy.where(numpy.arange(40) % 2 == 1, 3.0, -3.0)
        realised = _single_lead(expected + noise)
        honest = _single_lead(expected)
        inflated = _single_lead(numpy.where(numpy.arange(40) >= 32, expected + 3.0, expected))
        unit = _unit(1, realised, {'honest': honest, 'inflated': inflated})
        config = _config()
        fair = compute([unit], 'honest', config)
        gamed = compute([unit], 'inflated', config)
        # Unbiased, yet the players who scored most were mostly lucky.
        self.assertLess(fair['2A_actual_top_5_signed'], -2.0)
        self.assertGreater(fair['2A_predicted_top_5_signed'], fair['2A_actual_top_5_signed'] + 1.0)
        # Inflating the likely hauliers shows up on the model's own picks.
        self.assertGreater(gamed['2A_predicted_top_5_signed'], fair['2A_predicted_top_5_signed'] + 2.0)

    def test_blanks_drop_out_of_single_gw_rows_but_count_in_five_gw_totals(self):
        actual = numpy.array([[0.0, 2.0], [3.0, 1.0], [2.0, 2.0], [1.0, 5.0]])
        has_fixture = numpy.array([[False, True], [True, True], [True, True], [True, True]])
        preds = numpy.array([[0.0, 2.0], [3.0, 1.0], [2.0, 2.0], [1.0, 1.0]])
        unit = _unit(1, actual, {'m': preds}, has_fixture=has_fixture)
        metrics = compute([unit], 'm', _config(horizon=2))
        self.assertEqual(3, metrics['overall_n'])
        self.assertAlmostEqual(0.0, metrics['overall_mse'])
        self.assertGreater(metrics['2A_actual'], 0.0)

    def test_too_few_players_skips_the_correlation_instead_of_scoring_zero(self):
        actual = _single_lead([1.0, 2.0])
        unit = _unit(1, actual, {'m': actual})
        metrics = compute([unit], 'm', _config())
        self.assertIsNone(metrics['spearman_single_gw'])
        self.assertEqual(1, metrics['rank_skipped_single_gw'])

    def test_calibration_gap_is_the_mean_of_equal_count_bin_gaps(self):
        predicted = numpy.arange(10, dtype=float)
        actual = predicted + numpy.array([1, 1, -1, -1, 0, 0, 2, 2, 0, 0], dtype=float)
        unit = _unit(1, _single_lead(actual), {'m': _single_lead(predicted)})
        metrics = compute([unit], 'm', _config(), details=True)
        self.assertEqual([2] * 5, [row['n'] for row in metrics['calibration_bins']])
        self.assertAlmostEqual((1 + 1 + 0 + 2 + 0) / 5.0, metrics['calibration_gap'])

    def test_incomplete_windows_are_left_out_of_five_gw_metrics(self):
        actual = _single_lead(numpy.arange(10.0))
        unit = _unit(36, actual, {'m': actual}, valid_leads=3)
        metrics = compute([unit], 'm', _config())
        self.assertIsNone(metrics['2A_actual'])
        self.assertIsNone(metrics['spearman_five_gw'])
        self.assertIsNotNone(metrics['spearman_single_gw'])


class FixtureSwingTests(unittest.TestCase):
    """How well predicted rises and falls across a player's next five fixtures match the real ones."""

    def _swing_unit(self, actual, preds, minutes=None, fixture_count=None):
        actual = numpy.asarray(actual, dtype=float)
        unit = _unit(1, actual, preds)
        unit.minutes = numpy.full(actual.shape, 90.0) if minutes is None else numpy.asarray(minutes, dtype=float)
        unit.fixture_count = (
            numpy.ones(actual.shape, dtype=int) if fixture_count is None else numpy.asarray(fixture_count)
        )
        return unit

    def test_matching_swings_score_full_correlation_and_flat_scores_zero(self):
        actual = [[2.0, 8.0, 2.0, 2.0, 6.0], [1.0, 1.0, 9.0, 1.0, 3.0]]
        level = numpy.array([[3.0], [2.0]])
        tracking = level + (numpy.asarray(actual) - numpy.mean(actual, axis=1, keepdims=True)) * 0.5
        unit = self._swing_unit(actual, {'track': tracking, 'flat': numpy.repeat(level, 5, axis=1)})
        config = _config()
        tracked = compute([unit], 'track', config)
        flat = compute([unit], 'flat', config)
        self.assertAlmostEqual(1.0, tracked['fixture_swing_corr'])
        # Half-size swings: each predicted point of swing came with two real ones.
        self.assertAlmostEqual(2.0, tracked['fixture_swing_slope'])
        self.assertEqual(10, tracked['fixture_swing_n'])
        self.assertEqual(0.0, flat['fixture_swing_corr'])
        self.assertIsNone(flat['fixture_swing_slope'])
        self.assertLess(tracked['fixture_swing_mse'], flat['fixture_swing_mse'])

    def test_injuries_rotation_blanks_and_doubles_are_left_out(self):
        actual = [[2.0, 8.0, 2.0, 2.0, 6.0], [0.0, 0.0, 9.0, 1.0, 3.0], [2.0, 2.0, 12.0, 2.0, 2.0]]
        minutes = [[90] * 5, [0, 0, 90, 90, 90], [90] * 5]
        doubles = [[1] * 5, [1] * 5, [1, 1, 2, 1, 1]]
        unit = self._swing_unit(actual, {'m': numpy.asarray(actual)}, minutes=minutes, fixture_count=doubles)
        metrics = compute([unit], 'm', _config())
        self.assertEqual(5, metrics['fixture_swing_n'])

    def test_units_without_minutes_or_full_windows_skip_the_metric(self):
        unit = _unit(36, _single_lead(numpy.arange(10.0)), {'m': _single_lead(numpy.arange(10.0))}, valid_leads=3)
        metrics = compute([unit], 'm', _config())
        self.assertEqual(0, metrics['fixture_swing_n'])
        self.assertIsNone(metrics['fixture_swing_corr'])

    def test_fixture_swings_carry_their_own_weight_in_the_total(self):
        config = _config()
        rows = {component: weight for component, _key, _d, weight in scoring.components(config)}
        self.assertAlmostEqual(config['category_weights']['fixtures'], rows['fixture_swing'])
        keys = [key for _c, key, _d, _w in scoring.components(config)]
        base = {key: 1.0 for key in keys}
        model = dict(base, fixture_swing_corr=1.2)
        scored = scoring.score_model(model, base, {key: 0.1 for key in keys}, config)
        self.assertAlmostEqual(2.0, scored['categories']['fixtures'])
        self.assertAlmostEqual(2.0 * config['category_weights']['fixtures'], scored['total'])


class ScoringTests(unittest.TestCase):
    def test_component_weights_sum_to_one(self):
        total = sum(weight for _c, _k, _d, weight in scoring.components(DEFAULT_CONFIG))
        self.assertAlmostEqual(1.0, total)

    def test_noise_units_and_directions(self):
        self.assertAlmostEqual(2.0, scoring.noise_score(0.8, 1.0, 0.1, scoring.LOWER))
        self.assertAlmostEqual(-2.0, scoring.noise_score(0.3, 0.5, 0.1, scoring.HIGHER))
        self.assertIsNone(scoring.noise_score(0.8, 1.0, 0.0, scoring.LOWER))
        self.assertIsNone(scoring.noise_score(None, 1.0, 0.1, scoring.LOWER))

    def test_relative_pct_matches_the_plan_formulas(self):
        self.assertAlmostEqual(10.0, scoring.relative_pct(2.7, 3.0, scoring.LOWER))
        self.assertAlmostEqual(10.0, scoring.relative_pct(0.46, 0.40, scoring.HIGHER))
        self.assertIsNone(scoring.relative_pct(1.0, 0.0, scoring.LOWER))
        self.assertIsNone(scoring.relative_pct(1.0, 1.0, scoring.HIGHER))

    def test_baseline_scores_zero_and_a_missing_sigma_is_flagged_not_infinite(self):
        actual = _single_lead(numpy.arange(40) / 4.0 + numpy.where(numpy.arange(40) % 2, 1.0, -1.0))
        unit = _unit(1, actual, {'v1': _single_lead(numpy.arange(40) / 4.0)})
        unit.minutes = numpy.full(actual.shape, 90.0)
        unit.fixture_count = numpy.ones(actual.shape, dtype=int)
        config = _config()
        metrics = compute([unit], 'v1', config)
        sigmas = {key: 1.0 for _c, key, _d, _w in scoring.components(config)}
        self.assertAlmostEqual(0.0, scoring.score_model(metrics, metrics, sigmas, config)['total'])
        sigmas['calibration_gap'] = 0.0
        scored = scoring.score_model(metrics, metrics, sigmas, config)
        self.assertEqual(['calibration_gap'], scored['invalid'])
        self.assertTrue(numpy.isfinite(scored['total']))

    def test_each_metric_is_capped_before_weighting(self):
        config = _config()
        keys = [key for _c, key, _d, _w in scoring.components(config)]
        base = {key: 1.0 for key in keys}
        sigmas = {key: 0.01 for key in keys}
        # Calibration gap improves by 17 sd; everything else is unchanged.
        model = dict(base, calibration_gap=1.0 - 0.17)
        scored = scoring.score_model(model, base, sigmas, config)
        weight = config['category_weights']['calibration']
        self.assertAlmostEqual(3.0 * weight, scored['total'])
        self.assertAlmostEqual(17.0 * weight, scored['total_uncapped'])
        config['score_cap'] = None
        self.assertAlmostEqual(17.0 * weight, scoring.score_model(model, base, sigmas, config)['total'])

    def test_bootstrap_blocks_stay_inside_a_season(self):
        units = [_unit(gw, _single_lead([1.0, 2.0]), {}, season=season)
                 for season in (24, 25) for gw in range(1, 11)]
        draws = scoring.bootstrap_draws(units, block=5, resamples=10, seed=1)
        for drawn in draws:
            self.assertEqual(len(units), len(drawn))
            for start in range(0, len(drawn) - 4, 5):
                block = [units[i] for i in drawn[start:start + 5]]
                self.assertEqual(1, len({unit.season for unit in block}))
                origins = [unit.origin for unit in block]
                self.assertEqual(list(range(origins[0], origins[0] + 5)), origins)

    def test_guardrails_flag_overprediction_of_own_picks_and_a_bad_season(self):
        config = _config()
        base = {'overall_mse': 1.0, 'overall_mae': 1.0, 'calibration_gap': 0.1}
        model = dict(base, **{'overall_mse': 1.2, '2A_predicted_top_5_signed': 4.0})
        flags = scoring.guardrails(model, base, 0.5, {24: 0.8, 25: -2.0}, config)
        self.assertTrue(any('overall_mse' in flag for flag in flags))
        self.assertTrue(any('top_5 picks' in flag for flag in flags))
        self.assertTrue(any('season 25' in flag for flag in flags))


class CalibratorTests(unittest.TestCase):
    def test_only_fixtures_played_before_the_deadline_are_used(self):
        horizon = 2
        first = _unit(1, numpy.array([[2.0, 100.0], [4.0, 100.0], [6.0, 100.0]]),
                      {'m': numpy.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])})
        second = _unit(2, numpy.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
                       {'m': numpy.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])})
        for unit in (first, second):
            self.assertEqual(horizon, unit.horizon)
        records = rolling_linear([second, first], 'm', 'cal', min_pairs=3)
        self.assertIsNone(records[0]['slope'])
        numpy.testing.assert_allclose(first.preds['cal'], first.preds['m'])
        # At GW2 only GW1 has been played: actual = 2 x prediction exactly. The
        # lead-1 rows (target GW2, actual 100) are not known yet.
        self.assertEqual(3, records[1]['pairs'])
        self.assertAlmostEqual(2.0, records[1]['slope'])
        self.assertAlmostEqual(0.0, records[1]['intercept'], places=6)
        numpy.testing.assert_allclose(second.preds['cal'], 2.0 * second.preds['m'])

    def test_blanks_stay_zero_and_values_are_floored(self):
        history = _unit(1, numpy.array([[0.0], [0.0], [10.0]]), {'m': numpy.array([[4.0], [5.0], [6.0]])})
        target = _unit(2, numpy.array([[0.0], [0.0], [0.0]]), {'m': numpy.array([[0.0], [9.0], [5.0]])},
                       has_fixture=numpy.array([[True], [False], [True]]))
        rolling_linear([history, target], 'm', 'cal', min_pairs=3)
        self.assertEqual(0.0, target.preds['cal'][1, 0])
        self.assertGreaterEqual(target.preds['cal'][0, 0], 0.0)

    def _history(self):
        # actual = 1 + 0.5 x prediction for unflagged single-fixture players.
        return _unit(1, numpy.array([[2.0], [3.0], [4.0], [5.0]]),
                     {'m': numpy.array([[2.0], [4.0], [6.0], [8.0]])})

    def test_flag_is_applied_after_calibration_and_flagged_rows_are_not_fitted(self):
        history = self._history()
        # A 0%-flagged player who scored 10: must not pull the fit.
        history.parts['m'].flagged[0, 0] = True
        history.actual[0, 0] = 10.0
        target = _unit(2, numpy.zeros((2, 1)), {'m': numpy.array([[4.0], [4.0]])})
        target.parts['m'].multiplier[1, 0] = 0.0
        target.parts['m'].flagged[1, 0] = True
        records = rolling_linear([history, target], 'm', 'cal', min_pairs=3)
        self.assertEqual(3, records[1]['pairs'])
        self.assertAlmostEqual(0.5, records[1]['slope'])
        self.assertAlmostEqual(3.0, target.preds['cal'][0, 0])
        self.assertEqual(0.0, target.preds['cal'][1, 0])

    def test_doubles_are_calibrated_per_fixture_then_summed(self):
        history = self._history()
        target = _unit(2, numpy.zeros((1, 1)), {'m': numpy.array([[8.0]])})
        parts = target.parts['m']
        parts.base[0, 0, :2] = [4.0, 4.0]
        parts.count[0, 0] = 2
        rolling_linear([history, target], 'm', 'cal', min_pairs=3)
        # Two fixtures at 1 + 0.5 x 4 = 3 each, not 1 + 0.5 x 8 = 5 once.
        self.assertAlmostEqual(6.0, target.preds['cal'][0, 0])

    def test_sets_that_learned_the_flag_keep_flagged_rows_as_predicted(self):
        history = self._history()
        target = _unit(2, numpy.zeros((2, 1)), {'m': numpy.array([[4.0], [0.4]])})
        target.parts['m'].keep_flagged = True
        target.parts['m'].flagged[1, 0] = True
        rolling_linear([history, target], 'm', 'cal', min_pairs=3)
        self.assertAlmostEqual(3.0, target.preds['cal'][0, 0])
        self.assertAlmostEqual(0.4, target.preds['cal'][1, 0])


class PanelTests(unittest.TestCase):
    def test_both_models_get_chance_applied(self):
        v1_row = {'predicted_pp90': 5.625, 'predicted_minutes': 80.0, 'chance_of_playing': 75,
                  'predicted_points': 5.0}
        v2_row = {'predicted_pp90': None, 'predicted_minutes': 0.0, 'chance_of_playing': 75,
                  'predicted_points': 5.0}
        self.assertAlmostEqual(3.75, row_points(v1_row, SCORE_OPTIMISER))
        self.assertAlmostEqual(3.75, row_points(v2_row, SCORE_RAW_POINTS_CHANCE))
        self.assertAlmostEqual(5.0, row_points(v2_row, SCORE_RAW_POINTS))
        self.assertAlmostEqual(5.0, row_points(dict(v2_row, chance_of_playing=None), SCORE_RAW_POINTS_CHANCE))
        self.assertEqual(SCORE_RAW_POINTS_CHANCE, scorer_for_set({'config': {'model': 'learning_v2'}}))
        self.assertEqual(SCORE_RAW_POINTS, scorer_for_set(
            {'config': {'model': 'learning_v2', 'v2_chance_input': 'deadline_flags'}}))
        self.assertEqual(SCORE_OPTIMISER, scorer_for_set({'config': {'horizon_model': True}}))

    def test_double_gameweeks_sum_and_missing_predictions_are_counted(self):
        unit = Unit(25, 3, [10, 11], numpy.ones((2, 2), dtype=bool), numpy.zeros((2, 2)), 2)
        rows = [
            {'target_gameweek': 3, 'player_id': 10, 'predicted_points': 2.0, 'predicted_pp90': None},
            {'target_gameweek': 3, 'player_id': 10, 'predicted_points': 1.5, 'predicted_pp90': None},
            {'target_gameweek': 4, 'player_id': 10, 'predicted_points': 4.0, 'predicted_pp90': None},
            {'target_gameweek': 3, 'player_id': 99, 'predicted_points': 9.0, 'predicted_pp90': None},
            {'target_gameweek': 8, 'player_id': 11, 'predicted_points': 9.0, 'predicted_pp90': None},
        ]
        attach_predictions(unit, 'm', rows, SCORE_RAW_POINTS)
        numpy.testing.assert_allclose([[3.5, 4.0], [0.0, 0.0]], unit.preds['m'])
        self.assertEqual(2, unit.missing['m'])

    def test_eligibility_uses_only_the_deadline_and_last_season_early_on(self):
        class _Source:
            season_start_year = 25
            fixtures = [
                {'event': '1', 'team_h': '1', 'team_a': '2'},
                {'event': '2', 'team_h': '2', 'team_a': '1'},
                {'event': '3', 'team_h': '1', 'team_a': '2'},
            ]
            players_raw = {1: {'code': '101'}, 2: {'code': '102'}, 3: {'code': '103'}, 4: {'code': '104'}}
            player_rows = {
                1: [{'round': '1', 'minutes': '90'}, {'round': '2', 'minutes': '90'}],
                2: [{'round': '1', 'minutes': '0'}],
                3: [],
                4: [{'round': '1', 'minutes': '10'}],
            }

            def state_at_gameweek(self, _gameweek):
                return {1: {'team': 'A'}, 2: {'team': 'A'}, 3: {'team': 'B'}, 4: {'team': 'B'}}

            def team_id_from_snapshot(self, snapshot, _raw):
                return {'A': 1, 'B': 3}[snapshot['team']]

            def actuals_for_gameweek(self, gameweek):
                return {1: HistoricalActual(1, gameweek, 6.0, 90.0, 50, (gameweek * 10,))}

        class _Previous:
            players_raw = {9: {'code': '103'}}
            player_rows = {9: [{'minutes': '1000'}]}

        facts = SeasonFacts(_Source(), _Previous())
        rule = {'recent_gws': 5, 'last_season_min_minutes': 900.0}
        self.assertEqual({1: 1, 3: 3, 4: 3}, facts.eligible(2, **rule))
        # Once five gameweeks exist, last season no longer qualifies anyone.
        self.assertEqual({1: 1}, facts.eligible(7, **rule))
        unit = facts.build_unit(2, 2, rule)
        self.assertEqual([1, 3, 4], unit.player_ids.tolist())
        # Team 3 has no fixtures: a blank on both sides.
        self.assertEqual([[True, True], [False, False], [False, False]], unit.has_fixture.tolist())
        self.assertEqual([[6.0, 6.0], [0.0, 0.0], [0.0, 0.0]], unit.actual.tolist())


class ConfigTests(unittest.TestCase):
    def test_weights_that_do_not_sum_to_one_are_rejected(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config['category_weights']['ranking'] = 0.5
        with self.assertRaises(ValueError):
            validate_config(config)


class CacheHashTests(unittest.TestCase):
    def test_v2_builds_never_share_a_cache_with_v1_or_each_other(self):
        from premierlearning.learning_v2.score import LossWeighting
        from simulation.prediction_builder import HistoricalPredictionBuilder, _model_hash

        v1 = HistoricalPredictionBuilder(None)
        self.assertNotIn('model', v1.config)
        # v1 hashes exactly as before, so existing v1 caches stay valid.
        self.assertEqual(_model_hash(v1.config), v1.metadata['model_config_hash'])
        tier = HistoricalPredictionBuilder(None, learning_v2=True)
        smooth = HistoricalPredictionBuilder(
            None, learning_v2=True, v2_weighting=LossWeighting('smooth'),
        )
        self.assertEqual('learning_v2', tier.config['model'])
        hashes = {
            v1.metadata['model_config_hash'],
            tier.metadata['model_config_hash'],
            smooth.metadata['model_config_hash'],
        }
        self.assertEqual(3, len(hashes))

    def test_loss_and_tier_weight_options_get_their_own_caches(self):
        from simulation.prediction_builder import HistoricalPredictionBuilder

        v1 = HistoricalPredictionBuilder(None)
        v1_tier = HistoricalPredictionBuilder(None, points_tier_weight=True)
        self.assertNotIn('points_tier_weight', v1.config)
        huber = HistoricalPredictionBuilder(None, learning_v2=True)
        mse = HistoricalPredictionBuilder(None, learning_v2=True, v2_loss='mse')
        self.assertEqual('huber_raw', huber.config['v2_loss'])
        self.assertEqual('mse', mse.config['v2_loss'])
        hashes = {b.metadata['model_config_hash'] for b in (v1, v1_tier, huber, mse)}
        self.assertEqual(4, len(hashes))

    def test_only_a_non_default_patience_changes_the_cache(self):
        from simulation.prediction_builder import HistoricalPredictionBuilder

        default = HistoricalPredictionBuilder(None, learning_v2=True)
        test_run = HistoricalPredictionBuilder(None, learning_v2=True, v2_patience_floor=8)
        self.assertNotIn('v2_patience_floor', default.config)
        self.assertEqual(8, test_run.config['v2_patience_floor'])
        self.assertNotEqual(default.metadata['model_config_hash'], test_run.metadata['model_config_hash'])


class LatestSetTests(unittest.TestCase):
    def test_eval_sets_never_become_the_default(self):
        import os
        import tempfile
        from pathlib import Path

        from flask import Flask

        from persistence import DB_Handler

        with tempfile.TemporaryDirectory() as temp_dir:
            previous_path = os.environ.get('DATABASE_PATH')
            os.environ['DATABASE_PATH'] = str(Path(temp_dir) / 'latest.db')
            context = Flask(__name__).app_context()
            context.push()
            db = DB_Handler()
            try:
                db.init_db()
                ids = {}
                for name in ('strict-historical-rest-v1', 'eval-v2_huber'):
                    ids[name] = db.get_or_create_prediction_set({
                        'name': name,
                        'model_config_hash': name.ljust(64, 'x'),
                        'horizon_gameweeks': 5,
                        'training_policy': 'strict_point_in_time',
                        'availability_mode': 'deferred_chance',
                        'fixture_mode': 'final_calendar',
                        'random_seed': 42,
                        'config': {},
                    })
                    snapshot_id, _ = db.begin_prediction_snapshot(ids[name], 25, 1)
                    db.complete_prediction_snapshot(snapshot_id, 1)
                self.assertEqual(ids['strict-historical-rest-v1'], db.latest_prediction_set_id(25))
                self.assertEqual(ids['strict-historical-rest-v1'], db.latest_prediction_set_id())
            finally:
                db.close_connection()
                context.pop()
                if previous_path is None:
                    os.environ.pop('DATABASE_PATH', None)
                else:
                    os.environ['DATABASE_PATH'] = previous_path


class ManagerSetTests(unittest.TestCase):
    def setUp(self):
        import os
        import tempfile
        from pathlib import Path
        from unittest import mock

        from flask import Flask

        from persistence import DB_Handler
        from simulation.model_eval import manager_sets

        self.manager_sets = manager_sets
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_path = os.environ.get('DATABASE_PATH')
        os.environ['DATABASE_PATH'] = str(Path(self.temp_dir.name) / 'sets.db')
        self.context = Flask(__name__).app_context()
        self.context.push()
        self.db = DB_Handler()
        self.db.init_db()
        self.gameweeks = mock.patch.object(manager_sets, 'LAST_GAMEWEEK', 1)
        self.gameweeks.start()
        self.v1 = self._set('eval-v1_h5', minutes=80.0, pp90=4.5, points=4.0)
        self.v2 = self._set('eval-v2_log', minutes=0.0, pp90=None, points=6.0)
        self.h5 = self._set('eval-v2_log_h5', minutes=0.0, pp90=None, points=2.0)

    def tearDown(self):
        import os

        self.gameweeks.stop()
        self.db.close_connection()
        self.context.pop()
        if self.previous_path is None:
            os.environ.pop('DATABASE_PATH', None)
        else:
            os.environ['DATABASE_PATH'] = self.previous_path
        self.temp_dir.cleanup()

    def _set(self, name, minutes, pp90, points):
        set_id = self.db.get_or_create_prediction_set({
            'name': name,
            'model_config_hash': name.ljust(64, 'x'),
            'horizon_gameweeks': 5,
            'training_policy': 'strict_point_in_time',
            'availability_mode': 'deferred_chance',
            'fixture_mode': 'final_calendar',
            'random_seed': 42,
            'config': {'model': 'learning_v2'},
        })
        snapshot_id, _ = self.db.begin_prediction_snapshot(set_id, 25, 1)
        rows = [{
            'season_start_year': 25, 'as_of_gameweek': 1, 'target_gameweek': 1, 'fixture_id': 7,
            'fixture_order': 1, 'player_id': 10, 'player_code': 100, 'player_name': 'A Player',
            'position': 3, 'team_id': 1, 'opponent_team_id': 2, 'opponent': 'BBB', 'is_home': True,
            'price': 50, 'selectable': True, 'predicted_minutes': minutes, 'predicted_pp90': pp90,
            'predicted_points': points, 'chance_of_playing': 50, 'points_error': 0.0,
        }]
        self.db.persist_historical_predictions(snapshot_id, rows)
        self.db.complete_prediction_snapshot(snapshot_id, len(rows))
        return set_id

    def _pool(self, set_id):
        from squadtools.simulation_strategy import PredictionPool

        return PredictionPool.from_rows(self.db.get_historical_predictions(set_id, 25, 1))

    def test_stored_sets_tell_the_pool_whether_the_model_predicts_minutes(self):
        self.assertTrue(self._pool(self.v1).predicts_minutes)
        pool = self._pool(self.v2)
        self.assertFalse(pool.predicts_minutes)
        # Stored v2 points carry availability already, so no chance multiplier on top.
        self.assertAlmostEqual(6.0, pool.players[10].points(1))

    def test_blend_is_a_new_set_and_never_a_default(self):
        blend = self.manager_sets.blend_sets(self.db, self.v2, self.h5, seasons=(25,))

        row = self.db.get_historical_predictions(blend, 25, 1)[0]
        self.assertAlmostEqual(5.0, row['predicted_points'])
        self.assertIsNone(row['predicted_pp90'])
        self.assertFalse(self._pool(blend).predicts_minutes)
        self.assertEqual(6.0, self.db.get_historical_predictions(self.v2, 25, 1)[0]['predicted_points'])
        self.assertEqual(blend, self.manager_sets.blend_sets(self.db, self.v2, self.h5, seasons=(25,)))
        self.assertIsNone(self.db.latest_prediction_set_id(25))

    def test_only_eval_sets_can_be_blended(self):
        live = self._set('strict-historical-rest-v1', minutes=0.0, pp90=None, points=1.0)

        with self.assertRaises(ValueError):
            self.manager_sets.blend_sets(self.db, live, self.h5, seasons=(25,))
        with self.assertRaises(ValueError):
            self.manager_sets.blend_sets(self.db, self.v2, self.h5, name='strict-blend', seasons=(25,))


class SliceReportTests(unittest.TestCase):
    """Bias and error on the external-data eval slices, next gameweek only."""

    class _Slices:
        slice_columns = ['slice_dgw_leg', 'slice_new_manager_first10']

        def slices(self, season, fixture_id, fpl_code):
            assert season == 2025
            return {'slice_dgw_leg': int(fixture_id in (11, 12)), 'slice_new_manager_first10': int(fpl_code == 900)}

    def _unit(self):
        from simulation.model_eval.panel import Unit

        has = numpy.array([[True, True], [True, True], [False, True]])
        unit = Unit(25, 5, [1, 2, 3], has, numpy.array([[6.0, 1.0], [2.0, 2.0], [0.0, 3.0]]), 2,
                    codes=[900, 901, 902], fixtures=[[(11, 12), (20,)], [(13,), (21,)], [(), (22,)]])
        unit.preds['m'] = numpy.array([[4.0, 1.0], [3.0, 2.0], [9.0, 9.0]])
        return unit

    def test_slices_take_lead_zero_rows_that_played_and_any_flagged_fixture(self):
        from simulation.model_eval.slices import ALL, slice_stats, slice_text

        stats = slice_stats([self._unit()], ['m'], external=self._Slices())
        # The third player had no fixture at lead 0, so only two rows count.
        self.assertEqual(2, stats[ALL]['n'])
        self.assertAlmostEqual(((4 - 6) + (3 - 2)) / 2.0, stats[ALL]['m']['bias'])
        self.assertEqual(1, stats['slice_dgw_leg']['n'])
        self.assertAlmostEqual(-2.0, stats['slice_dgw_leg']['m']['bias'])
        self.assertAlmostEqual(2.0, stats['slice_new_manager_first10']['m']['mae'])
        self.assertIn('slice_dgw_leg', slice_text(stats, ['m']))

    def test_no_slice_file_means_no_report(self):
        from simulation.model_eval.slices import slice_stats

        class Empty:
            slice_columns = []

        self.assertEqual({}, slice_stats([self._unit()], ['m'], external=Empty()))


if __name__ == '__main__':
    unittest.main()
