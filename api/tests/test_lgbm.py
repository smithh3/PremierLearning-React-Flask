"""LightGBM points model: flattened columns, fitting, and the per-season wrapper."""
import math
import unittest

import numpy

from premierlearning.learning_v2.dataset import build_dataset
from premierlearning.learning_v2.plan import load_plan
from premierlearning.lgbm.features import FeatureTable, flatten
from premierlearning.lgbm.model import fit, params_for
from tests.test_learning_v2 import _fixture_world, _row, _teams_plan


def _simple_plan():
    plan = load_plan('simple_h1')
    plan.config.history_len = 4
    plan.config.max_squad = 3
    plan.config.team_form_n = 5
    return plan


class FlattenTests(unittest.TestCase):
    def test_player_history_summaries(self):
        season, saka = _fixture_world()
        plan = _simple_plan()
        train, _predict, _vocab = build_dataset([season], plan, predict_season=season)
        table = flatten(train, plan)
        self.assertEqual(len(table.names), len(set(table.names)))
        self.assertEqual((len(train.matches), len(table.names)), table.values.shape)
        row = table.values[_row(train, saka, 3)]

        def value(name):
            return float(row[table.names.index(name)])

        # Before fixture 3 Saka had 10 minutes (1 point) then a did-not-play.
        self.assertEqual(2.0, value('hist.count'))
        self.assertEqual(0.0, value('hist.minutes_played.mean1'))
        self.assertAlmostEqual(5.0, value('hist.minutes_played.mean3'))
        self.assertAlmostEqual(0.5, value('hist.played_share5'))
        self.assertEqual(1.0, value('hist.since_played'))
        self.assertTrue(math.isnan(value('hist.since_sixty')))
        self.assertAlmostEqual(9.0, value('hist.points_scored.per90_5'), places=4)
        # Faded mean: the newest match (0 points) weighs 1, the one before 2^(-1/3).
        older = 0.5 ** (1 / 3.0)
        self.assertAlmostEqual(older / (1 + older), value('hist.points_scored.faded3'), places=5)

    def test_first_match_has_unknown_history(self):
        season, saka = _fixture_world()
        plan = _simple_plan()
        train, _predict, _vocab = build_dataset([season], plan, predict_season=season)
        table = flatten(train, plan)
        row = table.values[_row(train, saka, 1)]
        self.assertEqual(0.0, row[table.names.index('hist.count')])
        self.assertTrue(math.isnan(row[table.names.index('hist.points_scored.mean5')]))

    def test_ids_are_categorical_and_player_id_is_left_out(self):
        season, _saka = _fixture_world()
        plan = _simple_plan()
        train, _predict, _vocab = build_dataset([season], plan, predict_season=season)
        table = flatten(train, plan)
        self.assertNotIn('player_id_embedding', table.names)
        self.assertIn('current_team.team_id_embedding', table.categorical)
        self.assertIn('current_opposition.opposition_team_id_embedding', table.categorical)
        self.assertIn('position', table.categorical)

    def test_squad_summaries(self):
        season, saka = _fixture_world()
        plan = _simple_plan()
        train, _predict, _vocab = build_dataset([season], plan, predict_season=season)
        table = flatten(train, plan)
        row = table.values[_row(train, saka, 3)]
        count = row[table.names.index('team_squad.count')]
        self.assertGreaterEqual(count, 1.0)
        total = row[table.names.index('team_squad.minutes_last_n.sum')]
        mean = row[table.names.index('team_squad.minutes_last_n.mean')]
        self.assertAlmostEqual(total / count, mean, places=4)

    def test_team_history_summaries_come_from_the_shared_table(self):
        season, saka = _fixture_world()
        plan = _teams_plan()
        train, predict, _vocab = build_dataset([season], plan, predict_season=season)
        table = flatten(train, plan)
        row = table.values[_row(train, saka, 3)]
        # Arsenal before fixture 3: won 2-0, drew 1-1.
        self.assertEqual(2.0, row[table.names.index('team_hist.count')])
        self.assertAlmostEqual(1.5, row[table.names.index('team_hist.goals_for.mean5')])
        self.assertAlmostEqual(0.5, row[table.names.index('team_hist.goals_against.mean5')])
        # Chelsea before fixture 3: lost 0-2, drew 1-1.
        self.assertAlmostEqual(0.5, row[table.names.index('opp_hist.goals_for.mean5')])
        self.assertAlmostEqual(1.5, row[table.names.index('opp_hist.goals_against.mean5')])
        # The future fixture sees all three past matches; same columns as training.
        predicted = flatten(predict, plan)
        self.assertEqual(table.names, predicted.names)
        future = predicted.values[_row(predict, saka, 4)]
        self.assertEqual(3.0, future[predicted.names.index('team_hist.count')])
        self.assertAlmostEqual(8.0 / 3.0, future[predicted.names.index('team_hist.goals_for.mean5')], places=5)


class FeatureSetTwoTests(unittest.TestCase):
    def _tables(self, plan):
        season, saka = _fixture_world()
        train, predict, _vocab = build_dataset([season], plan, predict_season=season)
        return saka, train, predict, flatten(train, plan), flatten(train, plan, features=2)

    def test_set_two_keeps_every_set_one_column_with_the_same_values(self):
        _saka, train, _predict, one, two = self._tables(_teams_plan())
        self.assertEqual(len(two.names), len(set(two.names)))
        missing = [name for name in one.names if name not in two.names]
        self.assertEqual([], missing)
        self.assertGreater(len(two.names), len(one.names))
        for name in one.names:
            numpy.testing.assert_array_equal(one.column(name), two.column(name), err_msg=name)

    def test_windows_trends_venue_and_matchups(self):
        saka, train, predict, _one, two = self._tables(_teams_plan())
        row = two.values[_row(train, saka, 3)]

        def value(name):
            return float(row[two.names.index(name)])

        # Before fixture 3: 1 point, then 0 (both at home, as this one is).
        self.assertAlmostEqual(0.0, value('hist.points_scored.mean1'))
        self.assertAlmostEqual(0.5, value('hist.points_scored.mean2'))
        self.assertAlmostEqual(0.0, value('hist.points_scored.trend3_10'))
        self.assertAlmostEqual(0.5, value('hist.points_scored.venue10'))
        # Arsenal 2-0, 1-1; Chelsea 0-2, 1-1. Windows of 1 see only the last match.
        self.assertAlmostEqual(1.0, value('team_hist.goals_for.mean1'))
        self.assertAlmostEqual(1.5, value('team_hist.goals_for.mean114'))
        self.assertAlmostEqual(1.5, value('team_hist.goals_for.venue10'))
        self.assertAlmostEqual(0.5, value('opp_hist.goals_for.venue10'))
        # Attack: Arsenal score 1.5, Chelsea concede 1.5. Defence: 0.5 and 0.5.
        self.assertAlmostEqual(1.5, value('fixture.attack_goals.mean5'))
        self.assertAlmostEqual(0.5, value('fixture.defence_goals.mean5'))
        self.assertAlmostEqual(1.0, value('fixture.attack_goals.ratio10'))
        self.assertIn('hist.xG.share10', two.names)
        # Predictions get the same columns.
        self.assertEqual(two.names, flatten(predict, _teams_plan(), features=2).names)

    def test_simple_plan_matchups_come_from_team_form(self):
        _saka, _train, _predict, _one, two = self._tables(_simple_plan())
        self.assertIn('fixture.attack_goals.last_n', two.names)
        self.assertIn('hist.points_scored.venue10', two.names)
        self.assertNotIn('fixture.attack_goals.mean5', two.names)

    def test_ids_can_be_left_out(self):
        season, _saka = _fixture_world()
        plan = _simple_plan()
        train, _predict, _vocab = build_dataset([season], plan, predict_season=season)
        table = flatten(train, plan, features=2, ids=False)
        self.assertEqual(['position'], table.categorical)
        self.assertFalse(any(name.endswith('team_id_embedding') for name in table.names))
        with self.assertRaises(ValueError):
            flatten(train, plan, features=9)


def _synthetic(rows=3000, seed=0):
    random = numpy.random.RandomState(seed)
    form = random.uniform(0, 6, rows)
    noise = random.uniform(-1, 1, rows)
    team = random.randint(1, 5, rows).astype(numpy.float32)
    values = numpy.column_stack([form, noise, team]).astype(numpy.float32)
    labels = numpy.maximum(0.0, form + 0.5 * (team == 2) + random.normal(0, 0.5, rows))
    years = numpy.where(numpy.arange(rows) < rows * 0.75, 23, 24)
    return FeatureTable(['form', 'noise', 'team'], values, ['team']), labels, years


class FitTests(unittest.TestCase):
    def test_fit_uses_the_latest_season_to_pick_rounds_and_learns_the_signal(self):
        table, labels, years = _synthetic()
        fitted = fit(table, labels, years, max_rounds=300, early_stop=30, verbose=False)
        self.assertEqual(24, fitted.report['holdout_season'])
        self.assertLessEqual(fitted.report['best_round'], 300)
        predicted = fitted.predict(table)
        self.assertTrue((predicted >= 0).all())
        self.assertGreater(numpy.corrcoef(predicted, labels)[0, 1], 0.9)
        gains = fitted.booster.feature_importance(importance_type='gain')
        self.assertGreater(gains[0], gains[1])

    def test_predict_refuses_other_columns(self):
        table, labels, years = _synthetic(rows=600)
        fitted = fit(table, labels, years, max_rounds=40, early_stop=10, verbose=False)
        other = FeatureTable(['noise', 'form', 'team'], table.values, ['team'])
        with self.assertRaises(ValueError):
            fitted.predict(other)

    def test_monotone_rule_holds_even_against_the_data(self):
        random = numpy.random.RandomState(1)
        rows = 3000
        form = random.uniform(0, 6, rows)
        noise = random.uniform(-1, 1, rows)
        team = random.randint(1, 5, rows).astype(numpy.float32)
        # Points dip in the middle of the form range: the free model follows it,
        # the 'form never lowers points' rule must not.
        labels = numpy.maximum(0.0, form - 3.0 * ((form > 2) & (form < 4)) + random.normal(0, 0.3, rows))
        years = numpy.where(numpy.arange(rows) < rows * 0.75, 23, 24)
        table = FeatureTable(['hist.xG.mean5', 'noise', 'team'],
                             numpy.column_stack([form, noise, team]).astype(numpy.float32), ['team'])
        grid = numpy.linspace(0, 6, 61)
        probe = FeatureTable(table.names, numpy.column_stack(
            [grid, numpy.zeros_like(grid), numpy.full_like(grid, 2)]).astype(numpy.float32), ['team'])
        free = fit(table, labels, years, max_rounds=200, early_stop=30, verbose=False).predict(probe)
        ruled = fit(table, labels, years, max_rounds=200, early_stop=30, verbose=False, rules='safe')
        self.assertIn('1 columns rising', ruled.report['rules'])
        steps = numpy.diff(ruled.predict(probe))
        self.assertTrue((steps >= -1e-9).all())
        self.assertTrue((numpy.diff(free) < -0.1).any())

    def test_safe_rules_match_the_real_columns(self):
        from premierlearning.lgbm.model import monotone_directions

        season, _saka = _fixture_world()
        plan = _teams_plan()
        train, _predict, _vocab = build_dataset([season], plan, predict_season=season)
        names = flatten(train, plan).names
        rules = dict(zip(names, monotone_directions(names)))
        expected = {
            'hist.xG.per90_10': 1,
            'hist.points_scored.faded3': 1,
            'hist.sixty_share5': 1,
            'hist.since_played': -1,
            'hist.xGC.mean5': -1,
            'context.chance_of_playing': 1,
            'team_hist.xg_for.mean10': 1,
            'team_hist.goals_against.faded10': -1,
            'opp_hist.xg_against.mean5': 1,
            'opp_hist.goals_for.mean38': -1,
            'current_opposition.team_features.goals_conceded_last_n': 1,
            'current_team.team_features.goals_conceded_last_n': -1,
            'team_squad.xG_last_n.sum': 0,
            'hist.count': 0,
            'position': 0,
        }
        for name, direction in expected.items():
            self.assertEqual(direction, rules[name], name)

    def test_seed_average_is_the_mean_of_the_boosters(self):
        table, labels, years = _synthetic(rows=1200)
        fitted = fit(table, labels, years, max_rounds=60, early_stop=10, verbose=False, seeds=3)
        self.assertEqual(2, len(fitted.others))
        self.assertEqual(3, fitted.report['seeds'])
        each = [b.predict(table.values, num_iteration=fitted.rounds) for b in [fitted.booster] + fitted.others]
        numpy.testing.assert_allclose(numpy.maximum(numpy.mean(each, axis=0), 0.0), fitted.predict(table))
        self.assertFalse(numpy.allclose(each[0], each[1]))

    def test_objectives(self):
        self.assertEqual('tweedie', params_for('tweedie')['objective'])
        self.assertEqual([300], params_for('lambdarank')['eval_at'])
        self.assertTrue(params_for('regression', linear=True)['linear_tree'])
        with self.assertRaises(ValueError):
            params_for('quantile')

    def test_ranking_orders_within_gameweeks_and_maps_back_to_points(self):
        table, labels, years = _synthetic(rows=3000)
        groups = years * 100 + numpy.arange(len(labels)) % 4
        with self.assertRaises(ValueError):
            fit(table, labels, years, objective='lambdarank', verbose=False)
        fitted = fit(table, labels, years, objective='lambdarank', max_rounds=300, early_stop=30,
                     verbose=False, groups=groups)
        self.assertIsNotNone(fitted.calibration)
        predicted = fitted.predict(table, groups=groups)
        self.assertGreater(numpy.corrcoef(predicted, labels)[0, 1], 0.85)
        # In points again: the average is close to the real one.
        self.assertLess(abs(predicted.mean() - labels.mean()), 0.3)

    def test_percentiles_within_groups(self):
        from premierlearning.lgbm.model import percentiles

        numpy.testing.assert_allclose([0.0, 1.0, 0.5, 1.0, 0.0, 0.5],
                                      percentiles([1, 9, 5, 3, 1, 2], [1, 1, 1, 2, 2, 2]))
        numpy.testing.assert_allclose([0.5], percentiles([4.0]))

    def test_linear_leaves(self):
        table, labels, years = _synthetic(rows=1500)
        fitted = fit(table, labels, years, max_rounds=60, early_stop=10, verbose=False, linear=True)
        self.assertTrue(fitted.report['linear'])
        self.assertGreater(numpy.corrcoef(fitted.predict(table), labels)[0, 1], 0.9)

    def test_recency_weights(self):
        from premierlearning.lgbm.model import recency_weights

        self.assertIsNone(recency_weights([20, 24], 0))
        numpy.testing.assert_allclose([0.25, 0.5, 1.0], recency_weights([20, 22, 24], 2))

    def _id_world(self, team_effect, rows=20000, seed=3):
        random = numpy.random.RandomState(seed)
        form = random.uniform(0, 6, rows)
        team = random.randint(1, 5, rows).astype(numpy.float32)
        position = random.randint(1, 4, rows).astype(numpy.float32)
        labels = form + team_effect * (team == 2) + random.normal(0, 0.5, rows)
        years = numpy.where(numpy.arange(rows) < rows * 0.75, 23, 24)
        values = numpy.column_stack([form, team, position]).astype(numpy.float32)
        return FeatureTable(['form', 'current_team.team_id_embedding', 'position'], values,
                            ['current_team.team_id_embedding', 'position']), labels, years

    def test_id_stage_keeps_ids_out_of_the_main_model_and_earns_rounds_when_they_help(self):
        table, labels, years = self._id_world(team_effect=1.0)
        fitted = fit(table, labels, years, max_rounds=200, early_stop=20, verbose=False, id_stage=True)
        self.assertEqual(['form', 'position'], fitted.booster.feature_name())
        self.assertGreater(fitted.report['id_stage']['rounds'], 0)
        self.assertLess(fitted.report['id_stage']['rmse_with_ids'], fitted.report['id_stage']['rmse_without_ids'])
        probe = FeatureTable(table.names, numpy.array([[3, 2, 1], [3, 3, 1]], dtype=numpy.float32), table.categorical)
        low_team, high_team = fitted.predict(probe)[1], fitted.predict(probe)[0]
        self.assertGreater(high_team - low_team, 0.5)

    def test_id_stage_is_dropped_when_ids_do_not_help(self):
        table, labels, years = self._id_world(team_effect=0.0)
        fitted = fit(table, labels, years, max_rounds=200, early_stop=20, verbose=False, id_stage=True)
        if fitted.report['id_stage']['rounds'] == 0:
            self.assertIsNone(fitted.id_booster)
        else:
            # Noise may earn a few rounds; it must stay a tiny correction.
            probe = FeatureTable(table.names, numpy.array([[3, t, 1] for t in (1, 2, 3, 4)], dtype=numpy.float32),
                                 table.categorical)
            self.assertLess(numpy.ptp(fitted.predict(probe)), 0.1)
        with self.assertRaises(ValueError):
            fit(table, labels, years, verbose=False, id_stage=True, objective='lambdarank', groups=years)


class _Records:
    """Stands in for TeamTimelines: {code: [record]}, oldest first."""

    def __init__(self, matches):
        from datetime import datetime, timedelta
        from types import SimpleNamespace

        self.records = {}
        for day, home, away, home_goals, away_goals in matches:
            time = datetime(2024, 8, 1) + timedelta(days=day)
            for code, opp, is_home, scored in ((home, away, 1.0, home_goals), (away, home, 0.0, away_goals)):
                self.records.setdefault(code, []).append(SimpleNamespace(
                    time=time, opponent=opp, home=is_home, goals_for=float(scored), xg_for=float(scored),
                    xg_known=1.0))


class FeatureSetThreeTests(unittest.TestCase):
    def _tables(self, priors=None):
        from simulation.season_model_lgbm import flatten_with_seasons

        season, saka = _fixture_world()
        plan = _teams_plan()
        train, predict, _vocab = build_dataset([season], plan, predict_season=season)
        three = flatten_with_seasons(train, plan, [season], features=3, priors=priors)
        return saka, train, predict, plan, season, three

    def test_ratings_follow_results_against_opponents_and_ignore_later_matches(self):
        from premierlearning.lgbm.extras import TeamRatings, day_number

        # Team 1 beats 2 and 3 well; 3 beats 2. Team 4 has never played (promoted).
        matches = [(day, 1, 2, 3, 0) for day in range(0, 60, 10)] + \
                  [(day + 5, 3, 2, 2, 1) for day in range(0, 60, 10)] + \
                  [(day + 3, 1, 3, 2, 1) for day in range(0, 60, 10)]
        ratings = TeamRatings(_Records(matches))
        cut = day_number(ratings_time(70))
        fit = ratings.at(cut)
        attack = {code: ratings.ratings(code, fit, cut)[0] for code in (1, 2, 3)}
        self.assertGreater(attack[1], attack[3])
        self.assertGreater(attack[3], attack[2])
        newcomer_attack, newcomer_defence = ratings.ratings(4, fit, cut)
        self.assertLess(newcomer_attack, 0.0)
        self.assertGreater(newcomer_defence, 0.0)
        # A match after the cut changes nothing.
        later = TeamRatings(_Records(matches + [(80, 2, 1, 9, 0)]))
        numpy.testing.assert_allclose(fit[2], later.at(cut)[2], rtol=1e-6)
        self.assertIsNone(ratings.at(day_number(ratings_time(-5))))

    def test_season_columns_in_the_fixture_world(self):
        saka, train, predict, plan, season, three = self._tables()

        def value(name, fixture=3):
            return float(three.values[_row(train, saka, fixture), three.names.index(name)])

        # Nothing before the first match: no ratings, no conceded history.
        self.assertTrue(math.isnan(value('dc.goals.exp_for', 1)))
        self.assertTrue(math.isnan(value('opp_conceded.points.mean5', 1)))
        # Before fixture 3 Arsenal won 2-0 and drew 1-1 against Chelsea.
        self.assertGreater(value('dc.goals.exp_for'), value('dc.goals.exp_against'))
        self.assertAlmostEqual(math.exp(-value('dc.goals.exp_against')), value('dc.goals.cs'), places=5)
        # Chelsea conceded to midfielders who played: 1 point (fixture 1), 0 (none played in fixture 2).
        self.assertAlmostEqual(0.5, value('opp_conceded.points.mean5'))
        self.assertAlmostEqual(0.2, value('opp_conceded.xg.mean5'), places=5)   # fixture 2's MID xG is unknown
        # League: midfielders' points per opponent match, three matches (1, 0, 0).
        self.assertAlmostEqual(1.5, value('opp_conceded.points.rel5'), places=5)
        from simulation.season_model_lgbm import flatten_with_seasons
        future = flatten_with_seasons(predict, plan, [season], features=3, priors=three.priors)
        self.assertEqual(three.names, future.names)

    def test_shrunk_per90_and_competition_for_place(self):
        priors = {'points_scored': [0.05] * 5}
        saka, train, _predict, _plan, _season, three = self._tables(priors=priors)

        def value(name):
            return float(three.values[_row(train, saka, 3), three.names.index(name)])

        # 1 point in 10 minutes, as if he also played 270 at 0.05 points a minute.
        self.assertAlmostEqual(90.0 * (1 + 270 * 0.05) / (10 + 270), value('hist.points_scored.shrunk90_5'), places=4)
        self.assertEqual(priors, three.priors)
        # His only teammate is a forward: no rival midfielders, a free place.
        self.assertEqual(0.0, value('place.rivals'))
        self.assertEqual(0.0, value('place.slot_rival_minutes'))
        self.assertAlmostEqual(5.0, value('place.own_minus_slot'))
        # Without priors given, the training batch works them out per position.
        _s, _t, _p, _pl, _se, own = self._tables()
        self.assertEqual(5, len(own.priors['points_scored']))


def ratings_time(day):
    from datetime import datetime, timedelta

    return datetime(2024, 8, 1) + timedelta(days=day)


def _staged_data(rows=6000, seed=1):
    random = numpy.random.RandomState(seed)
    position = random.randint(1, 5, rows).astype(numpy.float32)
    available = random.uniform(0, 1, rows)
    skill = random.uniform(0, 1, rows)
    defence = random.uniform(0, 1, rows)
    draw = random.uniform(0, 1, rows)
    minutes = numpy.where(draw < available * 0.8, 90.0, numpy.where(draw < available, 30.0, 0.0))
    clean = (random.uniform(0, 1, rows) < 0.2 + 0.5 * defence).astype(numpy.float64)
    cs_points = numpy.array([{1: 4, 2: 4, 3: 1, 4: 0}[int(p)] for p in position])
    rest = 3.0 * skill + random.normal(0, 0.3, rows)
    points = numpy.where(minutes >= 60, 2 + rest + cs_points * clean, numpy.where(minutes > 0, 1 + rest, 0.0))
    clean = numpy.where(minutes >= 60, clean, 0.0)
    values = numpy.column_stack([position, available, skill, defence]).astype(numpy.float32)
    years = numpy.where(numpy.arange(rows) < rows * 0.75, 23, 24)
    table = FeatureTable(['position', 'available', 'skill', 'defence'], values, ['position'])
    return table, points, years, minutes, clean


class StagedTests(unittest.TestCase):
    def test_stages_learn_minutes_clean_sheets_and_the_rest(self):
        from premierlearning.lgbm.staged import fit_staged

        table, points, years, minutes, clean = _staged_data()
        fitted = fit_staged(table, points, years, minutes, clean, verbose=False)
        bands = fitted.predict_bands(table)
        numpy.testing.assert_allclose(numpy.ones(len(bands)), bands.sum(axis=1), rtol=1e-6)
        predicted = fitted.predict(table)
        self.assertTrue((predicted >= 0).all())
        available = table.column('available')
        self.assertGreater(bands[available > 0.9, 2].mean(), bands[available < 0.1, 2].mean() + 0.5)
        self.assertLess(abs(predicted.mean() - points.mean()), 0.15)
        # Against the true expectation (the draws of minutes and clean sheets are noise).
        position = table.column('position').astype(int)
        cs_points = numpy.array([{1: 4, 2: 4, 3: 1, 4: 0}[p] for p in position])
        skill, defence = table.column('skill'), table.column('defence')
        truth = (0.8 * available * (2 + 3 * skill + cs_points * (0.2 + 0.5 * defence))
                 + 0.2 * available * (1 + 3 * skill))
        self.assertGreater(numpy.corrcoef(predicted, truth)[0, 1], 0.95)
        self.assertIn('holdout', fitted.report)
        with self.assertRaises(ValueError):
            fitted.predict(FeatureTable(['other'] * 4, table.values, []))

    def test_season_model_staged_on_feature_set_three(self):
        from unittest import mock

        from simulation import season_model_lgbm

        season, saka = _fixture_world()
        season.calculate_player_points_over_next_5_gameweeks = lambda: None
        plan = _teams_plan()
        with mock.patch.object(season_model_lgbm, 'load_plan', return_value=plan):
            models = season_model_lgbm.FixedSeasonModelsLGBM.fit(
                _Runner(season), plan_name='simple_h1_teams', features=3, staged=True)
        self.assertTrue(models.priors)
        models.predict(_Runner(season, as_of_gameweek=4), already_prepared=True)
        future = saka.future_matches[0]
        self.assertEqual('raw', future.points_basis)
        self.assertTrue(numpy.isfinite(future.points))
        self.assertAlmostEqual(1.0, sum(future.minutes_bands), places=5)

    def test_tweedie_trains_on_negative_points(self):
        table, labels, years = _synthetic(rows=1500)
        labels = labels.copy()
        labels[:20] = -1.0
        fitted = fit(table, labels, years, objective='tweedie', max_rounds=80, early_stop=10, verbose=False)
        self.assertTrue((fitted.predict(table) >= 0).all())

    def test_builder_and_experiments(self):
        from simulation.model_eval.experiments import EXPERIMENTS
        from simulation.prediction_builder import HistoricalPredictionBuilder

        base = dict(lightgbm=True, train_once_per_season=True, lgbm_plan='simple_h1_b1', lgbm_features=3,
                    lgbm_ids=False, lgbm_seeds=3)
        v4 = HistoricalPredictionBuilder(None, **base)
        v5 = HistoricalPredictionBuilder(None, lgbm_staged=True, **base)
        self.assertNotIn('lgbm_staged', v4.config)
        self.assertEqual('minutes_clean_sheet_rest', v5.config['lgbm_staged'])
        self.assertNotEqual(v4.metadata['model_config_hash'], v5.metadata['model_config_hash'])
        with self.assertRaises(ValueError):
            HistoricalPredictionBuilder(None, lgbm_staged=True, lgbm_objective='tweedie', **base)
        v3 = EXPERIMENTS['lgbm_v3']['args']
        self.assertEqual(v3 + ['--lgbm-objective', 'tweedie'], EXPERIMENTS['lgbm_v3_tweedie']['args'])
        v4_args = EXPERIMENTS['lgbm_v4']['args']
        self.assertEqual('3', v4_args[v4_args.index('--lgbm-features') + 1])
        self.assertEqual(len(v3), len(v4_args))
        self.assertEqual(v4_args + ['--lgbm-staged'], EXPERIMENTS['lgbm_v5']['args'])
        for name in ('lgbm_v3_tweedie', 'lgbm_v4', 'lgbm_v5'):
            self.assertNotIn(name + '_every5', EXPERIMENTS)


class _Runner:
    def __init__(self, season, as_of_gameweek=1):
        self.current_season = season
        self.past_seasons = []
        self.as_of_gameweek = as_of_gameweek

    def process_player_matches(self):
        pass

    def _trained_through_gameweek(self):
        return None


class SeasonModelTests(unittest.TestCase):
    def test_fit_and_predict_store_raw_points(self):
        from unittest import mock

        from simulation import season_model_lgbm

        season, saka = _fixture_world()
        season.calculate_player_points_over_next_5_gameweeks = lambda: None
        plan = _simple_plan()
        with mock.patch.object(season_model_lgbm, 'load_plan', return_value=plan):
            models = season_model_lgbm.FixedSeasonModelsLGBM.fit(_Runner(season), plan_name='simple_h1')
        models.predict(_Runner(season, as_of_gameweek=4), already_prepared=True)
        future = saka.future_matches[0]
        self.assertEqual('raw', future.points_basis)
        self.assertTrue(numpy.isfinite(future.points))
        self.assertGreaterEqual(future.points, 0.0)

    def test_builder_needs_train_once_and_hashes_apart_from_v2(self):
        from simulation.prediction_builder import HistoricalPredictionBuilder

        with self.assertRaises(ValueError):
            HistoricalPredictionBuilder(None, lightgbm=True)
        with self.assertRaises(ValueError):
            HistoricalPredictionBuilder(None, lightgbm=True, learning_v2=True, train_once_per_season=True)
        lgbm = HistoricalPredictionBuilder(None, lightgbm=True, train_once_per_season=True)
        lgbm_b1 = HistoricalPredictionBuilder(None, lightgbm=True, train_once_per_season=True,
                                              lgbm_plan='simple_h1_b1')
        net = HistoricalPredictionBuilder(None, learning_v2=True, train_once_per_season=True)
        ruled = HistoricalPredictionBuilder(None, lightgbm=True, train_once_per_season=True,
                                            lgbm_plan='simple_h1_b1', lgbm_rules='safe')
        v3 = HistoricalPredictionBuilder(None, lightgbm=True, train_once_per_season=True,
                                         lgbm_plan='simple_h1_b1', lgbm_features=2, lgbm_ids=False, lgbm_seeds=3)
        staged_ids = HistoricalPredictionBuilder(None, lightgbm=True, train_once_per_season=True,
                                                 lgbm_plan='simple_h1_b1', lgbm_id_stage=True, lgbm_recency=2)
        linear = HistoricalPredictionBuilder(None, lightgbm=True, train_once_per_season=True,
                                             lgbm_plan='simple_h1_b1', lgbm_linear=True)
        hashes = {b.metadata['model_config_hash'] for b in (lgbm, lgbm_b1, net, ruled, v3, staged_ids, linear)}
        self.assertEqual(7, len(hashes))
        for key in ('lgbm_rules', 'lgbm_features', 'lgbm_ids', 'lgbm_seeds', 'lgbm_linear', 'lgbm_id_stage',
                    'lgbm_recency'):
            self.assertNotIn(key, lgbm_b1.config)
        self.assertEqual(('residual', 2.0), (staged_ids.config['lgbm_id_stage'], staged_ids.config['lgbm_recency']))
        with self.assertRaises(ValueError):
            HistoricalPredictionBuilder(None, lightgbm=True, train_once_per_season=True, lgbm_ids=False,
                                        lgbm_id_stage=True)
        self.assertEqual((2, False, 3), (v3.config['lgbm_features'], v3.config['lgbm_ids'], v3.config['lgbm_seeds']))
        self.assertEqual('lightgbm', lgbm.config['model'])


class LiveTests(unittest.TestCase):
    def test_live_lightgbm_stores_raw_points_on_future_fixtures(self):
        from unittest import mock

        from premierlearning.lgbm import live

        season, saka = _fixture_world()
        runner = _Runner(season, as_of_gameweek=4)
        for version, plan in (('v3', _simple_plan()), ('v6', _teams_plan())):
            season, saka = _fixture_world()
            runner = _Runner(season, as_of_gameweek=4)
            with mock.patch.object(live, 'load_plan', return_value=plan):
                predictions = live.learn_lightgbm(runner, settings={'seeds': 2}, version=version)
            self.assertEqual(len(predictions), sum(len(p.future_matches) for p in season.players
                                                   if not getattr(p, 'skip_historical_predict', False)))
            future = saka.future_matches[0]
            self.assertEqual('raw', future.points_basis, version)
            self.assertTrue(numpy.isfinite(future.points), version)
            self.assertGreaterEqual(future.points, 0.0, version)
        self.assertEqual(live.V3['plan'], 'simple_h1_b1')

    def test_live_v6_matches_the_evaluated_lgbm_v6(self):
        from premierlearning.lgbm import live
        from simulation.model_eval.experiments import EXPERIMENTS

        args = EXPERIMENTS['lgbm_v6']['args']
        six = live.VERSIONS['v6']
        self.assertEqual('v6', live.live_version(['x']))
        self.assertEqual('v3', live.live_version(['x', '--lgbm-version', 'v3']))
        self.assertEqual(six['plan'], args[args.index('--lgbm-plan') + 1])
        self.assertEqual(six['features'], int(args[args.index('--lgbm-features') + 1]))
        self.assertEqual(six['seeds'], int(args[args.index('--lgbm-seeds') + 1]))
        self.assertIn('--lgbm-no-ids', args)
        self.assertFalse(six['ids'])
        self.assertNotIn('--lgbm-objective', args)
        self.assertEqual('regression', six['objective'])

    def test_runner_dispatches_on_both_flags(self):
        from unittest import mock

        from premierlearning import runner as runner_module

        stub = mock.Mock()
        with mock.patch.object(runner_module.sys, 'argv', ['x', '--learning-v2', '--lightgbm']), \
                mock.patch('premierlearning.lgbm.live.learn_lightgbm') as learn:
            runner_module.Runner.learn(stub)
        learn.assert_called_once_with(stub)


class RetrainEveryTests(unittest.TestCase):
    def _builder(self, **kwargs):
        from simulation.prediction_builder import HistoricalPredictionBuilder

        return HistoricalPredictionBuilder(None, lightgbm=True, train_once_per_season=True,
                                           lgbm_plan='simple_h1_b1', **kwargs)

    def test_schedule_and_hash(self):
        from simulation.prediction_builder import HistoricalPredictionBuilder

        with self.assertRaises(ValueError):
            HistoricalPredictionBuilder(None, lightgbm=True, retrain_every=5)
        once, every = self._builder(), self._builder(retrain_every=5)
        self.assertEqual('once_per_season', once.config['training_schedule'])
        self.assertEqual('every_5_gameweeks', every.config['training_schedule'])
        self.assertNotEqual(once.metadata['model_config_hash'], every.metadata['model_config_hash'])
        self.assertEqual([1, 1, 1, 1, 1, 6, 6, 11, 36, 36],
                         [every.fit_gameweek(g) for g in (1, 2, 3, 4, 5, 6, 10, 11, 36, 38)])
        self.assertEqual(1, once.fit_gameweek(30))

    def test_refits_every_n_and_skips_complete_snapshots_without_fitting(self):
        from unittest import mock

        builder = self._builder(retrain_every=5, gameweek_to=12)
        fits, built = [], []
        runners = {}

        def new_runner(season, gameweek):
            runners[gameweek] = mock.Mock(as_of_gameweek=gameweek)
            return runners[gameweek]

        def fit(runner):
            fits.append(runner.as_of_gameweek)
            return 'models@%s' % runner.as_of_gameweek

        def build_one(season, gameweek, fixed_models=None, prepared_runner=None):
            built.append((gameweek, fixed_models, prepared_runner is not None))
            return True

        complete = {(25, g) for g in (1, 2, 3, 4, 5, 6, 7)}
        with mock.patch.object(builder, '_complete_snapshots', return_value=complete), \
                mock.patch.object(builder, '_new_runner', side_effect=new_runner), \
                mock.patch.object(builder, '_fit_fixed_models', side_effect=fit), \
                mock.patch.object(builder, 'build_one', side_effect=build_one):
            count = builder._build_retraining(25, set(range(1, 39)))
        self.assertEqual([6, 11], fits)
        self.assertEqual(5, count)
        self.assertEqual([(8, 'models@6', False), (9, 'models@6', False), (10, 'models@6', False),
                          (11, 'models@11', True), (12, 'models@11', False)], built)

    def test_max_fits_stops_before_the_next_fit(self):
        """PL_RETRAIN_MAX_FITS=1: one fit and its snapshots per process (net fits leak memory)."""
        from unittest import mock

        builder = self._builder(retrain_every=5, gameweek_to=12)
        fits, built = [], []

        def fit(runner):
            fits.append(runner.as_of_gameweek)
            return 'models@%s' % runner.as_of_gameweek

        def build_one(season, gameweek, fixed_models=None, prepared_runner=None):
            built.append(gameweek)
            return True

        complete = {(25, g) for g in (1, 2, 3, 4, 5, 6, 7)}
        with mock.patch.dict('os.environ', {'PL_RETRAIN_MAX_FITS': '1'}), \
                mock.patch.object(builder, '_complete_snapshots', return_value=complete), \
                mock.patch.object(builder, '_new_runner', side_effect=lambda s, g: mock.Mock(as_of_gameweek=g)), \
                mock.patch.object(builder, '_fit_fixed_models', side_effect=fit), \
                mock.patch.object(builder, 'build_one', side_effect=build_one):
            count = builder._build_retraining(25, set(range(1, 39)))
        self.assertEqual([6], fits)
        self.assertEqual([8, 9, 10], built)
        self.assertEqual(3, count)

    def test_lgbm_holdout_is_the_newest_38_gameweeks(self):
        random = numpy.random.RandomState(4)
        weeks = [(23, g) for g in range(1, 39)] + [(24, g) for g in range(1, 39)] + [(25, g) for g in range(1, 6)]
        rows = []
        for year, gameweek in weeks:
            rows.extend([(year, gameweek)] * 40)
        years = numpy.array([year for year, _ in rows])
        groups = numpy.array([year * 100 + gameweek for year, gameweek in rows])
        form = random.uniform(0, 6, len(rows))
        labels = form + random.normal(0, 0.5, len(rows))
        table = FeatureTable(['form'], form.reshape(-1, 1).astype(numpy.float32), [])
        fitted = fit(table, labels, years, max_rounds=60, early_stop=10, verbose=False, groups=groups)
        self.assertEqual(2406, fitted.report['holdout_from'])
        self.assertEqual(38 * 40, fitted.report['holdout_rows'])
        from premierlearning.lgbm.model import describe

        self.assertIn('last 38 GWs (from 24 GW6)', describe(fitted))
        # A GW1 fit (two whole seasons): the hold-out is the latest season, as before.
        first = numpy.array([year < 25 for year in years])
        table_one = FeatureTable(['form'], table.values[first], [])
        one = fit(table_one, labels[first], years[first], max_rounds=60, early_stop=10, verbose=False,
                  groups=groups[first])
        self.assertEqual(2401, one.report['holdout_from'])
        self.assertIn('held-out season 24', describe(one))

    def test_experiments_and_pick(self):
        from simulation.model_eval import pick_lgbm
        from simulation.model_eval.experiments import EXPERIMENTS

        net = EXPERIMENTS['v2_b1_every5']['args']
        self.assertEqual(EXPERIMENTS['v2_b1_context']['args'] + ['--retrain-every', '5'], net)
        names = pick_lgbm.candidates()
        self.assertIn('lgbm_v3', names)
        self.assertIn('lgbm_v3_rank', names)
        self.assertNotIn('lgbm_v3_every5', names)
        found = {'lgbm_v3': (1.87, 'a'), 'lgbm_v3_rank': (1.95, 'b'), 'lgbm_v3_mix50_v2_b0_seed1': (2.01, 'a'),
                 'lgbm_v3_linear': (2.5, 'b')}
        self.assertEqual('lgbm_v3_rank', pick_lgbm.pick(found, names, built={'lgbm_v3', 'lgbm_v3_rank'})[0])
        self.assertEqual('lgbm_v3_linear', pick_lgbm.pick(found, names)[0])
        tied = {'lgbm_v3': (1.8729656, 'a'), 'lgbm_v3_id_stage': (1.8729656, 'b')}
        self.assertEqual('lgbm_v3', pick_lgbm.pick(tied, names)[0])


if __name__ == '__main__':
    unittest.main()
