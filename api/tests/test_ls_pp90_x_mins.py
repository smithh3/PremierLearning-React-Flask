import os
import unittest
from types import SimpleNamespace
from unittest import mock

from premierlearning.clean_match import (
    LAST_SEASON_POINTS_PER_90_KEY,
    LAST_SEASON_PP90_X_EARLY_KEY,
    LAST_SEASON_XGI_PER_90_KEY,
    LS_PP90_X_MINS_KEY,
    SEASON_DEFCON_PER_90_KEY,
    SEASON_POINTS_PER_90_KEY,
    SEASON_XGC_PER_90_KEY,
    SEASON_XGI_PER_90_KEY,
    TEAM_XG_NEXT_KEY,
    TEAM_XC_NEXT_KEY,
    TWO_SEASON_POINTS_PER_90_KEY,
    TWO_SEASON_PP90_X_EARLY_KEY,
    TWO_SEASON_XGI_PER_90_KEY,
    apply_feature_interactions,
    get_inputs_from_data_key,
    get_standard_stat_keys,
    use_ls_pp90_x_mins,
)
from premierlearning.player import Player
from premierlearning.runner import parse_as_of_args, parse_train_as_of_gw


class LastSeasonPp90MinsTests(unittest.TestCase):
    def test_interaction_is_pp90_times_minutes_fraction(self):
        clean = {
            'is_gkp': 0,
            'is_def': 0,
            'is_mid': 1,
            'is_att': 0,
            'new_stats_included': 1,
            'past_stats': {'defensive_contribution': {'avg': 0}},
            TEAM_XG_NEXT_KEY: 0,
            'matches_played': 0,
            'last_season_minutes': 0,
            'last_season_mins_per_match': 30.0,
            'club_move_team_diff': 1.0,
            LAST_SEASON_POINTS_PER_90_KEY: 12.0,
        }
        apply_feature_interactions(clean)
        self.assertAlmostEqual(4.0, clean[LS_PP90_X_MINS_KEY])

    def test_full_minutes_keeps_the_raw_pp90(self):
        clean = {
            'is_gkp': 0,
            'is_def': 0,
            'is_mid': 0,
            'is_att': 1,
            'new_stats_included': 0,
            'past_stats': {},
            TEAM_XG_NEXT_KEY: 0,
            'matches_played': 10,
            'last_season_minutes': 3420,
            LAST_SEASON_POINTS_PER_90_KEY: 8.0,
            'last_season_mins_per_match': 90.0,
        }
        apply_feature_interactions(clean)
        self.assertAlmostEqual(8.0, clean[LS_PP90_X_MINS_KEY])

    def test_feature_is_off_by_default_and_on_via_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('PL_LS_PP90_X_MINS', None)
            self.assertFalse(use_ls_pp90_x_mins())
            self.assertNotIn(LS_PP90_X_MINS_KEY, get_standard_stat_keys())
        with mock.patch.dict(os.environ, {'PL_LS_PP90_X_MINS': '1'}):
            self.assertTrue(use_ls_pp90_x_mins())
            self.assertIn(LS_PP90_X_MINS_KEY, get_standard_stat_keys())
        with mock.patch('sys.argv', ['premierlearning', '--ls-pp90-x-mins']):
            os.environ.pop('PL_LS_PP90_X_MINS', None)
            self.assertTrue(use_ls_pp90_x_mins())

    def test_as_of_flags_must_be_paired(self):
        self.assertEqual((None, None), parse_as_of_args(['premierlearning']))
        self.assertEqual(
            (25, 20),
            parse_as_of_args(
                ['premierlearning', '--as-of-season', '25', '--as-of-gw', '20']
            ),
        )
        with self.assertRaises(SystemExit):
            parse_as_of_args(['premierlearning', '--as-of-gw', '20'])

    def test_train_as_of_gw_is_optional(self):
        self.assertIsNone(parse_train_as_of_gw(['premierlearning']))
        self.assertEqual(
            1,
            parse_train_as_of_gw(
                ['premierlearning', '--as-of-season', '25', '--as-of-gw', '15',
                 '--train-as-of-gw', '1']
            ),
        )


    def test_early_season_interaction_does_not_scale_raw_pp90(self):
        clean = {
            'is_gkp': 0,
            'is_def': 0,
            'is_mid': 1,
            'is_att': 0,
            'new_stats_included': 1,
            'past_stats': {'defensive_contribution': {'avg': 0}},
            TEAM_XG_NEXT_KEY: 0,
            'matches_played': 0,
            'last_season_minutes': 0,
            'last_season_mins_per_match': 90.0,
            LAST_SEASON_POINTS_PER_90_KEY: 8.0,
            TWO_SEASON_POINTS_PER_90_KEY: 6.0,
        }
        apply_feature_interactions(clean)
        self.assertAlmostEqual(8.0, clean[LAST_SEASON_POINTS_PER_90_KEY])
        self.assertAlmostEqual(8.0, clean[LAST_SEASON_PP90_X_EARLY_KEY])
        self.assertAlmostEqual(6.0, clean[TWO_SEASON_PP90_X_EARLY_KEY])
        keys = get_standard_stat_keys()
        self.assertEqual(len(keys), len(set(keys)))
        self.assertIn(TWO_SEASON_POINTS_PER_90_KEY, keys)
        self.assertIn(LAST_SEASON_PP90_X_EARLY_KEY, keys)
        self.assertNotIn(TWO_SEASON_XGI_PER_90_KEY, keys)
        self.assertNotIn(TWO_SEASON_PP90_X_EARLY_KEY, keys)

    def test_horizon_feature_names_are_unique(self):
        clean = {
            'matches_played': 5,
            'is_gkp': 0,
            'is_def': 0,
            'is_mid': 1,
            'is_att': 0,
            'new_stats_included': 1,
            TEAM_XG_NEXT_KEY: 4.0,
            TEAM_XC_NEXT_KEY: 3.0,
            'team_xg_last_5': 2.0,
            'team_xc_last_5': 1.0,
            LAST_SEASON_POINTS_PER_90_KEY: 8.0,
            LAST_SEASON_XGI_PER_90_KEY: 0.5,
            TWO_SEASON_POINTS_PER_90_KEY: 7.0,
            TWO_SEASON_XGI_PER_90_KEY: 0.4,
            SEASON_POINTS_PER_90_KEY: 6.0,
            SEASON_XGI_PER_90_KEY: 0.35,
            SEASON_XGC_PER_90_KEY: 1.0,
            SEASON_DEFCON_PER_90_KEY: 7.5,
            'past_stats': {
                'total_points': {'last_5_avg': 6.0, 'avg': 5.5},
                'expected_goal_involvements': {'last_5_avg': 0.4, 'avg': 0.3},
                'expected_goals_conceded': {'last_5_avg': 1.1, 'avg': 1.2},
                'defensive_contribution': {'last_5_avg': 8.0, 'avg': 7.0},
            },
        }
        apply_feature_interactions(clean)
        feats = get_inputs_from_data_key(
            [clean], ['total_points', 'expected_goal_involvements',
                      'expected_goals_conceded', 'defensive_contribution'],
            True,
        )[0]
        self.assertEqual(len(feats), len(set(feats)))
        self.assertIn('points_last_3_per_90_excl_null', feats)
        self.assertIn('points_last_7_per_90_excl_null', feats)
        self.assertIn('xgi_last_3_per_90_excl_null', feats)
        self.assertIn('xgc_last_7_per_90_excl_null', feats)
        self.assertIn('defcon_last_3_per_90_excl_null', feats)
        self.assertIn(SEASON_POINTS_PER_90_KEY, feats)
        self.assertIn(SEASON_XGI_PER_90_KEY, feats)
        self.assertIn(SEASON_XGC_PER_90_KEY, feats)
        self.assertIn(SEASON_DEFCON_PER_90_KEY, feats)
        self.assertNotIn(TWO_SEASON_XGI_PER_90_KEY, feats)
        self.assertNotIn(TWO_SEASON_PP90_X_EARLY_KEY, feats)
        self.assertIn(TEAM_XG_NEXT_KEY, feats)
        self.assertNotIn('team_xg_over_n', feats)
        self.assertNotIn('total_points_last_5_avg', feats)


def _rate_match(minutes, points=0, xgi=0, xgc=0, defcon=0, carry=False):
    return SimpleNamespace(
        is_last_season_carry=carry,
        minutes=minutes,
        points=points,
        game_stats={
            'minutes': minutes,
            'total_points': points,
            'expected_goal_involvements': xgi,
            'expected_goals_conceded': xgc,
            'defensive_contribution': defcon,
        },
    )


class SeasonRateExclNullTests(unittest.TestCase):
    def test_last_season_pp90_skips_zero_minute_games(self):
        matches = [
            _rate_match(90, points=6),
            _rate_match(0, points=0),
            _rate_match(90, points=6),
            _rate_match(90, points=6),
        ]
        # 3×90-min 6pt games including appearance points
        self.assertAlmostEqual(18.0 / 270.0, Player.per_min_from_match_list(matches))
        self.assertAlmostEqual(
            18.0 / 270.0,
            Player.per_min_from_match_list(matches + [_rate_match(0, points=0)]),
        )

    def test_last_season_xgi_uses_the_same_appearance_filter(self):
        matches = [
            _rate_match(90, xgi=0.5),
            _rate_match(0, xgi=0.0),
            _rate_match(90, xgi=0.4),
            _rate_match(90, xgi=0.3),
        ]
        self.assertAlmostEqual(
            1.2 / 270.0,
            Player.per_min_from_match_list(matches, 'expected_goal_involvements'),
        )

    def test_needs_more_than_200_minutes(self):
        matches = [_rate_match(90, points=6), _rate_match(90, points=6)]
        self.assertEqual(0.0, Player.per_min_from_match_list(matches))

    def test_skips_last_season_carry_for_this_season_rate(self):
        matches = [
            _rate_match(90, points=12, carry=True),
            _rate_match(90, points=3),
            _rate_match(90, points=3),
            _rate_match(90, points=3),
        ]
        self.assertAlmostEqual(9.0 / 270.0, Player.per_min_from_match_list(matches))

    def test_points_rates_keep_playing_time_points(self):
        matches = [
            _rate_match(90, points=2, xgi=0.5),
            _rate_match(20, points=1, xgi=0.1),
            _rate_match(90, points=6, xgi=0.4),
        ]
        # 2+1+6 over 200 mins — exactly at the 200 floor, so 0
        self.assertEqual(0.0, Player.per_min_from_match_list(matches))
        matches_ok = matches + [_rate_match(90, points=2, xgi=0.2)]
        self.assertAlmostEqual(11.0 / 290.0, Player.per_min_from_match_list(matches_ok))
        self.assertAlmostEqual(
            1.2 / 290.0,
            Player.per_min_from_match_list(matches_ok, 'expected_goal_involvements'),
        )


if __name__ == '__main__':
    unittest.main()
