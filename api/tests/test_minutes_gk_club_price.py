import math
import unittest
from types import SimpleNamespace

from premierlearning.clean_match import (
    LAST_SEASON_POINTS_PER_90_KEY,
    MINUTES_ONLY_ROOT_KEYS,
    MOST_EXPENSIVE_POS_X_GKP_X_EARLY_KEY,
    SEASON_MINS_PER_MATCH_KEY,
    TEAM_XG_NEXT_KEY,
    apply_feature_interactions,
    get_inputs_from_data_key,
)
from premierlearning.config_constants import PRIOR_EARLY_TAU
from premierlearning.match import AbstractMatch
from premierlearning.player import mark_most_expensive_in_position


def _clean(**overrides):
    row = {
        'is_gkp': 0,
        'is_def': 0,
        'is_mid': 0,
        'is_att': 0,
        'new_stats_included': 0,
        'past_stats': {},
        TEAM_XG_NEXT_KEY: 0,
        'matches_played': 0,
        'last_season_minutes': 0,
        'last_season_mins_per_match': 0.0,
        'club_move_team_diff': 1.0,
        LAST_SEASON_POINTS_PER_90_KEY: 0.0,
        'is_most_expensive_in_pos': 0,
    }
    row.update(overrides)
    apply_feature_interactions(row)
    return row


class MostExpensiveGkEarlyTests(unittest.TestCase):
    def test_starting_gk_is_one_early_and_fades(self):
        early = _clean(is_gkp=1, is_most_expensive_in_pos=1, matches_played=0)
        self.assertAlmostEqual(1.0, early[MOST_EXPENSIVE_POS_X_GKP_X_EARLY_KEY])
        later = _clean(is_gkp=1, is_most_expensive_in_pos=1, matches_played=6)
        self.assertAlmostEqual(
            math.exp(-6.0 / PRIOR_EARLY_TAU),
            later[MOST_EXPENSIVE_POS_X_GKP_X_EARLY_KEY],
        )

    def test_reserve_gk_and_outfield_are_zero(self):
        reserve = _clean(is_gkp=1, is_most_expensive_in_pos=0, matches_played=0)
        self.assertAlmostEqual(0.0, reserve[MOST_EXPENSIVE_POS_X_GKP_X_EARLY_KEY])
        outfield = _clean(is_att=1, is_most_expensive_in_pos=1, matches_played=0)
        self.assertAlmostEqual(0.0, outfield[MOST_EXPENSIVE_POS_X_GKP_X_EARLY_KEY])

    def test_minutes_inputs_include_the_interaction(self):
        row = _clean(is_gkp=1, is_most_expensive_in_pos=1, matches_played=0)
        feats = get_inputs_from_data_key([row], ['minutes'], True)[0]
        self.assertAlmostEqual(1.0, feats[MOST_EXPENSIVE_POS_X_GKP_X_EARLY_KEY])

    def test_marks_the_highest_price_in_club_and_position(self):
        starter = SimpleNamespace(team_id=1, position=1, current_cost=55)
        reserve = SimpleNamespace(team_id=1, position=1, current_cost=40)
        other_club = SimpleNamespace(team_id=2, position=1, current_cost=45)
        defender = SimpleNamespace(team_id=1, position=2, current_cost=70)
        mark_most_expensive_in_position([starter, reserve, other_club, defender])
        self.assertEqual(1, starter.is_most_expensive_in_pos)
        self.assertEqual(0, reserve.is_most_expensive_in_pos)
        self.assertEqual(1, other_club.is_most_expensive_in_pos)
        self.assertEqual(1, defender.is_most_expensive_in_pos)


class SeasonMinsPerMatchTests(unittest.TestCase):
    def test_minutes_net_leads_with_this_season_rate(self):
        self.assertEqual(SEASON_MINS_PER_MATCH_KEY, MINUTES_ONLY_ROOT_KEYS[0])

    def test_minutes_inputs_include_season_mins_per_match(self):
        row = _clean()
        row[SEASON_MINS_PER_MATCH_KEY] = 72.0
        feats = get_inputs_from_data_key([row], ['minutes'], True)[0]
        self.assertAlmostEqual(72.0, feats[SEASON_MINS_PER_MATCH_KEY])
        self.assertEqual(SEASON_MINS_PER_MATCH_KEY, MINUTES_ONLY_ROOT_KEYS[0])

    def test_mean_of_this_season_fixtures_includes_dnps(self):
        self.assertEqual(0.0, AbstractMatch._mean_or_zero([]))
        self.assertAlmostEqual(45.0, AbstractMatch._mean_or_zero([90.0, 0.0, 45.0]))
