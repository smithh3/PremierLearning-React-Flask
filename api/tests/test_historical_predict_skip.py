import unittest

from simulation.historical_data import (
    VaastavSeasonData,
    season_usage_from_rows,
    skip_historical_predict_at_deadline,
    skip_historical_predict_for_season_usage,
)


class _Source:
    """Just enough of VaastavSeasonData for predict_skips_at."""

    def __init__(self, player_rows, codes, previous_usage):
        self.player_rows = player_rows
        self.players_raw = {pid: {'code': code} for pid, code in codes.items()}
        self._previous = previous_usage
        self._predict_skips = {}

    def previous_season_usage_by_code(self):
        return self._previous

    predict_skips_at = VaastavSeasonData.predict_skips_at


class HistoricalPredictSkipTests(unittest.TestCase):
    def test_skip_few_appearances_or_few_points(self):
        self.assertTrue(skip_historical_predict_for_season_usage(3, 80))
        self.assertTrue(skip_historical_predict_for_season_usage(10, 20))
        self.assertTrue(skip_historical_predict_for_season_usage(0, 0))
        self.assertFalse(skip_historical_predict_for_season_usage(4, 21))

    def test_usage_counts_appearances_with_minutes(self):
        totals = season_usage_from_rows({
            1: [
                {'minutes': 90, 'total_points': 6},
                {'minutes': 0, 'total_points': 0},
                {'minutes': 45, 'total_points': 2},
            ],
            2: [
                {'minutes': 90, 'total_points': 12},
            ],
        })
        self.assertEqual((2, 8.0), totals[1])
        self.assertEqual((1, 12.0), totals[2])

    def test_deadline_rule(self):
        # Recent minutes always predict.
        self.assertFalse(skip_historical_predict_at_deadline(10, True, True, 1, 1))
        # No last-season record: kept until five gameweeks have been played.
        self.assertFalse(skip_historical_predict_at_deadline(5, False, False, 0, 0))
        self.assertTrue(skip_historical_predict_at_deadline(6, False, False, 0, 0))
        # Otherwise the usual thresholds on last season plus this season so far.
        self.assertTrue(skip_historical_predict_at_deadline(1, False, True, 3, 80))
        self.assertFalse(skip_historical_predict_at_deadline(1, False, True, 4, 21))

    def test_skips_ignore_rows_at_or_after_the_deadline(self):
        rows = {
            # Plays only from GW10: the old whole-season rule predicted him at GW1.
            1: [{'round': gw, 'minutes': 90, 'total_points': 6} for gw in range(10, 30)],
            # Regular last season, nothing yet this season.
            2: [],
            # Barely played last season, nothing recent.
            3: [{'round': 1, 'minutes': 10, 'total_points': 1}],
        }
        source = _Source(rows, {1: 101, 2: 102, 3: 103}, {101: (2, 5.0), 102: (30, 120.0), 103: (1, 1.0)})
        self.assertEqual({1, 3}, source.predict_skips_at(8))
        self.assertEqual({3}, source.predict_skips_at(11))
