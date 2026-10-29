import unittest

from simulation.historical_data import (
    season_usage_from_rows,
    skip_historical_predict_for_season_usage,
)


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
