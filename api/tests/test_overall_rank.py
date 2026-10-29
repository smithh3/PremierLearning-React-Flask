import math
import unittest

from simulation.overall_rank import (
    estimate_total_managers,
    format_rank_line,
    interpolate_rank,
    season_label,
)


class OverallRankTests(unittest.TestCase):
    def test_season_label_accepts_short_and_full_years(self):
        self.assertEqual('2025/26', season_label(25))
        self.assertEqual('2025/26', season_label(2025))

    def test_interpolate_rank_uses_log_rank_between_samples(self):
        samples = [
            {'points': 1500, 'rank': 2000000, 'rank_percentage': 20.0},
            {'points': 2000, 'rank': 200000, 'rank_percentage': 2.0},
            {'points': 2500, 'rank': 5000, 'rank_percentage': 0.05},
        ]
        placed = interpolate_rank(2250, samples)
        expected = math.sqrt(200000 * 5000)
        self.assertEqual(round(expected), placed['rank'])
        self.assertAlmostEqual(1.025, placed['top_percent'], places=3)
        self.assertEqual(3, placed['sample_count'])

    def test_field_size_and_rank_line_include_raw_and_percentile(self):
        samples = [
            {'points': 1800, 'rank': 3000000, 'rank_percentage': 30.0},
            {'points': 2200, 'rank': 1000000, 'rank_percentage': 10.0},
        ]
        self.assertEqual(10000000, int(estimate_total_managers(samples)))
        line = format_rank_line({
            'rank': 12400,
            'top_percent': 0.10,
            'ahead_percent': 99.90,
            'total_managers': 12100000,
            'sample_count': 28,
            'season': '2025/26',
        })
        self.assertIn('~12,400', line)
        self.assertIn('of ~12,100,000', line)
        self.assertIn('top 0.10%', line)
        self.assertIn('ahead of 99.90% of managers', line)

    def test_missing_samples_are_reported(self):
        self.assertIsNone(interpolate_rank(2000, []))
        self.assertIn('not available', format_rank_line(None))
