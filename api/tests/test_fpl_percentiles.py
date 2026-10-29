import unittest

from fpl_manager.percentiles import (
    beaten_percent,
    describe_gameweek,
    describe_season,
    ordinal,
)


CURVES = {
    'seasons': {
        '2025/26': {
            'field_size': 10000,
            'samples': [[40, 8000], [80, 2000]],
        },
    },
    'gameweeks': {
        '2025/26': {
            '1': {
                'field_size': 10000,
                'samples': [[40, 8000], [80, 2000]],
            },
        },
    },
}


class PercentileTests(unittest.TestCase):
    def test_a_midway_score_is_the_geometric_rank(self):
        self.assertAlmostEqual(
            60.0,
            beaten_percent(60, [[40, 8000], [80, 2000]], 10000),
            places=5,
        )

    def test_scores_outside_the_sample_stay_on_the_end_rank(self):
        self.assertAlmostEqual(20.0, beaten_percent(10, [[40, 8000], [80, 2000]], 10000))
        self.assertAlmostEqual(80.0, beaten_percent(100, [[40, 8000], [80, 2000]], 10000))

    def test_descriptions_name_the_season_and_the_gameweek(self):
        self.assertEqual('20th', ordinal(20))
        self.assertEqual('1st', ordinal(1))
        self.assertEqual('22nd', ordinal(22))
        self.assertEqual('11th', ordinal(11))
        self.assertEqual(
            '60th percentile of 2025/26 gameweek 1',
            describe_gameweek(25, 1, 60, CURVES),
        )
        self.assertEqual(
            '60th percentile of 2025/26 finishes',
            describe_season(25, 60, CURVES),
        )
        self.assertEqual('', describe_gameweek(25, 9, 60, CURVES))
