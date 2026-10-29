import unittest

from premierlearning.match import AbstractMatch
from premierlearning.player import Player, is_form_appearance


class FormWindowTests(unittest.TestCase):
    def setUp(self):
        self.player = Player.__new__(Player)

    def test_form_appearance_floor_is_20(self):
        self.assertFalse(is_form_appearance(0))
        self.assertFalse(is_form_appearance(1))
        self.assertFalse(is_form_appearance(19))
        self.assertTrue(is_form_appearance(20))
        self.assertTrue(is_form_appearance(90))

    def test_sub_20_cameos_are_dropped_from_pooled_pp90(self):
        window = self.player.get_stat_window(
            [(1.0, 1.0), (6.0, 90.0), (2.0, 8.0), (8.0, 90.0)],
            per_90=True,
        )
        self.assertAlmostEqual(7.0, window['last_5_avg'])
        self.assertAlmostEqual(7.0, window['avg'])

    def test_cameo_only_window_is_zero(self):
        window = self.player.get_stat_window(
            [(1.0, 1.0), (1.0, 1.0), (2.0, 8.0)],
            per_90=True,
        )
        self.assertEqual(0.0, window['last_5_avg'])
        self.assertEqual(0.0, window['avg'])

    def test_twenty_minute_appearance_stays_in_window(self):
        window = self.player.get_stat_window(
            [(4.0, 20.0), (6.0, 90.0)],
            per_90=True,
        )
        self.assertAlmostEqual(10.0 * 90.0 / 110.0, window['last_5_avg'])

    def test_short_window_uses_last_three_qualifying(self):
        window = self.player.get_stat_window(
            [
                (2.0, 90.0),
                (4.0, 90.0),
                (6.0, 90.0),
                (8.0, 90.0),
                (10.0, 90.0),
            ],
            per_90=True,
        )
        self.assertAlmostEqual(8.0, window['last_5_avg'])
        self.assertAlmostEqual(6.0, window['avg'])


class Last7PtsPerAppearanceTests(unittest.TestCase):
    def test_means_raw_points_over_form_appearances(self):
        row = type('Row', (), {})()
        row._prior_matches = lambda: [
            type('M', (), {'game_stats': {'minutes': 90, 'total_points': 6}})(),
            type('M', (), {'game_stats': {'minutes': 20, 'total_points': 5}})(),
            type('M', (), {'game_stats': {'minutes': 8, 'total_points': 1}})(),
        ]
        # 8-min cameo dropped from form; 6 and 5 remain
        self.assertAlmostEqual(
            5.5,
            AbstractMatch._last_n_pts_per_appearance(row, 7),
        )
