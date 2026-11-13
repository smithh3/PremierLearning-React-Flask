"""Live v2 blend: window split, blend, raw-points scoring, saved minutes bands, email."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy

from squadtools.simulation_strategy import ChipCalendar


def _batch(rows):
    """rows: (owner, gameweek, h1 points)."""
    owners = [SimpleNamespace(name=owner) for owner, _gw, _points in rows]
    by_name = {}
    for index, (owner, _gw, _points) in enumerate(rows):
        owners[index] = by_name.setdefault(owner, owners[index])
    matches = [SimpleNamespace(gameweek=gw, points=points, game_stats={}) for _o, gw, points in rows]
    return SimpleNamespace(owners=owners, matches=matches)


class BlendMathsTests(unittest.TestCase):
    def test_window_points_match_the_harness_spread(self):
        from premierlearning.learning_v2.learn import spread_window_rates, window_points

        rows = [('a', 10, 2.0), ('a', 11, 6.0), ('a', 16, 3.0), ('b', 10, 0.0), ('b', 12, 0.0)]
        rates = numpy.array([4.0, 5.0, 7.0, 1.0, 3.0])
        h1 = numpy.array([points for _o, _g, points in rows])
        expected_batch = _batch(rows)
        spread_window_rates(expected_batch, rates, 10, 5)
        expected = [match.points for match in expected_batch.matches]
        actual = window_points(_batch(rows), h1, rates, 10, 5)
        numpy.testing.assert_allclose(expected, actual)
        # Player a: window total 4 + 5 = 9 split 2:6; GW16 is outside and keeps h1.
        numpy.testing.assert_allclose([2.25, 6.75, 3.0], actual[:3])
        # Player b had no h1 points: the total splits evenly.
        numpy.testing.assert_allclose([2.0, 2.0], actual[3:])

    def test_blend_is_three_quarters_base(self):
        from premierlearning.learning_v2.learn import blend_points

        numpy.testing.assert_allclose([7.0, 1.0], blend_points([8.0, 0.0], [4.0, 4.0]))

    def test_stored_points_are_marked_raw_and_keep_their_bands(self):
        from premierlearning.learning_v2.learn import store_minutes_bands, store_points

        matches = [SimpleNamespace(game_stats={}), SimpleNamespace(game_stats=None)]
        store_points(matches, [3.5, 1.25])
        store_minutes_bands(matches, numpy.array([[0.1, 0.2, 0.7], [0.5, 0.25, 0.25]]))
        self.assertEqual(['raw', 'raw'], [match.points_basis for match in matches])
        self.assertEqual(3.5, matches[0].game_stats['total_points'])
        self.assertEqual((0.5, 0.25, 0.25), matches[1].minutes_bands)

    def test_raw_points_skip_the_minutes_mapping_and_chance_cut(self):
        from squadtools.availability import match_optimiser_points

        raw = SimpleNamespace(points_basis='raw', points=4.2, minutes=0, game_stats={},
                              chance_of_playing=25, predicted_pp90=None)
        self.assertAlmostEqual(4.2, match_optimiser_points(raw))
        old = SimpleNamespace(points=4.2, minutes=0, game_stats={'minutes': 0},
                              chance_of_playing=None, predicted_pp90=None)
        self.assertEqual(0.0, match_optimiser_points(old))


class SavedBandsTests(unittest.TestCase):
    def test_bands_and_basis_survive_the_database(self):
        from flask import Flask

        from persistence import DB_Handler
        from premierlearning.match import RawMatch

        with tempfile.TemporaryDirectory() as directory:
            previous = os.environ.get('DATABASE_PATH')
            previous_url = os.environ.pop('DATABASE_URL', None)
            os.environ['DATABASE_PATH'] = str(Path(directory) / 'live.db')
            app = Flask(__name__)
            try:
                with app.app_context():
                    db = DB_Handler()
                    db.init_db()
                    match = RawMatch()
                    match.player_id = 7
                    match.minutes = 0
                    match.points = 3.25
                    match.gameweek = 10
                    match.opponent = 'LIV'
                    match.points_basis = 'raw'
                    match.minutes_bands = (0.1, 0.2, 0.7)
                    db.persist_future_match(match, player_id=7)
                    db.connection.commit()
                    loaded = db.get_future_match_jsons()[7][0]
                    restored = DB_Handler.convert_match_from_json(7, loaded)
                    db.close_connection()
            finally:
                if previous is None:
                    os.environ.pop('DATABASE_PATH', None)
                else:
                    os.environ['DATABASE_PATH'] = previous
                if previous_url is not None:
                    os.environ['DATABASE_URL'] = previous_url
        self.assertEqual('raw', restored.points_basis)
        numpy.testing.assert_allclose((0.1, 0.2, 0.7), restored.minutes_bands, rtol=1e-6)
        self.assertAlmostEqual(3.25, restored.points)


def _player(player_id, position, points, bands=(0.1, 0.2, 0.7), name=None):
    matches = [
        SimpleNamespace(
            gameweek=10 + week, opponent='OPP', points=points / 5.0, points_basis='raw',
            minutes=0, game_stats={}, minutes_bands=bands, chance_of_playing=None,
            predicted_pp90=None,
        )
        for week in range(5)
    ]
    return SimpleNamespace(
        id=player_id, first_name=name or 'P%s' % player_id, last_name='', team_name='TEA',
        position=position, current_cost=60, future_matches=matches, matches=[],
        get_points_within_n_gameweeks=lambda gw, n, total=points: float(total),
    )


class EmailTests(unittest.TestCase):
    def _opt(self):
        mids = [_player(100 + index, 3, 40.0 - index) for index in range(30)]
        owned_def = _player(1, 2, 0.2, name='OwnedDefender')
        owned_mid = mids[20]
        bought = _player(200, 4, 0.5, name='BoughtForward')
        sold = mids[25]
        junk = [_player(300 + index, 2, 0.0) for index in range(5)]
        everyone = mids + [owned_def, bought] + junk
        transfer = SimpleNamespace(player_out=sold, player_in=bought, selling_price=60, purchase_price=60)
        return SimpleNamespace(
            squad=SimpleNamespace(next_gameweek=10, built_from_scratch=False,
                                  squad=[owned_def, owned_mid, sold]),
            player_dict={player.id: player for player in everyone},
            transfers={'scheme': (1,), 'transfer_tuple': (transfer,)},
            chip_calendar=ChipCalendar(), chip_reason={}, horizon_plan=[],
            recommended_squad=[], this_week_scores={},
            get_this_week_transfers=lambda: [transfer], get_transfers=lambda: [],
        )

    def test_one_row_per_player_and_only_relevant_players(self):
        from squadtools.predictions_report import EMAIL_TOP_PER_POSITION, build_html_report

        html = build_html_report(self._opt(), applied=False)
        self.assertNotIn('raw 1GW', html)
        self.assertNotIn('scoreline', html.lower())
        # Top 10 midfielders, the owned midfielder outside them (sold one too),
        # the owned low-scoring defender and the forward being bought.
        self.assertEqual(10, EMAIL_TOP_PER_POSITION[3])
        self.assertIn('Player predictions (14 of 37)', html)
        self.assertIn('OwnedDefender', html)
        self.assertIn('BoughtForward', html)
        self.assertNotIn('P300', html)
        for tag in ('Yours', 'Buy', 'Sell'):
            self.assertIn('<td>%s</td>' % tag, html)

    def test_minutes_bands_are_shown_as_percentages(self):
        from unittest import mock

        from squadtools.predictions_report import build_html_report

        with mock.patch.dict('os.environ', {'PREMIERLEARNING_ARGS': '--v2-blend'}):
            html = build_html_report(self._opt(), applied=False)
        self.assertIn('Mins 0/1-59/60+', html)
        self.assertIn('<td>10/20/70</td>', html)
        self.assertIn('blend model', html)

    def test_columns_are_gameweeks_with_doubles_and_blanks(self):
        from unittest import mock

        from squadtools.predictions_report import build_html_report, email_player_row

        html = build_html_report(self._opt(), applied=False)
        for week in range(10, 15):
            self.assertIn('<th colspan="3">GW%d</th>' % week, html)
        self.assertNotIn('Match 1', html)
        player = _player(5, 3, 10.0)
        # GW11 blank, GW12 double.
        player.future_matches[1].gameweek = 12
        player.future_matches[1].opponent = 'TWO'
        row = email_player_row(player, 5, gameweeks=[10, 11, 12, 13, 14])
        cells = row[7:]
        self.assertEqual(['OPP', '2.0', '10/20/70'], cells[0:3])
        self.assertEqual(['-', '', ''], cells[3:6])
        # Fixture order (the moved fixture comes first in the list).
        self.assertEqual(['TWO + OPP', '4.0', '10/20/70 + 10/20/70'], cells[6:9])
        with mock.patch.dict('os.environ', {'PREMIERLEARNING_ARGS': '--learning-v2 --lightgbm'}):
            self.assertIn('LightGBM v6', build_html_report(self._opt(), applied=False))
        # The default (no model in the args) is LightGBM v6 too; v3 on request.
        with mock.patch.dict('os.environ', {'PREMIERLEARNING_ARGS': '--horizon-reps 5'}):
            self.assertIn('LightGBM v6', build_html_report(self._opt(), applied=False))
        with mock.patch.dict('os.environ', {'PREMIERLEARNING_ARGS': '--lightgbm --lgbm-version v3'}):
            self.assertIn('LightGBM v3', build_html_report(self._opt(), applied=False))

    def test_text_email_has_no_scorelines(self):
        from squadtools.predictions_report import build_text_report

        text = build_text_report(self._opt(), applied=False)
        self.assertNotIn('scoreline', text.lower())


if __name__ == '__main__':
    unittest.main()
