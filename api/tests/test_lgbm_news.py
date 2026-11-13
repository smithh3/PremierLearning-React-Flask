"""FPL news columns for LightGBM feature set 4 (premierlearning/lgbm/news.py)."""
import json
import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from premierlearning.lgbm.news import (
    NEWS_COLUMNS, NewsStore, certainty, news_cache_path, news_columns, reason_of, return_date,
)

YEAR = 23


def _entry(status='a', news='', added=None, chance=None):
    return {'status': status, 'news': news, 'news_added': added,
            'chance_of_playing_next_round': chance, 'chance_of_playing_this_round': chance}


def _write(root, gameweek, deadline, players):
    path = news_cache_path(YEAR, gameweek, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'gameweek': gameweek, 'deadline': deadline, 'players': players}),
                    encoding='utf-8')


def _batch(rows):
    """rows: (player id, gameweek, kickoff)."""
    return SimpleNamespace(
        owners=[SimpleNamespace(id=player) for player, _gw, _kickoff in rows],
        years=[YEAR] * len(rows),
        matches=[SimpleNamespace(gameweek=gw, kickoff_time=kickoff) for _player, gw, kickoff in rows],
    )


class ParsingTests(unittest.TestCase):
    def test_reasons(self):
        cases = {
            'Knee injury - Expected back 01 Jan': 'joint',
            'Hamstring injury - 75% chance of playing': 'muscle',
            'Knock - 50% chance of playing': 'knock',
            'Lack of match fitness - 25% chance of playing': 'fitness',
            'Anterior cruciate ligament - Unknown return date': 'serious',
            'Suspended until 26 Dec': 'suspended',
            'Season-long loan to Stoke City': 'left',
            "Not included in Everton's 25-man Premier League squad - Unknown return date": 'squad',
            'On international duty at the Africa Cup of Nations from Gameweek 17 onwards. - Unknown return date':
                'international',
            'Illness - 75% chance of playing': 'illness',
            'Will be assessed - 75% chance of playing': 'other',
            '': '',
        }
        for text, expected in cases.items():
            self.assertEqual(reason_of(text), expected, text)

    def test_return_date_rolls_into_next_year(self):
        added = datetime(2021, 9, 26)
        self.assertEqual(return_date('Knee injury - Expected back 01 Jan', added), datetime(2022, 1, 1))
        self.assertEqual(return_date('Suspended until 03 Oct', added), datetime(2021, 10, 3))
        # A date just before the news was added stays in that year (stale text).
        self.assertEqual(return_date('Calf injury - Expected back 20 Sep', added), datetime(2021, 9, 20))
        self.assertIsNone(return_date('Knock - 75% chance of playing', added))

    def test_certainty(self):
        self.assertEqual(certainty('a', 1.0, math.nan, False), 10.0)
        self.assertEqual(certainty('d', 0.75, math.nan, False), 7.5)
        self.assertEqual(certainty('i', 0.0, 2.0, False), 1.0)
        self.assertEqual(certainty('i', 0.0, -1.0, False), 1.0)
        self.assertEqual(certainty('i', 0.0, 10.0, False), 0.0)
        self.assertEqual(certainty('i', 0.0, math.nan, True), 0.0)
        self.assertEqual(certainty('s', 0.0, math.nan, False), 0.0)
        self.assertEqual(certainty('u', 0.0, math.nan, False), 0.0)


class ColumnTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        # Player 7: hamstring at GW1 (injured), 50% at GW2, clear at GW3.
        # Player 8: never flagged. Player 9: knee injury dated back 30 Aug at GW2.
        _write(self.root, 1, '2023-08-11T17:30:00+00:00', {
            '7': _entry('i', 'Hamstring injury - Expected back 26 Aug', '2023-08-09T10:00:00', 0),
            '8': _entry(), '9': _entry()})
        _write(self.root, 2, '2023-08-18T17:30:00+00:00', {
            '7': _entry('d', 'Hamstring injury - 50% chance of playing', '2023-08-16T17:30:00', 50),
            '8': _entry(), '9': _entry('i', 'Knee injury - Expected back 30 Aug', '2023-08-17T09:00:00', 0)})
        _write(self.root, 3, '2023-08-25T17:30:00+00:00', {
            '7': _entry(), '8': _entry(), '9': _entry('i', 'Knee injury - Expected back 30 Aug',
                                                       '2023-08-17T09:00:00', 0)})

    def _columns(self, rows, as_of=None):
        return news_columns(_batch(rows), as_of_gameweek=as_of, store=NewsStore(self.root))

    def test_played_rows_read_their_own_deadline(self):
        columns = self._columns([(7, 1, '2023-08-12T14:00:00Z'), (7, 2, '2023-08-19T14:00:00Z'),
                                 (7, 3, '2023-08-26T14:00:00Z'), (8, 3, '2023-08-26T14:00:00Z')])
        self.assertEqual(set(columns), set(NEWS_COLUMNS))
        self.assertEqual(list(columns['news.known']), [1, 1, 1, 1])
        self.assertEqual(list(columns['news.status_i']), [1, 0, 0, 0])
        self.assertEqual(list(columns['news.status_d']), [0, 1, 0, 0])
        self.assertEqual(list(columns['news.reason_muscle']), [1, 1, 0, 0])
        self.assertEqual(list(columns['news.chance']), [0.0, 0.5, 1.0, 1.0])
        self.assertEqual(list(columns['news.certainty']), [0.0, 5.0, 10.0, 10.0])
        # Expected back 26 Aug against a 12 Aug kickoff: 14 days.
        self.assertEqual(columns['news.days_to_return'][0], 14.0)
        self.assertTrue(math.isnan(columns['news.days_to_return'][1]))
        # Flagged since 16 Aug 17:30, deadline 18 Aug 17:30: two days.
        self.assertAlmostEqual(float(columns['news.days_flagged'][1]), 2.0, places=4)
        self.assertEqual(list(columns['news.flagged_deadlines']), [1, 2, 0, 0])
        self.assertEqual(columns['news.deadlines_since_flag'][2], 1.0)
        self.assertTrue(math.isnan(columns['news.deadlines_since_flag'][3]))
        self.assertTrue(math.isnan(columns['news.prev_chance'][0]))
        self.assertEqual(columns['news.prev_chance'][1], 0.0)
        self.assertEqual(columns['news.chance_change'][2], 0.5)

    def test_upcoming_rows_use_the_as_of_deadline_and_their_own_kickoff(self):
        # Predicting from GW2: player 9's GW3 and GW4 fixtures see GW2's news, not GW3's.
        columns = self._columns([(9, 3, '2023-08-26T14:00:00Z'), (9, 4, '2023-09-02T14:00:00Z')], as_of=2)
        self.assertEqual(list(columns['news.days_to_return']), [4.0, -3.0])
        self.assertEqual(list(columns['news.certainty']), [0.0, 1.0])
        self.assertEqual(list(columns['news.flagged_deadlines']), [1, 1])

    def test_no_file_means_unknown(self):
        columns = self._columns([(7, 5, '2023-09-16T14:00:00Z')])
        self.assertEqual(columns['news.known'][0], 0.0)
        self.assertTrue(math.isnan(columns['news.certainty'][0]))
        self.assertTrue(math.isnan(columns['news.chance'][0]))


class LiveSeasonTests(unittest.TestCase):
    EVENTS = [{'id': 1, 'deadline_time': '2026-08-21T17:30:00Z'},
              {'id': 2, 'deadline_time': '2026-08-28T17:30:00Z'},
              {'id': 3, 'deadline_time': '2026-09-04T17:30:00Z'}]

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_archive_saves_the_upcoming_deadline_from_the_live_api(self):
        from datetime import timezone

        from premierlearning.lgbm.news import archive_live_news

        bootstrap = {'events': self.EVENTS, 'elements': [
            dict(_entry('d', 'Knock - 75% chance of playing', '2026-08-27T10:00:00Z', 75), id=7)]}
        path = archive_live_news(26, bootstrap, now=datetime(2026, 8, 25, tzinfo=timezone.utc), data_root=self.root)
        self.assertEqual(path, news_cache_path(26, 2, self.root))
        store = NewsStore(self.root)
        self.assertEqual(store.player(26, 2, 7)['status'], 'd')
        self.assertEqual(store.player(26, 2, 7)['chance'], 0.75)
        self.assertIsNone(archive_live_news(26, bootstrap, now=datetime(2026, 9, 5, tzinfo=timezone.utc),
                                            data_root=self.root))

    def test_backfill_fills_passed_deadlines_from_fplcache(self):
        from datetime import timezone
        from unittest import mock

        from premierlearning.lgbm import news

        _write_live = news_cache_path(26, 1, self.root)
        _write_live.parent.mkdir(parents=True, exist_ok=True)
        _write_live.write_text('{"players": {}}', encoding='utf-8')
        calls = []

        def fetch(deadline):
            calls.append(deadline)
            return 'cache/2026/8/28/1700.json.xz', {'elements': [dict(_entry('i'), id=9)]}

        with mock.patch.object(news, '_fetch_pre_deadline_bootstrap', side_effect=fetch):
            written = news.backfill_live_news(26, self.EVENTS, now=datetime(2026, 9, 1, tzinfo=timezone.utc),
                                              data_root=self.root, verbose=False)
        # GW1 had a file, GW3 is still ahead: only GW2 is fetched.
        self.assertEqual(written, 1)
        self.assertEqual([deadline.isoformat() for deadline in calls], ['2026-08-28T17:30:00+00:00'])
        self.assertEqual(NewsStore(self.root).player(26, 2, 9)['status'], 'i')
        with mock.patch.object(news, '_fetch_pre_deadline_bootstrap', side_effect=FileNotFoundError('gap')):
            self.assertEqual(news.backfill_live_news(26, self.EVENTS, now=datetime(2026, 9, 10, tzinfo=timezone.utc),
                                                     data_root=self.root, verbose=False), 0)


class FeatureSetFourTests(unittest.TestCase):
    def test_set_four_is_set_three_plus_the_news_columns(self):
        from premierlearning.learning_v2.dataset import build_dataset
        from simulation.season_model_lgbm import flatten_with_seasons
        from tests.test_lgbm import _fixture_world, _teams_plan

        season, _saka = _fixture_world()
        plan = _teams_plan()
        train, predict, _vocab = build_dataset([season], plan, predict_season=season)
        three = flatten_with_seasons(train, plan, [season], features=3)
        four = flatten_with_seasons(train, plan, [season], features=4, priors=three.priors)
        self.assertEqual(four.names, three.names + list(NEWS_COLUMNS))
        numpy_equal = (three.values == four.values[:, :len(three.names)]) | (
            (three.values != three.values) & (four.values[:, :len(three.names)] != four.values[:, :len(three.names)]))
        self.assertTrue(numpy_equal.all())
        future = flatten_with_seasons(predict, plan, [season], features=4, priors=three.priors, as_of_gameweek=1)
        self.assertEqual(future.names, four.names)

    def test_experiments(self):
        from simulation.model_eval.experiments import EXPERIMENTS

        six = EXPERIMENTS['lgbm_v6']['args']
        self.assertEqual(six[six.index('--lgbm-features') + 1], '4')
        self.assertEqual(EXPERIMENTS['lgbm_v6_staged']['args'], six + ['--lgbm-staged'])
        four = EXPERIMENTS['lgbm_v4']['args']
        self.assertEqual(four[four.index('--lgbm-features') + 1], '3')


if __name__ == '__main__':
    unittest.main()
