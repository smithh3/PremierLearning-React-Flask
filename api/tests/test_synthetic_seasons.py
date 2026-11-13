import unittest
from datetime import timedelta

import numpy as np

from synthetic_seasons.events import team_goal_count
from synthetic_seasons.injuries import _matching_gw, is_flagged, shift_entry
from synthetic_seasons.source import DATA_ROOT, bonus_from_bps, event_points


class BonusAndScoringTests(unittest.TestCase):
    def test_bonus_tie_rules(self):
        self.assertEqual({1: 3, 2: 2, 3: 1, 4: 0}, bonus_from_bps({1: 40, 2: 30, 3: 20, 4: 10}))
        # Tie for first: both 3, the next player 1.
        self.assertEqual({1: 3, 2: 3, 3: 1, 4: 0}, bonus_from_bps({1: 40, 2: 40, 3: 20, 4: 10}))
        # Tie for second: both 2, nobody gets 1.
        self.assertEqual({1: 3, 2: 2, 3: 2, 4: 0}, bonus_from_bps({1: 40, 2: 30, 3: 30, 4: 10}))
        # Tie for third: all get 1.
        self.assertEqual({1: 3, 2: 2, 3: 1, 4: 1}, bonus_from_bps({1: 40, 2: 30, 3: 20, 4: 20}))

    def test_event_points(self):
        row = {'minutes': 90, 'goals_scored': 1, 'assists': 1, 'clean_sheets': 1, 'goals_conceded': 0,
               'saves': 0, 'bonus': 3, 'yellow_cards': 1, 'defensive_contribution': 10}
        self.assertEqual(2 + 6 + 3 + 4 + 3 - 1, event_points(row, 2, with_dc=False))
        self.assertEqual(2 + 6 + 3 + 4 + 3 - 1 + 2, event_points(row, 2, with_dc=True))
        keeper = {'minutes': 90, 'goals_conceded': 3, 'saves': 7, 'clean_sheets': 0}
        self.assertEqual(2 - 1 + 2, event_points(keeper, 1, with_dc=False))
        self.assertEqual(0, event_points({'minutes': 0, 'goals_scored': 1}, 4, with_dc=False))


class DrawTests(unittest.TestCase):
    def test_team_goals_keep_mean_with_less_spread(self):
        rng = np.random.default_rng(1)
        draws = np.array([team_goal_count(rng, 1.6, 0.26) for _ in range(40000)])
        self.assertAlmostEqual(1.6, draws.mean(), delta=0.03)
        self.assertAlmostEqual(1.6 * 0.74, draws.var(), delta=0.06)
        self.assertLess((draws == 0).mean(), np.exp(-1.6))


class FlagTests(unittest.TestCase):
    def test_flags_follow_the_missed_matches(self):
        # Template missed matches in GW10, 11, 12; new spell has a double in GW30.
        self.assertEqual(30, _matching_gw(10, (10, 11, 12), [30, 30, 31]))
        self.assertEqual(31, _matching_gw(12, (10, 11, 12), [30, 30, 31]))
        self.assertEqual(29, _matching_gw(9, (10, 11, 12), [30, 30, 31]))
        self.assertEqual(33, _matching_gw(14, (10, 11, 12), [30, 30, 31]))

    def test_shift_moves_added_time_and_return_date(self):
        entry = {'status': 'i', 'news': 'Hamstring injury - Expected back 12 Nov',
                 'news_added': '2023-11-01T10:00:00Z', 'chance_of_playing_next_round': 0}
        moved = shift_entry(entry, timedelta(days=70))
        self.assertEqual('Hamstring injury - Expected back 21 Jan', moved['news'])
        self.assertTrue(moved['news_added'].startswith('2024-01-10'))
        self.assertTrue(is_flagged(moved))
        self.assertFalse(is_flagged({'status': 'a', 'chance_of_playing_next_round': None}))


class PlayerFolderTests(unittest.TestCase):
    def test_renamed_player_reads_the_full_folder(self):
        import os
        import tempfile

        from premierlearning.season import player_folders

        with tempfile.TemporaryDirectory() as root:
            for name, rows in (('Michael_Olakigbe_620', 38), ('Michale_Olakigbe_620', 4), ('Bukayo_Saka_7', 3)):
                os.makedirs(os.path.join(root, name))
                with open(os.path.join(root, name, 'gw.csv'), 'w') as handle:
                    handle.write('round\n' + ''.join('%d\n' % i for i in range(rows)))
            self.assertEqual({620: 'Michael_Olakigbe_620', 7: 'Bukayo_Saka_7'}, player_folders(root))


class KickoffMatchingTests(unittest.TestCase):
    def test_moved_round_takes_the_slot_times(self):
        from datetime import datetime, timezone

        from synthetic_seasons.schedule import _match_kickoffs
        from synthetic_seasons.source import Fixture

        def fixture(fid, home, away, hour):
            return Fixture(fid, 1, home, away, 0, 0, datetime(2025, 8, 16, hour, tzinfo=timezone.utc))

        # Slot gameweek: 1-2 at 12:00, 3-4 at 15:00, 5-6 at 17:30 (hour 17).
        old = [fixture(1, 1, 2, 12), fixture(2, 3, 4, 15), fixture(3, 5, 6, 17)]
        # Moved round: 1-3, 4-5, 6-2.
        new = [fixture(10, 1, 3, 0), fixture(11, 4, 5, 0), fixture(12, 6, 2, 0)]
        for seed in range(5):
            times = _match_kickoffs(old, new, np.random.default_rng(seed))
            self.assertEqual({1, 2, 3}, {f.id for f in times.values()})
            for f in new:
                self.assertTrue({f.home, f.away} & {times[f.id].home, times[f.id].away})


class _FakeEpisode:
    def __init__(self, season):
        self.season = season
        self.gameweek = season.first
        self.points_total = 0.0
        self.season_start_year = season.season

    def observation(self):
        return [0.0]

    def mask(self):
        return [True]

    def step(self, _action):
        self.points_total += 50.0
        self.gameweek += 1
        return [0.0], 1.0, self.gameweek > self.season.last, {}


class _FakeSeason:
    def __init__(self, year, first=1, last=2):
        self.season, self.first, self.last = year, first, last

    def episode(self):
        return _FakeEpisode(self)


class SeasonMixTests(unittest.TestCase):
    def _mix(self, seed=0, copy_last=2):
        from synthetic_seasons.training import SeasonMix

        loaded = []

        def load(year, index, first, last):
            loaded.append((year, index))
            return _FakeSeason(year, first, copy_last)

        real = [_FakeSeason(23), _FakeSeason(24), _FakeSeason(25)]
        return SeasonMix(real, (1, 2, 3, 4, 5), (6, 7), load, np.random.default_rng(seed)), loaded

    def test_each_season_draws_real_or_a_training_copy(self):
        mix, _loaded = self._mix()
        labels = {23: set(), 24: set(), 25: set()}
        for _run in range(300):
            seasons = mix.training_seasons()
            self.assertEqual([23, 24, 25], [s.season for s in seasons])
            for season in seasons:
                labels[season.season].add(season.data_label)
        expected = {'real'} | {'copy %03d' % i for i in range(1, 6)}
        self.assertEqual({23: expected, 24: expected, 25: expected}, labels)

    def test_held_out_copies_are_never_drawn_for_training_and_cycle_in_evaluation(self):
        mix, loaded = self._mix()
        for _run in range(200):
            mix.training_seasons()
        self.assertFalse({index for _year, index in loaded} & {6, 7})
        labels = [mix.evaluation_seasons(k)[0] for k in range(6)]
        self.assertEqual(['real', 'copy 006', 'copy 007'] * 2, labels)
        label, seasons = mix.evaluation_seasons(2)
        self.assertEqual(['copy 007'] * 3, [s.data_label for s in seasons])

    def test_copy_with_other_gameweeks_is_refused(self):
        mix, _loaded = self._mix(copy_last=3)
        with self.assertRaises(ValueError):
            mix.evaluation_seasons(1)

    def test_evaluation_swaps_in_the_cycle_version(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        from synthetic_seasons import training

        mix, _loaded = self._mix(seed=1)
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'eval.csv').write_text('run\n1\n')      # one evaluation finished
            run = training.MixedRun(mix, log_path=Path(root) / 'runs.csv')
            self.assertTrue(run.at_start)
            run.evaluation = True
            with mock.patch.object(training, 'ensure_rules_baseline') as rules:
                run.begin_evaluation()
            rules.assert_called_once()
            self.assertEqual(['copy 006'] * 3, [s.data_label for s in run.seasons])
            self.assertIs(run.seasons[0], run.episode.season)
            self.assertTrue(run.at_start)

    def test_run_log_names_each_version(self):
        import csv
        import tempfile
        from pathlib import Path

        from synthetic_seasons.training import MixedSeasonByRun

        mix, _loaded = self._mix(seed=3)
        with tempfile.TemporaryDirectory() as root:
            log = Path(root) / 'runs.csv'
            factory = MixedSeasonByRun(mix, log_path=log)
            for _run in range(4):
                run = factory()
                while True:
                    _obs, _reward, done, info = run.step(0)
                    if done and info.get('run_finished'):
                        break
                    if done:
                        run = factory()
            with log.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(4, len(rows))
            for row in rows:
                for year in (23, 24, 25):
                    self.assertIn(row['%s_data' % year], {'real'} | {'copy %03d' % i for i in range(1, 6)})
                    self.assertEqual('100', row['%s_points' % year])


@unittest.skipUnless((DATA_ROOT / 'season_25_26' / 'players').is_dir(), 'needs the 2025/26 data')
class RealSeasonCopyTests(unittest.TestCase):
    """One copy of the real 2025/26 season keeps every rule (needs calibration.json)."""

    @classmethod
    def setUpClass(cls):
        from synthetic_seasons.events import calibration
        from synthetic_seasons.generate import SeasonCopier
        if not (DATA_ROOT / 'synthetic' / 'calibration.json').is_file():
            raise unittest.SkipTest('run python -m synthetic_seasons generate first')
        cls.copier = SeasonCopier(25, calib=calibration())
        cls.copy = cls.copier.make(1)

    def test_real_points_are_reproduced_by_the_scoring(self):
        source = self.copier.source
        wrong = [r for r in source.rows
                 if r['minutes'] > 0 and r['total_points'] != event_points(r, r['_position'], source.has_dc)]
        self.assertEqual([], wrong[:3])
        # Without minutes, only rare bench bookings score (kept as leftover points).
        benched = [r['total_points'] for r in source.rows if r['minutes'] == 0 and r['total_points']]
        self.assertTrue(all(points < 0 for points in benched))

    def test_copy_keeps_every_rule(self):
        from synthetic_seasons.report import check_copy
        source = self.copier.source
        rows = self.copy['rows']
        flags = {str(gw): {str(e): entry for e, entry in players.items()} for gw, players in self.copy['flags'].items()}
        problems = check_copy(source, rows, self.copy['scores'], self.copy['meta'], flags,
                              self.copy['reshuffle'].fixtures)
        self.assertEqual({}, {name: count for name, count in problems.items() if count})

    def test_fixtures_are_reshuffled_but_dates_stay(self):
        reshuffle = self.copy['reshuffle']
        self.assertGreater(self.copy['meta']['gameweeks_moved'], 20)
        self.assertEqual(0, reshuffle.cost)
        source = self.copier.source
        # Each team's row on a moved date now names the fixture played on it, against its opponent.
        rows = [r for r in self.copy['rows'] if r['fixture'] in reshuffle.kickoff_text]
        self.assertTrue(rows)
        for row in rows[:500]:
            fixture = reshuffle.fixtures[row['fixture']]
            self.assertIn(row['_team'], (fixture.home, fixture.away))
            self.assertEqual(row['_gw'], fixture.gameweek)
            self.assertNotEqual(source.fixtures[row['fixture']].gameweek, fixture.gameweek)

    def test_same_seed_same_copy(self):
        again = self.copier.make(1)
        self.assertEqual(self.copy['scores'], again['scores'])
        self.assertNotEqual(self.copy['scores'], self.copier.make(2)['scores'])

    def test_injuries_move(self):
        meta = self.copy['meta']
        self.assertGreater(meta['moved'], 0)
        # Every removed spell is replaced by an added one (unless no slot fits).
        self.assertLessEqual(abs(meta['removed'] - meta['added']), 3)
        self.assertLessEqual(meta['swaps']['out_no_deputy'], 10)


if __name__ == '__main__':
    unittest.main()
