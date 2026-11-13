import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from simulation.fixture_calendar import (
    MissingFixtureCalendar,
    apply_known_calendar,
    calendar_path,
    known_calendar,
    parse_fixture_events,
)


def _utc(*parts):
    return datetime(*parts, tzinfo=timezone.utc)


def _write(root, season, snapshots):
    path = calendar_path(season, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'snapshots': snapshots}), encoding='utf-8')


class FixtureCalendarTests(unittest.TestCase):
    def test_a_fixture_without_a_gameweek_reads_as_none(self):
        text = 'id,event,kickoff_time\n1,3,2023-08-26T14:00:00Z\n2,,\n3,nan,\n'
        self.assertEqual({1: 3, 2: None, 3: None}, parse_fixture_events(text))

    def test_the_calendar_is_the_last_one_saved_before_the_deadline(self):
        root = Path(tempfile.mkdtemp())
        _write(root, 23, [
            {'committed_at': '2024-02-01T10:00:00Z', 'events': {'7': 29, '8': 29}},
            {'committed_at': '2024-03-20T10:00:00Z', 'events': {'7': None, '8': 29}},
            {'committed_at': '2024-04-10T10:00:00Z', 'events': {'7': 34, '8': 29}},
        ])
        self.assertEqual({7: 29, 8: 29}, known_calendar(23, _utc(2024, 3, 16, 11), root))
        self.assertEqual({7: None, 8: 29}, known_calendar(23, _utc(2024, 3, 20, 10), root))
        self.assertEqual({7: 34, 8: 29}, known_calendar(23, _utc(2024, 4, 20), root))
        with self.assertRaises(MissingFixtureCalendar):
            known_calendar(23, _utc(2024, 1, 1), root)
        with self.assertRaises(MissingFixtureCalendar):
            known_calendar(22, _utc(2024, 1, 1), root)

    def test_a_change_inside_a_long_gap_counts_from_two_weeks_before_its_gameweek(self):
        root = Path(tempfile.mkdtemp())
        _write(root, 25, [
            {'committed_at': '2026-03-13T10:00:00Z', 'events': {'1': 34, '2': 30}},
            {'committed_at': '2026-06-17T10:00:00Z', 'events': {'1': 33, '2': 30}},
        ])
        (root / 'season_25_26' / 'fixtures.csv').write_text(
            'id,event,kickoff_time\n'
            '1,33,2026-04-21T19:00:00Z\n'
            '9,33,2026-04-18T13:30:00Z\n'
            '8,34,2026-04-25T14:00:00Z\n'
            '2,30,2026-03-14T15:00:00Z\n',
            encoding='utf-8',
        )
        # GW33 deadline 18 April 12:00: the move counts from 4 April 12:00.
        self.assertEqual(34, known_calendar(25, _utc(2026, 4, 4, 11), root)[1])
        self.assertEqual(33, known_calendar(25, _utc(2026, 4, 4, 12), root)[1])
        # Copy times only: not known until the June copy.
        self.assertEqual(34, known_calendar(25, _utc(2026, 4, 18, 12), root, lead_days=None)[1])

    def test_a_change_between_weekly_copies_counts_from_the_later_copy(self):
        root = Path(tempfile.mkdtemp())
        _write(root, 23, [
            {'committed_at': '2024-03-25T10:00:00Z', 'events': {'1': None}},
            {'committed_at': '2024-04-01T10:00:00Z', 'events': {'1': 34}},
        ])
        (root / 'season_23_24' / 'fixtures.csv').write_text(
            'id,event,kickoff_time\n1,34,2024-04-24T19:00:00Z\n', encoding='utf-8',
        )
        self.assertIsNone(known_calendar(23, _utc(2024, 3, 31), root)[1])
        self.assertEqual(34, known_calendar(23, _utc(2024, 4, 1, 10), root)[1])

    def test_saved_2025_26_calendar_knows_the_gw33_moves_by_the_gw33_deadline(self):
        if not calendar_path(25).is_file():
            self.skipTest('no saved 2025/26 fixture calendars')
        from simulation.fixture_calendar import final_deadlines

        deadline = final_deadlines(25)[33]
        at_gw33 = known_calendar(25, deadline)
        final = known_calendar(25, _utc(2026, 7, 1))
        for fixture_id in (332, 333, 334):
            self.assertEqual(33, final[fixture_id])
            self.assertEqual(33, at_gw33[fixture_id])
        # Copy times alone would still show them in GW34.
        self.assertEqual(34, known_calendar(25, deadline, lead_days=None)[332])

    def test_rows_move_to_the_known_week_and_unscheduled_ones_are_left_out(self):
        rows = [
            {'fixture_id': 1, 'target_gameweek': 30, 'player_id': 10},
            {'fixture_id': 2, 'target_gameweek': 34, 'player_id': 10},
            {'fixture_id': 3, 'target_gameweek': 34, 'player_id': 10},
            {'fixture_id': 4, 'target_gameweek': 35, 'player_id': 10},
            {'fixture_id': 5, 'target_gameweek': 36, 'player_id': 10},
        ]
        calendar = {1: 30, 2: 29, 3: None, 4: 27}
        moved = apply_known_calendar(rows, calendar, as_of_gameweek=28)
        self.assertEqual(
            [(1, 30), (2, 29), (5, 36)],
            [(row['fixture_id'], row['target_gameweek']) for row in moved],
        )
        # The stored rows are not changed.
        self.assertEqual(34, rows[1]['target_gameweek'])

    def test_saved_2023_24_calendar_hides_the_gw34_doubles_until_april(self):
        path = calendar_path(23)
        if not path.is_file():
            self.skipTest('no saved 2023/24 fixture calendars')
        final = known_calendar(23, _utc(2024, 6, 1))
        doubles = [fixture_id for fixture_id, event in final.items() if event == 34]
        self.assertGreater(len(doubles), 10)
        # GW30 deadline (30 March 2024): the rescheduled games had no gameweek yet.
        at_gw30 = known_calendar(23, _utc(2024, 3, 30, 11))
        self.assertGreater(sum(1 for fixture_id in doubles if at_gw30.get(fixture_id) != 34), 0)
        self.assertEqual(380, len(final))


class ManagerCalendarTests(unittest.TestCase):
    def test_v3_uses_the_known_calendar_and_its_own_cache_ids(self):
        from fpl_manager.season_source import (
            KNOWN_CALENDAR_ID_OFFSET,
            cache_snapshot_id,
            default_calendar,
        )

        self.assertEqual('known', default_calendar('v3'))
        self.assertEqual('final', default_calendar('v2'))
        self.assertEqual(KNOWN_CALENDAR_ID_OFFSET + 12, cache_snapshot_id(12, 'known'))
        self.assertEqual(12, cache_snapshot_id(12, 'final'))
        self.assertEqual(0, cache_snapshot_id(0, 'known'))

    def test_the_command_line_takes_a_calendar(self):
        from fpl_manager.__main__ import build_parser

        self.assertIsNone(build_parser().parse_args([]).calendar)
        self.assertEqual('final', build_parser().parse_args(['--calendar', 'final']).calendar)


if __name__ == '__main__':
    unittest.main()
