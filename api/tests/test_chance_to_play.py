import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from premierlearning.config_constants import SEASON_YR
from squadtools import chance_to_play as chance_mod
from squadtools.chance_to_play import (
    archive_live_chances,
    download_gameweek_chances,
    ensure_gameweek_chances,
    is_live_season,
    resolve_chance_of_playing,
)


class ChanceToPlayTests(unittest.TestCase):
    def test_live_season_skips_historic_lookup(self):
        self.assertTrue(is_live_season(SEASON_YR))
        self.assertEqual(
            {},
            ensure_gameweek_chances(SEASON_YR, 1, allow_download=False),
        )
        self.assertIsNone(
            resolve_chance_of_playing(SEASON_YR, 1, 10, allow_download=False)
        )

    def test_local_cache_is_used_without_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = (
                root
                / 'season_25_26'
                / 'chance_of_playing'
                / 'gw3.json'
            )
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps({
                    'players': {
                        '10': {
                            'chance': 75,
                            'chance_of_playing_this_round': 75,
                            'chance_of_playing_next_round': 50,
                            'status': 'd',
                        },
                        '11': {
                            'chance': None,
                            'status': 'a',
                        },
                    }
                }),
                encoding='utf-8',
            )
            chances = ensure_gameweek_chances(
                25,
                3,
                data_root=root,
                allow_download=False,
            )
            self.assertEqual(75, chances[10])
            self.assertIsNone(chances[11])
            self.assertEqual(
                75,
                resolve_chance_of_playing(
                    25, 3, 10, data_root=root, allow_download=False
                ),
            )

    def test_chance_prefers_this_round_when_event_is_current(self):
        element = {
            'chance_of_playing_this_round': 25,
            'chance_of_playing_next_round': 100,
        }
        events = [{'id': 5, 'is_current': True, 'is_next': False}]
        self.assertEqual(
            25,
            chance_mod._chance_for_gameweek(element, events, 5),
        )

    def test_chance_prefers_next_round_when_event_is_next(self):
        element = {
            'chance_of_playing_this_round': 25,
            'chance_of_playing_next_round': 75,
        }
        events = [{'id': 5, 'is_current': False, 'is_next': True}]
        self.assertEqual(
            75,
            chance_mod._chance_for_gameweek(element, events, 5),
        )

    def test_fixture_chance_falls_back_to_this_round_for_later_gws(self):
        element = {
            'chance_of_playing_this_round': 75,
            'chance_of_playing_next_round': None,
        }
        self.assertEqual(
            75,
            chance_mod.chance_for_fixture(element, 1, current_gameweek=0, next_gameweek=1),
        )
        self.assertEqual(
            75,
            chance_mod.chance_for_fixture(element, 2, current_gameweek=0, next_gameweek=1),
        )

    def test_fixture_chance_uses_this_round_for_current_gw(self):
        element = {
            'chance_of_playing_this_round': 75,
            'chance_of_playing_next_round': 100,
        }
        self.assertEqual(
            75,
            chance_mod.chance_for_fixture(element, 1, current_gameweek=1, next_gameweek=2),
        )
        self.assertEqual(
            100,
            chance_mod.chance_for_fixture(element, 2, current_gameweek=1, next_gameweek=2),
        )
        self.assertEqual(
            100,
            chance_mod.chance_for_fixture(element, 3, current_gameweek=1, next_gameweek=2),
        )

    def test_next_round_chance_scales_the_whole_horizon(self):
        element = {
            'chance_of_playing_this_round': None,
            'chance_of_playing_next_round': 75,
        }
        for gameweek in (1, 2, 5):
            self.assertEqual(
                75,
                chance_mod.chance_for_fixture(
                    element, gameweek, current_gameweek=0, next_gameweek=1
                ),
            )

    def test_unfit_status_is_zero_when_chance_fields_are_null(self):
        element = {
            'chance_of_playing_this_round': None,
            'chance_of_playing_next_round': None,
            'status': 'i',
        }
        self.assertEqual(
            0,
            chance_mod.chance_for_fixture(element, 1, current_gameweek=0, next_gameweek=1),
        )
        self.assertEqual(
            0,
            chance_mod.chance_for_fixture(element, 2, current_gameweek=0, next_gameweek=1),
        )
        returning = dict(element, chance_of_playing_next_round=75)
        self.assertEqual(
            75,
            chance_mod.chance_for_fixture(returning, 1, current_gameweek=0, next_gameweek=1),
        )
        self.assertEqual(
            75,
            chance_mod.chance_for_fixture(returning, 5, current_gameweek=0, next_gameweek=1),
        )

    def test_download_writes_only_player_chance_slice(self):
        bootstrap = {
            'events': [
                {
                    'id': 1,
                    'is_current': False,
                    'is_next': True,
                    'deadline_time': '2024-08-16T17:30:00Z',
                }
            ],
            'elements': [
                {
                    'id': 7,
                    'chance_of_playing_this_round': None,
                    'chance_of_playing_next_round': 50,
                    'status': 'd',
                }
            ],
        }
        deadline = datetime(2024, 8, 16, 17, 30, tzinfo=timezone.utc)
        with mock.patch.object(
            chance_mod,
            '_fetch_pre_deadline_bootstrap',
            return_value=('cache/2024/8/16/1238.json.xz', bootstrap),
        ):
            payload = download_gameweek_chances(1, deadline)
        self.assertEqual('cache/2024/8/16/1238.json.xz', payload['snapshot'])
        self.assertEqual(50, payload['players']['7']['chance'])
        self.assertEqual(['7'], list(payload['players']))

    def test_ensure_downloads_when_local_cache_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            deadline = datetime(2024, 8, 16, 17, 30, tzinfo=timezone.utc)
            payload = {
                'source': 'Randdalf/fplcache',
                'snapshot': 'cache/2024/8/16/1238.json.xz',
                'gameweek': 1,
                'deadline': deadline.isoformat(),
                'players': {
                    '9': {
                        'chance': 0,
                        'chance_of_playing_this_round': None,
                        'chance_of_playing_next_round': 0,
                        'status': 'i',
                    }
                },
            }
            with mock.patch.object(
                chance_mod,
                'download_gameweek_chances',
                return_value=payload,
            ) as download:
                chances = ensure_gameweek_chances(
                    24,
                    1,
                    deadline=deadline,
                    data_root=root,
                    allow_download=True,
                )
            download.assert_called_once()
            self.assertEqual(0, chances[9])
            saved = root / 'season_24_25' / 'chance_of_playing' / 'gw1.json'
            self.assertTrue(saved.is_file())
            # Second call should hit disk only.
            with mock.patch.object(
                chance_mod,
                'download_gameweek_chances',
            ) as download_again:
                again = ensure_gameweek_chances(
                    24,
                    1,
                    deadline=deadline,
                    data_root=root,
                    allow_download=True,
                )
            download_again.assert_not_called()
            self.assertEqual(0, again[9])

    def test_live_flags_are_archived_until_the_deadline(self):
        bootstrap = {
            'events': [
                {'id': 1, 'deadline_time': '2026-08-14T17:30:00Z', 'is_current': True},
                {'id': 2, 'deadline_time': '2026-08-21T17:30:00Z', 'is_next': True},
            ],
            'elements': [
                {'id': 5, 'chance_of_playing_next_round': 25, 'chance_of_playing_this_round': 0, 'status': 'd'},
                {'id': 6, 'chance_of_playing_next_round': None, 'chance_of_playing_this_round': None, 'status': 'i'},
                {'id': 7, 'chance_of_playing_next_round': None, 'chance_of_playing_this_round': None, 'status': 'a'},
            ],
        }
        with tempfile.TemporaryDirectory() as root:
            before = datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc)
            path = archive_live_chances(13, bootstrap, now=before, data_root=Path(root))
            self.assertTrue(path.name == 'gw2.json')
            # Read back the way historical builds read it once the season is over.
            self.assertEqual({5: 25, 6: 0, 7: None}, ensure_gameweek_chances(13, 2, data_root=Path(root), allow_download=False))
            after_all = datetime(2026, 8, 22, tzinfo=timezone.utc)
            self.assertIsNone(archive_live_chances(13, bootstrap, now=after_all, data_root=Path(root)))


if __name__ == '__main__':
    unittest.main()
