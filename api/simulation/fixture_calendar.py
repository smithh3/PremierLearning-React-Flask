"""The fixture calendar as FPL showed it at each past deadline.

The saved prediction sets put every fixture in its final gameweek, so a
replayed manager would know about blanks and doubles weeks or months before
FPL announced them. FPL's fixtures feed is not archived by FPL, but
vaastav/Fantasy-Premier-League commits a copy of it (data/<season>/fixtures.csv)
about once a gameweek, and git keeps every version with its date.

`download_calendar_history` saves, for one season, each version's gameweek per
fixture id (None while a postponed fixture has no gameweek) and when it was
committed, to data/season_YY_YY/fixture_calendars.json.

`known_calendar` gives each fixture's gameweek as known at a deadline. A
change seen between two copies A and B happened some time between them. When
the copies are close (most of 2021/22-2024/25 is about weekly) the change counts
from B, as the commit time is after FPL made it. Some gaps are long (2025/26
has gaps of three months, and four of its moves only appear after they were
played), so a change also counts from ANNOUNCE_LEAD_DAYS before the earlier
of its old and new gameweek deadlines, whichever is sooner, but never before
A. In the weekly-copy seasons FPL showed 37 of 40 moves at least that long
before the affected deadline (median about 3 weeks), so this rarely moves a
change earlier than FPL showed it, and only within a gap.
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

REPO = 'vaastav/Fantasy-Premier-League'
SOURCE = f'{REPO} data/<season>/fixtures.csv history (a copy of FPL\'s fixtures feed)'
CALENDAR_FILE = 'fixture_calendars.json'
USER_AGENT = 'premier-learning-app-fixture-calendar'
# A move FPL made between two copies counts from this long before the affected
# gameweek's deadline (but not before the earlier copy). Measured on the
# weekly-copy seasons: 37 of 40 moves were shown at least 14 days ahead.
ANNOUNCE_LEAD_DAYS = 14
# FPL deadlines in these seasons were 90 minutes before the first kick-off.
DEADLINE_BEFORE_KICKOFF = timedelta(minutes=90)


def data_root_default() -> Path:
    return Path(__file__).resolve().parents[1] / 'data'


def _short_year(season_start_year: int) -> int:
    year = int(season_start_year)
    return year % 100 if year >= 2000 else year


def season_dir(season_start_year: int, data_root: Optional[Path] = None) -> Path:
    year = _short_year(season_start_year)
    root = Path(data_root) if data_root is not None else data_root_default()
    return root / f'season_{year:02d}_{year + 1:02d}'


def calendar_path(season_start_year: int, data_root: Optional[Path] = None) -> Path:
    return season_dir(season_start_year, data_root) / CALENDAR_FILE


def vaastav_season_name(season_start_year: int) -> str:
    year = _short_year(season_start_year)
    return f'20{year:02d}-{year + 1:02d}'


def parse_fixture_events(text: str) -> Dict[int, Optional[int]]:
    """fixture id -> gameweek from one fixtures.csv; None when it has no gameweek."""
    events = {}
    for row in csv.DictReader(io.StringIO(text)):
        raw = str(row.get('event') or '').strip()
        try:
            event = int(float(raw)) if raw and raw.lower() != 'nan' else None
        except ValueError:
            event = None
        events[int(row['id'])] = event
    return events


def _parse_time(value: str) -> datetime:
    stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _http(url: str, accept: str = 'application/vnd.github+json') -> bytes:
    headers = {'User-Agent': USER_AGENT, 'Accept': accept}
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if token and 'api.github.com' in url:
        headers['Authorization'] = f'Bearer {token}'
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _commits(season_name: str):
    """(committed_at, sha) for every commit that changed this season's fixtures.csv."""
    found = []
    for page in range(1, 20):
        rows = json.loads(_http(
            f'https://api.github.com/repos/{REPO}/commits'
            f'?path=data/{season_name}/fixtures.csv&per_page=100&page={page}'
        ).decode('utf-8'))
        if not rows:
            break
        found += [(row['commit']['committer']['date'], row['sha']) for row in rows]
    return sorted(found)


def download_calendar_history(
    season_start_year: int, data_root: Optional[Path] = None, verbose: bool = True,
) -> Path:
    season_name = vaastav_season_name(season_start_year)
    snapshots = []
    for committed_at, sha in _commits(season_name):
        text = _http(
            f'https://raw.githubusercontent.com/{REPO}/{sha}/data/{season_name}/fixtures.csv',
            accept='text/plain',
        ).decode('utf-8')
        events = parse_fixture_events(text)
        snapshots.append({
            'committed_at': committed_at,
            'sha': sha,
            'events': {str(fixture_id): event for fixture_id, event in sorted(events.items())},
        })
        time.sleep(0.1)
    if not snapshots:
        raise FileNotFoundError(f'No fixtures.csv history for {season_name} in {REPO}')
    path = calendar_path(season_start_year, data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'source': SOURCE,
        'season': season_name,
        'snapshots': snapshots,
    }, indent=1) + '\n', encoding='utf-8')
    if verbose:
        print(f'Saved {len(snapshots)} {season_name} fixture calendars to {path}')
    return path


@lru_cache(maxsize=None)
def _load(path_text: str):
    payload = json.loads(Path(path_text).read_text(encoding='utf-8'))
    snapshots = []
    for snapshot in payload.get('snapshots') or []:
        events = {
            int(fixture_id): (None if event is None else int(event))
            for fixture_id, event in (snapshot.get('events') or {}).items()
        }
        snapshots.append((_parse_time(snapshot['committed_at']), snapshot.get('sha'), events))
    snapshots.sort(key=lambda item: item[0])
    return tuple(snapshots)


def final_deadlines(season_start_year: int, data_root: Optional[Path] = None) -> Dict[int, datetime]:
    """Gameweek -> deadline from the season's final fixtures.csv (empty if missing)."""
    path = season_dir(season_start_year, data_root) / 'fixtures.csv'
    if not path.is_file():
        return {}
    first_kickoff = {}
    with path.open(encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            raw_event = str(row.get('event') or '').strip()
            raw_kickoff = str(row.get('kickoff_time') or '').strip()
            if not raw_event or raw_event.lower() == 'nan' or not raw_kickoff:
                continue
            event = int(float(raw_event))
            kickoff = _parse_time(raw_kickoff)
            if event not in first_kickoff or kickoff < first_kickoff[event]:
                first_kickoff[event] = kickoff
    return {event: kickoff - DEADLINE_BEFORE_KICKOFF for event, kickoff in first_kickoff.items()}


@lru_cache(maxsize=None)
def _timelines(path_text: str, deadlines_key: tuple, lead_days):
    """fixture id -> ((known from, gameweek), ...), and when the first copy was taken.

    A change between copies A and B counts from B, or from `lead_days` before
    the earlier of its old and new gameweek deadlines if that is sooner, but
    never before A.
    """
    snapshots = _load(path_text)
    deadlines = dict(deadlines_key)
    lead = None if lead_days is None else timedelta(days=float(lead_days))
    first_time, _sha, first_events = snapshots[0]
    timelines = {fixture_id: [(first_time, event)] for fixture_id, event in first_events.items()}
    for (before_time, _a, before), (after_time, _b, after) in zip(snapshots, snapshots[1:]):
        for fixture_id, event in after.items():
            old = before.get(fixture_id)
            if fixture_id in before and old == event:
                continue
            known_from = after_time
            affected = [deadlines[week] for week in (old, event) if week in deadlines]
            if lead is not None and affected:
                known_from = min(known_from, max(before_time, min(affected) - lead))
            timelines.setdefault(fixture_id, []).append((known_from, event))
    return {fixture_id: tuple(steps) for fixture_id, steps in timelines.items()}, first_time


class MissingFixtureCalendar(LookupError):
    """No saved fixture calendar from before this deadline."""


def known_calendar(
    season_start_year: int, deadline: datetime, data_root: Optional[Path] = None,
    lead_days=ANNOUNCE_LEAD_DAYS,
) -> Dict[int, Optional[int]]:
    """fixture id -> gameweek as FPL showed it at `deadline` (None: no gameweek yet).

    `lead_days` None uses the copy times only (see the module notes).
    """
    path = calendar_path(season_start_year, data_root)
    if not path.is_file():
        raise MissingFixtureCalendar(
            f'No {path.name} for {vaastav_season_name(season_start_year)}. Run '
            f'python -m simulation.fixture_calendar {_short_year(season_start_year)}'
        )
    deadline = deadline if deadline.tzinfo else deadline.replace(tzinfo=timezone.utc)
    deadlines_key = tuple(sorted(final_deadlines(season_start_year, data_root).items()))
    timelines, first_time = _timelines(str(path), deadlines_key, lead_days)
    if deadline < first_time:
        raise MissingFixtureCalendar(
            f'No {vaastav_season_name(season_start_year)} fixture calendar saved '
            f'before {deadline.isoformat()}'
        )
    known = {}
    for fixture_id, steps in timelines.items():
        event = None
        for known_from, step_event in steps:
            if known_from > deadline:
                break
            event = step_event
        known[fixture_id] = event
    return known


def apply_known_calendar(rows, calendar: Dict[int, Optional[int]], as_of_gameweek: int):
    """Prediction rows moved to the gameweek FPL showed at the deadline.

    A fixture FPL had not given a gameweek yet, or still showed in a gameweek
    that has already gone, is left out: the manager would not have planned on
    it. A fixture missing from the calendar keeps its row as it is.
    """
    moved = []
    for row in rows:
        fixture_id = int(row['fixture_id'])
        if fixture_id not in calendar:
            moved.append(row)
            continue
        event = calendar[fixture_id]
        if event is None or int(event) < int(as_of_gameweek):
            continue
        if int(event) == int(row['target_gameweek']):
            moved.append(row)
            continue
        row = dict(row)
        row['target_gameweek'] = int(event)
        moved.append(row)
    return moved


def main(argv=None) -> int:
    import sys

    years = [int(value) for value in (argv if argv is not None else sys.argv[1:])] or [23, 24, 25]
    for year in years:
        download_calendar_history(year)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
