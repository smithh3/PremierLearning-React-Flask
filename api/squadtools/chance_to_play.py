"""Resolve FPL chance-of-playing for live and historical seasons.

Live seasons keep using bootstrap ``chance_of_playing_next_round`` from the
FPL API. Historical / simulation seasons load a thin per-gameweek cache under
``data/season_XX_YY/chance_of_playing/gwN.json``. Missing weeks are filled by
downloading only the last Randdalf/fplcache bootstrap snapshot at or before
that gameweek's deadline.
"""

from __future__ import annotations

import json
import lzma
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from premierlearning.config_constants import SEASON_YR


FPLCACHE_REPO = 'Randdalf/fplcache'
FPLCACHE_API = f'https://api.github.com/repos/{FPLCACHE_REPO}/contents'
FPLCACHE_RAW = f'https://raw.githubusercontent.com/{FPLCACHE_REPO}/main'
USER_AGENT = 'premier-learning-app-chance-to-play'
AVAILABILITY_MODE = 'fplcache_chance'


def data_root_default() -> Path:
    return Path(__file__).resolve().parents[1] / 'data'


def season_dir(season_start_year: int, data_root: Optional[Path] = None) -> Path:
    year = int(season_start_year) % 100 if int(season_start_year) >= 2000 else int(
        season_start_year
    )
    root = Path(data_root) if data_root is not None else data_root_default()
    return root / f'season_{year:02d}_{year + 1:02d}'


def chance_cache_path(
    season_start_year: int,
    gameweek: int,
    data_root: Optional[Path] = None,
) -> Path:
    return (
        season_dir(season_start_year, data_root)
        / 'chance_of_playing'
        / f'gw{int(gameweek)}.json'
    )


def is_live_season(season_start_year: int) -> bool:
    year = int(season_start_year) % 100 if int(season_start_year) >= 2000 else int(
        season_start_year
    )
    return year == int(SEASON_YR)


def resolve_chance_of_playing(
    season_start_year: int,
    gameweek: int,
    player_id: int,
    deadline: Optional[datetime] = None,
    data_root: Optional[Path] = None,
    allow_download: bool = True,
) -> Optional[int]:
    """Return 0/25/50/75/100-style chance, or None when unknown / live season."""
    if is_live_season(season_start_year):
        return None
    chances = ensure_gameweek_chances(
        season_start_year,
        gameweek,
        deadline=deadline,
        data_root=data_root,
        allow_download=allow_download,
    )
    if not chances:
        return None
    return chances.get(int(player_id))


def ensure_gameweek_chances(
    season_start_year: int,
    gameweek: int,
    deadline: Optional[datetime] = None,
    data_root: Optional[Path] = None,
    allow_download: bool = True,
) -> Dict[int, Optional[int]]:
    """Return ``{player_id: chance}`` for a historical gameweek.

    ``None`` values mean FPL reported no doubt (treat like live null → full
    points). Missing player ids are also treated as unknown/full availability by
    callers.
    """
    if is_live_season(season_start_year):
        return {}

    path = chance_cache_path(season_start_year, gameweek, data_root)
    cached = _load_chance_file(path)
    if cached is not None:
        return cached

    if not allow_download:
        return {}
    if deadline is None:
        raise ValueError(
            f'Cannot download fplcache chances for GW{gameweek} without a deadline'
        )

    payload = download_gameweek_chances(int(gameweek), deadline)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return _players_to_chance_map(payload.get('players') or {})


def download_gameweek_chances(gameweek: int, deadline: datetime) -> dict:
    """Fetch the minimal pre-deadline bootstrap slice from Randdalf/fplcache."""
    deadline = _as_utc(deadline)
    snapshot_path, bootstrap = _fetch_pre_deadline_bootstrap(deadline)
    players = {}
    for element in bootstrap.get('elements') or []:
        player_id = element.get('id')
        if player_id in (None, ''):
            continue
        chance = _chance_for_gameweek(element, bootstrap.get('events') or [], gameweek)
        players[str(int(player_id))] = {
            'chance_of_playing_this_round': element.get('chance_of_playing_this_round'),
            'chance_of_playing_next_round': element.get('chance_of_playing_next_round'),
            'chance': chance,
            'status': element.get('status'),
        }
    return {
        'source': FPLCACHE_REPO,
        'snapshot': snapshot_path,
        'gameweek': int(gameweek),
        'deadline': deadline.isoformat(),
        'players': players,
    }


def _load_chance_file(path: Path) -> Optional[Dict[int, Optional[int]]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return _players_to_chance_map(payload.get('players') or {})


def _players_to_chance_map(players: dict) -> Dict[int, Optional[int]]:
    result: Dict[int, Optional[int]] = {}
    for raw_id, row in players.items():
        try:
            player_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if isinstance(row, dict):
            chance = row.get('chance')
            if chance is None:
                chance = row.get('chance_of_playing_this_round')
            if chance is None:
                chance = row.get('chance_of_playing_next_round')
        else:
            chance = row
        result[player_id] = _normalise_chance(chance)
    return result


def _normalise_chance(value) -> Optional[int]:
    if value in (None, '', 'None', 'null'):
        return None
    try:
        chance = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, min(100, chance))


def chance_for_fixture(
    element: dict,
    fixture_gameweek,
    current_gameweek=0,
    next_gameweek=0,
) -> Optional[int]:
    """Scale every remaining fixture by the known FPL chance-to-play.

    The current gameweek uses ``this_round`` (falling back to ``next_round``).
    The next gameweek and every later fixture use ``next_round`` (falling back
    to ``this_round``). That is also how historical snapshots stash the resolved
    fplcache chance on ``chance_of_playing_next_round``. Unfit FPL statuses
    (injured / unavailable / suspended) count as 0% when chance fields are null.
    """
    this_round = _normalise_chance(element.get('chance_of_playing_this_round'))
    next_round = _normalise_chance(element.get('chance_of_playing_next_round'))
    try:
        gameweek = int(fixture_gameweek)
    except (TypeError, ValueError):
        return _apply_unfit_status(
            element, next_round if next_round is not None else this_round
        )
    current_gw = int(current_gameweek or 0)
    if current_gw and gameweek == current_gw:
        return _apply_unfit_status(
            element, this_round if this_round is not None else next_round
        )
    return _apply_unfit_status(
        element, next_round if next_round is not None else this_round
    )


def _apply_unfit_status(element: dict, chance: Optional[int]) -> Optional[int]:
    if chance is not None:
        return chance
    from premierlearning.fpl_apis import UNFIT_STATUSES
    status = str(element.get('status') or '').strip().lower()
    if status in UNFIT_STATUSES:
        return 0
    return None


def _chance_for_gameweek(element: dict, events: list, gameweek: int) -> Optional[int]:
    event = next((row for row in events if int(row.get('id') or 0) == int(gameweek)), None)
    this_round = _normalise_chance(element.get('chance_of_playing_this_round'))
    next_round = _normalise_chance(element.get('chance_of_playing_next_round'))
    if event is not None:
        if event.get('is_next'):
            return _apply_unfit_status(
                element, next_round if next_round is not None else this_round
            )
        if event.get('is_current'):
            return _apply_unfit_status(
                element, this_round if this_round is not None else next_round
            )
    if this_round is not None:
        return _apply_unfit_status(element, this_round)
    return _apply_unfit_status(element, next_round)


def _fetch_pre_deadline_bootstrap(deadline: datetime):
    deadline = _as_utc(deadline)
    errors = []
    for day_offset in range(0, 10):
        day = (deadline - timedelta(days=day_offset)).date()
        try:
            files = _list_snapshot_files(day.year, day.month, day.day)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            errors.append(str(exc))
            continue
        except Exception as exc:  # noqa: BLE001 - network / API failures
            errors.append(str(exc))
            continue
        candidates = []
        for name in files:
            try:
                stamp = _snapshot_datetime(day.year, day.month, day.day, name)
            except ValueError:
                continue
            if stamp <= deadline:
                candidates.append((stamp, name))
        if not candidates:
            continue
        stamp, name = max(candidates, key=lambda item: item[0])
        relative = f'cache/{day.year}/{day.month}/{day.day}/{name}'
        bootstrap = _download_bootstrap(relative)
        return relative, bootstrap
    detail = '; '.join(errors[:3])
    raise FileNotFoundError(
        f'No fplcache snapshot found at or before {deadline.isoformat()}'
        + (f' ({detail})' if detail else '')
    )


def _list_snapshot_files(year: int, month: int, day: int):
    url = f'{FPLCACHE_API}/cache/{year}/{month}/{day}'
    listing = _http_json(url)
    if not isinstance(listing, list):
        raise FileNotFoundError(f'Unexpected fplcache listing for {url}')
    return [
        row['name']
        for row in listing
        if str(row.get('name') or '').endswith('.json.xz')
    ]


def _download_bootstrap(relative_path: str) -> dict:
    url = f'{FPLCACHE_RAW}/{relative_path}'
    payload = _http_bytes(url)
    return json.loads(lzma.decompress(payload))


def _snapshot_datetime(year: int, month: int, day: int, filename: str) -> datetime:
    stem = str(filename).split('.', 1)[0]
    if len(stem) != 4 or not stem.isdigit():
        raise ValueError(f'Unexpected fplcache filename: {filename}')
    return datetime(
        int(year),
        int(month),
        int(day),
        int(stem[:2]),
        int(stem[2:]),
        tzinfo=timezone.utc,
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _http_json(url: str):
    return json.loads(_http_bytes(url).decode('utf-8'))


def _http_bytes(url: str) -> bytes:
    headers = {'User-Agent': USER_AGENT, 'Accept': 'application/vnd.github+json'}
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise urllib.error.HTTPError(
            exc.full_url,
            exc.code,
            f'{exc.reason} for {url}',
            exc.headers,
            exc.fp,
        ) from None
