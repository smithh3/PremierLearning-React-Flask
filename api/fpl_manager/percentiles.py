"""Official FPL score percentiles, saved locally.

FPL does not publish a table of every score. Each manager's history does.
A finished season keeps the final points and rank. The season still in
progress also keeps every gameweek's points and rank. Those rows are sampled
across the entry-id range and stored in ``api/data/fpl_percentiles.json``.
A score's percentile is read off that curve. Higher is better: the 90th
percentile beat about 90 percent of managers.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path


CURVES_PATH = Path(__file__).resolve().parents[1] / 'data' / 'fpl_percentiles.json'
_RAW_PATH = Path(__file__).resolve().parents[1] / 'run_metrics' / 'fpl_history_raw.json'
_CURVES = None


def season_label(season_start_year) -> str:
    start = int(season_start_year)
    if start < 100:
        start += 2000
    return '%d/%02d' % (start, (start + 1) % 100)


def load_curves(path=None) -> dict:
    global _CURVES
    target = Path(path) if path is not None else CURVES_PATH
    if path is None and _CURVES is not None:
        return _CURVES
    if not target.is_file():
        curves = {'seasons': {}, 'gameweeks': {}}
    else:
        curves = json.loads(target.read_text(encoding='utf-8'))
    if path is None:
        _CURVES = curves
    return curves


def beaten_percent(points, samples, field_size):
    """Percent of managers this score beats. Rank 1 is the best score."""
    rank = _interpolate_rank(points, samples)
    if rank is None or not field_size:
        return None
    beaten = 100.0 * (float(field_size) - float(rank)) / float(field_size)
    if beaten < 0.0:
        return 0.0
    if beaten > 100.0:
        return 100.0
    return beaten


def describe_gameweek(season_start_year, gameweek, points, curves=None) -> str:
    curves = load_curves() if curves is None else curves
    label = season_label(season_start_year)
    block = ((curves.get('gameweeks') or {}).get(label) or {}).get(str(int(gameweek)))
    if not block:
        return ''
    beaten = beaten_percent(points, block.get('samples') or [], block.get('field_size'))
    if beaten is None:
        return ''
    return '%s percentile of %s gameweek %s' % (ordinal(beaten), label, int(gameweek))


def season_beaten_percent(season_start_year, points, curves=None):
    """Percent of managers this season total beat. None when that season was not saved."""
    curves = load_curves() if curves is None else curves
    label = season_label(season_start_year)
    block = (curves.get('seasons') or {}).get(label)
    if not block:
        return None
    return beaten_percent(points, block.get('samples') or [], block.get('field_size'))


def describe_season(season_start_year, points, curves=None) -> str:
    beaten = season_beaten_percent(season_start_year, points, curves)
    if beaten is None:
        return ''
    return '%s percentile of %s finishes' % (
        ordinal(beaten), season_label(season_start_year),
    )


def ordinal(percent) -> str:
    number = int(round(float(percent)))
    if number < 0:
        number = 0
    if number > 100:
        number = 100
    if 10 <= number % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
    return '%d%s' % (number, suffix)


def _interpolate_rank(points, samples):
    rows = sorted(
        (
            (float(row[0]), float(row[1]))
            for row in samples
            if len(row) >= 2 and float(row[1]) > 0
        ),
        key=lambda row: (row[0], -row[1]),
    )
    if not rows:
        return None
    points = float(points)
    if points <= rows[0][0]:
        return rows[0][1]
    if points >= rows[-1][0]:
        return rows[-1][1]
    for left, right in zip(rows, rows[1:]):
        if left[0] <= points <= right[0]:
            span = right[0] - left[0]
            t = 0.0 if span <= 0 else (points - left[0]) / span
            low = max(1.0, left[1])
            high = max(1.0, right[1])
            return math.exp(math.log(low) + t * (math.log(high) - math.log(low)))
    return rows[-1][1]


def _entry_ids(total_players):
    from simulation.overall_rank import SAMPLE_ENTRY_IDS

    chosen = {int(entry_id) for entry_id in SAMPLE_ENTRY_IDS}
    total = max(1, int(total_players))
    step = max(1, total // 300)
    for entry_id in range(1, total + 1, step):
        chosen.add(int(entry_id))
    return sorted(entry_id for entry_id in chosen if entry_id > 0)


def _season_from_deadline(deadline) -> str:
    text = str(deadline or '')
    year = int(text[:4])
    month = int(text[5:7])
    start = year if month >= 7 else year - 1
    return season_label(start)


def _field_size(rows):
    estimates = []
    for row in rows:
        try:
            percent = float(row.get('rank_percentage'))
            rank = float(row.get('rank') or 0)
        except (TypeError, ValueError):
            continue
        if percent >= 1.0 and rank > 0:
            estimates.append(rank / (percent / 100.0))
    if not estimates:
        return None
    estimates.sort()
    return int(round(estimates[len(estimates) // 2]))


def _load_raw():
    if not _RAW_PATH.is_file():
        return {}
    try:
        payload = json.loads(_RAW_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    entries = payload.get('entries') if isinstance(payload, dict) else None
    return entries if isinstance(entries, dict) else {}


def _save_raw(entries):
    _RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RAW_PATH.write_text(
        json.dumps({'entries': entries}, sort_keys=True),
        encoding='utf-8',
    )


def pull_histories():
    """Download manager histories and write the percentile curves."""
    import requests

    from premierlearning.fpl_apis import FPL_USER_AGENT, GENERAL, MANAGER_GAMEWEEK_API

    session = requests.Session()
    headers = {'User-Agent': FPL_USER_AGENT}
    general = session.get(GENERAL, timeout=30, headers=headers).json()
    total_players = int(general.get('total_players') or 0)
    events = general.get('events') or []
    current_season = _season_from_deadline((events[0] or {}).get('deadline_time')) if events else ''
    ranked_count = {
        int(event['id']): int(event.get('ranked_count') or 0)
        for event in events
        if event.get('id') is not None
    }
    entries = _load_raw()
    wanted = [str(entry_id) for entry_id in _entry_ids(total_players)]
    print(
        'Pulling %s FPL histories (%s already saved)'
        % (len(wanted), sum(1 for entry_id in wanted if entry_id in entries)),
        flush=True,
    )
    try:
        for index, entry_id in enumerate(wanted, start=1):
            if entry_id in entries:
                continue
            try:
                response = session.get(
                    MANAGER_GAMEWEEK_API % int(entry_id),
                    timeout=20,
                    headers=headers,
                )
            except requests.RequestException:
                time.sleep(1.0)
                continue
            if response.status_code != 200:
                entries[entry_id] = {'missing': True}
            else:
                payload = response.json()
                entries[entry_id] = {
                    'past': payload.get('past') or [],
                    'current': [
                        {
                            'event': row.get('event'),
                            'points': row.get('points'),
                            'rank': row.get('rank'),
                        }
                        for row in (payload.get('current') or [])
                    ],
                }
            if index % 25 == 0:
                _save_raw(entries)
                print('Saved %s histories' % index, flush=True)
            time.sleep(0.08)
    finally:
        session.close()
        _save_raw(entries)
    curves = _curves_from_histories(entries, current_season, ranked_count, total_players)
    CURVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURVES_PATH.write_text(
        json.dumps(curves, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    global _CURVES
    _CURVES = curves
    print(
        'Wrote %s (%s season curves, %s gameweek curves)'
        % (
            CURVES_PATH,
            len(curves['seasons']),
            sum(len(weeks) for weeks in curves['gameweeks'].values()),
        ),
        flush=True,
    )
    return curves


def _curves_from_histories(entries, current_season, ranked_count, total_players):
    seasons = {}
    gameweeks = {}
    for payload in entries.values():
        if not isinstance(payload, dict) or payload.get('missing'):
            continue
        for row in payload.get('past') or []:
            label = str(row.get('season_name') or '')
            points = row.get('total_points')
            rank = row.get('rank')
            if not label or points is None or not rank:
                continue
            bucket = seasons.setdefault(label, {'rows': []})
            bucket['rows'].append({
                'points': int(points),
                'rank': int(rank),
                'rank_percentage': row.get('rank_percentage'),
            })
        if not current_season:
            continue
        weeks = gameweeks.setdefault(current_season, {})
        for row in payload.get('current') or []:
            event = row.get('event')
            points = row.get('points')
            rank = row.get('rank')
            if event is None or points is None or not rank:
                continue
            bucket = weeks.setdefault(str(int(event)), {'rows': []})
            bucket['rows'].append((int(points), int(rank)))
    season_curves = {}
    for label, bucket in seasons.items():
        field = _field_size(bucket['rows'])
        samples = sorted({(row['points'], row['rank']) for row in bucket['rows']})
        if field and samples:
            season_curves[label] = {
                'field_size': int(field),
                'sample_count': len(samples),
                'samples': [list(row) for row in samples],
            }
    gameweek_curves = {}
    for label, weeks in gameweeks.items():
        kept = {}
        for event, bucket in weeks.items():
            field = int(ranked_count.get(int(event)) or 0)
            samples = sorted(set(bucket['rows']))
            if field and samples:
                kept[event] = {
                    'field_size': field,
                    'sample_count': len(samples),
                    'samples': [list(row) for row in samples],
                }
        if kept:
            gameweek_curves[label] = kept
    return {
        'source': 'https://fantasy.premierleague.com/api/entry/{id}/history/',
        'total_players': int(total_players or 0),
        'current_season': current_season,
        'seasons': season_curves,
        'gameweeks': gameweek_curves,
    }


if __name__ == '__main__':
    pull_histories()
