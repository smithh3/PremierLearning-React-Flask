"""Estimate FPL overall rank from a season points total.

FPL does not publish a points-to-rank table. Finished seasons keep
``total_points``, ``rank`` and ``rank_percentage`` on each manager's
``/api/entry/{id}/history/`` past-season row. Sampling those and
interpolating is enough to place a simulated total.
"""
import json
import math
import time
from pathlib import Path

from premierlearning.fpl_apis import FPL_USER_AGENT, MANAGER_GAMEWEEK_API


SAMPLE_ENTRY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 75, 90,
    100, 125, 150, 175, 200, 250, 300, 400, 500, 600, 750, 900,
    1000, 1250, 1500, 2000, 3000, 4000, 5000, 6000, 7500, 9000,
    10000, 12500, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 75000, 90000,
    100000, 125000, 150000, 175000, 200000, 250000, 300000, 400000, 500000,
    600000, 750000, 900000,
    1000000, 1250000, 1500000, 1750000, 2000000, 2500000, 3000000, 4000000,
    5000000, 6000000, 7000000, 8000000, 9000000, 10000000, 11000000, 12000000,
)

CACHE_PATH = (
    Path(__file__).resolve().parents[1] / 'run_metrics' / 'fpl_rank_samples.json'
)


def season_label(season_start_year):
    start = int(season_start_year)
    if start < 100:
        start += 2000
    return '%d/%02d' % (start, (start + 1) % 100)


def _parse_percentage(value):
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _history_row_for_season(payload, label):
    for row in (payload or {}).get('past') or []:
        if str(row.get('season_name') or '') == label:
            return row
    current = (payload or {}).get('current') or []
    if not current:
        return None
    last = current[-1]
    points = last.get('total_points')
    rank = last.get('overall_rank')
    if points is None or rank is None:
        return None
    return {
        'season_name': label,
        'total_points': points,
        'rank': rank,
        'rank_percentage': None,
    }


def _load_cache():
    if not CACHE_PATH.is_file():
        return {}
    try:
        with CACHE_PATH.open(encoding='utf-8') as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def _fetch_history(entry_id, session):
    import requests
    url = MANAGER_GAMEWEEK_API % int(entry_id)
    response = session.get(
        url, timeout=12, headers={'User-Agent': FPL_USER_AGENT},
    )
    if response.status_code != 200:
        return None
    return response.json()


def collect_season_samples(season_start_year, entry_ids=None, fetch=True):
    label = season_label(season_start_year)
    cache = _load_cache()
    season_cache = cache.get(label) or {}
    wanted = tuple(entry_ids or SAMPLE_ENTRY_IDS)
    missing = [
        entry_id for entry_id in wanted
        if str(entry_id) not in season_cache
    ]
    if fetch and missing:
        import requests
        session = requests.Session()
        try:
            for entry_id in missing:
                try:
                    payload = _fetch_history(entry_id, session)
                except Exception:
                    continue
                row = _history_row_for_season(payload, label) if payload else None
                if row is None:
                    season_cache[str(entry_id)] = {
                        'entry_id': int(entry_id),
                        'missing': True,
                    }
                else:
                    season_cache[str(entry_id)] = {
                        'entry_id': int(entry_id),
                        'points': int(row.get('total_points') or 0),
                        'rank': int(row.get('rank') or 0),
                        'rank_percentage': _parse_percentage(row.get('rank_percentage')),
                    }
                time.sleep(0.1)
        finally:
            session.close()
        cache[label] = season_cache
        try:
            _save_cache(cache)
        except Exception:
            pass
    return [
        sample for sample in season_cache.values()
        if not sample.get('missing')
        and int(sample.get('points') or 0) > 0
        and int(sample.get('rank') or 0) > 0
    ]


def estimate_total_managers(samples):
    estimates = []
    for sample in samples:
        pct = sample.get('rank_percentage')
        rank = float(sample.get('rank') or 0)
        if pct and rank > 0:
            estimates.append(rank / (pct / 100.0))
    if not estimates:
        return None
    estimates.sort()
    return estimates[len(estimates) // 2]


def interpolate_rank(points, samples):
    rows = sorted(
        (sample for sample in samples if sample.get('rank') and sample.get('points')),
        key=lambda sample: (sample['points'], -sample['rank']),
    )
    if not rows:
        return None
    points = float(points)
    if points <= rows[0]['points']:
        lo = hi = rows[0]
        t = 0.0
    elif points >= rows[-1]['points']:
        lo = hi = rows[-1]
        t = 1.0
    else:
        lo = rows[0]
        hi = rows[-1]
        t = 0.0
        for left, right in zip(rows, rows[1:]):
            if left['points'] <= points <= right['points']:
                lo, hi = left, right
                span = float(right['points'] - left['points'])
                t = 0.0 if span <= 0 else (points - left['points']) / span
                break
    lo_rank = max(1.0, float(lo['rank']))
    hi_rank = max(1.0, float(hi['rank']))
    rank = math.exp(
        math.log(lo_rank) + t * (math.log(hi_rank) - math.log(lo_rank))
    )
    lo_pct = lo.get('rank_percentage')
    hi_pct = hi.get('rank_percentage')
    if lo_pct is None or hi_pct is None:
        top_percent = None
    else:
        top_percent = float(lo_pct) + t * (float(hi_pct) - float(lo_pct))
    return {
        'points': points,
        'rank': max(1, int(round(rank))),
        'top_percent': top_percent,
        'below_points': lo['points'],
        'above_points': hi['points'],
        'sample_count': len(rows),
    }


def estimate_overall_rank(season_start_year, points, fetch=True):
    samples = collect_season_samples(season_start_year, fetch=fetch)
    placed = interpolate_rank(points, samples)
    if placed is None:
        return None
    total = estimate_total_managers(samples)
    if total:
        top_percent = 100.0 * placed['rank'] / float(total)
    else:
        top_percent = placed.get('top_percent')
    ahead_percent = None
    if top_percent is not None:
        ahead_percent = max(0.0, min(100.0, 100.0 - float(top_percent)))
    placed['total_managers'] = int(round(total)) if total else None
    placed['top_percent'] = top_percent
    placed['ahead_percent'] = ahead_percent
    placed['season'] = season_label(season_start_year)
    return placed


def format_rank_line(estimate):
    if not estimate:
        return (
            'Overall rank: FPL history samples were not available for this season.'
        )
    rank = estimate['rank']
    ahead = estimate.get('ahead_percent')
    top = estimate.get('top_percent')
    total = estimate.get('total_managers')
    parts = ['Overall rank: ~{:,}'.format(rank)]
    if total:
        parts.append(' of ~{:,}'.format(total))
    extras = []
    if top is not None:
        extras.append('top %.2f%%' % top)
    if ahead is not None:
        extras.append('ahead of %.2f%% of managers' % ahead)
    if extras:
        parts.append(' (%s)' % ', '.join(extras))
    parts.append(
        ' [interpolated from %d FPL histories in %s]'
        % (estimate.get('sample_count') or 0, estimate.get('season'))
    )
    return ''.join(parts)
