"""Top-N FPL squad cost at season boundaries, for early team-form priors.

End of last season vs start of this season, using the N most expensive players
per club so roster-list bloat does not dominate. Deltas are league-centered
among staying clubs. Promoted sides have no prior list and get z = 0.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from premierlearning.config_constants import SQUAD_COST_TOP_N


def use_squad_cost() -> bool:
    """Top-15 YoY squad-cost prior. Off via --no-squad-cost or PL_SQUAD_COST=0."""
    if '--no-squad-cost' in sys.argv:
        return False
    raw = os.environ.get('PL_SQUAD_COST', '1')
    return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')


def data_root() -> Path:
    return Path(__file__).resolve().parents[1] / 'data'


def season_dir(year_from: int, root: Optional[Path] = None) -> Path:
    years = (int(year_from), int(year_from) + 1)
    return (root or data_root()) / ('season_%i_%i' % years)


def squad_cost_early_weight(gameweek) -> float:
    """1.0 at GW1, 0.0 from GW5 on. Unknown/missing GW → no fade (treat as GW1)."""
    from premierlearning.config_constants import SQUAD_COST_FADE_GAMES
    try:
        gw = int(gameweek)
    except (TypeError, ValueError):
        return 1.0
    if gw < 1:
        return 1.0
    fade = float(SQUAD_COST_FADE_GAMES) or 1.0
    return max(0.0, 1.0 - (gw - 1) / fade)


def nudge_rate(rate: float, z: float, weight: float, sign: float = 1.0) -> float:
    """Multiply a scoring rate by 1 ± scale * clip(z) * weight."""
    from premierlearning.config_constants import (
        SQUAD_COST_MIN_RATE,
        SQUAD_COST_RATE_SCALE,
        SQUAD_COST_Z_CLIP,
    )
    z = max(-SQUAD_COST_Z_CLIP, min(SQUAD_COST_Z_CLIP, float(z or 0.0)))
    w = max(0.0, min(1.0, float(weight or 0.0)))
    factor = 1.0 + float(sign) * SQUAD_COST_RATE_SCALE * z * w
    return max(SQUAD_COST_MIN_RATE, float(rate or 0.0) * factor)


def top_n_sum(values: Iterable[float], n: int = SQUAD_COST_TOP_N) -> float:
    ranked = sorted((float(v) for v in values), reverse=True)
    return float(sum(ranked[:n]))


def centered_z(deltas: Dict[int, float]) -> Dict[int, float]:
    """Mean-center then divide by std (population). Empty / tiny std → zeros."""
    if not deltas:
        return {}
    values = list(deltas.values())
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    std = var ** 0.5
    if std < 1e-6:
        return {code: 0.0 for code in deltas}
    return {code: (delta - mean) / std for code, delta in deltas.items()}


def _read_csv(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    for encoding in ('utf-8-sig', 'latin-1'):
        try:
            with path.open(newline='', encoding=encoding) as csv_file:
                return list(csv.DictReader(csv_file))
        except UnicodeDecodeError:
            continue
    return []


def _norm_name(value) -> str:
    return ' '.join(str(value or '').strip().casefold().split())


def _last_gw_csv(directory: Path) -> Optional[Path]:
    gws_dir = directory / 'gws'
    if not gws_dir.is_dir():
        return None
    best_n = -1
    best_path = None
    for path in gws_dir.glob('gw*.csv'):
        rest = path.stem[2:]
        if rest.isdigit():
            n = int(rest)
            if n > best_n:
                best_n = n
                best_path = path
    return best_path


def _club_maps_from_rows(rows: Iterable[dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    name_to_code: Dict[str, int] = {}
    code_to_short: Dict[int, str] = {}
    for row in rows:
        raw_code = row.get('code')
        try:
            code = int(float(raw_code)) if raw_code not in (None, '') else None
        except (TypeError, ValueError):
            code = None
        if code is None:
            continue
        short = str(row.get('short_name') or '').strip()
        if short:
            code_to_short[code] = short
            name_to_code[_norm_name(short)] = code
        for key in ('name', 'short_name'):
            value = row.get(key)
            if value:
                name_to_code[_norm_name(value)] = code
    return name_to_code, code_to_short


_CLUB_MAPS_CACHE: Dict[str, Tuple[Dict[str, int], Dict[int, str]]] = {}


def club_maps(root: Optional[Path] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Union of vaastav / FPL names → stable team.code across season folders."""
    root = root or data_root()
    key = str(root.resolve()) if root.exists() else str(root)
    cached = _CLUB_MAPS_CACHE.get(key)
    if cached is not None:
        return cached
    name_to_code: Dict[str, int] = {}
    code_to_short: Dict[int, str] = {}
    if root.is_dir():
        for path in sorted(root.glob('season_*_*')):
            if not path.is_dir():
                continue
            csv_rows = _read_csv(path / 'teams.csv')
            if csv_rows:
                names, shorts = _club_maps_from_rows(csv_rows)
                name_to_code.update(names)
                code_to_short.update(shorts)
                continue
            fantasy_path = path / 'fantasy.json'
            if fantasy_path.is_file():
                from premierlearning.helper_functions import read_json
                fantasy = read_json(str(fantasy_path)) or {}
                names, shorts = _club_maps_from_rows(fantasy.get('teams') or [])
                name_to_code.update(names)
                code_to_short.update(shorts)
    _CLUB_MAPS_CACHE[key] = (name_to_code, code_to_short)
    return name_to_code, code_to_short


def _top_n_from_gw_csv(
    path: Path,
    name_to_code: Dict[str, int],
    n: int = SQUAD_COST_TOP_N,
) -> Dict[int, float]:
    by_code: Dict[int, List[float]] = defaultdict(list)
    for row in _read_csv(path):
        code = name_to_code.get(_norm_name(row.get('team')))
        if code is None:
            continue
        try:
            value = float(row.get('value'))
        except (TypeError, ValueError):
            continue
        by_code[code].append(value)
    return {code: top_n_sum(values, n) for code, values in by_code.items()}


def _top_n_from_players(season, n: int = SQUAD_COST_TOP_N) -> Dict[int, float]:
    by_code: Dict[int, List[float]] = defaultdict(list)
    for player in getattr(season, 'players', None) or []:
        team = getattr(player, 'team', None)
        code = getattr(team, 'code', None) if team is not None else None
        if code is None:
            continue
        try:
            code_i = int(code)
        except (TypeError, ValueError):
            continue
        cost = getattr(player, 'current_cost', None)
        if cost is None:
            continue
        try:
            by_code[code_i].append(float(cost))
        except (TypeError, ValueError):
            continue
    return {code: top_n_sum(values, n) for code, values in by_code.items()}


def end_of_season_top_n(
    year_from: int,
    season=None,
    root: Optional[Path] = None,
    n: int = SQUAD_COST_TOP_N,
) -> Dict[int, float]:
    """Top-N costs at the last available GW snapshot, else in-memory player costs."""
    root = root or data_root()
    directory = season_dir(year_from, root)
    name_to_code, _ = club_maps(root)
    gw_path = _last_gw_csv(directory)
    if gw_path is not None:
        costs = _top_n_from_gw_csv(gw_path, name_to_code, n)
        if costs:
            return costs
    if season is not None:
        return _top_n_from_players(season, n)
    return {}


def start_of_season_top_n(
    year_from: int,
    season=None,
    root: Optional[Path] = None,
    n: int = SQUAD_COST_TOP_N,
) -> Dict[int, float]:
    """Top-N costs at GW1 if the file exists, else in-memory (live start prices)."""
    root = root or data_root()
    directory = season_dir(year_from, root)
    name_to_code, _ = club_maps(root)
    gw1 = directory / 'gws' / 'gw1.csv'
    if gw1.is_file():
        costs = _top_n_from_gw_csv(gw1, name_to_code, n)
        if costs:
            return costs
    if season is not None:
        return _top_n_from_players(season, n)
    return {}


def squad_cost_z_by_code(
    prev_year: int,
    curr_year: int,
    prev_season=None,
    curr_season=None,
    root: Optional[Path] = None,
    n: int = SQUAD_COST_TOP_N,
) -> Dict[int, float]:
    """Centered YoY z for clubs present in both end-of-prev and start-of-curr."""
    end_costs = end_of_season_top_n(prev_year, prev_season, root, n)
    start_costs = start_of_season_top_n(curr_year, curr_season, root, n)
    overlap = set(end_costs) & set(start_costs)
    deltas = {code: start_costs[code] - end_costs[code] for code in overlap}
    return centered_z(deltas)
