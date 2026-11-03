"""Optimiser caches that live with the simulation tables.

`init_db` drops the live prediction tables. These tables are created by
`ensure_simulation_schema` and are left in place.

The exact key includes every input that changes a solve, including purchase
prices, because those change what the squad can afford. The candidate key
does not. A candidate squad is checked against the current bank, prices and
squad rules before it is used. Sampled match points are not stored.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import time
import weakref

from squadtools.simulation_strategy import (
    _validate_squad,
    selling_price,
)

OPTIMISER_VERSION = 'mip-v1'
RULES_VERSION = 'hit-4'
# v2 never charges a hit. The version is part of both cache keys, so a v1
# squad solved with hits is not reused, and a v2 squad is not returned to v1.
RULES_VERSION_V2 = 'roll-5'
# v3 also values the bench a little, so its squads are kept apart from v2's.
RULES_VERSION_V3 = 'roll-5-bench'
# Range answers are written in batches. Each batch is one short transaction,
# so other workers are never kept waiting on a long open write.
RANGE_BATCH_ROWS = 25
RANGE_BATCH_SECONDS = 10.0
# About 1 KB each, so the in-memory copy stays near 100 MB at most.
RANGE_MEMORY_KEYS = 100000

_OPEN_CACHES = weakref.WeakSet()


def flush_all_caches():
    for cache in list(_OPEN_CACHES):
        try:
            cache.flush()
        except Exception:  # noqa: BLE001 - a closed connection at exit loses one batch
            pass


atexit.register(flush_all_caches)
# Fast runs reuse a squad when the team value is within £0.5m. Prices are
# stored in tenths of a million, so that band is 5.
RULES_VERSION_FAST = 'roll-5-fast'
VALUE_BAND = 5


def _digest(parts: dict) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def team_value(state, pool) -> int:
    """Bank plus what the 15 would sell for, in tenths of a million."""
    total = int(state.bank)
    for player_id in state.player_ids:
        owned = state.owned[player_id]
        total += selling_price(int(owned.purchase_price), int(pool[player_id].price))
    return int(total)


def team_values_close(current, stored, band=VALUE_BAND) -> bool:
    if stored is None:
        return False
    return abs(int(current) - int(stored)) <= int(band)


def squad_codes(pool, player_ids) -> tuple:
    return tuple(sorted(int(pool[int(player_id)].player_code) for player_id in player_ids))


def purchase_pairs(state, pool) -> tuple:
    pairs = []
    for player_id in state.player_ids:
        pairs.append((
            int(pool[player_id].player_code),
            int(state.owned[player_id].purchase_price),
        ))
    pairs.sort()
    return tuple(pairs)


def exact_key(
    snapshot_id,
    gameweek,
    horizon,
    transfer_count,
    bank,
    free_transfers,
    season_start_year,
    action,
    codes,
    purchases,
    charge_hits,
    rules_version=None,
) -> str:
    return _digest({
        'snapshot_id': int(snapshot_id),
        'gameweek': int(gameweek),
        'horizon': int(horizon),
        'transfer_count': int(transfer_count),
        'bank': int(bank),
        'free_transfers': int(free_transfers),
        'season_start_year': int(season_start_year),
        'rules_version': RULES_VERSION if rules_version is None else str(rules_version),
        'optimiser_version': OPTIMISER_VERSION,
        'action': str(action),
        'squad_codes': list(codes),
        'purchase_prices': [list(pair) for pair in purchases],
        'charge_hits': bool(charge_hits),
    })


def approximate_key(
    snapshot_id, gameweek, horizon, transfer_count, action, codes, rules_version=None,
) -> str:
    """Squad codes and the solve shape. Selling price and bank are not included.

    v1 omits the rules version so the existing rows still match. v2 includes
    it, which keeps a hit-squad out of a no-hit lookup.
    """
    payload = {
        'snapshot_id': int(snapshot_id),
        'gameweek': int(gameweek),
        'horizon': int(horizon),
        'transfer_count': int(transfer_count),
        'optimiser_version': OPTIMISER_VERSION,
        'action': str(action),
        'squad_codes': list(codes),
    }
    if rules_version is not None:
        payload['rules_version'] = str(rules_version)
    return _digest(payload)


def _by_code(pool):
    return {int(player.player_code): player for player in pool.players.values()}


def ids_for_codes(pool, codes):
    lookup = _by_code(pool)
    ids = []
    for code in codes:
        player = lookup.get(int(code))
        if player is None:
            return None
        ids.append(int(player.player_id))
    return tuple(ids)


def affordable(state, pool, selected_ids) -> bool:
    """Sells first, then buys, using the current selling rule and prices."""
    selected = {int(player_id) for player_id in selected_ids}
    bank = int(state.bank)
    owned = set(state.player_ids)
    for player_id in owned:
        if player_id not in selected:
            bank += selling_price(
                state.owned[player_id].purchase_price,
                pool[player_id].price,
            )
    for player_id in selected:
        if player_id not in owned:
            bank -= int(pool[player_id].price)
    return bank >= 0


def candidate_is_legal(state, pool, selected_ids) -> bool:
    try:
        _validate_squad(selected_ids, pool)
    except ValueError:
        return False
    return affordable(state, pool, selected_ids)


def range_key(parts: dict) -> str:
    """Everything that changes a ranged answer except the money it is bounded by."""
    payload = dict(parts)
    payload['optimiser_version'] = OPTIMISER_VERSION
    return _digest(payload)


class OptimiserCache:
    def __init__(self, db):
        self.db = db
        self.hits = 0
        self.misses = 0
        self.rejected = 0
        self.solve_seconds = 0.0
        # Ranged answers this process has seen: key -> [(lower, upper, result)].
        self._ranges = {}
        self._pending = []
        self._last_flush = time.monotonic()
        db.ensure_simulation_schema()
        _OPEN_CACHES.add(self)

    def get_range(self, key: str, value: int):
        """An answer whose range covers `value`, from memory, then the database."""
        value = int(value)
        for lower, upper, result in self._ranges.get(key, ()):
            if lower <= value <= upper:
                return result
        self.db.cursor.execute(self.db._sql(
            '''select lower_bound, upper_bound, result_json from manager_range_cache
               where cache_key = ? and lower_bound <= ? and upper_bound >= ?
               order by upper_bound limit 1'''
        ), (str(key), value, value))
        row = self.db.cursor.fetchone()
        if row is None:
            return None
        result = json.loads(row[2])
        self._remember(key, int(row[0]), int(row[1]), result)
        return result

    def put_range(self, key: str, lower: int, upper: int, result: dict, solve_seconds: float):
        lower, upper = int(lower), int(upper)
        if lower > upper:
            lower = upper
        self._remember(key, lower, upper, result)
        self._pending.append((
            str(key), lower, upper, json.dumps(result, sort_keys=True),
            float(solve_seconds) * 1000.0,
        ))
        if (
            len(self._pending) >= RANGE_BATCH_ROWS
            or time.monotonic() - self._last_flush >= RANGE_BATCH_SECONDS
        ):
            self.flush()

    def flush(self):
        """Write the waiting answers in one short transaction."""
        self._last_flush = time.monotonic()
        if not self._pending:
            return
        rows, self._pending = self._pending, []
        self.db.cursor.executemany(self.db._sql(
            '''insert into manager_range_cache (
                   cache_key, lower_bound, upper_bound, result_json, solve_ms
               ) values (?, ?, ?, ?, ?)'''
        ), rows)
        self.db.connection.commit()

    def _remember(self, key, lower, upper, result):
        if len(self._ranges) >= RANGE_MEMORY_KEYS:
            # The database still has them all; memory just starts again.
            self._ranges = {}
        entries = self._ranges.setdefault(key, [])
        for stored_lower, stored_upper, _result in entries:
            if stored_lower == lower and stored_upper == upper:
                return
        entries.append((lower, upper, result))

    def get_exact(self, key: str):
        self.db.cursor.execute(self.db._sql(
            'select result_json from manager_exact_cache where cache_key = ?'
        ), (str(key),))
        row = self.db.cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put_exact(self, key: str, parts: dict, result: dict, solve_seconds: float):
        payload = json.dumps(result, sort_keys=True)
        codes = json.dumps(list(parts['squad_codes']))
        self.db.cursor.execute(self.db._sql(
            'delete from manager_exact_cache where cache_key = ?'
        ), (str(key),))
        self.db.cursor.execute(self.db._sql(
            '''insert into manager_exact_cache (
                   cache_key, prediction_snapshot_id, gameweek, horizon,
                   transfer_count, bank, free_transfers, season_start_year,
                   rules_version, optimiser_version, action, squad_codes,
                   result_json, solve_ms
               ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        ), (
            str(key),
            int(parts['snapshot_id']),
            int(parts['gameweek']),
            int(parts['horizon']),
            int(parts['transfer_count']),
            int(parts['bank']),
            int(parts['free_transfers']),
            int(parts['season_start_year']),
            str(parts.get('rules_version') or RULES_VERSION),
            OPTIMISER_VERSION,
            str(parts['action']),
            codes,
            payload,
            float(solve_seconds) * 1000.0,
        ))
        self.db.connection.commit()

    def get_candidate(self, key: str):
        self.db.cursor.execute(self.db._sql(
            'select result_json from manager_candidate_cache where cache_key = ?'
        ), (str(key),))
        row = self.db.cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put_candidate(self, key: str, parts: dict, result: dict):
        self.db.cursor.execute(self.db._sql(
            'delete from manager_candidate_cache where cache_key = ?'
        ), (str(key),))
        self.db.cursor.execute(self.db._sql(
            '''insert into manager_candidate_cache (
                   cache_key, prediction_snapshot_id, gameweek, horizon,
                   transfer_count, optimiser_version, squad_codes, result_json
               ) values (?, ?, ?, ?, ?, ?, ?, ?)'''
        ), (
            str(key),
            int(parts['snapshot_id']),
            int(parts['gameweek']),
            int(parts['horizon']),
            int(parts['transfer_count']),
            OPTIMISER_VERSION,
            json.dumps(list(parts['squad_codes'])),
            json.dumps(result, sort_keys=True),
        ))
        self.db.connection.commit()

    def get_fixed(self, snapshot_id: int, gameweek: int):
        self.db.cursor.execute(self.db._sql(
            '''select payload_json from manager_fixed_cache
               where prediction_snapshot_id = ? and gameweek = ?'''
        ), (int(snapshot_id), int(gameweek)))
        row = self.db.cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put_fixed(self, snapshot_id: int, gameweek: int, payload: dict):
        self.db.cursor.execute(self.db._sql(
            '''delete from manager_fixed_cache
               where prediction_snapshot_id = ? and gameweek = ?'''
        ), (int(snapshot_id), int(gameweek)))
        self.db.cursor.execute(self.db._sql(
            '''insert into manager_fixed_cache (
                   prediction_snapshot_id, gameweek, payload_json
               ) values (?, ?, ?)'''
        ), (int(snapshot_id), int(gameweek), json.dumps(payload, sort_keys=True)))
        self.db.connection.commit()
