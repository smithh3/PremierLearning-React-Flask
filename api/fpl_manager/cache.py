"""Optimiser caches that live with the simulation tables.

`init_db` drops the live prediction tables. These tables are created by
`ensure_simulation_schema` and are left in place.

The exact key includes every input that changes a solve, including purchase
prices, because those change what the squad can afford. The candidate key
does not. A candidate squad is checked against the current bank, prices and
squad rules before it is used. Sampled match points are not stored.
"""
from __future__ import annotations

import hashlib
import json

from squadtools.simulation_strategy import (
    _validate_squad,
    selling_price,
)

OPTIMISER_VERSION = 'mip-v1'
RULES_VERSION = 'hit-4'


def _digest(parts: dict) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


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
) -> str:
    return _digest({
        'snapshot_id': int(snapshot_id),
        'gameweek': int(gameweek),
        'horizon': int(horizon),
        'transfer_count': int(transfer_count),
        'bank': int(bank),
        'free_transfers': int(free_transfers),
        'season_start_year': int(season_start_year),
        'rules_version': RULES_VERSION,
        'optimiser_version': OPTIMISER_VERSION,
        'action': str(action),
        'squad_codes': list(codes),
        'purchase_prices': [list(pair) for pair in purchases],
        'charge_hits': bool(charge_hits),
    })


def approximate_key(snapshot_id, gameweek, horizon, transfer_count, action, codes) -> str:
    """Squad codes and the solve shape. Selling price and bank are not included."""
    return _digest({
        'snapshot_id': int(snapshot_id),
        'gameweek': int(gameweek),
        'horizon': int(horizon),
        'transfer_count': int(transfer_count),
        'optimiser_version': OPTIMISER_VERSION,
        'action': str(action),
        'squad_codes': list(codes),
    })


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


class OptimiserCache:
    def __init__(self, db):
        self.db = db
        self.hits = 0
        self.misses = 0
        self.rejected = 0
        self.solve_seconds = 0.0
        db.ensure_simulation_schema()

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
            RULES_VERSION,
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
