"""High-level actions solved by the existing squad MIP.

Hits of 4 points sit in the solver objective, so a transfer that is not worth
the points is left on the bench of the action: the squad stays as it is.
Free Hit is solved for this gameweek only, because that squad is thrown away
afterwards. Bench Boost and Triple Captain may spend free transfers and do
not buy hits. Wildcard and Free Hit do not pay hits.

Bench weights stay off. The 15 the solver returns is the squad the hit was
charged for, with no second pass that could add a move afterwards.
"""
from __future__ import annotations

import time

from squadtools.simulation_strategy import (
    CHIP_BENCH_BOOST,
    CHIP_FREE_HIT,
    CHIP_TRIPLE_CAPTAIN,
    CHIP_WILDCARD,
    ChipCalendar,
    SimulationStrategy,
    StrategyConfig,
    TransferPlan,
)

from fpl_manager.actions import (
    BENCH_BOOST,
    FREE_HIT,
    HOLD,
    TRIPLE_CAPTAIN,
    WILDCARD,
    chip_for,
    transfer_count,
)
from fpl_manager.cache import (
    approximate_key,
    candidate_is_legal,
    exact_key,
    ids_for_codes,
    purchase_pairs,
    squad_codes,
)
from fpl_manager.lineup import HIT_COST, greedy_lineup, hit_penalty, score_lineup
from fpl_manager.observation import LAST_GAMEWEEK
from fpl_manager.optimiser import _empty_plan


class MipOptimiser:
    def __init__(self, horizon_gameweeks: int = 6, cache=None, snapshot_id: int = 0, season_start_year: int = 25):
        self.horizon_gameweeks = int(horizon_gameweeks)
        self.cache = cache
        self.snapshot_id = int(snapshot_id)
        self.season_start_year = int(season_start_year)

    def execute(self, action: str, state, pool, gameweek: int):
        chip = chip_for(action)
        if action == HOLD:
            return _empty_plan(state), None
        horizon = 1 if action == FREE_HIT else self.horizon_gameweeks
        max_transfers, charge_hits = _limits(action, state)
        plan = self._solve(
            action, state, pool, gameweek, max_transfers, horizon, charge_hits, chip,
        )
        return plan, chip

    def transfer_gains(self, state, pool, gameweek: int) -> dict:
        """Best squad using at most that many transfers, minus the hold, minus one hit."""
        gains = {}
        for count in (1, 2, 3, 4, 5):
            for horizon in (3, 6):
                try:
                    plan = self._solve(
                        'TRANSFER_%s' % count,
                        state,
                        pool,
                        gameweek,
                        count,
                        horizon,
                        True,
                        None,
                    )
                except ValueError:
                    gains[(count, horizon)] = 0.0
                    continue
                hits = hit_penalty(len(plan.transfers), state.free_transfers, False)
                after = _horizon_points(plan.selected_player_ids, pool, gameweek, horizon, None)
                before = _horizon_points(state.player_ids, pool, gameweek, horizon, None)
                gains[(count, horizon)] = after - before - hits
        return gains

    def _solve(self, action, state, pool, gameweek, max_transfers, horizon, charge_hits, chip):
        codes = squad_codes(pool, state.player_ids)
        purchases = purchase_pairs(state, pool)
        parts = {
            'snapshot_id': self.snapshot_id,
            'gameweek': int(gameweek),
            'horizon': int(horizon),
            'transfer_count': int(max_transfers),
            'bank': int(state.bank),
            'free_transfers': int(state.free_transfers),
            'season_start_year': self.season_start_year,
            'action': str(action),
            'squad_codes': codes,
            'purchase_prices': purchases,
            'charge_hits': bool(charge_hits),
        }
        key = exact_key(
            parts['snapshot_id'],
            parts['gameweek'],
            parts['horizon'],
            parts['transfer_count'],
            parts['bank'],
            parts['free_transfers'],
            parts['season_start_year'],
            parts['action'],
            codes,
            purchases,
            charge_hits,
        )
        approx = approximate_key(
            parts['snapshot_id'],
            parts['gameweek'],
            parts['horizon'],
            parts['transfer_count'],
            parts['action'],
            codes,
        )
        if self.cache is not None:
            stored = self.cache.get_exact(key)
            if stored is not None:
                plan = _safe_plan(stored, state, pool, gameweek, horizon, chip)
                if plan is not None:
                    self.cache.hits += 1
                    return plan
            candidate = self.cache.get_candidate(approx)
            if candidate is not None:
                selected = ids_for_codes(pool, candidate.get('selected_codes') or ())
                if selected is not None and candidate_is_legal(state, pool, selected):
                    self.cache.hits += 1
                    return _plan_from_selected(state, pool, gameweek, horizon, selected, chip)
                self.cache.rejected += 1
            self.cache.misses += 1

        started = time.perf_counter()
        selected = _solve_squad(
            state,
            pool,
            gameweek,
            horizon,
            max_transfers,
            charge_hits,
            chip,
            self.season_start_year,
        )
        elapsed = time.perf_counter() - started
        plan = _plan_from_selected(state, pool, gameweek, horizon, selected, chip)
        if self.cache is not None:
            self.cache.solve_seconds += elapsed
            payload = _serialise(pool, plan)
            self.cache.put_exact(key, parts, payload, elapsed)
            self.cache.put_candidate(approx, parts, payload)
        return plan


def _limits(action, state):
    if action in (WILDCARD, FREE_HIT):
        return 15, False
    if action in (BENCH_BOOST, TRIPLE_CAPTAIN):
        return max(0, int(state.free_transfers)), False
    return transfer_count(action), True


def _calendar(chip, gameweek) -> ChipCalendar:
    gameweek = int(gameweek)
    if chip == CHIP_WILDCARD:
        return ChipCalendar(wildcards=(gameweek,))
    if chip == CHIP_FREE_HIT:
        return ChipCalendar(free_hits=(gameweek,))
    if chip == CHIP_BENCH_BOOST:
        return ChipCalendar(bench_boosts=(gameweek,))
    if chip == CHIP_TRIPLE_CAPTAIN:
        return ChipCalendar(triple_captains=(gameweek,))
    return ChipCalendar()


def _solve_squad(state, pool, gameweek, horizon, max_transfers, charge_hits, chip, season_start_year):
    weeks = _weeks(gameweek, horizon)
    if not weeks:
        return tuple(state.player_ids)
    if int(max_transfers) <= 0:
        return tuple(state.player_ids)
    strategy = SimulationStrategy(StrategyConfig(
        horizon_gameweeks=len(weeks),
        min_current_gw_gain=0.0,
        min_horizon_gain=0.0,
        max_transfers_per_gameweek=max(1, int(max_transfers)),
        chips=_calendar(chip, gameweek),
        bench_fill_weight=0.0,
        bench_weights=(),
        season_start_year=int(season_start_year),
    ))
    return strategy._optimise_squad(
        pool=pool,
        as_of_gameweek=int(gameweek),
        current_state=state,
        max_transfers=int(max_transfers),
        budget=int(state.bank),
        hold_current=0.0,
        hold_horizon=0.0,
        require_transfer=False,
        enforce_thresholds=False,
        gameweeks=weeks,
        hit_cost=HIT_COST if charge_hits else 0.0,
        free_transfers=int(state.free_transfers),
    )


def _weeks(gameweek, horizon):
    weeks = []
    for offset in range(int(horizon)):
        week = int(gameweek) + offset
        if week > LAST_GAMEWEEK:
            break
        weeks.append(week)
    return tuple(weeks)


def _xi(player_ids, pool, gameweek, bench_boost, triple_captain) -> float:
    starters, bench, captain_id, _vice = greedy_lineup(
        player_ids,
        lambda player_id: pool[player_id].points(gameweek),
        lambda player_id: pool[player_id].position,
    )
    return score_lineup(
        lambda player_id: pool[player_id].points(gameweek),
        starters,
        bench,
        captain_id,
        bench_boost,
        triple_captain,
    )


def _horizon_points(player_ids, pool, gameweek, horizon, chip) -> float:
    total = 0.0
    for offset, week in enumerate(_weeks(gameweek, horizon)):
        total += _xi(
            player_ids,
            pool,
            week,
            bench_boost=chip == CHIP_BENCH_BOOST and offset == 0,
            triple_captain=chip == CHIP_TRIPLE_CAPTAIN and offset == 0,
        )
    return total


def _plan_from_selected(state, pool, gameweek, horizon, selected, chip) -> TransferPlan:
    selected = tuple(sorted(int(player_id) for player_id in selected))
    transfers = tuple(SimulationStrategy._pair_transfers(state, selected, pool))
    return TransferPlan(
        transfers=transfers,
        selected_player_ids=selected,
        expected_hold_current=_horizon_points(state.player_ids, pool, gameweek, 1, chip),
        expected_selected_current=_horizon_points(selected, pool, gameweek, 1, chip),
        expected_hold_horizon=_horizon_points(state.player_ids, pool, gameweek, horizon, chip),
        expected_selected_horizon=_horizon_points(selected, pool, gameweek, horizon, chip),
    )


def _serialise(pool, plan: TransferPlan) -> dict:
    return {
        'selected_codes': [int(pool[player_id].player_code) for player_id in plan.selected_player_ids],
        'transfers': [
            {
                'out_code': int(pool[transfer.player_out_id].player_code),
                'in_code': int(pool[transfer.player_in_id].player_code),
            }
            for transfer in plan.transfers
        ],
    }


def _hydrate(payload, state, pool, gameweek, horizon, chip):
    selected = ids_for_codes(pool, payload.get('selected_codes') or ())
    if selected is None:
        return None
    return _plan_from_selected(state, pool, gameweek, horizon, selected, chip)


def _safe_plan(payload, state, pool, gameweek, horizon, chip):
    try:
        return _hydrate(payload, state, pool, gameweek, horizon, chip)
    except ValueError:
        return None
