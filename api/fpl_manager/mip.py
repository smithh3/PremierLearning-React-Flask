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
    FreeTransferRules,
    SimulationStrategy,
    SquadState,
    StrategyConfig,
    TransferPlan,
    copy_squad_state,
    optimiser_candidate_ids,
    selling_price,
)

from fpl_manager import highs_solver
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
    RULES_VERSION,
    RULES_VERSION_FAST,
    RULES_VERSION_V2,
    RULES_VERSION_V3,
    approximate_key,
    candidate_is_legal,
    exact_key,
    ids_for_codes,
    purchase_pairs,
    range_key,
    squad_codes,
    team_value,
    team_values_close,
)

# HiGHS gets this long first. Most squad problems finish well inside it.
# A few hard ones take HiGHS several seconds, so those go to CBC instead.
HIGHS_FIRST_SECONDS = 0.75
# Lower end of a stored "no legal squad" answer: it covers any smaller amount.
_NO_LOWER_BOUND = -10 ** 9
_RULES_FOR_VERSION = {
    'v2': RULES_VERSION_V2,
    'fast': RULES_VERSION_FAST,
    'v3': RULES_VERSION_V3,
}
from fpl_manager.lineup import HIT_COST, greedy_lineup, hit_penalty, score_lineup
from fpl_manager.observation import LAST_GAMEWEEK
from fpl_manager.optimiser import _empty_plan


class MipOptimiser:
    def __init__(
        self,
        horizon_gameweeks: int = 6,
        cache=None,
        snapshot_id: int = 0,
        season_start_year: int = 25,
        observation_version: str = 'v1',
        solver: str = 'cbc',
        ranged: bool = False,
        bench_weight: float = 0.0,
    ):
        """`solver` 'highs' tries HiGHS first and hands hard problems to CBC.
        `ranged` stores answers by budget or bank range (see cache.py).
        `bench_weight` counts that share of the bench's predicted points.
        """
        self.horizon_gameweeks = int(horizon_gameweeks)
        self.cache = cache
        self.snapshot_id = int(snapshot_id)
        self.season_start_year = int(season_start_year)
        self.observation_version = str(observation_version or 'v1')
        self.solver = str(solver or 'cbc')
        self.ranged = bool(ranged)
        self.bench_weight = float(bench_weight)
        self.highs_solves = 0
        self.cbc_solves = 0

    def execute(self, action: str, state, pool, gameweek: int):
        chip = chip_for(action)
        if action == HOLD:
            return _empty_plan(state), None
        try:
            if self.observation_version in ('v2', 'fast', 'v3'):
                return self._execute_v2(action, state, pool, gameweek, chip)
            horizon = 1 if action == FREE_HIT else self.horizon_gameweeks
            max_transfers, charge_hits = _limits(action, state)
            plan = self._solve(
                action, state, pool, gameweek, max_transfers, horizon, charge_hits, chip,
            )
            return plan, chip
        except ValueError:
            return _empty_plan(state), None

    def _execute_v2(self, action, state, pool, gameweek, chip):
        from fpl_manager.plan_v2 import PLAN_WEEKS

        rules = FreeTransferRules(self.season_start_year)
        free_transfers = rules.at_start_of_gameweek(gameweek, state.free_transfers)
        prepared = copy_squad_state(state)
        prepared.free_transfers = free_transfers
        if action == FREE_HIT:
            weeks = _weeks(gameweek, 1)
            max_transfers = 15
        elif action == WILDCARD:
            weeks = _weeks(gameweek, PLAN_WEEKS)
            max_transfers = 15
        elif action in (BENCH_BOOST, TRIPLE_CAPTAIN):
            weeks = _weeks(gameweek, PLAN_WEEKS)
            max_transfers = free_transfers
        else:
            weeks = _weeks(gameweek, PLAN_WEEKS)
            max_transfers = min(transfer_count(action), free_transfers)
        plan = self._solve(
            action,
            prepared,
            pool,
            gameweek,
            max_transfers,
            len(weeks) or 1,
            False,
            chip,
            gameweeks=weeks,
            rules_version=_RULES_FOR_VERSION.get(self.observation_version, RULES_VERSION_V2),
        )
        return plan, chip

    def transfer_gains(self, state, pool, gameweek: int, played=None) -> dict:
        """Points the plan adds over holding the current squad."""
        if self.observation_version == 'v3':
            from fpl_manager.plan_v3 import v3_transfer_gains
            return v3_transfer_gains(self, state, pool, gameweek, played)
        if self.observation_version == 'v2':
            from fpl_manager.plan_v2 import v2_transfer_gains
            return v2_transfer_gains(self, state, pool, gameweek, played)
        if self.observation_version == 'fast':
            from fpl_manager.plan_fast import fast_transfer_gains
            return fast_transfer_gains(self, state, pool, gameweek, played)
        # v1: best squad using at most that many transfers, minus the hold, minus hits.
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

    def _solve(
        self,
        action,
        state,
        pool,
        gameweek,
        max_transfers,
        horizon,
        charge_hits,
        chip,
        gameweeks=None,
        rules_version=None,
        value_band=False,
        input_question=False,
    ):
        if gameweeks is None:
            gameweeks = _weeks(gameweek, horizon)
        else:
            gameweeks = tuple(int(week) for week in gameweeks)
        horizon = len(gameweeks)
        rules_version = RULES_VERSION if rules_version is None else str(rules_version)
        if self.ranged and not charge_hits and rules_version in (RULES_VERSION_V2, RULES_VERSION_V3):
            return self._solve_ranged(
                state, pool, gameweek, max_transfers, chip, gameweeks, rules_version,
                input_question,
            )
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
            'rules_version': rules_version,
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
            rules_version=rules_version,
        )
        approx = approximate_key(
            parts['snapshot_id'],
            parts['gameweek'],
            parts['horizon'],
            parts['transfer_count'],
            parts['action'],
            codes,
            rules_version=None if rules_version == RULES_VERSION else rules_version,
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
                legal = selected is not None and candidate_is_legal(state, pool, selected)
                banded = (
                    value_band
                    and selected is not None
                    and team_values_close(team_value(state, pool), candidate.get('team_value'))
                )
                if legal or banded:
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
            gameweeks=gameweeks,
            bench_weight=self.bench_weight,
        )
        elapsed = time.perf_counter() - started
        plan = _plan_from_selected(state, pool, gameweek, horizon, selected, chip)
        if self.cache is not None:
            self.cache.solve_seconds += elapsed
            payload = _serialise(pool, plan)
            if rules_version == RULES_VERSION_FAST:
                payload['team_value'] = team_value(state, pool)
            self.cache.put_exact(key, parts, payload, elapsed)
            self.cache.put_candidate(approx, parts, payload)
        return plan

    def _solve_ranged(
        self, state, pool, gameweek, max_transfers, chip, gameweeks, rules_version,
        input_question,
    ):
        """Answer from a stored money range, or solve and store the range.

        A wildcard or free-hit squad does not depend on the squad it replaces,
        only on the money available: bank plus what the owned players sell
        for. An owned player still matters when keeping him costs less than
        buying him (his price rose after you bought him) or when he would not
        otherwise be a candidate, so those players are part of the key. A
        squad found with budget B that costs C is the best squad for every
        budget from C to B.

        A transfer plan depends on the owned squad and its sale prices, and is
        bounded by the bank instead: a plan found with bank B that needs N is
        the best plan for every bank from N to B.

        `input_question` is for the inputs the policy sees, never for an
        action it plays: owned players are priced at today's price and are
        left out of a squad key. The answer is still affordable, because
        keeping a player never costs more than today's price.
        """
        weeks = tuple(int(week) for week in gameweeks)
        horizon = len(weeks)
        current = list(state.player_ids)
        keep_cost = {
            player_id: selling_price(state.owned[player_id].purchase_price, pool[player_id].price)
            for player_id in current
        }
        base = {
            'snapshot_id': int(self.snapshot_id),
            'season_start_year': int(self.season_start_year),
            'rules_version': str(rules_version),
            'gameweeks': list(weeks),
            'scoring': _scoring_name(chip),
            'bench_weight': float(self.bench_weight),
        }
        squad_chip = chip in (CHIP_WILDCARD, CHIP_FREE_HIT) and int(max_transfers) >= 15
        if squad_chip:
            regular = set(optimiser_candidate_ids(pool, (), weeks))
            specials = [] if input_question else sorted(
                player_id for player_id in current
                if keep_cost[player_id] != int(pool[player_id].price) or player_id not in regular
            )
            parts = dict(base, kind='squad', specials=sorted(
                (int(pool[player_id].player_code), int(keep_cost[player_id]))
                for player_id in specials
            ))
            money = int(state.bank) + sum(keep_cost.values())
        else:
            parts = dict(base, kind='transfers', max_transfers=int(max_transfers), owned=sorted(
                (int(pool[player_id].player_code), int(keep_cost[player_id]))
                for player_id in current
            ))
            money = int(state.bank)
        key = range_key(parts)
        if self.cache is not None:
            stored = self.cache.get_range(key, money)
            if stored is not None and stored.get('infeasible'):
                self.cache.hits += 1
                raise ValueError('No legal squad for this question (stored)')
            if stored is not None:
                plan = _safe_plan(stored, state, pool, gameweek, horizon, chip)
                if plan is not None:
                    self.cache.hits += 1
                    return plan
            self.cache.misses += 1

        started = time.perf_counter()
        try:
            selected, lower = self._solve_range_problem(
                state, pool, gameweek, max_transfers, chip, weeks, squad_chip,
                keep_cost, money, regular if squad_chip else None,
                specials if squad_chip else None,
            )
        except ValueError:
            # No legal squad with this much money means none with less either.
            if self.cache is not None:
                self.cache.put_range(
                    key, _NO_LOWER_BOUND, money, {'infeasible': True},
                    time.perf_counter() - started,
                )
            raise
        elapsed = time.perf_counter() - started
        plan = _plan_from_selected(state, pool, gameweek, horizon, selected, chip)
        if self.cache is not None:
            self.cache.solve_seconds += elapsed
            self.cache.put_range(key, lower, money, _serialise(pool, plan), elapsed)
        return plan

    def _solve_range_problem(
        self, state, pool, gameweek, max_transfers, chip, weeks, squad_chip, keep_cost,
        money, regular, specials,
    ):
        """Solve one ranged question. Returns the 15 and the least money it needs."""
        current = list(state.player_ids)
        scoring = _calendar(chip, gameweek)
        if squad_chip:
            candidates = sorted(regular | set(specials))
            costs = {player_id: int(pool[player_id].price) for player_id in candidates}
            for player_id in specials:
                costs[player_id] = int(keep_cost[player_id])
            fallback_state = SquadState(
                owned={player_id: state.owned[player_id] for player_id in specials},
                bank=money - sum(int(keep_cost[player_id]) for player_id in specials),
                free_transfers=0,
            )
            selected = self._run_solver(
                pool, weeks, candidates, costs, money, (), 15, scoring, fallback_state,
                gameweek, chip,
            )
            lower = sum(int(costs[player_id]) for player_id in selected)
        else:
            candidates = optimiser_candidate_ids(pool, set(current), weeks)
            costs = {player_id: int(pool[player_id].price) for player_id in candidates}
            costs.update(keep_cost)
            budget = int(state.bank) + sum(keep_cost.values())
            selected = self._run_solver(
                pool, weeks, candidates, costs, budget, current, int(max_transfers), scoring,
                state, gameweek, chip,
            )
            lower = sum(int(costs[player_id]) for player_id in selected) - sum(keep_cost.values())
        return selected, lower

    def _run_solver(
        self, pool, weeks, candidates, costs, budget, keep_ids, max_transfers, scoring,
        fallback_state, gameweek, chip,
    ):
        """HiGHS first when chosen and installed; CBC for anything it does not prove in time."""
        if self.solver == 'highs' and highs_solver.available():
            try:
                picked = highs_solver.solve_squad(
                    pool, weeks, candidates, costs, budget,
                    keep_ids=keep_ids,
                    max_transfers=max_transfers,
                    chips=scoring,
                    bench_weight=self.bench_weight,
                    time_limit=HIGHS_FIRST_SECONDS,
                )
                self.highs_solves += 1
                return picked
            except highs_solver.NotProven:
                pass
        self.cbc_solves += 1
        return _solve_squad(
            fallback_state, pool, gameweek, len(weeks), max_transfers, False, chip,
            self.season_start_year, gameweeks=weeks, bench_weight=self.bench_weight,
        )


def _scoring_name(chip) -> str:
    """Bench Boost and Triple Captain change the objective; the others do not."""
    if chip in (CHIP_BENCH_BOOST, CHIP_TRIPLE_CAPTAIN):
        return str(chip)
    return 'normal'


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


def _solve_squad(
    state, pool, gameweek, horizon, max_transfers, charge_hits, chip, season_start_year,
    gameweeks=None, bench_weight=0.0,
):
    weeks = _weeks(gameweek, horizon) if gameweeks is None else tuple(int(week) for week in gameweeks)
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
        bench_fill_weight=float(bench_weight),
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
