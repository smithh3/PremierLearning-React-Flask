"""The short 2025/26 plans.

One roll: hold this week, then spend every free transfer available next week.
One wildcard: play it this week and score that squad over the next five.
Both numbers are the difference against holding the current squad for those five.
"""
from __future__ import annotations

from squadtools.simulation_strategy import CHIP_WILDCARD, ChipCalendar, FreeTransferRules

from fpl_manager.cache import RULES_VERSION_FAST
from fpl_manager.plan_v2 import (
    PLAN_WEEKS,
    _chip_legal,
    _ft_after_holds,
    _points,
    _solve_chip,
    _solve_transfers,
    _weeks,
)


def fast_transfer_gains(optimiser, state, pool, gameweek, played=None) -> dict:
    played = played or ChipCalendar()
    rules = FreeTransferRules(optimiser.season_start_year)
    weeks = _weeks(gameweek, PLAN_WEEKS)
    baseline = _points(state.player_ids, pool, weeks)
    return {
        'roll_1_gain_5': _roll_one(optimiser, state, pool, rules, weeks) - baseline,
        'wildcard_now_gain_5': _wildcard_now(
            optimiser, state, pool, played, weeks,
        ) - baseline,
    }


def _roll_one(optimiser, state, pool, rules, weeks) -> float:
    if len(weeks) < 2:
        return _points(state.player_ids, pool, weeks)
    allowance = _ft_after_holds(rules, weeks[0], state.free_transfers, 1)
    if allowance <= 0:
        return _points(state.player_ids, pool, weeks)
    plan = _solve_transfers(
        optimiser,
        state,
        pool,
        weeks[1],
        allowance,
        weeks[1:],
        rules_version=RULES_VERSION_FAST,
        value_band=True,
    )
    if plan is None:
        return _points(state.player_ids, pool, weeks)
    return (
        _points(state.player_ids, pool, weeks[:1])
        + _points(plan.selected_player_ids, pool, weeks[1:])
    )


def _wildcard_now(optimiser, state, pool, played, weeks) -> float:
    if not weeks or not _chip_legal(
        optimiser.season_start_year, played, CHIP_WILDCARD, weeks[0],
    ):
        return _points(state.player_ids, pool, weeks)
    plan = _solve_chip(
        optimiser,
        state,
        pool,
        weeks[0],
        CHIP_WILDCARD,
        weeks,
        rules_version=RULES_VERSION_FAST,
        value_band=True,
    )
    if plan is None:
        return _points(state.player_ids, pool, weeks)
    return _points(plan.selected_player_ids, pool, weeks)
