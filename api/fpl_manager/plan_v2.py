"""Squad plans that fill the v2 observation.

The 5-week window is the plan. No path pays a hit, and nothing is solved past
gameweek 38. Free transfers are banked at most four ahead, and never above the
season's real cap (2 in 2023/24, 5 after that).

`roll_gain_5` spends or rolls those free transfers and uses no chip.
`free_hit_gain_5` does that with one free-hit week inside the five. The
free-hit week is left out of the transfer plan.
`wildcard_gain_5` plays a wildcard on one of the five weeks, with transfers
rolled or spent before it and after it, and a free hit when one is still
available.
`wildcard_in_5_gain_10` holds today's squad for five weeks, wildcards on the
next one, and holds that 15 for the rest of a 10-week view. The first five
weeks are the same squad on both sides, so the value is only the last five.
"""
from __future__ import annotations

from squadtools.chip_strategy import chip_window, inventory_for
from squadtools.simulation_strategy import (
    CHIP_FREE_HIT,
    CHIP_WILDCARD,
    ChipCalendar,
    FreeTransferRules,
    apply_transfer_plan,
    copy_squad_state,
)

from fpl_manager.actions import FREE_HIT, WILDCARD
from fpl_manager.cache import RULES_VERSION_V2
from fpl_manager.mip import _xi
from fpl_manager.observation import LAST_GAMEWEEK

PLAN_WEEKS = 5
LONG_WEEKS = 10
ROLL_CAP = 4
RULES_V2 = RULES_VERSION_V2

_HELD = {
    CHIP_WILDCARD: 'wildcard',
    CHIP_FREE_HIT: 'free_hit',
}


def v2_transfer_gains(optimiser, state, pool, gameweek, played=None) -> dict:
    played = played or ChipCalendar()
    rules = FreeTransferRules(optimiser.season_start_year)
    weeks = _weeks(gameweek, PLAN_WEEKS)
    baseline = _points(state.player_ids, pool, weeks)
    roll = _best_roll(optimiser, state, pool, rules, gameweek, weeks)
    free_hit = _best_free_hit(
        optimiser, state, pool, rules, gameweek, weeks, played, baseline,
    )
    wildcard = _best_wildcard(
        optimiser, state, pool, rules, gameweek, weeks, played, baseline,
    )
    return {
        'roll_gain_5': roll - baseline,
        'free_hit_gain_5': free_hit - baseline,
        'wildcard_gain_5': wildcard - baseline,
        'wildcard_in_5_gain_10': _wildcard_in_five(
            optimiser, state, pool, rules, gameweek, played,
        ),
    }


def roll_steps(rules, start_week, free_transfers, weeks, skip=()):
    """One spend inside `weeks`: (week, free transfers that week, objective weeks)."""
    weeks = tuple(int(week) for week in weeks)
    if not weeks:
        return []
    skip = {int(week) for week in skip}
    banked = rules.at_start_of_gameweek(start_week, free_transfers)
    limit = min(ROLL_CAP, int(rules.max_bank))
    if banked <= 0 or limit <= 0:
        return []
    steps = []
    cursor = int(start_week)
    for week in weeks:
        while cursor < week:
            banked = rules.after_gameweek(cursor, banked, 0, None)
            cursor += 1
        if week in skip or banked <= 0:
            continue
        objective = tuple(item for item in weeks if item >= week and item not in skip)
        if not objective:
            continue
        steps.append((week, int(banked), objective))
        if banked >= limit:
            break
    return steps


def _best_roll(optimiser, state, pool, rules, gameweek, weeks) -> float:
    best = _points(state.player_ids, pool, weeks)
    for spend_week, count, objective in roll_steps(
        rules, gameweek, state.free_transfers, weeks,
    ):
        plan = _solve_transfers(
            optimiser, state, pool, spend_week, count, objective,
        )
        if plan is None:
            continue
        best = max(best, _rolled_points(
            state.player_ids, plan.selected_player_ids, pool, weeks, spend_week,
        ))
    return best


def _best_free_hit(optimiser, state, pool, rules, gameweek, weeks, played, baseline) -> float:
    best = None
    for fh_week in _chip_weeks(optimiser, played, CHIP_FREE_HIT, weeks):
        for spend_week, count, objective, plan in _roll_options(
            optimiser, state, pool, rules, gameweek, weeks, skip=(fh_week,),
        ):
            fh_state = _state_on_roll(
                state, plan, rules, gameweek, spend_week, fh_week, count,
            )
            if fh_state is None:
                continue
            fh_plan = _solve_chip(
                optimiser, fh_state, pool, fh_week, CHIP_FREE_HIT, (fh_week,),
            )
            if fh_plan is None:
                continue
            total = _score_assigned(pool, weeks, _ids_with_chip(
                weeks,
                state.player_ids,
                None,
                None,
                spend_week,
                None if plan is None else plan.selected_player_ids,
                fh_week,
                fh_plan.selected_player_ids,
            ))
            best = total if best is None else max(best, total)
    return baseline if best is None else best


def _best_wildcard(optimiser, state, pool, rules, gameweek, weeks, played, baseline) -> float:
    best = None
    wc_weeks = _chip_weeks(optimiser, played, CHIP_WILDCARD, weeks)
    for wc_week in wc_weeks:
        fh_choices = [None]
        fh_choices.extend(
            week for week in _chip_weeks(optimiser, played, CHIP_FREE_HIT, weeks)
            if week != wc_week
        )
        for fh_week in fh_choices:
            skip = () if fh_week is None else (fh_week,)
            before = tuple(week for week in weeks if week < wc_week)
            for spend_week, count, _objective, plan in _roll_options(
                optimiser, state, pool, rules, gameweek, before, skip=skip,
            ):
                arrived = _state_on_roll(
                    state, plan, rules, gameweek, spend_week, wc_week, count,
                )
                if arrived is None:
                    continue
                wc_objective = tuple(
                    week for week in weeks
                    if week >= wc_week and week != fh_week
                )
                wc_plan = _solve_chip(
                    optimiser, arrived, pool, wc_week, CHIP_WILDCARD, wc_objective,
                )
                if wc_plan is None:
                    continue
                wc_state = _apply_chip_squad(arrived, wc_plan, rules, wc_week)
                if wc_state is None:
                    continue
                after = tuple(week for week in weeks if week > wc_week)
                after_options = _roll_options(
                    optimiser,
                    wc_state,
                    pool,
                    rules,
                    wc_week + 1,
                    after,
                    skip=skip,
                )
                for after_week, after_count, _after_objective, after_plan in after_options:
                    fh_ids = None
                    if fh_week is not None:
                        fh_base = arrived if fh_week < wc_week else wc_state
                        fh_origin = gameweek if fh_week < wc_week else wc_week + 1
                        fh_plan_used = plan if fh_week < wc_week else after_plan
                        fh_spend = spend_week if fh_week < wc_week else after_week
                        fh_allowance = count if fh_week < wc_week else after_count
                        if fh_week > wc_week and (
                            after_week is None or fh_week < after_week
                        ):
                            fh_state = _held_state(wc_state, rules, wc_week + 1, fh_week)
                        elif fh_week < wc_week:
                            fh_state = _state_on_roll(
                                state, plan, rules, gameweek, spend_week, fh_week, count,
                            )
                        else:
                            fh_state = _state_on_roll(
                                fh_base, fh_plan_used, rules, fh_origin,
                                fh_spend, fh_week, fh_allowance,
                            )
                        if fh_state is None:
                            continue
                        fh_solved = _solve_chip(
                            optimiser, fh_state, pool, fh_week, CHIP_FREE_HIT, (fh_week,),
                        )
                        if fh_solved is None:
                            continue
                        fh_ids = fh_solved.selected_player_ids
                    ids = _ids_with_chip(
                        weeks,
                        state.player_ids,
                        wc_week,
                        wc_plan.selected_player_ids,
                        spend_week,
                        None if plan is None else plan.selected_player_ids,
                        fh_week,
                        fh_ids,
                        after_week,
                        None if after_plan is None else after_plan.selected_player_ids,
                    )
                    total = _score_assigned(pool, weeks, ids)
                    best = total if best is None else max(best, total)
    return baseline if best is None else best


def _wildcard_in_five(optimiser, state, pool, rules, gameweek, played) -> float:
    weeks = _weeks(gameweek, LONG_WEEKS)
    if len(weeks) <= PLAN_WEEKS:
        return 0.0
    back = weeks[PLAN_WEEKS:]
    wc_week = back[0]
    if not _chip_legal(optimiser.season_start_year, played, CHIP_WILDCARD, wc_week):
        return 0.0
    arrived = _held_state(state, rules, gameweek, wc_week)
    plan = _solve_chip(
        optimiser, arrived, pool, wc_week, CHIP_WILDCARD, back,
    )
    if plan is None:
        return 0.0
    return (
        _points(plan.selected_player_ids, pool, back)
        - _points(state.player_ids, pool, back)
    )


def _roll_options(optimiser, state, pool, rules, origin, weeks, skip=()):
    """The hold, then each ladder spend. A spend that fails to solve is skipped."""
    options = [(None, 0, (), None)]
    if not weeks:
        return options
    for spend_week, count, objective in roll_steps(
        rules, origin, state.free_transfers, weeks, skip=skip,
    ):
        plan = _solve_transfers(
            optimiser, state, pool, spend_week, count, objective,
        )
        if plan is None:
            continue
        options.append((spend_week, count, objective, plan))
    return options


def _solve_transfers(
    optimiser, state, pool, week, count, objective, rules_version=None, value_band=False,
):
    trial = copy_squad_state(state)
    trial.free_transfers = int(count)
    action = _action_name('TRANSFER_%s' % int(count), objective)
    return _ask(
        optimiser, trial, pool, week, int(count), action, objective, None,
        rules_version=rules_version, value_band=value_band,
    )


def _solve_chip(
    optimiser, state, pool, week, chip, objective, rules_version=None, value_band=False,
):
    if not objective:
        return None
    name = WILDCARD if chip == CHIP_WILDCARD else FREE_HIT
    if chip == CHIP_FREE_HIT:
        name = FREE_HIT
    action = _action_name(name, objective)
    return _ask(
        optimiser, state, pool, week, 15, action, objective, chip,
        rules_version=rules_version, value_band=value_band,
    )


def _ask(
    optimiser, state, pool, week, transfers, action, weeks, chip,
    rules_version=None, value_band=False,
):
    weeks = tuple(int(item) for item in weeks)
    if not weeks or int(transfers) <= 0:
        return None
    try:
        return optimiser._solve(
            action,
            state,
            pool,
            int(week),
            int(transfers),
            len(weeks),
            False,
            chip,
            gameweeks=weeks,
            rules_version=rules_version or RULES_V2,
            value_band=value_band,
        )
    except ValueError:
        return None


def _action_name(base, weeks) -> str:
    weeks = tuple(int(week) for week in weeks)
    if weeks == tuple(range(weeks[0], weeks[0] + len(weeks))):
        return base
    return base + 'w' + '-'.join(str(week) for week in weeks)


def _weeks(gameweek, length):
    weeks = []
    for offset in range(int(length)):
        week = int(gameweek) + offset
        if week > LAST_GAMEWEEK:
            break
        weeks.append(week)
    return tuple(weeks)


def _chip_weeks(optimiser, played, chip, weeks):
    return tuple(
        week for week in weeks
        if _chip_legal(optimiser.season_start_year, played, chip, week)
    )


def _chip_legal(season_start_year, played, chip, week) -> bool:
    inventory = inventory_for(season_start_year, week, played)
    held = getattr(inventory, _HELD[chip])
    if not held:
        return False
    start, end = chip_window(season_start_year, chip, week)
    return start <= int(week) <= end


def _held_state(state, rules, origin, target):
    arrived = copy_squad_state(state)
    if int(target) <= int(origin):
        arrived.free_transfers = rules.at_start_of_gameweek(target, state.free_transfers)
        return arrived
    arrived.free_transfers = _ft_after_holds(
        rules, origin, state.free_transfers, int(target) - int(origin),
    )
    return arrived


def _ft_after_holds(rules, gameweek, free_transfers, holds) -> int:
    available = rules.at_start_of_gameweek(gameweek, free_transfers)
    week = int(gameweek)
    for _step in range(int(holds)):
        available = rules.after_gameweek(week, available, 0, None)
        week += 1
    return int(available)


def _state_on_roll(state, plan, rules, origin, spend_week, target, allowance):
    if spend_week is None or plan is None or int(target) < int(spend_week):
        return _held_state(state, rules, origin, target)
    arrived = _state_after_spend(state, plan, rules, spend_week, target, allowance)
    return arrived


def _state_after_spend(state, plan, rules, spend_week, target, allowance):
    try:
        arrived = apply_transfer_plan(state, plan) if plan.transfers else copy_squad_state(state)
    except ValueError:
        return None
    if int(target) <= int(spend_week):
        arrived.free_transfers = int(allowance)
        return arrived
    available = rules.after_gameweek(
        spend_week, int(allowance), len(plan.transfers), None,
    )
    week = int(spend_week) + 1
    while week < int(target):
        available = rules.after_gameweek(week, available, 0, None)
        week += 1
    arrived.free_transfers = int(available)
    return arrived


def _apply_chip_squad(state, plan, rules, chip_week):
    try:
        arrived = apply_transfer_plan(state, plan) if plan.transfers else copy_squad_state(state)
    except ValueError:
        return None
    arrived.free_transfers = rules.after_gameweek(
        chip_week, state.free_transfers, 0, CHIP_WILDCARD,
    )
    return arrived


def _ids_with_chip(
    weeks,
    base_ids,
    chip_week,
    chip_ids,
    spend_week,
    spend_ids,
    fh_week,
    fh_ids,
    after_week=None,
    after_ids=None,
):
    chosen = {}
    for week in weeks:
        ids = base_ids
        if spend_week is not None and spend_ids is not None and week >= spend_week:
            if chip_week is None or week < chip_week:
                ids = spend_ids
        if chip_week is not None and chip_ids is not None and week >= chip_week:
            ids = chip_ids
        if after_week is not None and after_ids is not None and week >= after_week:
            ids = after_ids
        if fh_week is not None and week == fh_week and fh_ids is not None:
            ids = fh_ids
        chosen[week] = ids
    return chosen


def _rolled_points(base_ids, new_ids, pool, weeks, spend_week) -> float:
    total = 0.0
    for week in weeks:
        ids = new_ids if week >= spend_week else base_ids
        total += _xi(ids, pool, week, False, False)
    return total


def _score_assigned(pool, weeks, chosen) -> float:
    return sum(_xi(chosen[week], pool, week, False, False) for week in weeks)


def _points(player_ids, pool, weeks) -> float:
    return sum(_xi(player_ids, pool, week, False, False) for week in weeks)
