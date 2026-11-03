"""Inputs for the v3 manager. They are cheaper than v2's and compare now with later.

The action the policy plays is still solved exactly. These inputs answer
simpler questions, so most of them are stored squads looked up by budget:

`spend_now_gain_5` spends every free transfer this week.
`transfer_1_gain_5` makes one transfer this week.
`roll_gain_5` is the best of holding, spending now, and holding this week
then spending next week.
`wildcard_now_gain_5` wildcards this week. `wildcard_gain_5` is the best
week in the five to wildcard, keeping today's squad until then. No transfers
are planned around it and no free hit is paired with it.
`free_hit_now_gain` and `free_hit_gain_5` are the same for a free hit: that
week's free-hit squad against today's squad in that week.
`wildcard_in_5_gain_10` keeps today's squad for five weeks and wildcards on
the sixth, over weeks six to ten.

Every gain is the plan's predicted points minus keeping today's squad over
the same weeks. A chip that cannot be played in a week is skipped, and a gain
with nothing to compare is 0.

For these questions only, the bank is rounded down to the nearest £0.5m and
owned players are valued at today's price. That lets many squads share one
stored answer. A plan built that way is still affordable, because the real
bank is at least as big and keeping a player never costs more than today's
price.
"""
from __future__ import annotations

from squadtools.simulation_strategy import (
    CHIP_FREE_HIT,
    CHIP_WILDCARD,
    ChipCalendar,
    FreeTransferRules,
    OwnedPlayer,
    SquadState,
)

from fpl_manager.actions import MAX_TRANSFER_ACTIONS
from fpl_manager.cache import RULES_VERSION_V3
from fpl_manager.mip import _xi
from fpl_manager.observation import LAST_GAMEWEEK
from fpl_manager.plan_v2 import _chip_legal

PLAN_WEEKS = 5
LONG_WEEKS = 10
# Tenths of a million: £0.5m.
INPUT_BANK_STEP = 5

GAIN_NAMES_V3 = (
    'roll_gain_5',
    'spend_now_gain_5',
    'transfer_1_gain_5',
    'wildcard_gain_5',
    'wildcard_now_gain_5',
    'wildcard_in_5_gain_10',
    'free_hit_gain_5',
    'free_hit_now_gain',
)


def v3_transfer_gains(optimiser, state, pool, gameweek, played=None) -> dict:
    played = played or ChipCalendar()
    gameweek = int(gameweek)
    rules = FreeTransferRules(optimiser.season_start_year)
    weeks = _weeks(gameweek, PLAN_WEEKS)
    gains = {name: 0.0 for name in GAIN_NAMES_V3}
    if not weeks:
        return gains
    held = {week: _xi(state.player_ids, pool, week, False, False) for week in weeks}
    baseline = sum(held.values())
    free_transfers = rules.at_start_of_gameweek(gameweek, state.free_transfers)
    trial = input_state(state, pool, free_transfers)

    spend_now = None
    if free_transfers >= 1:
        plan = _transfers(optimiser, trial, pool, gameweek, min(free_transfers, MAX_TRANSFER_ACTIONS), weeks)
        if plan is not None:
            spend_now = _points(plan.selected_player_ids, pool, weeks) - baseline
    one_now = spend_now if free_transfers == 1 else None
    if free_transfers >= 2:
        plan = _transfers(optimiser, trial, pool, gameweek, 1, weeks)
        if plan is not None:
            one_now = _points(plan.selected_player_ids, pool, weeks) - baseline
    spend_next = None
    later = weeks[1:]
    if later:
        next_transfers = rules.after_gameweek(gameweek, free_transfers, 0, None)
        count = min(int(next_transfers), MAX_TRANSFER_ACTIONS)
        if count >= 1:
            waiting = input_state(state, pool, next_transfers)
            plan = _transfers(optimiser, waiting, pool, later[0], count, later)
            if plan is not None:
                spend_next = (
                    held[weeks[0]] + _points(plan.selected_player_ids, pool, later) - baseline
                )
    gains['spend_now_gain_5'] = spend_now or 0.0
    gains['transfer_1_gain_5'] = one_now or 0.0
    gains['roll_gain_5'] = max(0.0, spend_now or 0.0, spend_next or 0.0)

    season = optimiser.season_start_year
    best_wildcard = None
    for week in weeks:
        if not _chip_legal(season, played, CHIP_WILDCARD, week):
            continue
        objective = tuple(item for item in weeks if item >= week)
        plan = _chip(optimiser, trial, pool, week, CHIP_WILDCARD, objective)
        if plan is None:
            continue
        before = sum(held[item] for item in weeks if item < week)
        gain = before + _points(plan.selected_player_ids, pool, objective) - baseline
        if week == gameweek:
            gains['wildcard_now_gain_5'] = gain
        best_wildcard = gain if best_wildcard is None else max(best_wildcard, gain)
    gains['wildcard_gain_5'] = max(0.0, best_wildcard or 0.0)

    best_free_hit = None
    for week in weeks:
        if not _chip_legal(season, played, CHIP_FREE_HIT, week):
            continue
        plan = _chip(optimiser, trial, pool, week, CHIP_FREE_HIT, (week,))
        if plan is None:
            continue
        gain = _xi(plan.selected_player_ids, pool, week, False, False) - held[week]
        if week == gameweek:
            gains['free_hit_now_gain'] = gain
        best_free_hit = gain if best_free_hit is None else max(best_free_hit, gain)
    gains['free_hit_gain_5'] = max(0.0, best_free_hit or 0.0)

    long_weeks = _weeks(gameweek, LONG_WEEKS)
    back = long_weeks[PLAN_WEEKS:]
    if back and _chip_legal(season, played, CHIP_WILDCARD, back[0]):
        plan = _chip(optimiser, trial, pool, back[0], CHIP_WILDCARD, back)
        if plan is not None:
            gains['wildcard_in_5_gain_10'] = (
                _points(plan.selected_player_ids, pool, back)
                - _points(state.player_ids, pool, back)
            )
    return gains


def input_state(state, pool, free_transfers) -> SquadState:
    """Bank rounded down to £0.5m; owned players valued at today's price."""
    bank = (int(state.bank) // INPUT_BANK_STEP) * INPUT_BANK_STEP
    return SquadState(
        owned={
            int(player_id): OwnedPlayer(int(player_id), int(pool[player_id].price))
            for player_id in state.player_ids
        },
        bank=max(0, bank),
        free_transfers=int(free_transfers),
    )


def _transfers(optimiser, state, pool, week, count, objective):
    try:
        return optimiser._solve(
            'TRANSFER_%s' % int(count), state, pool, int(week), int(count), len(objective),
            False, None, gameweeks=objective, rules_version=RULES_VERSION_V3,
            input_question=True,
        )
    except ValueError:
        return None


def _chip(optimiser, state, pool, week, chip, objective):
    name = 'WILDCARD' if chip == CHIP_WILDCARD else 'FREE_HIT'
    try:
        return optimiser._solve(
            name, state, pool, int(week), 15, len(objective), False, chip,
            gameweeks=objective, rules_version=RULES_VERSION_V3, input_question=True,
        )
    except ValueError:
        return None


def _points(player_ids, pool, weeks) -> float:
    return sum(_xi(player_ids, pool, week, False, False) for week in weeks)


def _weeks(gameweek, length):
    return tuple(
        int(gameweek) + offset
        for offset in range(int(length))
        if int(gameweek) + offset <= LAST_GAMEWEEK
    )
