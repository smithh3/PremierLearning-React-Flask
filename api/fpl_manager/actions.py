"""One categorical action per gameweek. At most one chip can be chosen."""
from __future__ import annotations

from squadtools.chip_strategy import chip_window, inventory_for
from squadtools.simulation_strategy import (
    CHIP_BENCH_BOOST,
    CHIP_FREE_HIT,
    CHIP_TRIPLE_CAPTAIN,
    CHIP_WILDCARD,
    ChipCalendar,
    FreeTransferRules,
    SquadState,
)

HOLD = 'HOLD'
TRANSFER_1 = 'TRANSFER_1'
TRANSFER_2 = 'TRANSFER_2'
TRANSFER_3 = 'TRANSFER_3'
TRANSFER_4 = 'TRANSFER_4'
TRANSFER_5 = 'TRANSFER_5'
WILDCARD = 'WILDCARD'
FREE_HIT = 'FREE_HIT'
BENCH_BOOST = 'BENCH_BOOST'
TRIPLE_CAPTAIN = 'TRIPLE_CAPTAIN'

ACTIONS = (
    HOLD,
    TRANSFER_1,
    TRANSFER_2,
    TRANSFER_3,
    TRANSFER_4,
    TRANSFER_5,
    WILDCARD,
    FREE_HIT,
    BENCH_BOOST,
    TRIPLE_CAPTAIN,
)

_TRANSFER_COUNTS = {
    TRANSFER_1: 1,
    TRANSFER_2: 2,
    TRANSFER_3: 3,
    TRANSFER_4: 4,
    TRANSFER_5: 5,
}

_CHIP_BY_ACTION = {
    WILDCARD: CHIP_WILDCARD,
    FREE_HIT: CHIP_FREE_HIT,
    BENCH_BOOST: CHIP_BENCH_BOOST,
    TRIPLE_CAPTAIN: CHIP_TRIPLE_CAPTAIN,
}

# The existing transfer MIP refuses more than five moves in a normal week.
MAX_TRANSFER_ACTIONS = 5


def transfer_count(action: str) -> int:
    return int(_TRANSFER_COUNTS.get(action, 0))


def chip_for(action: str):
    return _CHIP_BY_ACTION.get(action)


def legal_mask(
    season_start_year: int,
    gameweek: int,
    played: ChipCalendar,
    squad: SquadState,
    allow_hits: bool = True,
) -> tuple:
    """HOLD is always legal. Chips follow the season's real windows, not the planner's GW5 habit.

    v1 leaves a transfer count available when it would cost a hit. v2 masks any
    count above the free transfers already in hand. A count is also masked when
    the squad has fewer players than that count, which cannot happen for a
    15-man squad and the five actions above.
    """
    inventory = inventory_for(season_start_year, gameweek, played)
    held = {
        CHIP_WILDCARD: inventory.wildcard,
        CHIP_FREE_HIT: inventory.free_hit,
        CHIP_BENCH_BOOST: inventory.bench_boost,
        CHIP_TRIPLE_CAPTAIN: inventory.triple_captain,
    }
    owned = len(squad.player_ids)
    free_transfers = FreeTransferRules(season_start_year).at_start_of_gameweek(
        gameweek, squad.free_transfers,
    )
    mask = []
    for action in ACTIONS:
        chip = chip_for(action)
        if chip is None:
            count = transfer_count(action)
            allowed = count <= owned
            if count and not allow_hits:
                allowed = allowed and count <= free_transfers
            mask.append(allowed)
            continue
        start, end = chip_window(season_start_year, chip, gameweek)
        mask.append(bool(held[chip]) and start <= int(gameweek) <= end)
    return tuple(mask)
