"""XI, captain and realised points without calling the mixed-integer solver.

Autosubs are not applied. A starter who scores zero stays in the side. Bench
points count only when Bench Boost is played.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

from squadtools.simulation_strategy import (
    LINEUP_POSITION_MAX,
    LINEUP_POSITION_MIN,
    SQUAD_POSITION_COUNTS,
)

HIT_COST = 4


def greedy_lineup(player_ids: Sequence[int], points_of: Callable, position_of: Callable):
    """Highest-scoring legal XI. Captain and vice are the top two starters."""
    ids = [int(player_id) for player_id in player_ids]
    if len(ids) != sum(SQUAD_POSITION_COUNTS.values()):
        raise ValueError('a lineup needs 15 players')
    by_position = {position: [] for position in SQUAD_POSITION_COUNTS}
    for player_id in ids:
        by_position[int(position_of(player_id))].append(player_id)
    for position, group in by_position.items():
        group.sort(key=lambda player_id: (float(points_of(player_id)), -player_id), reverse=True)
        if len(group) < SQUAD_POSITION_COUNTS[position]:
            raise ValueError('squad is short at position %s' % position)

    starters = []
    for position, minimum in LINEUP_POSITION_MIN.items():
        starters.extend(by_position[position][:minimum])
        del by_position[position][:minimum]
    rest = []
    for position, group in by_position.items():
        already = sum(1 for player_id in starters if int(position_of(player_id)) == position)
        room = LINEUP_POSITION_MAX[position] - already
        rest.extend(group[:room])
    rest.sort(key=lambda player_id: (float(points_of(player_id)), -player_id), reverse=True)
    starters.extend(rest[: 11 - len(starters)])
    if len(starters) != 11:
        raise ValueError('could not fill a legal XI')
    starter_set = set(starters)
    bench = [player_id for player_id in ids if player_id not in starter_set]
    outfield = sorted(
        (player_id for player_id in bench if int(position_of(player_id)) != 1),
        key=lambda player_id: (float(points_of(player_id)), -player_id),
        reverse=True,
    )
    keepers = [player_id for player_id in bench if int(position_of(player_id)) == 1]
    ranked = sorted(starters, key=lambda player_id: (float(points_of(player_id)), -player_id), reverse=True)
    return tuple(starters), tuple(outfield + keepers), ranked[0], ranked[1]


def score_lineup(points_of: Callable, starters, bench, captain_id: int, bench_boost: bool, triple_captain: bool) -> float:
    """XI plus the captain once. Bench Boost adds the bench. Triple Captain adds the captain again."""
    total = sum(float(points_of(player_id)) for player_id in starters)
    total += float(points_of(captain_id))
    if triple_captain:
        total += float(points_of(captain_id))
    if bench_boost:
        total += sum(float(points_of(player_id)) for player_id in bench)
    return float(total)


def hit_penalty(transfers_made: int, free_transfers: int, chip_ignores_hits: bool) -> int:
    if chip_ignores_hits:
        return 0
    extra = max(0, int(transfers_made) - int(free_transfers))
    return extra * HIT_COST
