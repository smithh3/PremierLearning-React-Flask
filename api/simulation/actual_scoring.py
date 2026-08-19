"""Apply FPL lineup, captain and automatic-substitution rules to actuals."""

import itertools
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from squadtools.simulation_strategy import (
    LINEUP_POSITION_MAX,
    LINEUP_POSITION_MIN,
    Lineup,
    PredictionPool,
)


@dataclass
class ActualScore:
    points: float
    scoring_player_ids: Tuple[int, ...]
    autosubs: Tuple[dict, ...]
    captain_scored_id: Optional[int]
    unexpected_zero_minute_starters: int


def _minutes(actuals, player_id):
    actual = actuals.get(int(player_id))
    return float(actual.minutes if actual is not None else 0.0)


def _points(actuals, player_id):
    actual = actuals.get(int(player_id))
    return float(actual.points if actual is not None else 0.0)


def _position_counts(player_ids, pool):
    counts = {position: 0 for position in LINEUP_POSITION_MIN}
    for player_id in player_ids:
        position = pool[player_id].position
        counts[position] = counts.get(position, 0) + 1
    return counts


def _full_lineup_is_legal(player_ids, pool):
    if len(player_ids) != 11:
        return False
    counts = _position_counts(player_ids, pool)
    return all(
        LINEUP_POSITION_MIN[position]
        <= counts.get(position, 0)
        <= LINEUP_POSITION_MAX[position]
        for position in LINEUP_POSITION_MIN
    )


def _best_outfield_subs(base_ids, candidates, missing_slots, pool):
    """Choose the highest-priority legal bench subset (three players at most)."""
    candidates = tuple(candidates)
    best = ()
    best_key = None
    for count in range(0, min(int(missing_slots), len(candidates)) + 1):
        for indexes in itertools.combinations(range(len(candidates)), count):
            chosen = tuple(candidates[index] for index in indexes)
            final_ids = tuple(base_ids) + chosen
            if len(final_ids) == 11:
                legal = _full_lineup_is_legal(final_ids, pool)
            else:
                counts = _position_counts(final_ids, pool)
                legal = all(
                    counts.get(position, 0) <= LINEUP_POSITION_MAX[position]
                    for position in LINEUP_POSITION_MAX
                )
            if not legal:
                continue
            # More substitutions first, then earlier bench places.
            priority = tuple(-index for index in indexes)
            key = (len(chosen), priority)
            if best_key is None or key > best_key:
                best_key = key
                best = chosen
    return best


def score_lineup(
    lineup: Lineup,
    pool: PredictionPool,
    actuals: Dict[int, object],
) -> ActualScore:
    starter_ids = tuple(lineup.starters)
    played_starters = [
        player_id for player_id in starter_ids if _minutes(actuals, player_id) > 0
    ]
    missing_starters = [
        player_id for player_id in starter_ids if _minutes(actuals, player_id) <= 0
    ]
    scoring_ids = list(played_starters)
    autosubs = []

    missing_keeper = next(
        (
            player_id
            for player_id in missing_starters
            if pool[player_id].position == 1
        ),
        None,
    )
    bench_keeper = next(
        (
            player_id
            for player_id in lineup.bench
            if pool[player_id].position == 1
        ),
        None,
    )
    if (
        missing_keeper is not None
        and bench_keeper is not None
        and _minutes(actuals, bench_keeper) > 0
    ):
        scoring_ids.append(bench_keeper)
        autosubs.append({'out': missing_keeper, 'in': bench_keeper})

    missing_outfield = [
        player_id
        for player_id in missing_starters
        if pool[player_id].position != 1
    ]
    played_outfield_bench = [
        player_id
        for player_id in lineup.bench
        if pool[player_id].position != 1 and _minutes(actuals, player_id) > 0
    ]
    chosen_subs = _best_outfield_subs(
        scoring_ids,
        played_outfield_bench,
        len(missing_outfield),
        pool,
    )
    remaining_missing = list(missing_outfield)
    for player_in in chosen_subs:
        # Pair logs position-first when possible; scoring only depends on the set.
        same_position = next(
            (
                player_out
                for player_out in remaining_missing
                if pool[player_out].position == pool[player_in].position
            ),
            None,
        )
        player_out = same_position if same_position is not None else remaining_missing[0]
        remaining_missing.remove(player_out)
        scoring_ids.append(player_in)
        autosubs.append({'out': player_out, 'in': player_in})

    points = sum(_points(actuals, player_id) for player_id in scoring_ids)
    captain_scored_id = None
    if _minutes(actuals, lineup.captain_id) > 0:
        captain_scored_id = lineup.captain_id
    elif _minutes(actuals, lineup.vice_captain_id) > 0:
        captain_scored_id = lineup.vice_captain_id
    if captain_scored_id is not None:
        points += _points(actuals, captain_scored_id)

    return ActualScore(
        points=float(points),
        scoring_player_ids=tuple(sorted(scoring_ids)),
        autosubs=tuple(autosubs),
        captain_scored_id=captain_scored_id,
        unexpected_zero_minute_starters=len(missing_starters),
    )
