"""The squad MIP solved inside this process with HiGHS.

It is the same model as `SimulationStrategy._optimise_squad` with the bench
weights off: pick 15, start 11 in a legal formation every week, captain one
starter, at most three per club, inside the budget, and at most
`max_transfers` of the owned players leave. The objective is the starters'
predicted points plus the captain's extra copy, with the same tiny
player-id tie-break. `bench_weight` adds that share of the bench's points.

Running CBC meant starting a program, writing the model to a file and
waiting for one of a few machine-wide slots. That was most of each solve,
and the bundled CBC sometimes froze. HiGHS runs in this process.

Prices are what including a player costs: today's price, or the sale price
for an owned player. The budget is the bank plus what the owned players sell
for. That is the existing rule rearranged: buying the incoming players must
not cost more than the bank plus the sales.

highspy 1.14 crashes on import on this laptop because its C++ runtime is
older than the one that build needs. 1.7.2 works and is the pinned version.
"""
from __future__ import annotations

import numpy as np

from squadtools.simulation_strategy import (
    LINEUP_POSITION_MAX,
    LINEUP_POSITION_MIN,
    MAX_PLAYERS_PER_TEAM,
    SQUAD_POSITION_COUNTS,
    ChipCalendar,
)

TIME_LIMIT_SECONDS = 120.0
_TIE_BREAK = 1e-9
# Stop once the best squad is proven within this many points of the best
# possible. Proving the last 1e-9 tie-break took HiGHS seconds on squads
# with many equal bench fillers, for no difference in points.
MIP_ABS_GAP = 1e-4


class SolverUnavailable(RuntimeError):
    """highspy could not be imported."""


class NotProven(RuntimeError):
    """HiGHS ran out of time before proving its squad is the best."""


def available() -> bool:
    try:
        import highspy  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means use CBC
        return False
    return True


def solve_squad(
    pool,
    gameweeks,
    candidate_ids,
    costs,
    budget,
    keep_ids=(),
    max_transfers=15,
    chips=None,
    bench_weight=0.0,
    time_limit=TIME_LIMIT_SECONDS,
    options=None,
):
    """Best 15 of `candidate_ids`. Raises ValueError when there is no legal squad.

    `options` overrides HiGHS settings, for tuning.
    """
    try:
        import highspy
    except Exception as error:  # noqa: BLE001
        raise SolverUnavailable(str(error)) from error

    weeks = [int(week) for week in gameweeks]
    ids = prune_dominated(pool, candidate_ids, weeks, costs, keep_ids)
    if len(ids) < 15 or not weeks:
        raise ValueError('Not enough selectable players to build a squad')
    chips = chips or ChipCalendar()
    n = len(ids)
    w_count = len(weeks)
    positions = np.array([int(pool[player_id].position) for player_id in ids])
    teams = np.array([int(pool[player_id].team_id) for player_id in ids])
    price = np.array([int(costs[player_id]) for player_id in ids], dtype=float)
    points = np.array(
        [[float(pool[player_id].points(week)) for player_id in ids] for week in weeks]
    )
    # Columns: picked x (n), then starts s[w] (n each), then captains c[w] (n each).
    ncol = n * (1 + 2 * w_count)

    def s_col(week_index):
        return n + week_index * n

    def c_col(week_index):
        return n + w_count * n + week_index * n

    cost = np.zeros(ncol)
    cost[:n] = _TIE_BREAK * (10000 - np.array(ids, dtype=float))
    bench = float(bench_weight)
    for index, week in enumerate(weeks):
        week_points = points[index]
        extra = float(chips.captain_extra(week))
        cost[c_col(index):c_col(index) + n] = extra * week_points
        if chips.is_bench_boost(week):
            cost[:n] += week_points
            continue
        cost[s_col(index):s_col(index) + n] = (1.0 - bench) * week_points
        if bench > 0:
            cost[:n] += bench * week_points

    rows = _Rows()
    everyone = np.arange(n)
    rows.add(everyone, np.ones(n), 15, 15)
    for position, required in SQUAD_POSITION_COUNTS.items():
        members = np.flatnonzero(positions == position)
        rows.add(members, np.ones(len(members)), required, required)
    for team in sorted(set(teams.tolist())):
        members = np.flatnonzero(teams == team)
        rows.add(members, np.ones(len(members)), -np.inf, MAX_PLAYERS_PER_TEAM)
    for index in range(w_count):
        starts = s_col(index) + everyone
        captains = c_col(index) + everyone
        rows.add(starts, np.ones(n), 11, 11)
        rows.add(captains, np.ones(n), 1, 1)
        rows.add_pairs(starts, everyone, -np.inf, 0.0)  # start <= picked
        rows.add_pairs(captains, starts, -np.inf, 0.0)  # captain <= start
        for position in SQUAD_POSITION_COUNTS:
            members = s_col(index) + np.flatnonzero(positions == position)
            rows.add(
                members, np.ones(len(members)),
                LINEUP_POSITION_MIN[position], LINEUP_POSITION_MAX[position],
            )
    keep = [ids.index(int(player_id)) for player_id in keep_ids if int(player_id) in ids]
    if keep and int(max_transfers) < len(keep):
        rows.add(np.array(keep), np.ones(len(keep)), len(keep) - int(max_transfers), np.inf)
    rows.add(everyone, price, -np.inf, float(budget))

    inf = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_col_ = ncol
    lp.num_row_ = rows.count
    lp.col_cost_ = cost
    lp.col_lower_ = np.zeros(ncol)
    lp.col_upper_ = np.ones(ncol)
    lower, upper = rows.bounds()
    lp.row_lower_ = np.where(np.isinf(lower), -inf, lower)
    lp.row_upper_ = np.where(np.isinf(upper), inf, upper)
    starts, index, value = rows.matrix()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.num_col_ = ncol
    lp.a_matrix_.num_row_ = rows.count
    lp.a_matrix_.start_ = starts
    lp.a_matrix_.index_ = index
    lp.a_matrix_.value_ = value
    lp.integrality_ = [highspy.HighsVarType.kInteger] * ncol
    lp.sense_ = highspy.ObjSense.kMaximize

    settings = {
        'output_flag': False,
        'threads': 1,
        'mip_rel_gap': 0.0,
        'mip_abs_gap': MIP_ABS_GAP,
        'time_limit': float(time_limit),
    }
    settings.update(options or {})
    solver = highspy.Highs()
    for key, value in settings.items():
        solver.setOptionValue(key, value)
    solver.passModel(lp)
    start = _starting_answer(pool, ids, weeks, keep_ids, price, budget, ncol, n, s_col, c_col)
    if start is not None:
        solution = highspy.HighsSolution()
        solution.col_value = start
        solution.value_valid = True
        solver.setSolution(solution)
    solver.run()
    status = solver.getModelStatus()
    if status == highspy.HighsModelStatus.kTimeLimit:
        raise NotProven('HiGHS did not prove a best squad within %ss' % time_limit)
    if status != highspy.HighsModelStatus.kOptimal:
        raise ValueError('No feasible squad: %s' % solver.modelStatusToString(status))
    values = np.asarray(solver.getSolution().col_value[:n])
    picked = tuple(sorted(ids[i] for i in np.flatnonzero(values > 0.5)))
    if len(picked) != 15:
        raise ValueError('HiGHS returned %s players' % len(picked))
    return picked


def _starting_answer(pool, ids, weeks, keep_ids, price, budget, ncol, n, s_col, c_col):
    """Keeping the owned squad, with its best XI and captain each week.

    Handing HiGHS a legal answer up front lets it rule out weaker branches
    from the start. None when the owned squad is not all among the candidates
    or does not fit the budget.
    """
    from fpl_manager.lineup import greedy_lineup

    position = {player_id: index for index, player_id in enumerate(ids)}
    owned = [int(player_id) for player_id in keep_ids]
    if len(owned) != 15 or any(player_id not in position for player_id in owned):
        return None
    if sum(price[position[player_id]] for player_id in owned) > float(budget) + 1e-9:
        return None
    values = np.zeros(ncol)
    for player_id in owned:
        values[position[player_id]] = 1.0
    try:
        for index, week in enumerate(weeks):
            starters, _bench, captain, _vice = greedy_lineup(
                owned,
                lambda player_id, week=week: pool[player_id].points(week),
                lambda player_id: pool[player_id].position,
            )
            for player_id in starters:
                values[s_col(index) + position[player_id]] = 1.0
            values[c_col(index) + position[captain]] = 1.0
    except ValueError:
        return None
    return values


def prune_dominated(pool, candidate_ids, gameweeks, costs, keep_ids=()):
    """Drop players who can never be needed. The best squad is unchanged.

    A player can go when he is not owned and enough others in his position
    cost no more and score at least as much in every week. "Enough" means
    they come from at least (squad places in that position + 4) clubs. If he
    were in a best squad, at most places-1 of those others could be in it
    too, and the other 14 players can fill at most four clubs to the limit
    of three. So at least one of the others could replace him: same
    position, no dearer, no fewer points any week, and his club has room.
    Swapping him in keeps the squad legal and never scores less.

    Exact ties go to the lower player id, so of two identical players one
    always stays. Beating is then a strict order: repeating the swap only
    moves up it, so it ends at a best squad made of kept players.
    """
    ids = sorted(int(player_id) for player_id in candidate_ids)
    owned = {int(player_id) for player_id in keep_ids}
    weeks = [int(week) for week in gameweeks]
    by_position = {}
    for player_id in ids:
        by_position.setdefault(int(pool[player_id].position), []).append(player_id)
    kept = []
    for position, members in by_position.items():
        need = SQUAD_POSITION_COUNTS[position] + 4
        cost = np.array([int(costs[player_id]) for player_id in members])
        score = np.array([[float(pool[player_id].points(week)) for week in weeks] for player_id in members])
        clubs = np.array([int(pool[player_id].team_id) for player_id in members])
        member_ids = np.array(members)
        for index, player_id in enumerate(members):
            if player_id in owned:
                kept.append(player_id)
                continue
            at_least = (cost <= cost[index]) & np.all(score >= score[index], axis=1)
            strictly = (cost < cost[index]) | np.any(score > score[index], axis=1)
            better = at_least & (strictly | (member_ids < player_id))
            if len(set(clubs[better].tolist())) >= need:
                continue
            kept.append(player_id)
    return sorted(kept)


class _Rows:
    """Row-wise sparse matrix built from numpy pieces."""

    def __init__(self):
        self._index = []
        self._value = []
        self._lengths = []
        self._lower = []
        self._upper = []

    @property
    def count(self):
        return len(self._lengths)

    def add(self, columns, values, lower, upper):
        self._index.append(np.asarray(columns, dtype=np.int32))
        self._value.append(np.asarray(values, dtype=float))
        self._lengths.append(len(columns))
        self._lower.append(float(lower))
        self._upper.append(float(upper))

    def add_pairs(self, plus_columns, minus_columns, lower, upper):
        """One row per pair: plus - minus, inside the bounds."""
        count = len(plus_columns)
        stacked = np.empty(2 * count, dtype=np.int32)
        stacked[0::2] = plus_columns
        stacked[1::2] = minus_columns
        signs = np.empty(2 * count)
        signs[0::2] = 1.0
        signs[1::2] = -1.0
        self._index.append(stacked)
        self._value.append(signs)
        self._lengths.extend([2] * count)
        self._lower.extend([float(lower)] * count)
        self._upper.extend([float(upper)] * count)

    def bounds(self):
        return np.array(self._lower), np.array(self._upper)

    def matrix(self):
        lengths = np.array(self._lengths, dtype=np.int32)
        starts = np.zeros(len(lengths) + 1, dtype=np.int32)
        np.cumsum(lengths, out=starts[1:])
        return starts, np.concatenate(self._index), np.concatenate(self._value)
