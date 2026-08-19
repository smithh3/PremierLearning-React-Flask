"""Pure offline squad and transfer optimisation for historical replays."""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pulp import (
    LpMaximize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
)


SQUAD_POSITION_COUNTS = {1: 2, 2: 5, 3: 5, 4: 3}
LINEUP_POSITION_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
LINEUP_POSITION_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
MAX_PLAYERS_PER_TEAM = 3
INITIAL_BUDGET = 1000


@dataclass
class PlayerProjection:
    player_id: int
    player_code: int
    name: str
    position: int
    team_id: int
    price: int
    selectable: bool
    points_by_gameweek: Dict[int, float] = field(default_factory=dict)
    minutes_by_gameweek: Dict[int, float] = field(default_factory=dict)

    def points(self, gameweek: int) -> float:
        if not self.selectable:
            return 0.0
        return float(self.points_by_gameweek.get(int(gameweek), 0.0))

    def minutes(self, gameweek: int) -> float:
        if not self.selectable:
            return 0.0
        return float(self.minutes_by_gameweek.get(int(gameweek), 0.0))


class PredictionPool:
    def __init__(self, players: Dict[int, PlayerProjection]):
        self.players = players

    @classmethod
    def from_rows(cls, rows):
        players: Dict[int, PlayerProjection] = {}
        for row in rows:
            player_id = int(row['player_id'])
            player = players.get(player_id)
            if player is None:
                player = PlayerProjection(
                    player_id=player_id,
                    player_code=int(row['player_code']),
                    name=str(row['player_name']),
                    position=int(row['position']),
                    team_id=int(row['team_id']),
                    price=int(round(float(row['price']))),
                    selectable=bool(row['selectable']),
                )
                players[player_id] = player
            gameweek = int(row['target_gameweek'])
            player.points_by_gameweek[gameweek] = (
                player.points_by_gameweek.get(gameweek, 0.0)
                + float(row['predicted_points'] or 0.0)
            )
            player.minutes_by_gameweek[gameweek] = (
                player.minutes_by_gameweek.get(gameweek, 0.0)
                + float(row['predicted_minutes'] or 0.0)
            )
        return cls(players)

    def __getitem__(self, player_id):
        return self.players[int(player_id)]

    def __contains__(self, player_id):
        return int(player_id) in self.players


@dataclass
class OwnedPlayer:
    player_id: int
    purchase_price: int


@dataclass
class SquadState:
    owned: Dict[int, OwnedPlayer]
    bank: int
    free_transfers: int

    @property
    def player_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self.owned))


@dataclass
class Lineup:
    starters: Tuple[int, ...]
    bench: Tuple[int, ...]
    captain_id: int
    vice_captain_id: int
    expected_points: float


@dataclass
class PlannedTransfer:
    player_out_id: int
    player_in_id: int
    sell_price: int
    buy_price: int


@dataclass
class TransferPlan:
    transfers: Tuple[PlannedTransfer, ...]
    selected_player_ids: Tuple[int, ...]
    expected_hold_current: float
    expected_selected_current: float
    expected_hold_horizon: float
    expected_selected_horizon: float

    @property
    def current_gain(self):
        return self.expected_selected_current - self.expected_hold_current

    @property
    def horizon_gain(self):
        return self.expected_selected_horizon - self.expected_hold_horizon


@dataclass
class StrategyConfig:
    horizon_gameweeks: int = 5
    min_current_gw_gain: float = 0.0
    min_horizon_gain: float = 0.0
    max_transfers_per_gameweek: int = 5


def selling_price(purchase_price: int, current_price: int) -> int:
    purchase_price = int(purchase_price)
    current_price = int(current_price)
    price_change = current_price - purchase_price
    if price_change <= 0:
        return current_price
    return purchase_price + math.floor(price_change / 2)


class FreeTransferRules:
    """Season-aware free-transfer state, excluding hits and chips."""

    def __init__(self, season_start_year: int):
        self.season_start_year = int(season_start_year)
        self.max_bank = 2 if self.season_start_year <= 23 else 5

    def at_start_of_gameweek(self, gameweek: int, available: int) -> int:
        if self.season_start_year == 25 and int(gameweek) == 16:
            return 5
        return min(self.max_bank, max(0, int(available)))

    def after_gameweek(self, gameweek: int, available: int, used: int) -> int:
        if int(gameweek) == 1:
            next_available = 1
        else:
            next_available = min(
                self.max_bank,
                max(0, int(available) - int(used)) + 1,
            )
        return self.at_start_of_gameweek(int(gameweek) + 1, next_available)


def _validate_squad(player_ids: Iterable[int], pool: PredictionPool):
    ids = tuple(player_ids)
    if len(ids) != 15 or len(set(ids)) != 15:
        raise ValueError('A squad must contain 15 unique players')
    position_counts = {position: 0 for position in SQUAD_POSITION_COUNTS}
    team_counts = {}
    for player_id in ids:
        player = pool[player_id]
        position_counts[player.position] = position_counts.get(player.position, 0) + 1
        team_counts[player.team_id] = team_counts.get(player.team_id, 0) + 1
    if position_counts != SQUAD_POSITION_COUNTS:
        raise ValueError(f'Invalid squad position counts: {position_counts}')
    if any(count > MAX_PLAYERS_PER_TEAM for count in team_counts.values()):
        raise ValueError(f'Invalid squad club counts: {team_counts}')


def pick_lineup(
    player_ids: Sequence[int],
    pool: PredictionPool,
    gameweek: int,
) -> Lineup:
    """Pick a legal XI, captain, vice-captain and deterministic bench order."""
    ids = tuple(int(player_id) for player_id in player_ids)
    if len(ids) != 15:
        raise ValueError('Lineup selection requires a 15-player squad')
    model = LpProblem(name=f'lineup_gw_{gameweek}', sense=LpMaximize)
    starts = {
        player_id: LpVariable(f'start_{gameweek}_{player_id}', cat='Binary')
        for player_id in ids
    }
    model += lpSum(starts.values()) == 11
    for position in SQUAD_POSITION_COUNTS:
        count = lpSum(
            starts[player_id] * int(pool[player_id].position == position)
            for player_id in ids
        )
        model += count >= LINEUP_POSITION_MIN[position]
        model += count <= LINEUP_POSITION_MAX[position]
    model += lpSum(
        starts[player_id]
        * (
            pool[player_id].points(gameweek)
            + (1e-9 * (10000 - player_id))
        )
        for player_id in ids
    )
    model.solve(PULP_CBC_CMD(msg=0))
    if LpStatus[model.status] != 'Optimal':
        raise ValueError(f'Could not select legal GW{gameweek} lineup')
    starter_ids = tuple(sorted(
        player_id for player_id in ids if starts[player_id].value() == 1
    ))
    ranked_starters = sorted(
        starter_ids,
        key=lambda player_id: (
            pool[player_id].points(gameweek),
            -player_id,
        ),
        reverse=True,
    )
    captain_id = ranked_starters[0]
    vice_captain_id = ranked_starters[1]
    bench_ids = [player_id for player_id in ids if player_id not in starter_ids]
    outfield_bench = sorted(
        (
            player_id
            for player_id in bench_ids
            if pool[player_id].position != 1
        ),
        key=lambda player_id: (
            pool[player_id].points(gameweek),
            -player_id,
        ),
        reverse=True,
    )
    keeper_bench = [
        player_id for player_id in bench_ids if pool[player_id].position == 1
    ]
    bench = tuple(outfield_bench + keeper_bench)
    expected = (
        sum(pool[player_id].points(gameweek) for player_id in starter_ids)
        + pool[captain_id].points(gameweek)
    )
    return Lineup(
        starters=starter_ids,
        bench=bench,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        expected_points=float(expected),
    )


class SimulationStrategy:
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()

    def _gameweeks(self, as_of_gameweek):
        return tuple(
            gameweek
            for gameweek in range(
                int(as_of_gameweek),
                min(39, int(as_of_gameweek) + self.config.horizon_gameweeks),
            )
        )

    @staticmethod
    def _expected_scores(player_ids, pool, gameweeks):
        scores = {
            gameweek: pick_lineup(player_ids, pool, gameweek).expected_points
            for gameweek in gameweeks
        }
        return scores

    def build_initial_squad(
        self,
        pool: PredictionPool,
        as_of_gameweek: int = 1,
        budget: int = INITIAL_BUDGET,
    ) -> Tuple[int, ...]:
        return self._optimise_squad(
            pool=pool,
            as_of_gameweek=as_of_gameweek,
            current_state=None,
            max_transfers=15,
            budget=int(budget),
            hold_current=0.0,
            hold_horizon=0.0,
            require_transfer=False,
            enforce_thresholds=False,
        )

    def plan(
        self,
        state: SquadState,
        pool: PredictionPool,
        as_of_gameweek: int,
    ) -> TransferPlan:
        _validate_squad(state.player_ids, pool)
        gameweeks = self._gameweeks(as_of_gameweek)
        hold_scores = self._expected_scores(state.player_ids, pool, gameweeks)
        hold_current = hold_scores.get(int(as_of_gameweek), 0.0)
        hold_horizon = sum(hold_scores.values())
        max_transfers = min(
            max(0, int(state.free_transfers)),
            max(0, int(self.config.max_transfers_per_gameweek)),
        )
        if max_transfers <= 0:
            return self._hold_plan(state, hold_current, hold_horizon)
        try:
            selected = self._optimise_squad(
                pool=pool,
                as_of_gameweek=as_of_gameweek,
                current_state=state,
                max_transfers=max_transfers,
                budget=state.bank,
                hold_current=hold_current,
                hold_horizon=hold_horizon,
                require_transfer=True,
                enforce_thresholds=True,
            )
        except ValueError:
            return self._hold_plan(state, hold_current, hold_horizon)

        selected_scores = self._expected_scores(selected, pool, gameweeks)
        selected_current = selected_scores.get(int(as_of_gameweek), 0.0)
        selected_horizon = sum(selected_scores.values())
        current_gain = selected_current - hold_current
        horizon_gain = selected_horizon - hold_horizon
        if (
            current_gain + 1e-8 < float(self.config.min_current_gw_gain)
            or horizon_gain + 1e-8 < float(self.config.min_horizon_gain)
            or (
                self.config.min_current_gw_gain <= 0
                and self.config.min_horizon_gain <= 0
                and horizon_gain <= 1e-8
            )
        ):
            return self._hold_plan(state, hold_current, hold_horizon)
        transfers = self._pair_transfers(state, selected, pool)
        return TransferPlan(
            transfers=tuple(transfers),
            selected_player_ids=tuple(sorted(selected)),
            expected_hold_current=float(hold_current),
            expected_selected_current=float(selected_current),
            expected_hold_horizon=float(hold_horizon),
            expected_selected_horizon=float(selected_horizon),
        )

    @staticmethod
    def _hold_plan(state, hold_current, hold_horizon):
        return TransferPlan(
            transfers=(),
            selected_player_ids=state.player_ids,
            expected_hold_current=float(hold_current),
            expected_selected_current=float(hold_current),
            expected_hold_horizon=float(hold_horizon),
            expected_selected_horizon=float(hold_horizon),
        )

    def _optimise_squad(
        self,
        pool,
        as_of_gameweek,
        current_state,
        max_transfers,
        budget,
        hold_current,
        hold_horizon,
        require_transfer,
        enforce_thresholds,
    ):
        gameweeks = self._gameweeks(as_of_gameweek)
        current_ids = set(current_state.player_ids if current_state else ())
        valid_ids = sorted(
            player_id
            for player_id, player in pool.players.items()
            if player.selectable or player_id in current_ids
        )
        if len(valid_ids) < 15:
            raise ValueError('Not enough selectable players to build a squad')
        model = LpProblem(name=f'simulation_squad_gw_{as_of_gameweek}', sense=LpMaximize)
        selected = {
            player_id: LpVariable(f'selected_{player_id}', cat='Binary')
            for player_id in valid_ids
        }
        starts = {
            gameweek: {
                player_id: LpVariable(
                    f'start_{gameweek}_{player_id}', cat='Binary'
                )
                for player_id in valid_ids
            }
            for gameweek in gameweeks
        }
        captains = {
            gameweek: {
                player_id: LpVariable(
                    f'captain_{gameweek}_{player_id}', cat='Binary'
                )
                for player_id in valid_ids
            }
            for gameweek in gameweeks
        }

        model += lpSum(selected.values()) == 15
        for position, required in SQUAD_POSITION_COUNTS.items():
            model += lpSum(
                selected[player_id]
                * int(pool[player_id].position == position)
                for player_id in valid_ids
            ) == required
        team_ids = sorted({pool[player_id].team_id for player_id in valid_ids})
        for team_id in team_ids:
            model += lpSum(
                selected[player_id]
                * int(pool[player_id].team_id == team_id)
                for player_id in valid_ids
            ) <= MAX_PLAYERS_PER_TEAM

        gw_score_expressions = {}
        for gameweek in gameweeks:
            model += lpSum(starts[gameweek].values()) == 11
            model += lpSum(captains[gameweek].values()) == 1
            for player_id in valid_ids:
                model += starts[gameweek][player_id] <= selected[player_id]
                model += captains[gameweek][player_id] <= starts[gameweek][player_id]
            for position in SQUAD_POSITION_COUNTS:
                lineup_count = lpSum(
                    starts[gameweek][player_id]
                    * int(pool[player_id].position == position)
                    for player_id in valid_ids
                )
                model += lineup_count >= LINEUP_POSITION_MIN[position]
                model += lineup_count <= LINEUP_POSITION_MAX[position]
            gw_score_expressions[gameweek] = lpSum(
                (
                    starts[gameweek][player_id]
                    + captains[gameweek][player_id]
                )
                * pool[player_id].points(gameweek)
                for player_id in valid_ids
            )

        if current_state is None:
            model += lpSum(
                selected[player_id] * pool[player_id].price
                for player_id in valid_ids
            ) <= int(budget)
        else:
            outgoing = lpSum(
                1 - selected[player_id] for player_id in current_ids
            )
            model += outgoing <= int(max_transfers)
            if require_transfer:
                model += outgoing >= 1
            incoming_cost = lpSum(
                selected[player_id] * pool[player_id].price
                for player_id in valid_ids
                if player_id not in current_ids
            )
            sale_proceeds = lpSum(
                (1 - selected[player_id])
                * selling_price(
                    current_state.owned[player_id].purchase_price,
                    pool[player_id].price,
                )
                for player_id in current_ids
            )
            model += incoming_cost <= int(current_state.bank) + sale_proceeds

        horizon_score = lpSum(gw_score_expressions.values())
        if enforce_thresholds:
            model += (
                gw_score_expressions[int(as_of_gameweek)]
                >= float(hold_current) + float(self.config.min_current_gw_gain)
            )
            model += (
                horizon_score
                >= float(hold_horizon) + float(self.config.min_horizon_gain)
            )
        model += horizon_score + lpSum(
            selected[player_id] * (1e-9 * (10000 - player_id))
            for player_id in valid_ids
        )
        model.solve(PULP_CBC_CMD(msg=0))
        if LpStatus[model.status] != 'Optimal':
            raise ValueError(
                f'No feasible squad for GW{as_of_gameweek}: {LpStatus[model.status]}'
            )
        result = tuple(sorted(
            player_id
            for player_id in valid_ids
            if selected[player_id].value() == 1
        ))
        _validate_squad(result, pool)
        return result

    @staticmethod
    def _pair_transfers(state, selected, pool):
        current_ids = set(state.player_ids)
        selected_ids = set(selected)
        outgoing = sorted(current_ids - selected_ids)
        incoming = sorted(selected_ids - current_ids)
        by_position_out = {}
        by_position_in = {}
        for player_id in outgoing:
            by_position_out.setdefault(pool[player_id].position, []).append(player_id)
        for player_id in incoming:
            by_position_in.setdefault(pool[player_id].position, []).append(player_id)
        transfers = []
        for position in SQUAD_POSITION_COUNTS:
            position_out = by_position_out.get(position, [])
            position_in = by_position_in.get(position, [])
            if len(position_out) != len(position_in):
                raise ValueError(
                    f'Transfer position mismatch at position {position}: '
                    f'{position_out} -> {position_in}'
                )
            for player_out_id, player_in_id in zip(position_out, position_in):
                transfers.append(PlannedTransfer(
                    player_out_id=player_out_id,
                    player_in_id=player_in_id,
                    sell_price=selling_price(
                        state.owned[player_out_id].purchase_price,
                        pool[player_out_id].price,
                    ),
                    buy_price=pool[player_in_id].price,
                ))
        return transfers


def apply_transfer_plan(
    state: SquadState,
    plan: TransferPlan,
) -> SquadState:
    owned = dict(state.owned)
    bank = int(state.bank)
    for transfer in plan.transfers:
        if transfer.player_out_id not in owned:
            raise ValueError(f'Cannot sell unowned player {transfer.player_out_id}')
        del owned[transfer.player_out_id]
        bank += int(transfer.sell_price)
    for transfer in plan.transfers:
        if transfer.player_in_id in owned:
            raise ValueError(f'Cannot buy owned player {transfer.player_in_id}')
        if bank < int(transfer.buy_price):
            raise ValueError('Transfer plan exceeds available bank')
        bank -= int(transfer.buy_price)
        owned[transfer.player_in_id] = OwnedPlayer(
            player_id=transfer.player_in_id,
            purchase_price=int(transfer.buy_price),
        )
    return SquadState(
        owned=owned,
        bank=bank,
        free_transfers=max(0, int(state.free_transfers) - len(plan.transfers)),
    )
