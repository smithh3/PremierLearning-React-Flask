"""Turns one high-level action into a transfer list.

The mixed-integer optimiser in squadtools is the production implementation of
this job. GreedyOptimiser is the stand-in that lets an episode run, and be
tested, without that solver. It sells the lowest forecast and buys the best
affordable player in the same position.
"""
from __future__ import annotations

from squadtools.simulation_strategy import (
    MAX_PLAYERS_PER_TEAM,
    SQUAD_POSITION_COUNTS,
    OwnedPlayer,
    PlannedTransfer,
    PredictionPool,
    SquadState,
    TransferPlan,
    copy_squad_state,
    selling_price,
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
from fpl_manager.lineup import greedy_lineup, hit_penalty, score_lineup


def _empty_plan(state: SquadState) -> TransferPlan:
    return TransferPlan(
        transfers=(),
        selected_player_ids=state.player_ids,
        expected_hold_current=0.0,
        expected_selected_current=0.0,
        expected_hold_horizon=0.0,
        expected_selected_horizon=0.0,
    )


def _forecast_xi(state: SquadState, pool: PredictionPool, gameweek: int) -> float:
    starters, bench, captain_id, _vice = greedy_lineup(
        state.player_ids,
        lambda player_id: pool[player_id].points(gameweek),
        lambda player_id: pool[player_id].position,
    )
    return score_lineup(
        lambda player_id: pool[player_id].points(gameweek),
        starters, bench, captain_id, False, False,
    )


class GreedyOptimiser:
    def execute(self, action: str, state: SquadState, pool: PredictionPool, gameweek: int):
        chip = chip_for(action)
        if action in (HOLD, BENCH_BOOST, TRIPLE_CAPTAIN):
            return _empty_plan(state), chip
        if action in (WILDCARD, FREE_HIT):
            return self._rebuild(state, pool, gameweek), chip
        return self._transfers(state, pool, gameweek, transfer_count(action)), None

    def transfer_gains(self, state: SquadState, pool: PredictionPool, gameweek: int) -> dict:
        """Gain over holding, minus a hit once, summed across the horizon."""
        gains = {}
        for count in (1, 2, 3, 4, 5):
            try:
                plan = self._transfers(state, pool, gameweek, count)
            except ValueError:
                for horizon in (3, 6):
                    gains[(count, horizon)] = 0.0
                continue
            trial = _apply_owned(state, plan)
            hits = hit_penalty(len(plan.transfers), state.free_transfers, False)
            for horizon in (3, 6):
                after = 0.0
                before = 0.0
                for offset in range(horizon):
                    week = int(gameweek) + offset
                    after += _forecast_xi(trial, pool, week)
                    before += _forecast_xi(state, pool, week)
                gains[(count, horizon)] = after - before - hits
        return gains

    def _transfers(self, state, pool, gameweek, count):
        owned_ids = set(state.player_ids)
        bank = int(state.bank)
        purchase = {player_id: state.owned[player_id].purchase_price for player_id in owned_ids}
        transfers = []
        for _step in range(count):
            team_counts = {}
            for player_id in owned_ids:
                team_id = int(pool[player_id].team_id)
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
            best = None
            for seller in owned_ids:
                position = int(pool[seller].position)
                seller_team = int(pool[seller].team_id)
                proceeds = selling_price(purchase[seller], pool[seller].price)
                budget = bank + proceeds
                candidates = [
                    player for player in pool.players.values()
                    if player.selectable
                    and int(player.position) == position
                    and player.player_id not in owned_ids
                    and int(player.price) <= budget
                    and team_counts.get(int(player.team_id), 0) - int(int(player.team_id) == seller_team) < MAX_PLAYERS_PER_TEAM
                ]
                if not candidates:
                    continue
                buyer = max(
                    candidates,
                    key=lambda player: (float(player.points(gameweek)), -int(player.price), -player.player_id),
                )
                gain = float(buyer.points(gameweek)) - float(pool[seller].points(gameweek))
                if best is None or gain > best[0] or (gain == best[0] and seller < best[1]):
                    best = (gain, seller, buyer, proceeds)
            if best is None:
                raise ValueError('no legal buyer for transfer %s' % (len(transfers) + 1))
            _gain, seller, buyer, proceeds = best
            transfers.append(PlannedTransfer(
                player_out_id=seller,
                player_in_id=buyer.player_id,
                sell_price=proceeds,
                buy_price=int(buyer.price),
            ))
            owned_ids.remove(seller)
            owned_ids.add(buyer.player_id)
            bank = bank + proceeds - int(buyer.price)
            purchase[buyer.player_id] = int(buyer.price)
        plan = TransferPlan(
            transfers=tuple(transfers),
            selected_player_ids=tuple(sorted(owned_ids)),
            expected_hold_current=0.0,
            expected_selected_current=0.0,
            expected_hold_horizon=0.0,
            expected_selected_horizon=0.0,
        )
        return plan

    def _rebuild(self, state, pool, gameweek):
        budget = int(state.bank) + sum(
            selling_price(owned.purchase_price, pool[player_id].price)
            for player_id, owned in state.owned.items()
        )
        chosen = []
        spent = 0
        for position, count in SQUAD_POSITION_COUNTS.items():
            options = sorted(
                (
                    player for player in pool.players.values()
                    if player.selectable and int(player.position) == position
                ),
                key=lambda player: (
                    -float(player.points(gameweek)), int(player.price), player.player_id,
                ),
            )
            picked = 0
            for player in options:
                if picked >= count:
                    break
                if spent + int(player.price) > budget:
                    continue
                chosen.append(player)
                spent += int(player.price)
                picked += 1
            if picked < count:
                raise ValueError('wildcard cannot fill position %s' % position)
        chosen_ids = {player.player_id for player in chosen}
        transfers = _pair_rebuild(state, pool, chosen)
        return TransferPlan(
            transfers=tuple(transfers),
            selected_player_ids=tuple(sorted(chosen_ids)),
            expected_hold_current=0.0,
            expected_selected_current=0.0,
            expected_hold_horizon=0.0,
            expected_selected_horizon=0.0,
        )


def _pair_rebuild(state, pool, chosen):
    outgoing = {}
    incoming = {}
    chosen_ids = {player.player_id for player in chosen}
    for player_id in state.owned:
        if player_id not in chosen_ids:
            outgoing.setdefault(int(pool[player_id].position), []).append(player_id)
    for player in chosen:
        if player.player_id not in state.owned:
            incoming.setdefault(int(player.position), []).append(player)
    transfers = []
    for position, sellers in outgoing.items():
        buyers = incoming.get(position, [])
        if len(sellers) != len(buyers):
            raise ValueError('rebuild position mismatch at %s' % position)
        for seller, buyer in zip(sellers, buyers):
            transfers.append(PlannedTransfer(
                player_out_id=seller,
                player_in_id=buyer.player_id,
                sell_price=selling_price(state.owned[seller].purchase_price, pool[seller].price),
                buy_price=int(buyer.price),
            ))
    return transfers


def _apply_owned(state: SquadState, plan: TransferPlan) -> SquadState:
    owned = {
        player_id: OwnedPlayer(player_id, owned.purchase_price)
        for player_id, owned in state.owned.items()
    }
    bank = int(state.bank)
    for transfer in plan.transfers:
        del owned[transfer.player_out_id]
        bank += int(transfer.sell_price)
        owned[transfer.player_in_id] = OwnedPlayer(transfer.player_in_id, int(transfer.buy_price))
        bank -= int(transfer.buy_price)
    trial = copy_squad_state(SquadState(owned=owned, bank=bank, free_transfers=state.free_transfers))
    return trial
