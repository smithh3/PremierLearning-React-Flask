import unittest

from simulation.actual_scoring import score_lineup
from squadtools.simulation_strategy import (
    FreeTransferRules,
    OwnedPlayer,
    PlayerProjection,
    PredictionPool,
    SimulationStrategy,
    SquadState,
    StrategyConfig,
    SQUAD_POSITION_COUNTS,
    Lineup,
    pick_lineup,
    selling_price,
)


def make_pool():
    players = {}
    player_id = 1
    current_ids = []
    for position, count in SQUAD_POSITION_COUNTS.items():
        for _ in range(count):
            players[player_id] = PlayerProjection(
                player_id=player_id,
                player_code=1000 + player_id,
                name=f'Player {player_id}',
                position=position,
                team_id=((player_id - 1) % 8) + 1,
                price=50,
                selectable=True,
                points_by_gameweek={gw: 1.0 for gw in range(1, 6)},
                minutes_by_gameweek={gw: 90.0 for gw in range(1, 6)},
            )
            current_ids.append(player_id)
            player_id += 1
    strong_defender_id = player_id
    players[strong_defender_id] = PlayerProjection(
        player_id=strong_defender_id,
        player_code=1000 + strong_defender_id,
        name='Strong defender',
        position=2,
        team_id=20,
        price=50,
        selectable=True,
        points_by_gameweek={gw: 10.0 for gw in range(1, 6)},
        minutes_by_gameweek={gw: 90.0 for gw in range(1, 6)},
    )
    return PredictionPool(players), tuple(current_ids), strong_defender_id


class SimulationStrategyTests(unittest.TestCase):
    def test_selling_price_uses_half_profit(self):
        self.assertEqual(55, selling_price(50, 60))
        self.assertEqual(49, selling_price(50, 49))

    def test_free_transfer_rules_are_season_specific(self):
        old_rules = FreeTransferRules(23)
        self.assertEqual(2, old_rules.after_gameweek(5, 2, 0))
        new_rules = FreeTransferRules(24)
        self.assertEqual(5, new_rules.after_gameweek(5, 5, 1))
        afcon_rules = FreeTransferRules(25)
        self.assertEqual(5, afcon_rules.after_gameweek(15, 1, 1))

    def test_lineup_is_legal_and_has_captain(self):
        pool, current_ids, _ = make_pool()
        lineup = pick_lineup(current_ids, pool, 1)
        self.assertEqual(11, len(lineup.starters))
        counts = {position: 0 for position in SQUAD_POSITION_COUNTS}
        for player_id in lineup.starters:
            counts[pool[player_id].position] += 1
        self.assertEqual(1, counts[1])
        self.assertGreaterEqual(counts[2], 3)
        self.assertGreaterEqual(counts[3], 2)
        self.assertGreaterEqual(counts[4], 1)
        self.assertIn(lineup.captain_id, lineup.starters)
        self.assertIn(lineup.vice_captain_id, lineup.starters)

    def test_initial_squad_respects_budget_and_club_limit(self):
        pool, _, _ = make_pool()
        selected = SimulationStrategy().build_initial_squad(pool, 1, budget=1000)
        self.assertEqual(15, len(selected))
        self.assertLessEqual(sum(pool[player_id].price for player_id in selected), 1000)
        team_counts = {}
        for player_id in selected:
            team_id = pool[player_id].team_id
            team_counts[team_id] = team_counts.get(team_id, 0) + 1
        self.assertLessEqual(max(team_counts.values()), 3)

    def test_beneficial_free_transfer_is_selected(self):
        pool, current_ids, strong_defender_id = make_pool()
        state = SquadState(
            owned={
                player_id: OwnedPlayer(player_id, pool[player_id].price)
                for player_id in current_ids
            },
            bank=0,
            free_transfers=1,
        )
        plan = SimulationStrategy().plan(state, pool, 1)
        self.assertEqual(1, len(plan.transfers))
        self.assertEqual(strong_defender_id, plan.transfers[0].player_in_id)
        self.assertGreater(plan.current_gain, 0)
        self.assertGreater(plan.horizon_gain, 0)

    def test_threshold_can_force_hold(self):
        pool, current_ids, _ = make_pool()
        state = SquadState(
            owned={
                player_id: OwnedPlayer(player_id, pool[player_id].price)
                for player_id in current_ids
            },
            bank=0,
            free_transfers=1,
        )
        strategy = SimulationStrategy(StrategyConfig(
            min_current_gw_gain=100.0,
            min_horizon_gain=100.0,
        ))
        plan = strategy.plan(state, pool, 1)
        self.assertEqual((), plan.transfers)
        self.assertEqual(current_ids, plan.selected_player_ids)

    def test_each_gain_threshold_is_enforced_independently(self):
        pool, current_ids, _ = make_pool()
        state = SquadState(
            owned={
                player_id: OwnedPlayer(player_id, pool[player_id].price)
                for player_id in current_ids
            },
            bank=0,
            free_transfers=1,
        )
        current_threshold = SimulationStrategy(StrategyConfig(
            min_current_gw_gain=20.0,
            min_horizon_gain=0.0,
        )).plan(state, pool, 1)
        horizon_threshold = SimulationStrategy(StrategyConfig(
            min_current_gw_gain=0.0,
            min_horizon_gain=100.0,
        )).plan(state, pool, 1)
        self.assertEqual((), current_threshold.transfers)
        self.assertEqual((), horizon_threshold.transfers)

    def test_captain_fallback_and_autosub(self):
        pool, current_ids, _ = make_pool()
        lineup = pick_lineup(current_ids, pool, 1)
        first_bench = lineup.bench[0]
        captain = next(
            player_id
            for player_id in lineup.starters
            if pool[player_id].position == pool[first_bench].position
        )
        vice_captain = next(
            player_id for player_id in lineup.starters if player_id != captain
        )
        lineup = Lineup(
            starters=lineup.starters,
            bench=lineup.bench,
            captain_id=captain,
            vice_captain_id=vice_captain,
            expected_points=lineup.expected_points,
        )

        class Actual:
            def __init__(self, points, minutes):
                self.points = points
                self.minutes = minutes

        actuals = {
            player_id: Actual(points=2, minutes=90)
            for player_id in current_ids
        }
        actuals[captain] = Actual(points=0, minutes=0)
        actuals[first_bench] = Actual(points=5, minutes=90)
        result = score_lineup(lineup, pool, actuals)
        self.assertEqual(vice_captain, result.captain_scored_id)
        self.assertEqual(1, result.unexpected_zero_minute_starters)
        self.assertTrue(any(sub['in'] == first_bench for sub in result.autosubs))


if __name__ == '__main__':
    unittest.main()
