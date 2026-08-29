import os
import unittest

from simulation.actual_scoring import score_lineup
from squadtools.simulation_strategy import (
    ChipCalendar,
    FreeTransferRules,
    OwnedPlayer,
    PlayerProjection,
    PredictionPool,
    SimulationStrategy,
    SquadState,
    StrategyConfig,
    SQUAD_POSITION_COUNTS,
    Lineup,
    apply_transfer_plan,
    copy_squad_state,
    expected_gameweek_points,
    fill_bench,
    flat_expected_points,
    optimiser_candidate_ids,
    pick_lineup,
    predicted_points_line,
    selling_price,
)


def _default_fixture_ids(team_id, gameweeks):
    return {int(gw): {int(team_id) * 1000 + int(gw)} for gw in gameweeks}


def make_pool(gameweeks=None, extra_players=None):
    gameweeks = tuple(gameweeks or range(1, 6))
    players = {}
    player_id = 1
    current_ids = []
    for position, count in SQUAD_POSITION_COUNTS.items():
        for _ in range(count):
            team_id = ((player_id - 1) % 8) + 1
            players[player_id] = PlayerProjection(
                player_id=player_id,
                player_code=1000 + player_id,
                name=f'Player {player_id}',
                position=position,
                team_id=team_id,
                price=50,
                selectable=True,
                points_by_gameweek={gw: 2.0 for gw in gameweeks},
                minutes_by_gameweek={gw: 90.0 for gw in gameweeks},
                fixture_ids_by_gameweek=_default_fixture_ids(team_id, gameweeks),
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
        points_by_gameweek={gw: 10.0 for gw in gameweeks},
        minutes_by_gameweek={gw: 90.0 for gw in gameweeks},
        fixture_ids_by_gameweek=_default_fixture_ids(20, gameweeks),
    )
    for extra in extra_players or ():
        player_id += 1
        team_id = extra.get('team_id', 20)
        extra_weeks = tuple(extra['points_by_gameweek'])
        players[player_id] = PlayerProjection(
            player_id=player_id,
            player_code=1000 + player_id,
            name=extra.get('name', f'Extra {player_id}'),
            position=extra['position'],
            team_id=team_id,
            price=extra.get('price', 50),
            selectable=True,
            points_by_gameweek=dict(extra['points_by_gameweek']),
            minutes_by_gameweek={
                gw: extra.get('minutes', 90.0)
                for gw in extra['points_by_gameweek']
            },
            fixture_ids_by_gameweek=_default_fixture_ids(team_id, extra_weeks),
        )
        extra['player_id'] = player_id
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

    def _state_for(self, pool, current_ids):
        return SquadState(
            owned={
                player_id: OwnedPlayer(player_id, pool[player_id].price)
                for player_id in current_ids
            },
            bank=0,
            free_transfers=1,
        )

    def _transfer_count(self, gain, horizon, reference):
        pool, current_ids, _ = make_pool()
        strategy = SimulationStrategy(StrategyConfig(
            horizon_gameweeks=horizon,
            min_current_gw_gain=0.0,
            min_horizon_gain=gain,
            min_gain_reference_horizon=reference,
        ))
        plan = strategy.plan(self._state_for(pool, current_ids), pool, 1)
        return len(plan.transfers)

    def test_min_horizon_gain_is_quoted_over_the_reference_window(self):
        """This pool's best single transfer gains just under 64 over 4 weeks.

        A 70 bar quoted over 5 weeks asks for 56 across 4, so it lets the
        transfer through, while taking the same number literally blocks it.
        """
        self.assertEqual(1, self._transfer_count(70.0, 4, 5))
        self.assertEqual(1, self._transfer_count(56.0, 4, 0))
        self.assertEqual(0, self._transfer_count(70.0, 4, 0))

    def test_min_horizon_gain_is_untouched_at_the_reference_horizon(self):
        self.assertEqual(
            self._transfer_count(70.0, 5, 0), self._transfer_count(70.0, 5, 5)
        )

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

    def test_pick_lineup_uses_chance_adjusted_keeper_points(self):
        from squadtools.simulation_strategy import PredictionPool

        pool, current_ids, _ = make_pool()
        rows = []
        for player_id in current_ids:
            player = pool[player_id]
            chance = 75 if player_id == 1 else None
            unadjusted = 4.1 if player_id == 1 else (4.0 if player_id == 2 else 2.0)
            rows.append({
                'player_id': player_id,
                'player_code': player.player_code,
                'player_name': player.name,
                'position': player.position,
                'team_id': player.team_id,
                'price': player.price,
                'selectable': True,
                'target_gameweek': 1,
                'predicted_pp90': unadjusted,
                'predicted_minutes': 90.0,
                'predicted_points': unadjusted,
                'chance_of_playing': chance,
            })
        adjusted = PredictionPool.from_rows(rows)
        self.assertAlmostEqual(4.1 * 0.75, adjusted[1].points(1))
        self.assertAlmostEqual(4.0, adjusted[2].points(1))
        lineup = pick_lineup(current_ids, adjusted, 1)
        self.assertIn(2, lineup.starters)
        self.assertNotIn(1, lineup.starters)

    def test_from_rows_keeps_fixture_ids_even_when_minutes_are_zero(self):
        from squadtools.simulation_strategy import PredictionPool

        rows = [
            {
                'player_id': 1,
                'player_code': 1,
                'player_name': 'A',
                'position': 3,
                'team_id': 4,
                'price': 50,
                'selectable': True,
                'target_gameweek': 1,
                'predicted_pp90': 0.0,
                'predicted_minutes': 0.0,
                'predicted_points': 0.0,
                'fixture_id': 101,
            },
            {
                'player_id': 1,
                'player_code': 1,
                'player_name': 'A',
                'position': 3,
                'team_id': 4,
                'price': 50,
                'selectable': True,
                'target_gameweek': 1,
                'predicted_pp90': 0.0,
                'predicted_minutes': 0.0,
                'predicted_points': 0.0,
                'fixture_id': 102,
            },
        ]
        pool = PredictionPool.from_rows(rows)
        self.assertEqual({101, 102}, pool[1].fixture_ids_by_gameweek[1])

    def test_plus_two_plus_four_holds_unless_at_max_free_transfers(self):
        pool, current_ids, strong_defender_id = make_pool()
        pool.players[strong_defender_id].points_by_gameweek = {
            gw: 2.2 for gw in range(1, 6)
        }
        owned = {
            player_id: OwnedPlayer(player_id, pool[player_id].price)
            for player_id in current_ids
        }
        config = StrategyConfig(
            min_current_gw_gain=2.0,
            min_horizon_gain=4.0,
            max_free_transfers=5,
            relax_thresholds_at_max_ft=True,
        )
        below_cap = SimulationStrategy(config).plan(
            SquadState(owned=dict(owned), bank=0, free_transfers=1),
            pool,
            1,
        )
        at_cap = SimulationStrategy(config).plan(
            SquadState(owned=dict(owned), bank=0, free_transfers=5),
            pool,
            1,
        )
        self.assertEqual((), below_cap.transfers)
        self.assertEqual(1, len(at_cap.transfers))
        self.assertEqual(strong_defender_id, at_cap.transfers[0].player_in_id)
        self.assertGreater(at_cap.horizon_gain, 0)
        self.assertLess(at_cap.current_gain, 2.0)
        self.assertLess(at_cap.horizon_gain, 4.0)

    def test_roll_holds_this_week_when_two_frees_next_week_score_more(self):
        weeks = range(5, 9)
        pool, current_ids, strong_defender_id = make_pool(
            gameweeks=weeks,
            extra_players=(
                {
                    'position': 2,
                    'team_id': 20,
                    'points_by_gameweek': {gw: 8.0 for gw in weeks},
                },
                {
                    'position': 3,
                    'team_id': 19,
                    'points_by_gameweek': {gw: 8.0 for gw in weeks},
                },
            ),
        )
        pool.players[strong_defender_id].points_by_gameweek = {
            gw: 1.0 for gw in weeks
        }
        state = SquadState(
            owned={
                player_id: OwnedPlayer(player_id, pool[player_id].price)
                for player_id in current_ids
            },
            bank=0,
            free_transfers=1,
        )
        config = StrategyConfig(
            horizon_gameweeks=4,
            min_horizon_gain=0.0,
            min_current_gw_gain=0.0,
            season_start_year=25,
        )
        spend = SimulationStrategy(config).plan(state, pool, 5)
        config.roll_free_transfers = True
        rolled = SimulationStrategy(config).plan(state, pool, 5)
        self.assertEqual(1, len(spend.transfers))
        self.assertEqual((), rolled.transfers)
        self.assertEqual('wait_1', rolled.roll_path)

    def test_roll_does_not_wait_through_a_wildcard(self):
        pool, current_ids, strong_defender_id = make_pool()
        state = SquadState(
            owned={
                player_id: OwnedPlayer(player_id, pool[player_id].price)
                for player_id in current_ids
            },
            bank=0,
            free_transfers=1,
        )
        strategy = SimulationStrategy(StrategyConfig(
            horizon_gameweeks=4,
            min_horizon_gain=0.0,
            min_current_gw_gain=0.0,
            roll_free_transfers=True,
            chips=ChipCalendar(wildcards=(2,)),
        ))
        plan = strategy.plan(state, pool, 1)
        self.assertEqual(1, len(plan.transfers))
        self.assertEqual(strong_defender_id, plan.transfers[0].player_in_id)
        self.assertEqual('spend', plan.roll_path)

    def test_two_free_transfers_roll_up_to_four(self):
        strategy = SimulationStrategy(StrategyConfig(season_start_year=25))
        self.assertEqual(3, strategy._ft_after_holds(8, 2, 1))
        self.assertEqual(4, strategy._ft_after_holds(8, 2, 2))
        self.assertEqual(5, strategy._ft_after_holds(8, 3, 2))

    def test_chip_calendar_matches_season_inventory(self):
        old = ChipCalendar.resolve(23)
        self.assertEqual((6, 24), old.wildcards)
        self.assertEqual((35,), old.free_hits)
        self.assertEqual((7,), old.bench_boosts)
        self.assertEqual((13,), old.triple_captains)
        current = ChipCalendar.resolve(25)
        self.assertEqual((6, 24), current.wildcards)
        self.assertEqual((35,), current.free_hits)
        self.assertEqual((7, 25), current.bench_boosts)
        self.assertEqual((13, 30), current.triple_captains)
        afcon_bb = ChipCalendar.resolve(25, bench_boosts=(16, 25))
        self.assertEqual((16, 25), afcon_bb.bench_boosts)

    def test_planning_window_stops_before_wildcard_and_skips_free_hit(self):
        chips = ChipCalendar(wildcards=(6, 24), free_hits=(35,), bench_boosts=(7, 25))
        self.assertEqual((4, 5), chips.planning_gameweeks(4, 5))
        self.assertEqual((6, 7, 8, 9, 10), chips.planning_gameweeks(6, 5))
        self.assertEqual((35, 36, 37, 38), chips.planning_gameweeks(35, 5))
        self.assertEqual((33, 34, 36, 37, 38), chips.planning_gameweeks(33, 5))
        fh_week = ChipCalendar(free_hits=(8,))
        self.assertEqual((8, 9, 10, 11), fh_week.planning_gameweeks(8, 4))
        self.assertEqual((5, 6, 7, 9), fh_week.planning_gameweeks(5, 4))
        self.assertEqual(
            ChipCalendar(wildcards=(8,)).planning_gameweeks(8, 4),
            fh_week.planning_gameweeks(8, 4),
        )

    def test_future_wildcard_points_are_ignored_before_the_reset(self):
        extras = [{
            'position': 2,
            'team_id': 20,
            'points_by_gameweek': {
                4: 1.0,
                5: 1.0,
                6: 20.0,
                7: 20.0,
                8: 20.0,
            },
        }]
        pool, current_ids, strong_defender_id = make_pool(
            gameweeks=range(4, 9),
            extra_players=extras,
        )
        pool.players[strong_defender_id].points_by_gameweek = {
            gw: 1.0 for gw in range(4, 9)
        }
        late_id = extras[0]['player_id']
        state = SquadState(
            owned={
                player_id: OwnedPlayer(player_id, pool[player_id].price)
                for player_id in current_ids
            },
            bank=0,
            free_transfers=1,
        )
        strategy = SimulationStrategy(StrategyConfig(
            chips=ChipCalendar(wildcards=(6,)),
        ))
        plan = strategy.plan(state, pool, 4)
        self.assertEqual((), plan.transfers)
        self.assertNotIn(late_id, plan.selected_player_ids)

    def test_planned_free_hit_week_is_ignored_when_choosing_transfers(self):
        extras = [{
            'position': 2,
            'team_id': 20,
            'points_by_gameweek': {
                5: 1.0,
                6: 1.0,
                7: 40.0,
                8: 1.0,
            },
        }]
        pool, current_ids, strong_defender_id = make_pool(
            gameweeks=range(5, 9),
            extra_players=extras,
        )
        pool.players[strong_defender_id].points_by_gameweek = {
            gw: 1.0 for gw in range(5, 9)
        }
        late_id = extras[0]['player_id']
        state = SquadState(
            owned={
                player_id: OwnedPlayer(player_id, pool[player_id].price)
                for player_id in current_ids
            },
            bank=0,
            free_transfers=1,
        )
        strategy = SimulationStrategy(StrategyConfig(
            horizon_gameweeks=4,
            chips=ChipCalendar(free_hits=(7,)),
        ))
        plan = strategy.plan(state, pool, 5)
        self.assertEqual((), plan.transfers)
        self.assertNotIn(late_id, plan.selected_player_ids)

    def test_wildcard_can_exceed_the_free_transfer_cap(self):
        extras = [
            {
                'position': 2,
                'team_id': 18,
                'points_by_gameweek': {gw: 12.0 for gw in range(6, 11)},
            },
            {
                'position': 3,
                'team_id': 19,
                'points_by_gameweek': {gw: 12.0 for gw in range(6, 11)},
            },
        ]
        pool, current_ids, strong_defender_id = make_pool(
            gameweeks=range(6, 11),
            extra_players=extras,
        )
        state = SquadState(
            owned={
                player_id: OwnedPlayer(player_id, pool[player_id].price)
                for player_id in current_ids
            },
            bank=0,
            free_transfers=1,
        )
        strategy = SimulationStrategy(StrategyConfig(
            max_transfers_per_gameweek=1,
            chips=ChipCalendar(wildcards=(6,)),
        ))
        plan = strategy.plan(state, pool, 6)
        incoming = {transfer.player_in_id for transfer in plan.transfers}
        self.assertGreaterEqual(len(plan.transfers), 3)
        self.assertIn(strong_defender_id, incoming)
        self.assertTrue(incoming.issuperset(
            extra['player_id'] for extra in extras
        ))

    def test_bench_boost_counts_bench_points(self):
        pool, current_ids, _ = make_pool()
        lineup = pick_lineup(current_ids, pool, 1)
        without_boost = expected_gameweek_points(current_ids, pool, 1)
        with_boost = expected_gameweek_points(
            current_ids, pool, 1, bench_boost=True
        )
        self.assertGreater(with_boost, without_boost)
        self.assertAlmostEqual(
            with_boost,
            without_boost + sum(pool[player_id].points(1) for player_id in lineup.bench),
        )

        class Actual:
            def __init__(self, points, minutes):
                self.points = points
                self.minutes = minutes

        actuals = {
            player_id: Actual(points=2, minutes=90)
            for player_id in current_ids
        }
        for player_id in lineup.bench:
            actuals[player_id] = Actual(points=5, minutes=90)
        boosted = score_lineup(lineup, pool, actuals, bench_boost=True)
        normal = score_lineup(lineup, pool, actuals)
        self.assertEqual(4 * 5, boosted.points - normal.points)
        self.assertEqual((), boosted.autosubs)

    def test_triple_captain_adds_a_third_copy_of_captain_points(self):
        pool, current_ids, _ = make_pool()
        lineup = pick_lineup(current_ids, pool, 1)
        doubled = expected_gameweek_points(current_ids, pool, 1)
        tripled = expected_gameweek_points(
            current_ids, pool, 1, triple_captain=True
        )
        self.assertAlmostEqual(
            tripled,
            doubled + pool[lineup.captain_id].points(1),
        )

        class Actual:
            def __init__(self, points, minutes):
                self.points = points
                self.minutes = minutes

        actuals = {
            player_id: Actual(points=2, minutes=90)
            for player_id in current_ids
        }
        actuals[lineup.captain_id] = Actual(points=8, minutes=90)
        normal = score_lineup(lineup, pool, actuals)
        tripled_actual = score_lineup(lineup, pool, actuals, triple_captain=True)
        self.assertEqual(8, tripled_actual.points - normal.points)

    def test_free_hit_reverts_to_the_previous_squad(self):
        pool, current_ids, strong_defender_id = make_pool()
        state = SquadState(
            owned={
                player_id: OwnedPlayer(player_id, pool[player_id].price)
                for player_id in current_ids
            },
            bank=0,
            free_transfers=2,
        )
        saved = copy_squad_state(state)
        plan = SimulationStrategy(StrategyConfig(
            chips=ChipCalendar(free_hits=(1,)),
        )).plan(state, pool, 1)
        self.assertEqual(1, len(plan.transfers))
        self.assertEqual(strong_defender_id, plan.transfers[0].player_in_id)
        play_state = apply_transfer_plan(state, plan)
        self.assertIn(strong_defender_id, play_state.owned)
        self.assertEqual(saved.player_ids, state.player_ids)
        restored = saved
        self.assertNotIn(strong_defender_id, restored.owned)
        self.assertEqual(2, restored.free_transfers)

    def test_bench_weights_count_first_bench_in_expected_points(self):
        pool, current_ids, _ = make_pool()
        lineup = pick_lineup(current_ids, pool, 1)
        without = expected_gameweek_points(current_ids, pool, 1)
        weights = (0.6, 0.21, 0.075)
        keeper = 0.075
        with_weights = expected_gameweek_points(
            current_ids,
            pool,
            1,
            bench_weights=weights,
            reserve_keeper_weight=keeper,
        )
        bench_contrib = sum(
            weight * pool[player_id].points(1)
            for weight, player_id in zip(weights + (keeper,), lineup.bench)
        )
        self.assertAlmostEqual(without + bench_contrib, with_weights)

    def test_flat_expected_points_are_xi_and_captain_only(self):
        pool, current_ids, _ = make_pool()
        lineup = pick_lineup(current_ids, pool, 1)
        flat = flat_expected_points(lineup, pool, 1)
        self.assertAlmostEqual(lineup.expected_points, flat)
        boosted = flat_expected_points(lineup, pool, 1, bench_boost=True)
        self.assertAlmostEqual(
            flat + sum(pool[player_id].points(1) for player_id in lineup.bench),
            boosted,
        )
        tripled = flat_expected_points(lineup, pool, 1, triple_captain=True)
        self.assertAlmostEqual(
            flat + pool[lineup.captain_id].points(1),
            tripled,
        )

    def test_predicted_points_line_lists_all_fifteen_highest_first(self):
        pool, current_ids, _ = make_pool()
        lineup = pick_lineup(current_ids, pool, 1)
        line = predicted_points_line(lineup, pool, 1)
        self.assertTrue(line.startswith('pred: '))
        self.assertNotIn('\n', line)
        parts = line[len('pred: '):].split(', ')
        self.assertEqual(15, len(parts))
        scores = []
        for part in parts:
            if part.endswith('C') or part.endswith('b'):
                part = part[:-1]
            scores.append(float(part.rsplit(' ', 1)[1]))
        self.assertEqual(sorted(scores, reverse=True), scores)
        self.assertTrue(any(part.endswith('C') for part in parts))
        self.assertEqual(4, sum(1 for part in parts if part.endswith('b')))

    def test_bench_weights_select_a_high_scoring_bench_defender(self):
        extras = [{
            'position': 2,
            'team_id': 19,
            'points_by_gameweek': {1: 8.0},
        }]
        pool, _, _ = make_pool(gameweeks=(1,), extra_players=extras)
        spiked_id = extras[0]['player_id']
        starter_ids = (1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)
        for player_id, player in pool.players.items():
            if player_id == spiked_id:
                continue
            player.points_by_gameweek = {
                1: 10.0 if player_id in starter_ids else 2.0
            }
        selected = SimulationStrategy(StrategyConfig(
            horizon_gameweeks=1,
            bench_weights=(0.6, 0.21, 0.075),
            reserve_keeper_weight=0.075,
        )).build_initial_squad(pool, 1, budget=1000)
        lineup = pick_lineup(selected, pool, 1)
        self.assertIn(spiked_id, selected)
        self.assertEqual(spiked_id, lineup.bench[0])

    def test_fill_bench_keeps_the_frozen_xi(self):
        pool, _, _ = make_pool(gameweeks=(1,))
        starters = (1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14)
        result = fill_bench(
            starters,
            pool,
            (1,),
            budget=1000,
            bench_weights=(0.5, 0.3, 0.1),
            reserve_keeper_weight=0.2,
        )
        self.assertIsNotNone(result)
        self.assertTrue(set(starters).issubset(result))
        self.assertEqual(15, len(result))

    def test_fill_bench_respects_club_slots_left_by_the_xi(self):
        extras = [{
            'position': 2,
            'team_id': 1,
            'price': 40,
            'points_by_gameweek': {1: 20.0},
        }]
        pool, _, _ = make_pool(gameweeks=(1,), extra_players=extras)
        spiked_id = extras[0]['player_id']
        pool[3].team_id = 1
        pool[4].team_id = 1
        pool[9].team_id = 8
        # 1, 3, 4 are team 1. Player 9 is the leftover mid, moved off team 1.
        starters = (1, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14)
        self.assertEqual(3, sum(1 for pid in starters if pool[pid].team_id == 1))
        result = fill_bench(
            starters,
            pool,
            (1,),
            budget=1000,
            bench_weights=(0.5, 0.3, 0.1),
            reserve_keeper_weight=0.2,
        )
        self.assertIsNotNone(result)
        self.assertNotIn(spiked_id, result)

    def test_fill_bench_prefers_a_strong_first_sub_over_even_bench(self):
        extras = [
            {'position': 2, 'team_id': 11, 'price': 40, 'points_by_gameweek': {1: 8.0}},
            {'position': 3, 'team_id': 12, 'price': 40, 'points_by_gameweek': {1: 1.0}},
            {'position': 4, 'team_id': 13, 'price': 40, 'points_by_gameweek': {1: 1.0}},
            {'position': 2, 'team_id': 14, 'price': 40, 'points_by_gameweek': {1: 4.0}},
            {'position': 3, 'team_id': 15, 'price': 40, 'points_by_gameweek': {1: 4.0}},
            {'position': 4, 'team_id': 16, 'price': 40, 'points_by_gameweek': {1: 4.0}},
        ]
        pool, _, _ = make_pool(gameweeks=(1,), extra_players=extras)
        pool[16].points_by_gameweek = {1: 1.0}
        strong_def = extras[0]['player_id']
        even_def = extras[3]['player_id']
        starters = (1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14)
        result = fill_bench(
            starters,
            pool,
            (1,),
            budget=1000,
            bench_weights=(0.5, 0.3, 0.1),
            reserve_keeper_weight=0.2,
        )
        self.assertIn(strong_def, result)
        self.assertNotIn(even_def, result)


def _projection(
    player_id,
    position,
    team_id,
    price,
    points,
    minutes,
    gameweeks=(1, 2, 3, 4),
    selectable=True,
):
    return PlayerProjection(
        player_id=player_id,
        player_code=1000 + player_id,
        name=f'P{player_id}',
        position=position,
        team_id=team_id,
        price=price,
        selectable=selectable,
        points_by_gameweek={gw: points for gw in gameweeks},
        minutes_by_gameweek={gw: minutes for gw in gameweeks},
        fixture_ids_by_gameweek=_default_fixture_ids(team_id, gameweeks),
    )


class OptimiserCandidateFilterTests(unittest.TestCase):
    GAMEWEEKS = (1, 2, 3, 4)

    def _ids(self, pool, current_ids=()):
        return set(optimiser_candidate_ids(pool, current_ids, self.GAMEWEEKS))

    def test_drops_low_minutes_and_low_points(self):
        pool = PredictionPool({
            1: _projection(1, 3, 1, 80, 5.0, 90.0),
            2: _projection(2, 3, 2, 80, 5.0, 40.0),
            3: _projection(3, 3, 3, 80, 1.5, 90.0),
            20: _projection(20, 3, 10, 40, 0.0, 0.0),
            21: _projection(21, 3, 11, 41, 0.0, 0.0),
        })
        kept = self._ids(pool)
        self.assertEqual({1, 20, 21}, kept)

    def test_keeps_owned_players_who_fail_the_floor(self):
        pool = PredictionPool({
            1: _projection(1, 3, 1, 80, 0.0, 0.0),
            2: _projection(2, 3, 2, 80, 5.0, 90.0),
        })
        kept = self._ids(pool, current_ids=(1,))
        self.assertIn(1, kept)
        self.assertIn(2, kept)

    def test_keeps_two_cheapest_from_different_clubs_per_position(self):
        pool = PredictionPool({
            1: _projection(1, 3, 1, 40, 0.0, 0.0),
            2: _projection(2, 3, 1, 40, 0.0, 0.0),
            3: _projection(3, 3, 2, 45, 0.0, 0.0),
            4: _projection(4, 3, 3, 50, 0.0, 0.0),
            5: _projection(5, 3, 4, 80, 5.0, 90.0),
        })
        kept = self._ids(pool)
        self.assertEqual({1, 3, 5}, kept)

    def test_prefers_cheap_enablers_from_clubs_we_can_still_sign(self):
        owned = (
            _projection(10, 2, 1, 50, 2.0, 90.0),
            _projection(11, 3, 1, 50, 2.0, 90.0),
            _projection(12, 4, 1, 50, 2.0, 90.0),
        )
        pool = PredictionPool({
            10: owned[0],
            11: owned[1],
            12: owned[2],
            1: _projection(1, 3, 1, 39, 0.0, 0.0),
            2: _projection(2, 3, 2, 40, 0.0, 0.0),
            3: _projection(3, 3, 3, 41, 0.0, 0.0),
            4: _projection(4, 3, 4, 80, 5.0, 90.0),
        })
        kept = self._ids(pool, current_ids=(10, 11, 12))
        self.assertIn(2, kept)
        self.assertIn(3, kept)
        self.assertNotIn(1, kept)
        self.assertIn(4, kept)
        self.assertTrue({10, 11, 12}.issubset(kept))

    def test_initial_squad_still_builds_after_dropping_duds(self):
        extras = [
            {
                'position': 3,
                'team_id': 18,
                'price': 45,
                'points_by_gameweek': {gw: 0.0 for gw in range(1, 6)},
                'minutes': 10.0,
            }
            for _ in range(20)
        ]
        pool, _, _ = make_pool(extra_players=extras)
        selected = SimulationStrategy().build_initial_squad(pool, 1, budget=1000)
        self.assertEqual(15, len(selected))
        dud_ids = {extra['player_id'] for extra in extras}
        # At most the two cheapest dud mids are kept as enablers.
        self.assertLessEqual(len(dud_ids.intersection(selected)), 2)


class QuietCbcTests(unittest.TestCase):
    def test_cbc_commands_are_detected(self):
        from squadtools.simulation_strategy import _is_cbc_command

        self.assertTrue(_is_cbc_command(['cbc', 'model.lp']))
        self.assertTrue(_is_cbc_command([r'C:\Solvers\cbc.exe', '-solve']))
        self.assertFalse(_is_cbc_command(['python', '-m', 'simulation.sweep']))
        self.assertFalse(_is_cbc_command([]))

    def test_quiet_cbc_kwargs_drop_inherited_stdio(self):
        from squadtools.simulation_strategy import _quiet_cbc_kwargs
        import subprocess

        kwargs = _quiet_cbc_kwargs({'stdin': 1, 'stdout': 2, 'stderr': 3})
        self.assertIs(kwargs['stdin'], subprocess.DEVNULL)
        self.assertIs(kwargs['stdout'], subprocess.DEVNULL)
        self.assertIs(kwargs['stderr'], subprocess.DEVNULL)
        if os.name == 'nt':
            self.assertFalse(kwargs['close_fds'])
            self.assertTrue(
                kwargs['creationflags']
                & getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
            )
        else:
            self.assertTrue(kwargs['close_fds'])

    def test_ensure_quiet_cbc_is_idempotent(self):
        from squadtools.simulation_strategy import ensure_quiet_cbc

        ensure_quiet_cbc()
        ensure_quiet_cbc()

    def test_solver_never_passes_a_thread_count(self):
        from unittest import mock

        from squadtools.simulation_strategy import _cbc_solver

        for threads in (None, '1', '4'):
            env = {} if threads is None else {'SIM_CBC_THREADS': threads}
            with mock.patch.dict(os.environ, env):
                if threads is None:
                    os.environ.pop('SIM_CBC_THREADS', None)
                solver = _cbc_solver()
            self.assertIsNone(solver.optionsDict.get('threads'))
            self.assertNotIn(
                'threads', ' '.join(solver.options + solver.getOptions())
            )

    def test_restore_unpatched_createprocess_puts_back_the_original(self):
        from squadtools.simulation_strategy import _restore_unpatched_createprocess

        if os.name != 'nt':
            self.assertFalse(_restore_unpatched_createprocess())
            return
        import _winapi

        real = _winapi.CreateProcess
        saved = getattr(_winapi, 'original_CreateProcess', None)

        def fake(*args, **kwargs):
            raise AssertionError('still patched')

        try:
            _winapi.original_CreateProcess = real
            _winapi.CreateProcess = fake
            self.assertTrue(_restore_unpatched_createprocess())
            self.assertIs(_winapi.CreateProcess, real)
        finally:
            _winapi.CreateProcess = real
            if saved is None:
                delattr(_winapi, 'original_CreateProcess')
            else:
                _winapi.original_CreateProcess = saved


if __name__ == '__main__':
    unittest.main()
