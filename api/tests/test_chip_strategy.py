import unittest

from squadtools.chip_strategy import (
    ChipInventory,
    ChipPlanner,
    ChipStrategyConfig,
    add_played_chip,
    adjust_chip_thresholds,
    chip_window,
    chips_sharing_window,
    dgw_half_multiplier,
    effective_threshold,
    expiring_inventory,
    fixture_imbalance,
    remaining_fixtures_by_team,
    force_remaining_chips,
    free_hit_blocked_after_wildcard,
    inventory_for,
    low_minute_owned_count,
    triple_captain_threshold,
    visible_chip_plan,
    wc_early_delay_factor,
    wc_early_threshold_bonus,
)
from squadtools.simulation_strategy import (
    CHIP_FREE_HIT,
    CHIP_WILDCARD,
    ChipCalendar,
    OwnedPlayer,
    SquadState,
)
from tests.test_simulation_strategy import make_pool


def make_state(pool, player_ids, free_transfers=2, bank=0):
    return SquadState(
        owned={
            player_id: OwnedPlayer(player_id, pool[player_id].price)
            for player_id in player_ids
        },
        bank=int(bank),
        free_transfers=int(free_transfers),
    )


def make_planner(**kwargs):
    config = dict(
        season_start_year=25,
        horizon_gameweeks=3,
        fh_threshold=8.0,
        wc_threshold=15.0,
        fh_recovery_threshold=8.0,
        tc_above_average_threshold=4.0,
        tc_window=5,
        forecast_future_halves=False,
        # Most chip tests only need the rebuild probe. The 2-FT WC branch is
        # covered by test_wildcard_eval_rolls_up_to_two_transfers.
        wc_eval_max_transfers=0,
        # The bars above were picked against this helper's 3-week horizon, so
        # quote them over it and leave the scaling to the tests that want it.
        chip_bar_reference_horizon=3,
    )
    config.update(kwargs)
    return ChipPlanner(ChipStrategyConfig(**config))


def spike_extras(gameweek, points, weeks):
    return [
        {
            'position': position,
            'team_id': 10 + position,
            'points_by_gameweek': {
                gw: points if gw == gameweek else 0.0 for gw in weeks
            },
        }
        for position in (2, 3, 4)
    ]


def lasting_extras(points, weeks):
    return [
        {
            'position': position,
            'team_id': 10 + position,
            'points_by_gameweek': {gw: points for gw in weeks},
        }
        for position in (2, 3, 4)
    ]


class ChipStrategyTests(unittest.TestCase):
    def test_inventory_is_per_half_in_season_25(self):
        played = ChipCalendar(
            wildcards=(6,),
            free_hits=(8,),
            bench_boosts=(7,),
            triple_captains=(9,),
        )
        first_half = inventory_for(25, 10, played)
        self.assertEqual(
            ChipInventory(False, False, False, False),
            first_half,
        )
        second_half = inventory_for(25, 20, played)
        self.assertEqual(ChipInventory(True, True, True, True), second_half)

    def test_older_seasons_use_one_free_hit_bench_boost_and_triple_captain(self):
        played = ChipCalendar(free_hits=(8,), bench_boosts=(7,), triple_captains=(13,))
        self.assertFalse(inventory_for(23, 20, played).free_hit)
        self.assertFalse(inventory_for(23, 20, played).bench_boost)
        self.assertFalse(inventory_for(23, 20, played).triple_captain)
        self.assertTrue(inventory_for(23, 20, played).wildcard)

    def test_chip_window_runs_to_gw38_before_2025_except_the_wildcard(self):
        self.assertEqual((1, 19), chip_window(24, CHIP_WILDCARD, 16))
        self.assertEqual((20, 38), chip_window(24, CHIP_WILDCARD, 20))
        self.assertEqual((1, 38), chip_window(24, CHIP_FREE_HIT, 16))
        self.assertEqual((1, 19), chip_window(25, CHIP_FREE_HIT, 16))
        self.assertEqual((20, 38), chip_window(25, CHIP_FREE_HIT, 20))

    def test_only_the_wildcard_expires_at_halfway_before_2025(self):
        held = ChipInventory(True, True, True, True)
        self.assertEqual(
            ChipInventory(True, False, False, False),
            expiring_inventory(24, held, 16),
        )
        # Second half: everything really does expire at GW38.
        self.assertEqual(held, expiring_inventory(24, held, 30))
        self.assertEqual(held, expiring_inventory(25, held, 16))

    def test_chips_sharing_window_counts_rivals_for_the_same_weeks(self):
        held = ChipInventory(True, True, True, True)
        self.assertEqual(1, chips_sharing_window(24, held, 16, CHIP_WILDCARD))
        self.assertEqual(3, chips_sharing_window(24, held, 16, CHIP_FREE_HIT))
        self.assertEqual(4, chips_sharing_window(25, held, 16, CHIP_FREE_HIT))

    def test_free_hit_bar_survives_the_halfway_deadline_before_2025(self):
        held = ChipInventory(True, True, True, True)
        before = effective_threshold(
            11.0,
            16,
            chips_sharing_window(24, held, 16, CHIP_FREE_HIT),
            *chip_window(24, CHIP_FREE_HIT, 16),
        )
        per_half = effective_threshold(
            11.0,
            16,
            chips_sharing_window(25, held, 16, CHIP_FREE_HIT),
            *chip_window(25, CHIP_FREE_HIT, 16),
        )
        self.assertGreater(before, 0.0)
        self.assertEqual(0.0, per_half)

    def test_chip_bar_stretches_onto_the_horizon_it_is_measured_over(self):
        """A bar quoted over 5 weeks has to ask for twice as much over 10.

        Chip gaps are horizon totals, so without this the horizon knob would
        keep doubling as a threshold knob.
        """
        for horizon, expected in ((5, 50.0), (10, 100.0), (2, 20.0), (4, 40.0)):
            planner = make_planner(
                horizon_gameweeks=horizon,
                chip_bar_reference_horizon=5,
            )
            self.assertAlmostEqual(expected, planner._week_bar(50.0, 7))

    def test_early_wildcard_bonus_stretches_with_the_bar_it_is_added_to(self):
        planner = make_planner(
            horizon_gameweeks=10, chip_bar_reference_horizon=5
        )
        self.assertAlmostEqual(60.0, planner._week_bar(lambda _gw: 30.0, 4))

    def test_chip_week_is_picked_against_the_stretched_bar(self):
        evals = {6: (60.0, None), 9: (120.0, None)}
        at_reference = make_planner(
            horizon_gameweeks=5, chip_bar_reference_horizon=5
        )
        self.assertEqual(9, at_reference._best_week(evals, 100.0, set()))
        # The same quoted bar over twice the weeks wants 200, so neither week
        # qualifies any more.
        stretched = make_planner(
            horizon_gameweeks=10, chip_bar_reference_horizon=5
        )
        self.assertIsNone(stretched._best_week(evals, 100.0, set()))

    def test_zero_reference_horizon_leaves_the_bar_alone(self):
        planner = make_planner(
            horizon_gameweeks=9, chip_bar_reference_horizon=0
        )
        self.assertAlmostEqual(40.0, planner._week_bar(40.0, 3))

    def test_thresholds_decay_to_zero_when_slack_is_gone(self):
        self.assertAlmostEqual(
            8.0, effective_threshold(8.0, 1, 4, 1, 19)
        )
        self.assertEqual(0.0, effective_threshold(8.0, 16, 4, 1, 19))
        self.assertGreater(
            effective_threshold(8.0, 1, 4, 1, 19),
            effective_threshold(8.0, 10, 4, 1, 19),
        )

    def test_triple_captain_threshold_holds_then_tapers_to_a_floor(self):
        self.assertAlmostEqual(1.3, triple_captain_threshold(5, 1.3))
        self.assertAlmostEqual(1.3, triple_captain_threshold(9, 1.3))
        self.assertAlmostEqual(1.3, triple_captain_threshold(10, 1.3))
        self.assertAlmostEqual(0.8, triple_captain_threshold(19, 1.3))
        self.assertAlmostEqual(
            1.3 + (0.8 - 1.3) * (4.0 / 9.0),
            triple_captain_threshold(14, 1.3),
        )
        self.assertAlmostEqual(1.3, triple_captain_threshold(20, 1.3))
        self.assertAlmostEqual(1.3, triple_captain_threshold(29, 1.3))
        self.assertAlmostEqual(0.8, triple_captain_threshold(38, 1.3))
        self.assertLess(
            triple_captain_threshold(33, 1.3),
            triple_captain_threshold(29, 1.3),
        )
        self.assertGreater(triple_captain_threshold(19, 1.3), 0.0)

    def test_free_hit_is_blocked_for_three_weeks_after_wildcard(self):
        self.assertFalse(free_hit_blocked_after_wildcard(5, 7))
        self.assertTrue(free_hit_blocked_after_wildcard(7, 7))
        self.assertTrue(free_hit_blocked_after_wildcard(10, 7))
        self.assertFalse(free_hit_blocked_after_wildcard(11, 7))
        self.assertTrue(free_hit_blocked_after_wildcard(8, 7, weeks=1))
        self.assertFalse(free_hit_blocked_after_wildcard(9, 7, weeks=1))
        self.assertTrue(free_hit_blocked_after_wildcard(12, 7, weeks=5))
        self.assertFalse(free_hit_blocked_after_wildcard(13, 7, weeks=5))

    def test_triple_captain_easy_until_can_hold_the_bar_later(self):
        self.assertAlmostEqual(
            1.6, triple_captain_threshold(19, 1.6, floor=1.2, easy_until=19)
        )
        self.assertAlmostEqual(
            1.2, triple_captain_threshold(19, 1.6, floor=1.2, easy_until=10)
        )

    def test_force_remaining_uses_wildcard_then_bench_boost(self):
        assigned = force_remaining_chips(
            16,
            19,
            ChipInventory(True, True, True, True),
        )
        self.assertEqual(
            {
                16: 'wildcard',
                17: 'bench_boost',
                18: 'free_hit',
                19: 'triple_captain',
            },
            assigned,
        )
        self.assertEqual(4, len(set(assigned)))

    def test_deadline_forces_remaining_chips_in_order(self):
        weeks = range(1, 20)
        pool, current_ids, _ = make_pool(gameweeks=weeks)
        result = make_planner(
            fh_threshold=1000.0,
            wc_threshold=1000.0,
            tc_above_average_threshold=1000.0,
        ).plan(make_state(pool, current_ids), pool, 16)
        self.assertEqual((16,), result.calendar.wildcards)
        self.assertEqual((17,), result.calendar.bench_boosts)
        self.assertEqual((18,), result.calendar.free_hits)
        self.assertEqual((19,), result.calendar.triple_captains)
        self.assertTrue(result.reason['forced'])
        occupied = [
            result.calendar.chip_at(gameweek)
            for gameweek in range(16, 20)
        ]
        self.assertEqual(4, len([chip for chip in occupied if chip]))
        self.assertEqual(len(set(occupied)), len(occupied))

    def test_pre_2025_seasons_hold_chips_past_the_halfway_deadline(self):
        weeks = range(1, 20)
        pool, current_ids, _ = make_pool(gameweeks=weeks)
        result = make_planner(
            season_start_year=24,
            fh_threshold=1000.0,
            wc_threshold=1000.0,
            tc_above_average_threshold=1000.0,
        ).plan(make_state(pool, current_ids), pool, 16)
        self.assertFalse(result.reason['forced'])
        self.assertEqual(1, result.reason['expiring_chips_left'])
        self.assertEqual((), result.calendar.free_hits)
        self.assertEqual((), result.calendar.bench_boosts)
        self.assertEqual((), result.calendar.triple_captains)

    def test_pre_2025_wildcard_is_still_forced_before_it_expires(self):
        weeks = range(1, 20)
        pool, current_ids, _ = make_pool(gameweeks=weeks)
        result = make_planner(
            season_start_year=24,
            fh_threshold=1000.0,
            wc_threshold=1000.0,
            tc_above_average_threshold=1000.0,
        ).plan(make_state(pool, current_ids), pool, 19)
        self.assertTrue(result.reason['forced'])
        self.assertEqual((19,), result.calendar.wildcards)
        self.assertEqual((), result.calendar.free_hits)
        self.assertEqual((), result.calendar.triple_captains)

    def test_free_hit_is_chosen_when_only_its_differential_clears(self):
        weeks = range(1, 16)
        pool, current_ids, _ = make_pool(
            gameweeks=weeks,
            extra_players=spike_extras(6, 40.0, weeks),
        )
        result = make_planner(
            fh_threshold=5.0,
            wc_threshold=1000.0,
            tc_above_average_threshold=1000.0,
        ).plan(make_state(pool, current_ids), pool, 5)
        self.assertEqual((6,), result.calendar.free_hits)
        self.assertEqual((), result.calendar.wildcards)
        self.assertIsNone(result.reason.get('recovery'))
        self.assertIn(6, result.free_hit_plans)
        self.assertEqual(15, len(result.free_hit_plans[6].selected_player_ids))

    def test_wildcard_is_chosen_when_recovery_fails(self):
        weeks = range(1, 16)
        pool, current_ids, _ = make_pool(
            gameweeks=weeks,
            extra_players=lasting_extras(20.0, weeks),
        )
        result = make_planner(
            fh_threshold=1.0,
            wc_threshold=1.0,
            fh_recovery_threshold=0.0,
            tc_above_average_threshold=1000.0,
        ).plan(make_state(pool, current_ids), pool, 5)
        self.assertTrue(result.calendar.wildcards)
        wildcard_gw = result.calendar.wildcards[0]
        self.assertEqual((wildcard_gw + 1,), result.calendar.bench_boosts)
        recovery = result.reason.get('recovery')
        if recovery is not None:
            self.assertFalse(recovery['chose_free_hit'])
            self.assertNotEqual(
                (recovery['gameweek'],), result.calendar.free_hits
            )

    def test_free_hit_is_chosen_when_the_existing_team_recovers(self):
        weeks = range(1, 16)
        pool, current_ids, _ = make_pool(
            gameweeks=weeks,
            extra_players=spike_extras(5, 50.0, weeks),
        )
        result = make_planner(
            fh_threshold=1.0,
            wc_threshold=1.0,
            fh_recovery_threshold=8.0,
            tc_above_average_threshold=1000.0,
        ).plan(make_state(pool, current_ids), pool, 5)
        self.assertIn(5, result.calendar.free_hits)
        recovery = result.reason.get('recovery')
        self.assertIsNotNone(recovery)
        self.assertTrue(recovery['chose_free_hit'])
        self.assertNotIn(5, result.calendar.wildcards)

    def test_bench_boost_is_the_week_after_wildcard(self):
        weeks = range(1, 16)
        pool, current_ids, _ = make_pool(
            gameweeks=weeks,
            extra_players=lasting_extras(20.0, weeks),
        )
        result = make_planner(
            fh_threshold=1000.0,
            wc_threshold=1.0,
            tc_above_average_threshold=1000.0,
        ).plan(make_state(pool, current_ids), pool, 5)
        self.assertTrue(result.calendar.wildcards)
        self.assertEqual(
            (result.calendar.wildcards[0] + 1,),
            result.calendar.bench_boosts,
        )
        self.assertEqual(
            result.calendar.wildcards[0] + 1,
            result.reason['bench_boost_after_wildcard'],
        )

    def test_later_free_hits_are_rescored_from_the_post_wildcard_team(self):
        weeks = range(1, 16)
        pool, current_ids, _ = make_pool(
            gameweeks=weeks,
            extra_players=lasting_extras(20.0, weeks),
        )
        result = make_planner(
            fh_threshold=0.0,
            wc_threshold=1.0,
            tc_above_average_threshold=1000.0,
        ).plan(make_state(pool, current_ids), pool, 5)
        self.assertTrue(result.calendar.wildcards)
        wildcard_gw = result.calendar.wildcards[0]
        later = result.reason.get('fh_after_wildcard_weeks') or []
        self.assertTrue(all(gameweek > wildcard_gw for gameweek in later))
        self.assertTrue(
            all(
                not free_hit_blocked_after_wildcard(gameweek, wildcard_gw)
                for gameweek in later
            )
        )
        self.assertNotIn(wildcard_gw + 1, later)
        if result.calendar.free_hits:
            self.assertGreater(result.calendar.free_hits[0], wildcard_gw + 3)

    def test_free_hit_waits_three_weeks_after_wildcard_when_horizon_allows(self):
        weeks = range(1, 20)
        pool, current_ids, _ = make_pool(
            gameweeks=weeks,
            extra_players=lasting_extras(20.0, weeks),
        )
        result = make_planner(
            horizon_gameweeks=8,
            fh_threshold=0.0,
            wc_threshold=1.0,
            tc_above_average_threshold=1000.0,
        ).plan(make_state(pool, current_ids), pool, 5)
        self.assertTrue(result.calendar.wildcards)
        wildcard_gw = result.calendar.wildcards[0]
        later = result.reason.get('fh_after_wildcard_weeks') or []
        self.assertTrue(later)
        self.assertGreaterEqual(min(later), wildcard_gw + 4)
        if result.calendar.free_hits:
            self.assertGreaterEqual(result.calendar.free_hits[0], wildcard_gw + 4)

    def test_triple_captain_uses_the_window_peak_above_average(self):
        weeks = range(1, 16)
        pool, current_ids, _ = make_pool(gameweeks=weeks)
        captain = current_ids[0]
        pool.players[captain].points_by_gameweek = {
            gw: 2.0 for gw in weeks
        }
        pool.players[captain].points_by_gameweek[7] = 20.0
        result = make_planner(
            fh_threshold=1000.0,
            wc_threshold=1000.0,
            tc_above_average_threshold=4.0,
            tc_window=5,
        ).plan(make_state(pool, current_ids), pool, 5)
        self.assertEqual((7,), result.calendar.triple_captains)
        self.assertEqual(7, result.reason['triple_captain']['gameweek'])
        self.assertEqual(captain, result.reason['triple_captain']['player_id'])

    def test_triple_captain_requires_a_minimum_captain_score(self):
        weeks = range(1, 16)
        pool, current_ids, strong_id = make_pool(gameweeks=weeks)
        pool.players[strong_id].points_by_gameweek = {gw: 1.0 for gw in weeks}
        captain = current_ids[0]
        for player_id in current_ids:
            pool.players[player_id].points_by_gameweek = {
                gw: 2.0 for gw in weeks
            }
        pool.players[captain].points_by_gameweek[7] = 7.9
        skipped = make_planner(
            fh_threshold=1000.0,
            wc_threshold=1000.0,
            tc_above_average_threshold=4.0,
            tc_window=5,
        ).plan(make_state(pool, current_ids), pool, 5)
        self.assertEqual((), skipped.calendar.triple_captains)
        pool.players[captain].points_by_gameweek[7] = 8.0
        picked = make_planner(
            fh_threshold=1000.0,
            wc_threshold=1000.0,
            tc_above_average_threshold=4.0,
            tc_window=5,
        ).plan(make_state(pool, current_ids), pool, 5)
        self.assertEqual((7,), picked.calendar.triple_captains)

    def test_triple_captain_is_skipped_on_free_hit_or_wildcard_weeks(self):
        weeks = range(1, 16)
        pool, current_ids, _ = make_pool(
            gameweeks=weeks,
            extra_players=spike_extras(5, 40.0, weeks),
        )
        captain = current_ids[0]
        pool.players[captain].points_by_gameweek = {
            gw: 2.0 for gw in weeks
        }
        pool.players[captain].points_by_gameweek[5] = 80.0
        result = make_planner(
            fh_threshold=5.0,
            wc_threshold=1000.0,
            tc_above_average_threshold=0.0,
        ).plan(make_state(pool, current_ids), pool, 5)
        self.assertEqual((5,), result.calendar.free_hits)
        self.assertNotIn(5, result.calendar.triple_captains)

    def test_bench_boost_plays_immediately_after_an_already_used_wildcard(self):
        weeks = range(1, 16)
        pool, current_ids, _ = make_pool(gameweeks=weeks)
        result = make_planner(
            fh_threshold=1000.0,
            wc_threshold=1000.0,
            tc_above_average_threshold=1000.0,
        ).plan(
            make_state(pool, current_ids),
            pool,
            6,
            played=ChipCalendar(wildcards=(5,)),
        )
        self.assertEqual((5,), result.calendar.wildcards)
        self.assertEqual((6,), result.calendar.bench_boosts)

    def test_no_chips_in_the_first_four_gameweeks(self):
        weeks = range(1, 16)
        pool, current_ids, _ = make_pool(
            gameweeks=weeks,
            extra_players=spike_extras(3, 80.0, weeks),
        )
        result = make_planner(
            fh_threshold=0.0,
            wc_threshold=0.0,
            fh_recovery_threshold=0.0,
            tc_above_average_threshold=0.0,
        ).plan(make_state(pool, current_ids), pool, 1)
        for gameweek in range(1, 5):
            self.assertIsNone(result.calendar.chip_at(gameweek))
        assigned = force_remaining_chips(
            1,
            19,
            ChipInventory(True, True, True, True),
        )
        self.assertTrue(all(gameweek >= 5 for gameweek in assigned))

    def test_add_played_chip_locks_used_weeks(self):
        played = add_played_chip(ChipCalendar(), 6, 'wildcard')
        played = add_played_chip(played, 7, 'bench_boost')
        self.assertEqual((6,), played.wildcards)
        self.assertEqual((7,), played.bench_boosts)
        self.assertEqual(played, add_played_chip(played, 8, None))

    def test_first_half_forecasts_second_half_wildcard_and_stops_planning_before_it(self):
        weeks = range(1, 32)
        extras = [
            {
                'position': position,
                'team_id': 11 + index,
                'points_by_gameweek': {
                    gw: 25.0 if gw >= 21 else 1.0 for gw in weeks
                },
            }
            for index, position in enumerate((2, 2, 3, 3, 4, 4, 3, 2))
        ]
        pool, current_ids, _ = make_pool(gameweeks=weeks, extra_players=extras)
        result = make_planner(
            horizon_gameweeks=5,
            fh_threshold=1000.0,
            wc_threshold=8.0,
            tc_above_average_threshold=1000.0,
            forecast_future_halves=True,
        ).plan(make_state(pool, current_ids), pool, 15)
        second_half_wildcards = [
            gameweek for gameweek in result.calendar.wildcards if gameweek >= 20
        ]
        self.assertTrue(second_half_wildcards)
        next_wildcard = result.calendar.next_wildcard_after(15)
        self.assertEqual(second_half_wildcards[0], next_wildcard)
        window = result.calendar.planning_gameweeks(15, 5)
        self.assertTrue(window)
        self.assertLess(max(window), next_wildcard)
        self.assertTrue(result.reason.get('second_half', {}).get('deferred'))
        self.assertEqual(
            20, result.reason.get('second_half', {}).get('placeholder_wildcard')
        )
        self.assertTrue(all(gameweek < 20 for gameweek in result.calendar.free_hits))
        self.assertTrue(
            all(gameweek < 20 for gameweek in result.calendar.triple_captains)
        )
        self.assertTrue(all(gameweek <= 19 for gameweek in result.free_hit_plans))
        self.assertTrue(all(gameweek <= 19 for gameweek in result.wildcard_plans))

    def test_fh_and_wc_probes_only_cover_the_next_horizon(self):
        weeks = range(1, 16)
        pool, current_ids, _ = make_pool(gameweeks=weeks)
        planner = make_planner(
            horizon_gameweeks=5,
            forecast_future_halves=True,
            fh_threshold=0.0,
            wc_threshold=0.0,
            tc_above_average_threshold=1000.0,
        )
        result = planner.plan(make_state(pool, current_ids), pool, 1)
        self.assertTrue(set(result.free_hit_plans) <= {5})
        self.assertTrue(set(result.wildcard_plans) <= {5})
        self.assertEqual(
            20, result.reason.get('second_half', {}).get('placeholder_wildcard')
        )
        self.assertNotIn(
            20, visible_chip_plan(result.calendar, result.reason)['wildcards']
        )
        self.assertIn(20, result.calendar.wildcards)
        self.assertTrue(
            all(gameweek <= 10 for gameweek in result.calendar.triple_captains)
        )

    def test_one_optimise_per_kind_probes_this_week_only(self):
        weeks = range(1, 16)
        pool, current_ids, _ = make_pool(
            gameweeks=weeks,
            extra_players=spike_extras(5, 40.0, weeks),
        )
        planner = make_planner(
            one_optimise_per_kind=True,
            forecast_future_halves=True,
            fh_threshold=5.0,
            wc_threshold=1000.0,
            tc_above_average_threshold=1000.0,
        )
        rolls = []
        resets = []
        original_roll = planner._roll
        original_reset = planner._reset_plan

        def counting_roll(*args, **kwargs):
            rolls.append(1)
            return original_roll(*args, **kwargs)

        def counting_reset(*args, **kwargs):
            resets.append(kwargs.get('free_hit'))
            return original_reset(*args, **kwargs)

        planner._roll = counting_roll
        planner._reset_plan = counting_reset
        result = planner.plan(make_state(pool, current_ids), pool, 5)
        self.assertEqual([], rolls)
        self.assertEqual([True, False], resets)
        self.assertEqual((5,), tuple(result.free_hit_plans))
        self.assertEqual((5,), tuple(result.wildcard_plans))
        self.assertEqual(
            20, result.reason.get('second_half', {}).get('placeholder_wildcard')
        )
        self.assertTrue(result.reason.get('second_half', {}).get('deferred'))

    def test_fixture_imbalance_is_mean_abs_remaining_games(self):
        weeks = range(1, 8)
        pool, current_ids, _ = make_pool(gameweeks=weeks)
        self.assertAlmostEqual(0.0, fixture_imbalance(pool, 1))
        for player in pool.players.values():
            if int(player.team_id) != 1:
                continue
            for gameweek in (5, 6, 7):
                player.fixture_ids_by_gameweek.pop(gameweek, None)
        self.assertGreater(fixture_imbalance(pool, 1), 0.0)

    def test_remaining_fixtures_count_ids_not_minutes(self):
        from squadtools.simulation_strategy import PlayerProjection, PredictionPool

        def player(player_id, team_id, fixtures, minutes=90.0):
            weeks = tuple(fixtures)
            return PlayerProjection(
                player_id=player_id,
                player_code=player_id,
                name='P%s' % player_id,
                position=3,
                team_id=team_id,
                price=50,
                selectable=True,
                points_by_gameweek={gw: 2.0 for gw in weeks},
                minutes_by_gameweek={gw: minutes for gw in weeks},
                fixture_ids_by_gameweek={
                    gw: set(ids) for gw, ids in fixtures.items()
                },
            )

        pool = PredictionPool({
            1: player(1, 1, {1: {11}, 2: {12}, 3: {13}}, minutes=0.0),
            2: player(2, 2, {1: {21}, 2: {22}, 3: {23}}),
        })
        counts = remaining_fixtures_by_team(pool, 1)
        self.assertEqual(3.0, counts[1])
        self.assertEqual(3.0, counts[2])
        pool.players[1].fixture_ids_by_gameweek[2] = {12, 120}
        self.assertEqual(4.0, remaining_fixtures_by_team(pool, 1)[1])
        del pool.players[1].fixture_ids_by_gameweek[3]
        self.assertEqual(3.0, remaining_fixtures_by_team(pool, 1)[1])
        self.assertEqual(2.0, remaining_fixtures_by_team(pool, 2)[1])

    def test_dgw_weight_raises_thresholds_more_in_the_second_half(self):
        self.assertEqual(1.0, dgw_half_multiplier(19, 3.0))
        self.assertEqual(3.0, dgw_half_multiplier(20, 3.0))
        fh1, wc1 = adjust_chip_thresholds(
            10.0,
            45.0,
            fixture_imbalance_value=2.0,
            half_multiplier=1.0,
            fh_dgw_weight=2.0,
            wc_dgw_weight=4.0,
            low_minute_count=0,
            wc_rotation_weight=0.0,
        )
        fh2, wc2 = adjust_chip_thresholds(
            10.0,
            45.0,
            fixture_imbalance_value=2.0,
            half_multiplier=2.0,
            fh_dgw_weight=2.0,
            wc_dgw_weight=4.0,
            low_minute_count=0,
            wc_rotation_weight=0.0,
        )
        self.assertAlmostEqual(14.0, fh1)
        self.assertAlmostEqual(53.0, wc1)
        self.assertAlmostEqual(18.0, fh2)
        self.assertGreater(wc2, wc1)

    def test_rotation_weight_lowers_wildcard_threshold(self):
        weeks = range(1, 8)
        pool, current_ids, _ = make_pool(gameweeks=weeks)
        state = make_state(pool, current_ids)
        self.assertEqual(0, low_minute_owned_count(state, pool, 1))
        for player_id in current_ids[:4]:
            pool.players[player_id].minutes_by_gameweek[1] = 30.0
        self.assertEqual(4, low_minute_owned_count(state, pool, 1))
        _fh, wc = adjust_chip_thresholds(
            10.0,
            45.0,
            fixture_imbalance_value=0.0,
            half_multiplier=1.0,
            fh_dgw_weight=0.0,
            wc_dgw_weight=0.0,
            low_minute_count=4,
            wc_rotation_weight=1.5,
        )
        self.assertAlmostEqual(39.0, wc)

    def test_a_pool_without_predicted_minutes_counts_low_points(self):
        weeks = range(1, 8)
        pool, current_ids, _ = make_pool(gameweeks=weeks)
        state = make_state(pool, current_ids)
        for player in pool.players.values():
            player.minutes_by_gameweek = {}
        for player_id in current_ids:
            pool.players[player_id].points_by_gameweek[1] = 5.0
        self.assertEqual(0, low_minute_owned_count(state, pool, 1))
        for player_id in current_ids[:3]:
            pool.players[player_id].points_by_gameweek[1] = 1.5
        self.assertEqual(3, low_minute_owned_count(state, pool, 1))

    def test_early_wildcard_bonus_fades_linearly_to_gameweek_six(self):
        self.assertAlmostEqual(1.0, wc_early_delay_factor(3, earliest=3, until=6))
        self.assertAlmostEqual(2.0 / 3.0, wc_early_delay_factor(4, earliest=3, until=6))
        self.assertAlmostEqual(1.0 / 3.0, wc_early_delay_factor(5, earliest=3, until=6))
        self.assertAlmostEqual(0.0, wc_early_delay_factor(6, earliest=3, until=6))
        self.assertAlmostEqual(0.0, wc_early_delay_factor(10, earliest=3, until=6))
        self.assertAlmostEqual(15.0, wc_early_threshold_bonus(3, 15.0, earliest=3, until=6))
        self.assertAlmostEqual(10.0, wc_early_threshold_bonus(4, 15.0, earliest=3, until=6))
        self.assertAlmostEqual(5.0, wc_early_threshold_bonus(5, 15.0, earliest=3, until=6))
        self.assertAlmostEqual(0.0, wc_early_threshold_bonus(6, 15.0, earliest=3, until=6))
        self.assertAlmostEqual(0.0, wc_early_threshold_bonus(3, 0.0, earliest=3, until=6))

    def test_early_wildcard_weight_blocks_wildcards_before_gameweek_six(self):
        weeks = range(1, 16)
        pool, current_ids, _ = make_pool(
            gameweeks=weeks,
            extra_players=lasting_extras(20.0, weeks),
        )
        open_plan = make_planner(
            earliest_chip_gameweek=3,
            wc_early_weight=0.0,
            horizon_gameweeks=4,
            fh_threshold=1000.0,
            wc_threshold=1.0,
            tc_above_average_threshold=1000.0,
        ).plan(make_state(pool, current_ids), pool, 3)
        self.assertIn(3, open_plan.calendar.wildcards)
        delayed = make_planner(
            earliest_chip_gameweek=3,
            wc_early_weight=1000.0,
            horizon_gameweeks=4,
            fh_threshold=1000.0,
            wc_threshold=1.0,
            tc_above_average_threshold=1000.0,
        ).plan(make_state(pool, current_ids), pool, 3)
        self.assertNotIn(3, delayed.calendar.wildcards)
        self.assertNotIn(4, delayed.calendar.wildcards)
        self.assertNotIn(5, delayed.calendar.wildcards)

    def test_used_free_hit_skips_rebuild_probes(self):
        weeks = range(1, 16)
        pool, current_ids, _ = make_pool(gameweeks=weeks)
        planner = make_planner(
            forecast_future_halves=False,
            fh_threshold=0.0,
            wc_threshold=1000.0,
            tc_above_average_threshold=1000.0,
        )
        calls = []
        original = planner._evaluate_free_hits

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        planner._evaluate_free_hits = counting
        played = add_played_chip(ChipCalendar(), 6, 'free_hit')
        result = planner.plan(
            make_state(pool, current_ids), pool, 8, played
        )
        self.assertEqual([], calls)
        self.assertEqual({}, result.free_hit_plans)

    def test_wildcard_eval_rolls_up_to_two_transfers(self):
        weeks = range(1, 16)
        pool, current_ids, _ = make_pool(gameweeks=weeks)
        planner = make_planner(
            horizon_gameweeks=3,
            forecast_future_halves=False,
            fh_threshold=1000.0,
            wc_threshold=1000.0,
            tc_above_average_threshold=1000.0,
            wc_eval_max_transfers=2,
        )
        seen = []
        original = planner._roll

        def counting(*args, **kwargs):
            seen.append(kwargs.get('max_transfers'))
            return original(*args, **kwargs)

        planner._roll = counting
        planner.plan(make_state(pool, current_ids), pool, 5)
        self.assertIn(2, seen)

    def test_future_chip_choice_is_not_locked_until_played(self):
        weeks = range(1, 16)
        pool, current_ids, _ = make_pool(gameweeks=weeks)
        planner = make_planner(forecast_future_halves=False)
        state = make_state(pool, current_ids)
        planner.plan(state, pool, 5, ChipCalendar())
        later = planner.plan(state, pool, 6, ChipCalendar())
        self.assertTrue(inventory_for(25, 6, ChipCalendar()).free_hit)
        self.assertTrue(inventory_for(25, 6, ChipCalendar()).wildcard)
        self.assertIsNotNone(later.reason.get('as_of_gameweek'))
        locked = add_played_chip(ChipCalendar(), 5, 'free_hit')
        self.assertFalse(inventory_for(25, 6, locked).free_hit)
        skipped = planner.plan(state, pool, 6, locked)
        self.assertEqual({}, skipped.free_hit_plans)

    def test_decided_free_hit_is_skipped_when_replanning_transfers(self):
        weeks = range(1, 16)
        pool, current_ids, _ = make_pool(
            gameweeks=weeks,
            extra_players=spike_extras(7, 80.0, weeks),
        )
        planner = make_planner(
            horizon_gameweeks=3,
            fh_threshold=0.0,
            wc_threshold=1000.0,
            tc_above_average_threshold=1000.0,
            fh_recovery_threshold=1000.0,
        )
        result = planner.plan(make_state(pool, current_ids), pool, 5)
        fh_week = result.reason.get('free_hit_gameweek')
        self.assertIsNotNone(fh_week)
        self.assertTrue(result.reason.get('committed_chip_roll'))
        window = result.calendar.planning_gameweeks(5, 3)
        if int(fh_week) == 5:
            self.assertEqual((5, 6, 7), window)
        else:
            self.assertNotIn(int(fh_week), window)
        self.assertEqual(
            3,
            len(result.calendar.planning_gameweeks(int(fh_week), 3)),
        )


if __name__ == '__main__':
    unittest.main()
