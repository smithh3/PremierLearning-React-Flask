import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from squadtools.fpl_deadline import (
    is_transfer_deadline_day,
    last_sunday,
    parse_fpl_deadline,
    to_london,
    uk_is_dst,
)
from squadtools.live_strategy import live_chip_config, played_chips_from_user_json
from squadtools.predictions_report import (
    _last_n_appearance_pp90,
    build_html_report,
    build_text_report,
)
from squadtools.simulation_strategy import ChipCalendar, MIN_GAIN_REFERENCE_HORIZON


class LiveChipConfigTests(unittest.TestCase):
    # Sweep trial 39_stuck_rot's own bars, as absolute totals over its horizon.
    SIM_BARS = {
        'fh_threshold': 10.0,
        'wc_threshold': 45.0,
        'fh_recovery_threshold': 10.0,
        'fh_dgw_weight': 2.0,
        'wc_dgw_weight': 4.0,
        'wc_rotation_weight': 6.0,
        'wc_early_weight': 15.0,
    }

    def test_live_bars_stretch_back_out_to_the_winning_trials_numbers(self):
        config = live_chip_config(25)
        self.assertEqual(6, config.horizon_gameweeks)
        scale = config.horizon_gameweeks / config.chip_bar_reference_horizon
        for name, sim_total in self.SIM_BARS.items():
            self.assertAlmostEqual(
                sim_total, getattr(config, name) * scale, msg=name
            )

    def test_live_gain_floors_match_the_winning_trial(self):
        config = live_chip_config(25)
        gain_scale = config.horizon_gameweeks / MIN_GAIN_REFERENCE_HORIZON
        self.assertAlmostEqual(5.0, config.min_horizon_gain * gain_scale)
        # One week's gain, so the horizon never rescales it.
        self.assertAlmostEqual(1.0, config.min_current_gw_gain)

    def test_live_chip_config_carries_the_trials_unscaled_knobs(self):
        config = live_chip_config(25)
        self.assertEqual(3, config.earliest_chip_gameweek)
        self.assertTrue(config.roll_free_transfers)
        self.assertAlmostEqual(2.0, config.dgw_h2_mult)
        self.assertAlmostEqual(1.6, config.tc_above_average_threshold)
        self.assertEqual(8, config.tc_window)

    def test_live_chip_config_values_first_two_bench_slots(self):
        config = live_chip_config(25)
        self.assertAlmostEqual(0.0, config.bench_fill_weight)
        self.assertEqual((0.25, 0.25, 0.0), config.bench_weights)
        self.assertIsNone(config.reserve_keeper_weight)

    def test_skip_bench_fill_drops_all_slot_weights(self):
        import os
        previous = os.environ.get('SKIP_BENCH_FILL')
        os.environ['SKIP_BENCH_FILL'] = '1'
        try:
            config = live_chip_config(25)
            self.assertAlmostEqual(0.0, config.bench_fill_weight)
            self.assertEqual((), config.bench_weights)
            self.assertIsNone(config.reserve_keeper_weight)
        finally:
            if previous is None:
                os.environ.pop('SKIP_BENCH_FILL', None)
            else:
                os.environ['SKIP_BENCH_FILL'] = previous

    def test_pre_gw1_allows_a_full_squad_rebuild(self):
        config = live_chip_config(25, pre_gw1=True)
        self.assertEqual(15, config.max_transfers_per_gameweek)
        self.assertAlmostEqual(0.0, config.min_horizon_gain)
        self.assertAlmostEqual(0.0, config.min_current_gw_gain)


class DeadlineDayTests(unittest.TestCase):
    def test_last_sunday_of_march_2026(self):
        self.assertEqual(29, last_sunday(2026, 3).day)

    def test_uk_dst_bounds_2026(self):
        before = datetime(2026, 3, 29, 0, 59, tzinfo=timezone.utc)
        after = datetime(2026, 3, 29, 1, 0, tzinfo=timezone.utc)
        self.assertFalse(uk_is_dst(before))
        self.assertTrue(uk_is_dst(after))

    def test_is_deadline_day_on_the_uk_date_before_the_deadline(self):
        deadline = parse_fpl_deadline('2026-08-21T17:30:00Z')
        morning = datetime(2026, 8, 21, 5, 0, tzinfo=timezone.utc)  # 06:00 BST
        self.assertTrue(is_transfer_deadline_day(deadline, morning))
        self.assertEqual(21, to_london(morning).day)

    def test_is_not_deadline_day_the_morning_after(self):
        deadline = parse_fpl_deadline('2026-08-21T17:30:00Z')
        next_morning = datetime(2026, 8, 22, 5, 0, tzinfo=timezone.utc)
        self.assertFalse(is_transfer_deadline_day(deadline, next_morning))

    def test_is_not_deadline_day_after_the_deadline_passes(self):
        deadline = parse_fpl_deadline('2026-08-21T17:30:00Z')
        after = datetime(2026, 8, 21, 18, 0, tzinfo=timezone.utc)
        self.assertFalse(is_transfer_deadline_day(deadline, after))

    def test_run_near_a_deadline_books_the_one_after(self):
        from squadtools.fpl_deadline import deadline_for_following_run, london_at_expression
        events = [
            {'id': 8, 'deadline_time': '2026-10-23T17:30:00Z'},
            {'id': 9, 'deadline_time': '2026-10-31T11:00:00Z'},
            {'id': 10, 'deadline_time': '2026-11-06T18:30:00Z'},
        ]
        during_gw8 = datetime(2026, 10, 23, 16, 0, tzinfo=timezone.utc)
        booked = deadline_for_following_run(events, during_gw8)
        self.assertEqual(datetime(2026, 10, 31, 11, 0, tzinfo=timezone.utc), booked)
        # 31 Oct 2026 is GMT, so two hours before 11:00 UTC is 09:00 UK.
        self.assertEqual(
            'at(2026-10-31T09:00:00)',
            london_at_expression(booked - timedelta(hours=2)),
        )

    def test_run_on_an_ordinary_day_books_the_soonest_deadline(self):
        from squadtools.fpl_deadline import deadline_for_following_run
        events = [
            {'id': 9, 'deadline_time': '2026-10-31T11:00:00Z'},
            {'id': 10, 'deadline_time': '2026-11-06T18:30:00Z'},
        ]
        ordinary = datetime(2026, 10, 26, 6, 0, tzinfo=timezone.utc)
        booked = deadline_for_following_run(events, ordinary)
        self.assertEqual(datetime(2026, 10, 31, 11, 0, tzinfo=timezone.utc), booked)

    def test_a_run_ending_just_outside_the_window_books_the_following_deadline(self):
        from squadtools.fpl_deadline import deadline_for_following_run
        events = [
            {'id': 10, 'deadline_time': '2026-11-06T18:30:00Z'},
            {'id': 11, 'deadline_time': '2026-11-21T11:00:00Z'},
        ]
        # The GW10 run started 13:30 (5 hours early) and finished 15:28.
        finished = datetime(2026, 11, 6, 15, 28, tzinfo=timezone.utc)
        self.assertEqual(datetime(2026, 11, 21, 11, 0, tzinfo=timezone.utc),
                         deadline_for_following_run(events, finished, hours_before=5))
        # A run more than 5 hours ahead still books GW10 itself.
        early = datetime(2026, 11, 6, 9, 0, tzinfo=timezone.utc)
        self.assertEqual(datetime(2026, 11, 6, 18, 30, tzinfo=timezone.utc),
                         deadline_for_following_run(events, early, hours_before=5))


class PlayedChipsTests(unittest.TestCase):
    def test_parses_fpl_my_team_chip_names(self):
        user_json = {
            'fpl_api': {
                'picks': {
                    'chips': [
                        {
                            'name': 'wildcard',
                            'status_for_entry': 'played',
                            'played_by_entry': [6],
                        },
                        {
                            'name': 'freehit',
                            'status_for_entry': 'available',
                            'played_by_entry': [],
                        },
                        {
                            'name': 'bboost',
                            'status_for_entry': 'played',
                            'played_by_entry': [7],
                        },
                        {
                            'name': '3xc',
                            'status_for_entry': 'available',
                            'played_by_entry': [],
                        },
                    ]
                }
            }
        }
        played = played_chips_from_user_json(user_json, season_start_year=26)
        self.assertEqual((6,), played.wildcards)
        self.assertEqual((7,), played.bench_boosts)
        self.assertEqual((), played.free_hits)
        self.assertEqual((), played.triple_captains)


class PredictionsEmailTests(unittest.TestCase):
    def test_email_includes_chip_and_transfer_strategy(self):
        opt = SimpleNamespace(
            squad=SimpleNamespace(next_gameweek=8, built_from_scratch=False),
            player_dict={},
            transfers={
                'scheme': (1,),
                'transfer_tuple': (),
            },
            chip_calendar=ChipCalendar(
                wildcards=(8,),
                bench_boosts=(9,),
                free_hits=(12,),
                triple_captains=(19,),
            ),
            chip_reason={'wildcard_gameweek': 8, 'bench_boost_after_wildcard': 9},
            horizon_plan=[
                {
                    'gameweek': 8,
                    'chip': 'wildcard',
                    'transfers': ['A -> B', 'C -> D'],
                },
                {
                    'gameweek': 9,
                    'chip': 'bench_boost',
                    'transfers': [],
                },
            ],
            recommended_squad=[],
            get_this_week_transfers=lambda: [],
            get_transfers=lambda: [
                {'gameweek': 8, 'chip': 'wildcard', 'transfers': ['A -> B']},
            ],
        )
        html = build_html_report(opt, applied=False)
        text = build_text_report(opt, applied=False)
        self.assertIn('Chip strategy', html)
        self.assertIn('Wildcard', html)
        self.assertIn('Transfer strategy', html)
        self.assertIn('A -&gt; B', html)
        self.assertIn('Chip strategy', text)
        self.assertIn('Transfer strategy', text)
        self.assertIn('GW 8 [Wildcard]', text)
        self.assertIn('A -> B', text)


class PrevPp90Tests(unittest.TestCase):
    def test_skips_dnps_and_pools_last_five_appearances(self):
        def match(minutes, points):
            return SimpleNamespace(
                minutes=minutes,
                points=points,
                game_stats={'minutes': minutes, 'total_points': points},
            )

        player = SimpleNamespace(matches=[
            match(90, 2),
            match(90, 6),
            match(0, 0),
            match(90, 8),
            match(45, 4),
            match(90, 5),
            match(90, 10),
        ])
        # Last 5 form appearances, raw FPL total_points like the model's
        # pooled per-90 window: 6+8+4+5+10 over 90+90+45+90+90. The DNP and
        # the oldest 90/2 row fall outside the window.
        self.assertAlmostEqual(
            33.0 * 90.0 / 405.0,
            _last_n_appearance_pp90(player, n=5),
        )

    def test_uses_model_snapshot_when_present(self):
        player = SimpleNamespace(
            past_stats={'total_points': {'last_5_avg': 7.4}},
            matches=[],
        )
        self.assertAlmostEqual(7.4, _last_n_appearance_pp90(player))

    def test_skips_sub_20_minute_cameos_from_form_window(self):
        player = SimpleNamespace(matches=[
            SimpleNamespace(minutes=0, points=0, game_stats={'minutes': 0, 'total_points': 0}),
            SimpleNamespace(minutes=8, points=2, game_stats={'minutes': 8, 'total_points': 2}),
            SimpleNamespace(minutes=90, points=6, game_stats={'minutes': 90, 'total_points': 6}),
        ])
        # Only the 90-minute row qualifies: 6 raw points per 90.
        self.assertAlmostEqual(6.0, _last_n_appearance_pp90(player))

    def test_cameo_only_form_window_is_zero(self):
        player = SimpleNamespace(matches=[
            SimpleNamespace(minutes=1, points=1, game_stats={'minutes': 1, 'total_points': 1}),
            SimpleNamespace(minutes=1, points=1, game_stats={'minutes': 1, 'total_points': 1}),
            SimpleNamespace(minutes=8, points=2, game_stats={'minutes': 8, 'total_points': 2}),
        ])
        self.assertEqual(0.0, _last_n_appearance_pp90(player))


class TransferEmailTests(unittest.TestCase):
    def test_email_shows_xi_gain_separate_from_player_delta(self):
        out_player = SimpleNamespace(
            first_name='Out', last_name='Player', team_name='NFO',
            position=3, current_cost=55, future_matches=[],
            get_points_within_n_gameweeks=lambda gw, n: 20.0,
        )
        in_player = SimpleNamespace(
            first_name='In', last_name='Player', team_name='MCI',
            position=3, current_cost=70, future_matches=[],
            get_points_within_n_gameweeks=lambda gw, n: 12.0,
        )
        transfer = SimpleNamespace(
            player_out=out_player,
            player_in=in_player,
            selling_price=55,
            purchase_price=70,
        )
        opt = SimpleNamespace(
            squad=SimpleNamespace(next_gameweek=2, built_from_scratch=False),
            player_dict={},
            transfers={'scheme': (1,), 'transfer_tuple': (transfer,)},
            chip_calendar=ChipCalendar(),
            chip_reason={},
            horizon_plan=[],
            this_week_scores={
                'hold_current': 40.0,
                'selected_current': 42.0,
                'hold_horizon': 180.0,
                'selected_horizon': 188.0,
                'current_gain': 2.0,
                'horizon_gain': 8.0,
            },
            recommended_squad=[],
            get_this_week_transfers=lambda: [transfer],
            get_transfers=lambda: [],
        )
        html = build_html_report(opt, applied=True)
        text = build_text_report(opt, applied=True)
        self.assertIn('Player 5GW Δ', html)
        self.assertIn('XI + captain', html)
        self.assertIn('Δ +8.0', html)
        self.assertIn('player Δ -8.0', text)
        self.assertIn('5GW horizon: hold 180.0 -> 188.0 (Δ +8.0)', text)


class PredictionRowsTests(unittest.TestCase):
    # Column layout from csv_headers: the per-match blocks start after Scale.
    BASIS = 4
    PRED = 5
    SCALE = 9
    M1_PP90 = 11
    M1_ADJ = 13

    def _player(self, h1_points=4.0, scale=1.5):
        match = SimpleNamespace(
            gameweek=3,
            opponent='LIV',
            minutes=90,
            game_stats={'minutes': 90},
            points=6.0,
            predicted_pp90=6.0,
            chance_of_playing=None,
            points_error=0.0,
            h1_points=h1_points,
            horizon_scale=scale,
        )
        return SimpleNamespace(
            first_name='A',
            last_name='B',
            team_name='ARS',
            position=3,
            current_cost=70,
            future_matches=[match],
        )

    def test_each_player_gets_a_final_and_a_raw_row(self):
        from squadtools.predictions_report import player_rows

        final, raw = player_rows(self._player(), 1)
        self.assertEqual('final', final[self.BASIS])
        self.assertEqual('raw 1GW', raw[self.BASIS])

    def test_final_is_the_raw_row_times_the_quoted_scale(self):
        from squadtools.predictions_report import player_rows

        final, raw = player_rows(self._player(h1_points=4.0, scale=1.5), 1)
        self.assertEqual('1.50', final[self.SCALE])
        self.assertEqual('1.50', raw[self.SCALE])
        self.assertEqual('6.0', final[self.M1_ADJ])
        self.assertEqual('4.0', raw[self.M1_ADJ])
        self.assertAlmostEqual(
            float(final[self.M1_ADJ]),
            float(raw[self.M1_ADJ]) * float(final[self.SCALE]),
        )
        # pp90 carries the same factor, since both rows share predicted minutes.
        self.assertEqual('6.0', final[self.M1_PP90])
        self.assertEqual('4.0', raw[self.M1_PP90])

    def test_scale_divides_out_even_where_the_minutes_mapping_bites(self):
        from squadtools.predictions_report import _match_scores

        # 88 predicted minutes gets mapped up to 90 for scoring. The raw row needs
        # the same mapping, or the rows stop differing by only the scale factor.
        match = SimpleNamespace(
            gameweek=3, opponent='LIV', minutes=88, game_stats={'minutes': 88},
            points=6.0, predicted_pp90=6.0 * 90.0 / 88.0, chance_of_playing=75,
            points_error=0.0, h1_points=4.0, horizon_scale=1.5,
        )
        _final_pp90, final_adj = _match_scores(match, raw=False)
        _raw_pp90, raw_adj = _match_scores(match, raw=True)
        self.assertAlmostEqual(1.5, final_adj / raw_adj)

    def test_raw_row_is_blank_when_the_single_week_points_were_not_kept(self):
        from squadtools.predictions_report import player_rows

        final, raw = player_rows(self._player(h1_points=None, scale=None), 1)
        self.assertEqual('6.0', final[self.M1_ADJ])
        self.assertEqual('', raw[self.M1_ADJ])
        self.assertEqual('', raw[self.M1_PP90])
        self.assertEqual('', raw[self.PRED])
        self.assertEqual('', final[self.SCALE])

    def test_table_has_a_basis_and_scale_column(self):
        from squadtools.predictions_report import csv_headers

        headers = csv_headers(1)
        self.assertEqual('Basis', headers[self.BASIS])
        self.assertEqual('Scale', headers[self.SCALE])
        self.assertEqual('M1 adj', headers[self.M1_ADJ])


class PredictionPoolFromPlayersTests(unittest.TestCase):
    def test_records_fixture_ids_from_future_matches(self):
        from squadtools.live_strategy import prediction_pool_from_players

        match = SimpleNamespace(
            gameweek=3,
            fixture_id=77,
            minutes=0,
            game_stats={},
            points=0,
            predicted_pp90=0,
        )
        player = SimpleNamespace(
            id=1,
            team_id=2,
            position=3,
            code=9,
            current_cost=50,
            first_name='A',
            last_name='B',
            future_matches=[match],
        )
        pool = prediction_pool_from_players({1: player}, selectable_ids={1})
        self.assertEqual({77}, pool[1].fixture_ids_by_gameweek[3])


if __name__ == '__main__':
    unittest.main()
