import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from simulation.rebuild_and_sweep import (
    DEFAULT_SEASONS,
    _predict_argv,
    apply_h1_poisson_defaults,
    predict_name,
    resolve_seasons,
)
from squadtools.chip_strategy import CHIP_BAR_REFERENCE_HORIZON
from squadtools.simulation_strategy import MIN_GAIN_REFERENCE_HORIZON
from simulation.sweep import (
    H1_POISSON_TRIALS,
    TRIAL_BY_NAME,
    TRIALS,
    _describe_prediction_set,
    apply_results_tag,
    config_kwargs,
    chips_used,
    consistency_lines,
    latest_by_trial_season,
    selected_trials,
    trial_overrides,
    write_comparison,
)


def _trial(name):
    for trial in TRIALS:
        if trial['name'] == name:
            return trial
    raise AssertionError('missing trial %s' % name)


class SimulationSweepTests(unittest.TestCase):
    def test_grid_has_twenty_named_trials(self):
        self.assertEqual(20, len(TRIALS))
        names = [trial['name'] for trial in TRIALS]
        self.assertEqual(20, len(set(names)))
        self.assertEqual('01_hz3', names[0])
        self.assertEqual('20_dgw_canary', names[-1])
        # Both grid axes are stated on every trial so the comparison table
        # shows the horizon and the bar even at their baseline values.
        for trial in TRIALS:
            overrides = trial_overrides(trial)
            self.assertIn('hz', overrides)
            self.assertIn('wc', overrides)

    def test_baseline_cell_starts_the_dead_knobs_at_zero(self):
        base = config_kwargs(_trial('02_hz4'), 5, 25)
        self.assertEqual(4, base['horizon_gameweeks'])
        self.assertEqual(0.0, base['fh_dgw_weight'])
        self.assertEqual(0.0, base['wc_dgw_weight'])
        self.assertEqual(0.0, base['wc_rotation_weight'])
        self.assertEqual(0.0, base['wc_early_weight'])
        self.assertTrue(base['roll_free_transfers'])
        self.assertEqual(3, base['earliest_chip_gameweek'])
        self.assertEqual(25, base['season_start_year'])

    def test_hz4_cell_reproduces_the_previous_grids_mid_trial(self):
        """The old grid ran hz 4 against a 45 WC bar, 10 FH bar, 3.0 min gain.

        Those were absolute totals; the same behaviour now comes from quoted
        bars scaled by 4/5, which is what keeps the two runs comparable.
        """
        cell = config_kwargs(_trial('02_hz4'), 5, 25)
        self.assertEqual(4, cell['horizon_gameweeks'])
        bar_scale = cell['horizon_gameweeks'] / CHIP_BAR_REFERENCE_HORIZON
        self.assertAlmostEqual(45.0, cell['wc_threshold'] * bar_scale)
        self.assertAlmostEqual(10.0, cell['fh_threshold'] * bar_scale)
        self.assertAlmostEqual(10.0, cell['fh_recovery_threshold'] * bar_scale)
        gain_scale = cell['horizon_gameweeks'] / MIN_GAIN_REFERENCE_HORIZON
        self.assertAlmostEqual(3.0, cell['min_horizon_gain'] * gain_scale)

    def test_star_walks_both_axes_and_pins_the_corners(self):
        horizons = [
            config_kwargs(_trial(name), 5, 25)['horizon_gameweeks']
            for name in ('01_hz3', '02_hz4', '03_hz5', '04_hz6')
        ]
        self.assertEqual([3, 4, 5, 6], horizons)
        lo = config_kwargs(_trial('05_wc_lo'), 5, 25)
        hi = config_kwargs(_trial('06_wc_hi'), 5, 25)
        self.assertLess(lo['wc_threshold'], hi['wc_threshold'])
        # A corner reuses an axis bar at another horizon, so the additive
        # prediction from the two axes is what it gets compared against.
        for name, bar, horizon in (
            ('07_hz3_lo', lo['wc_threshold'], 3),
            ('08_hz3_hi', hi['wc_threshold'], 3),
            ('09_hz6_lo', lo['wc_threshold'], 6),
            ('10_hz6_hi', hi['wc_threshold'], 6),
        ):
            corner = config_kwargs(_trial(name), 5, 25)
            self.assertEqual(bar, corner['wc_threshold'])
            self.assertEqual(horizon, corner['horizon_gameweeks'])

    def test_no_trial_uses_a_horizon_we_cannot_afford(self):
        """hz 8 more than doubled hz 6's cost, which was already 3x hz 4."""
        for trial in TRIALS:
            kwargs = config_kwargs(trial, 5, 25)
            self.assertLessEqual(kwargs['horizon_gameweeks'], 6, trial['name'])

    def test_winner_trial_is_exactly_its_two_factors_at_the_same_horizon(self):
        patience = config_kwargs(_trial('12_mh_max_hz6'), 5, 25)
        rotation = config_kwargs(_trial('13_rot_hi_hz6'), 5, 25)
        both = config_kwargs(_trial('14_stuck_hz6'), 5, 25)
        arm = config_kwargs(_trial('04_hz6'), 5, 25)
        self.assertEqual(6, both['horizon_gameweeks'])
        self.assertEqual(patience['min_horizon_gain'], both['min_horizon_gain'])
        self.assertEqual(
            rotation['wc_rotation_weight'], both['wc_rotation_weight']
        )
        # Each factor is alone in its own trial, so the triple is decomposable.
        self.assertEqual(0.0, patience['wc_rotation_weight'])
        self.assertEqual(arm['min_horizon_gain'], rotation['min_horizon_gain'])

    def test_rotation_trial_and_its_flat_bar_equivalent_agree(self):
        """Four rotting players at the rotation weight is the cut 17 makes."""
        rot = config_kwargs(_trial('16_rot_hi'), 5, 25)
        flat = config_kwargs(_trial('17_wc_equiv'), 5, 25)
        rotting = 4
        self.assertAlmostEqual(
            rot['wc_threshold'] - rotting * rot['wc_rotation_weight'],
            flat['wc_threshold'],
        )
        self.assertEqual(0.0, flat['wc_rotation_weight'])
        self.assertEqual(
            rot['horizon_gameweeks'], flat['horizon_gameweeks']
        )

    def test_from_and_only_select_trials(self):
        only = selected_trials(mock.Mock(only=2, start=1))
        self.assertEqual([(2, TRIALS[1])], only)
        rest = selected_trials(mock.Mock(only=None, start=2))
        self.assertEqual(19, len(rest))
        self.assertEqual((2, TRIALS[1]), rest[0])
        self.assertEqual((20, TRIALS[19]), rest[-1])

    def test_trials_flag_selects_named_last_run_variants(self):
        args = mock.Mock(
            only=None,
            start=1,
            baseline=False,
            trials=','.join(H1_POISSON_TRIALS),
        )
        chosen = selected_trials(args)
        names = [trial['name'] for _index, trial in chosen]
        self.assertEqual(list(H1_POISSON_TRIALS), names)
        for name in H1_POISSON_TRIALS:
            self.assertIn(name, TRIAL_BY_NAME)

    def test_unknown_trial_name_is_rejected(self):
        with self.assertRaises(SystemExit):
            selected_trials(mock.Mock(
                only=None, start=1, baseline=False, trials='not_a_trial'
            ))

    def test_no_horizon_cache_does_not_warn_about_horizon_reps(self):
        db = mock.Mock()
        db.get_prediction_set.return_value = {
            'id': 9,
            'name': 'h1-poisson',
            'model_config_hash': 'abc',
            'config': {
                'minutes_reps': 5,
                'horizon_reps': 0,
                'horizon_model': False,
                'points_loss': 'poisson',
            },
        }
        with mock.patch('builtins.print') as printed:
            _describe_prediction_set(db, 9)
        text = ' '.join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn('points_loss=poisson', text)
        self.assertNotIn('want 5', text)

    def test_resume_skips_finished_trials_for_that_season_only(self):
        from simulation.sweep import remaining_trials

        chosen = list(enumerate(TRIALS, start=1))
        rows = [
            {'name': '01_hz3', 'season': 25},
            {'name': '02_hz4', 'season': 25},
            {'name': '04_hz6', 'season': 25},
            {'name': '01_hz3', 'season': 24},
        ]
        remaining, skipped = remaining_trials(chosen, 25, rows=rows)
        skipped_names = [trial['name'] for _index, trial in skipped]
        remaining_names = [trial['name'] for _index, trial in remaining]
        self.assertEqual(['01_hz3', '02_hz4', '04_hz6'], skipped_names)
        self.assertIn('03_hz5', remaining_names)
        self.assertIn('09_hz6_lo', remaining_names)
        self.assertEqual(17, len(remaining))
        later, later_skipped = remaining_trials(chosen, 24, rows=rows)
        self.assertEqual(['01_hz3'], [trial['name'] for _index, trial in later_skipped])
        self.assertEqual(19, len(later))

    def test_resume_restarts_trials_that_never_reached_the_log(self):
        from simulation.sweep import remaining_trials

        chosen = [(9, TRIALS[8]), (10, TRIALS[9])]
        remaining, skipped = remaining_trials(
            chosen, 25, rows=[{'name': '01_hz3', 'season': 25}]
        )
        self.assertEqual([], skipped)
        self.assertEqual(chosen, remaining)

    def test_baseline_uses_pre_sweep_knobs_not_the_grid(self):
        from simulation.sweep import BASELINE, selected_trials

        chosen = selected_trials(mock.Mock(baseline=True, only=None, start=1))
        self.assertEqual([(0, BASELINE)], chosen)
        kwargs = config_kwargs(BASELINE, 6, 25)
        self.assertEqual('00_baseline', BASELINE['name'])
        self.assertFalse(kwargs['roll_free_transfers'])
        self.assertEqual(30.0, kwargs['wc_threshold'])
        self.assertEqual(11.0, kwargs['fh_threshold'])
        self.assertEqual(5, kwargs['horizon_gameweeks'])
        self.assertEqual(2.0, kwargs['min_horizon_gain'])
        self.assertEqual(0.0, kwargs['min_current_gw_gain'])
        self.assertEqual(1.3, kwargs['tc_above_average_threshold'])
        self.assertEqual(10, kwargs['tc_window'])
        self.assertEqual(0.3, kwargs['bench_fill_weight'])
        self.assertEqual((0.5, 0.3, 0.1), kwargs['bench_weights'])
        self.assertEqual(0.2, kwargs['reserve_keeper_weight'])
        self.assertEqual(5, kwargs['earliest_chip_gameweek'])
        self.assertEqual(0.0, kwargs['fh_dgw_weight'])
        self.assertEqual(0.0, kwargs['wc_dgw_weight'])
        self.assertEqual(0.0, kwargs['wc_rotation_weight'])
        self.assertEqual(0.0, kwargs['wc_early_weight'])
        self.assertTrue(kwargs['chip_strategy'])

    def test_comparison_table_ranks_by_points_and_shows_season(self):
        rows = [
            {
                'name': '01_mid',
                'season': 24,
                'total_points': 2000,
                'transfers_made': 40,
                'hold_gameweeks': 8,
                'options': {'roll': True, 'fh_dgw': 2.0},
                'chips_label': 'WC 5,22',
                'overall_rank': {'rank': 1900000, 'top_percent': 14.8},
            },
            {
                'name': '03_dgw_heavy',
                'season': 25,
                'total_points': 2100,
                'transfers_made': 64,
                'hold_gameweeks': 0,
                'options': {'roll': True, 'fh_dgw': 4.0},
                'chips_label': 'WC 5,22',
                'overall_rank': {'rank': 800000, 'top_percent': 6.2},
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl = Path(temp_dir) / 'hyperparam_sweep.jsonl'
            table = Path(temp_dir) / 'hyperparam_sweep.txt'
            with mock.patch('simulation.sweep.JSONL_PATH', jsonl), mock.patch(
                'simulation.sweep.TABLE_PATH', table
            ), mock.patch('simulation.sweep.RESULTS_DIR', Path(temp_dir)):
                jsonl.write_text(
                    '\n'.join(json.dumps(row) for row in rows) + '\n',
                    encoding='utf-8',
                )
                text = write_comparison()
                self.assertIn('03_dgw_heavy', text)
                self.assertLess(text.find('03_dgw_heavy'), text.find('01_mid'))
                self.assertIn('800000', text)
                self.assertIn(' 25 ', text)
                self.assertIn(' 24 ', text)
                self.assertTrue(
                    table.read_text(encoding='utf-8').startswith(
                        'Simulation hyperparameter sweep'
                    )
                )

    def test_stale_minutes_reps_warns_before_the_sweep(self):
        db = mock.Mock()
        db.get_prediction_set.return_value = {
            'id': 2,
            'name': 'old-cache',
            'model_config_hash': 'abc123def456',
            'config': {'minutes_reps': 1, 'horizon_reps': 5},
        }
        with mock.patch('builtins.print') as printed:
            _describe_prediction_set(db, 2)
        text = ' '.join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn('minutes_reps=1', text)
        self.assertIn('want 5', text)

    def test_chips_used_groups_gameweeks(self):
        self.assertEqual(
            {'wildcard': [5, 20], 'triple_captain': [19]},
            chips_used([
                {'gameweek': 5, 'chip': 'wildcard'},
                {'gameweek': 6, 'chip': None},
                {'gameweek': 19, 'chip': 'triple_captain'},
                {'gameweek': 20, 'chip': 'wildcard'},
            ]),
        )


class RebuildAndSweepTests(unittest.TestCase):
    def test_default_runs_24_then_25_on_the_same_plan(self):
        from simulation.rebuild_and_sweep import main as rebuild_main

        self.assertEqual((24, 25), DEFAULT_SEASONS)
        args = mock.Mock(season=None, seasons=(24, 25))
        self.assertEqual((24, 25), resolve_seasons(args))
        with mock.patch(
            'simulation.rebuild_and_sweep.build_predictions'
        ) as build:
            with mock.patch('simulation.rebuild_and_sweep.run_sweep') as sweep:
                sweep.return_value = 0
                self.assertEqual(0, rebuild_main(['--dry-run']))
        build.assert_not_called()
        self.assertEqual(
            [
                mock.call(['--season', '24', '--dry-run', '--workers', '1']),
                mock.call(['--season', '25', '--dry-run', '--workers', '1']),
            ],
            sweep.call_args_list,
        )

    def test_skip_build_sweeps_each_season(self):
        from simulation.rebuild_and_sweep import main as rebuild_main

        with mock.patch(
            'simulation.rebuild_and_sweep.build_predictions'
        ) as build:
            with mock.patch('simulation.rebuild_and_sweep.run_sweep') as sweep:
                sweep.return_value = 0
                self.assertEqual(
                    0, rebuild_main(['--skip-build', '--from', '10'])
                )
        build.assert_not_called()
        self.assertEqual(
            [
                mock.call(['--season', '24', '--from', '10', '--workers', '1']),
                mock.call(['--season', '25', '--from', '10', '--workers', '1']),
            ],
            sweep.call_args_list,
        )

    def test_single_season_flag_still_runs_one_year(self):
        from simulation.rebuild_and_sweep import main as rebuild_main

        with mock.patch(
            'simulation.rebuild_and_sweep.build_predictions'
        ) as build:
            with mock.patch('simulation.rebuild_and_sweep.run_sweep') as sweep:
                sweep.return_value = 0
                self.assertEqual(0, rebuild_main(['--dry-run', '--season', '24']))
        build.assert_not_called()
        sweep.assert_called_once_with(['--season', '24', '--dry-run', '--workers', '1'])
        self.assertEqual('strict-historical-24-25-form3-7', predict_name(24))
        argv = _predict_argv(predict_name(24), 24)
        self.assertEqual('24', argv[argv.index('--seasons') + 1])
        self.assertIn('--train-once-per-season', argv)
        self.assertNotIn('--no-horizon-model', argv)
        self.assertNotIn('--points-loss', argv)

    def test_h1_poisson_pack_trains_without_horizon_and_sweeps_last_run_variants(self):
        from simulation.rebuild_and_sweep import main as rebuild_main

        with mock.patch(
            'simulation.rebuild_and_sweep.build_predictions'
        ) as build:
            with mock.patch('simulation.rebuild_and_sweep.run_sweep') as sweep:
                sweep.return_value = 0
                self.assertEqual(
                    0, rebuild_main(['--dry-run', '--h1-poisson', '--season', '25'])
                )
        build.assert_not_called()
        sweep_argv = sweep.call_args.args[0]
        self.assertEqual('25', sweep_argv[sweep_argv.index('--season') + 1])
        self.assertEqual(
            ','.join(H1_POISSON_TRIALS),
            sweep_argv[sweep_argv.index('--trials') + 1],
        )
        self.assertEqual('h1_poisson_mw', sweep_argv[sweep_argv.index('--tag') + 1])

        args = apply_h1_poisson_defaults(
            mock.Mock(
                h1_poisson=True,
                points_reps=None,
                trials=None,
                tag=None,
                no_horizon_model=False,
                points_loss='huber',
                points_minutes_weight=False,
            )
        )
        self.assertEqual('strict-historical-25-26-h1-poisson-mw', predict_name(25, args))
        predict = _predict_argv(predict_name(25, args), 25, args)
        self.assertIn('--no-horizon-model', predict)
        self.assertEqual('poisson', predict[predict.index('--points-loss') + 1])
        self.assertIn('--points-minutes-weight', predict)
        self.assertEqual('5', predict[predict.index('--points-reps') + 1])
        self.assertEqual('1', predict[predict.index('--horizon-reps') + 1])

    def test_old_winners_pack_replays_horizon_cells_on_existing_poisson_preds(self):
        from simulation.rebuild_and_sweep import H1_POISSON_OLD_WINNER_TRIALS
        from simulation.rebuild_and_sweep import main as rebuild_main

        with mock.patch(
            'simulation.rebuild_and_sweep.build_predictions'
        ) as build:
            with mock.patch('simulation.rebuild_and_sweep.run_sweep') as sweep:
                sweep.return_value = 0
                self.assertEqual(
                    0,
                    rebuild_main([
                        '--dry-run',
                        '--h1-poisson',
                        '--old-winners',
                        '--skip-build',
                        '--season',
                        '25',
                        '--prediction-set-id',
                        '7',
                    ]),
                )
        build.assert_not_called()
        sweep_argv = sweep.call_args.args[0]
        self.assertEqual(
            ','.join(H1_POISSON_OLD_WINNER_TRIALS),
            sweep_argv[sweep_argv.index('--trials') + 1],
        )
        self.assertEqual(
            'h1_poisson_old_winners',
            sweep_argv[sweep_argv.index('--tag') + 1],
        )
        self.assertEqual('7', sweep_argv[sweep_argv.index('--prediction-set-id') + 1])

    def test_resume_flag_is_forwarded_to_the_sweep(self):
        from simulation.rebuild_and_sweep import main as rebuild_main

        with mock.patch(
            'simulation.rebuild_and_sweep.build_predictions'
        ) as build:
            with mock.patch('simulation.rebuild_and_sweep.run_sweep') as sweep:
                sweep.return_value = 0
                self.assertEqual(
                    0,
                    rebuild_main(
                        [
                            '--skip-build',
                            '--seasons',
                            '25,24,23',
                            '--workers',
                            '4',
                            '--resume',
                        ]
                    ),
                )
        build.assert_not_called()
        self.assertEqual(
            [
                mock.call(
                    ['--season', '25', '--resume', '--workers', '4']
                ),
                mock.call(
                    ['--season', '24', '--resume', '--workers', '4']
                ),
                mock.call(
                    ['--season', '23', '--resume', '--workers', '4']
                ),
            ],
            sweep.call_args_list,
        )

    def test_workers_flag_is_forwarded_to_the_sweep(self):
        from simulation.rebuild_and_sweep import main as rebuild_main

        with mock.patch(
            'simulation.rebuild_and_sweep.build_predictions'
        ) as build:
            with mock.patch('simulation.rebuild_and_sweep.run_sweep') as sweep:
                sweep.return_value = 0
                self.assertEqual(
                    0,
                    rebuild_main(['--dry-run', '--seasons', '25', '--workers', '4']),
                )
        build.assert_not_called()
        sweep.assert_called_once_with(
            ['--season', '25', '--dry-run', '--workers', '4']
        )

    def test_baseline_flag_is_forwarded_to_the_sweep(self):
        from simulation.rebuild_and_sweep import main as rebuild_main

        with mock.patch(
            'simulation.rebuild_and_sweep.build_predictions'
        ) as build:
            with mock.patch('simulation.rebuild_and_sweep.run_sweep') as sweep:
                sweep.return_value = 0
                self.assertEqual(
                    0,
                    rebuild_main(
                        ['--dry-run', '--skip-build', '--seasons', '25', '--baseline']
                    ),
                )
        build.assert_not_called()
        sweep.assert_called_once_with(
            ['--season', '25', '--dry-run', '--baseline', '--workers', '1']
        )

    def test_sweep_never_asks_cbc_for_a_thread_count(self):
        from simulation.sweep import _configure_workers

        with mock.patch.dict(os.environ, {'SIM_CBC_THREADS': '8'}):
            self.assertEqual(1, _configure_workers(1))
            self.assertNotIn('SIM_CBC_THREADS', os.environ)
            self.assertEqual(4, _configure_workers(4))
            self.assertNotIn('SIM_CBC_THREADS', os.environ)


    def test_parallel_workers_run_isolated_python_subprocesses(self):
        from simulation.sweep import _run_trials_parallel

        row = {
            'name': '01_mid',
            'total_points': 2000,
            'overall_rank': {},
            'chips': {},
            'chips_label': '-',
        }
        with mock.patch(
            'simulation.sweep._await_trial_process',
            return_value=(1, row, '01_mid_s25.log'),
        ) as await_job:
            with mock.patch('simulation.sweep._append_jsonl'):
                with mock.patch('simulation.sweep.write_comparison'):
                    with mock.patch('simulation.sweep._print_trial_ok'):
                        with mock.patch(
                            'simulation.sweep._use_real_python_executable'
                        ) as use_python:
                            _run_trials_parallel(
                                [(1, TRIALS[0])], 6, 25, 4
                            )
        use_python.assert_called_once()
        await_job.assert_called_once_with((1, TRIALS[0], 6, 25), True)

    def test_parallel_workers_do_not_stream_when_several_trials_run(self):
        from simulation.sweep import _run_trials_parallel

        row = {
            'name': '01_mid',
            'total_points': 2000,
            'overall_rank': {},
            'chips': {},
            'chips_label': '-',
        }
        with mock.patch(
            'simulation.sweep._await_trial_process',
            return_value=(1, row, '01_mid_s25.log'),
        ) as await_job:
            with mock.patch('simulation.sweep._append_jsonl'):
                with mock.patch('simulation.sweep.write_comparison'):
                    with mock.patch('simulation.sweep._print_trial_ok'):
                        with mock.patch(
                            'simulation.sweep._use_real_python_executable'
                        ):
                            _run_trials_parallel(
                                [(1, TRIALS[0]), (2, TRIALS[1])], 6, 25, 2
                            )
        self.assertEqual(2, await_job.call_count)
        for call in await_job.call_args_list:
            self.assertEqual(False, call.args[1])

    def test_worker_env_strips_debugger_variables(self):
        from simulation.sweep import _worker_env

        with mock.patch.dict(
            os.environ,
            {
                'DEBUGPY_RUNNING': 'true',
                'PYDEVD_USE_CYTHON': 'NO',
                'PYTHONHOME': r'C:\Python38',
                'PYTHONPATH': r'C:\debugpy\libs;C:\project',
                'SIM_CBC_THREADS': '8',
                'KEEP': '1',
            },
        ):
            env = _worker_env()
        self.assertNotIn('DEBUGPY_RUNNING', env)
        self.assertNotIn('PYDEVD_USE_CYTHON', env)
        self.assertNotIn('PYTHONHOME', env)
        self.assertEqual('1', env['KEEP'])
        self.assertNotIn('SIM_CBC_THREADS', env)
        self.assertEqual(sys.prefix, env['VIRTUAL_ENV'])
        self.assertEqual(r'C:\project', env['PYTHONPATH'])

    def test_worker_command_starts_an_internal_sweep_child(self):
        from simulation.sweep import _worker_command

        apply_results_tag(None)
        with mock.patch(
            'simulation.sweep._worker_python', return_value='python.exe'
        ):
            cmd = _worker_command(2, 6, 25)
        self.assertEqual('python.exe', cmd[0])
        self.assertIn('--internal-worker', cmd)
        self.assertEqual(['--only', '2'], cmd[-2:])
        self.assertNotIn('--tag', cmd)

    def test_worker_command_passes_baseline_instead_of_only(self):
        from simulation.sweep import BASELINE, _worker_command

        with mock.patch(
            'simulation.sweep._worker_python', return_value='python.exe'
        ):
            cmd = _worker_command(0, 6, 25, BASELINE)
        self.assertIn('--baseline', cmd)
        self.assertNotIn('--only', cmd)

    def test_real_python_executable_prefers_venv_interpreter(self):
        from simulation.sweep import _use_real_python_executable

        previous = sys.executable
        try:
            with mock.patch(
                'simulation.sweep._worker_python',
                return_value='/venv/Scripts/python.exe',
            ):
                with mock.patch('simulation.sweep.sys') as fake_sys:
                    fake_sys.executable = '/debugpy/launcher'
                    _use_real_python_executable()
                    self.assertEqual(
                        '/venv/Scripts/python.exe', fake_sys.executable
                    )
        finally:
            pass
        self.assertEqual(previous, sys.executable)

    def test_worker_python_uses_venv_not_system_base(self):
        from simulation.sweep import _worker_python

        with tempfile.TemporaryDirectory() as tmp:
            scripts = Path(tmp) / 'Scripts'
            scripts.mkdir()
            venv_py = scripts / 'python.exe'
            venv_py.write_bytes(b'')
            with mock.patch('simulation.sweep.sys') as fake_sys:
                fake_sys.prefix = tmp
                fake_sys.executable = (
                    r'C:\Users\hal\.cursor\extensions\debugpy\launcher'
                )
                fake_sys._base_executable = r'C:\Python38\python.exe'
                with mock.patch('simulation.sweep.os.name', 'nt'):
                    self.assertEqual(str(venv_py), _worker_python())

    def test_in_process_runner_uses_the_grid_baseline_cell(self):
        from simulation import __main__ as sim_main

        self.assertIs(TRIAL_BY_NAME['02_hz4'], sim_main.TRIAL)
        kwargs = config_kwargs(sim_main.TRIAL, 6, 25)
        self.assertTrue(kwargs['roll_free_transfers'])
        self.assertEqual(4, kwargs['horizon_gameweeks'])
        self.assertEqual(3, kwargs['earliest_chip_gameweek'])
        self.assertEqual(11.25 * CHIP_BAR_REFERENCE_HORIZON, kwargs['wc_threshold'])
        self.assertEqual(2.5 * CHIP_BAR_REFERENCE_HORIZON, kwargs['fh_threshold'])


class WorkerShutdownTests(unittest.TestCase):
    def setUp(self):
        from simulation import sweep

        self.sweep = sweep
        sweep._LIVE_WORKERS.clear()
        self.addCleanup(sweep._LIVE_WORKERS.clear)

    def _popen_calls(self, os_name):
        fake = mock.Mock()
        fake.pid = 4321
        with mock.patch.object(self.sweep.os, 'name', os_name):
            with mock.patch.object(
                self.sweep.subprocess, 'Popen', return_value=fake
            ) as popen:
                with mock.patch.object(
                    self.sweep, '_kill_workers_with_this_process'
                ):
                    with mock.patch.object(self.sweep, '_add_worker_to_job'):
                        with mock.patch.object(self.sweep, '_worker_env'):
                            with mock.patch.object(self.sweep, '_worker_cwd'):
                                self.sweep._popen_worker(['python'])
        return popen, fake

    def test_workers_stay_in_the_job_so_they_die_with_the_sweep(self):
        popen, _fake = self._popen_calls('nt')
        flags = popen.call_args.kwargs['creationflags']
        breakaway = 0x01000000
        self.assertFalse(flags & breakaway)

    def test_spawned_workers_are_tracked_for_cleanup(self):
        _popen, fake = self._popen_calls('nt')
        self.assertIn(fake, self.sweep._LIVE_WORKERS)

    def test_kill_live_workers_takes_out_the_whole_tree_on_windows(self):
        proc = mock.Mock()
        proc.pid = 777
        proc.poll.return_value = None
        self.sweep._LIVE_WORKERS.add(proc)
        with mock.patch.object(self.sweep.os, 'name', 'nt'):
            with mock.patch.object(self.sweep.subprocess, 'run') as run:
                self.sweep.kill_live_workers()
        self.assertEqual(
            ['taskkill', '/PID', '777', '/T', '/F'], run.call_args.args[0]
        )
        self.assertEqual(set(), self.sweep._LIVE_WORKERS)

    def test_finished_workers_are_not_killed_again(self):
        proc = mock.Mock()
        proc.poll.return_value = 0
        self.sweep._LIVE_WORKERS.add(proc)
        with mock.patch.object(self.sweep.os, 'name', 'nt'):
            with mock.patch.object(self.sweep.subprocess, 'run') as run:
                self.sweep.kill_live_workers()
        run.assert_not_called()
        self.assertEqual(set(), self.sweep._LIVE_WORKERS)

    def test_job_setup_is_skipped_off_windows(self):
        with mock.patch.object(self.sweep.os, 'name', 'posix'):
            self.assertIsNone(self.sweep._kill_workers_with_this_process())


class ConsistencyTableTests(unittest.TestCase):
    def _row(self, name, season, points, top):
        return {
            'name': name,
            'season': season,
            'total_points': points,
            'overall_rank': {'rank': 1000, 'top_percent': top},
        }

    def _line_for(self, lines, name):
        matches = [line for line in lines if line.startswith(name)]
        self.assertEqual(1, len(matches), lines)
        return matches[0]

    def test_trials_are_ranked_by_mean_top_percent(self):
        rows = [
            self._row('01_mid', 24, 2100, 8.0),
            self._row('01_mid', 25, 2000, 4.0),
            self._row('02_dgw_off', 24, 2200, 2.0),
            self._row('02_dgw_off', 25, 1900, 3.0),
        ]
        lines = consistency_lines(rows)
        body = [line for line in lines if line[:2].isdigit()]
        self.assertEqual(2, len(body))
        self.assertTrue(body[0].startswith('02_dgw_off'))
        self.assertTrue(body[1].startswith('01_mid'))

    def test_spread_and_worst_come_from_the_season_columns(self):
        rows = [
            self._row('01_mid', 23, 2050, 10.0),
            self._row('01_mid', 24, 2100, 8.0),
            self._row('01_mid', 25, 2000, 4.0),
        ]
        line = self._line_for(consistency_lines(rows), '01_mid')
        # n, then 23/24/25, then mean 7.33, worst 10.00, spread 6.00.
        self.assertEqual(
            ['01_mid', '3', '10.00', '8.00', '4.00', '7.33', '10.00', '6.00', '2050'],
            line.split(),
        )

    def test_only_the_newest_run_per_trial_and_season_counts(self):
        rows = [
            self._row('01_mid', 25, 900, 50.0),
            self._row('01_mid', 24, 2100, 8.0),
            self._row('01_mid', 25, 2000, 4.0),
        ]
        self.assertEqual(
            {('01_mid', 24), ('01_mid', 25)},
            set(latest_by_trial_season(rows)),
        )
        line = self._line_for(consistency_lines(rows), '01_mid')
        self.assertIn('4.00', line)
        self.assertNotIn('50.00', line)

    def test_legacy_rows_that_stored_the_season_label_still_parse(self):
        rows = [
            self._row('01_mid', '25/26', 2000, 4.0),
            self._row('01_mid', 24, 2100, 8.0),
        ]
        self.assertEqual(
            {('01_mid', 24), ('01_mid', 25)},
            set(latest_by_trial_season(rows)),
        )
        line = self._line_for(consistency_lines(rows), '01_mid')
        self.assertEqual(
            ['01_mid', '2', '8.00', '4.00', '6.00', '8.00', '4.00', '2050'],
            line.split(),
        )

    def test_a_single_season_needs_no_consistency_section(self):
        self.assertEqual([], consistency_lines([self._row('01_mid', 25, 2000, 4.0)]))

    def test_missing_seasons_and_ranks_are_blank(self):
        rows = [
            self._row('01_mid', 24, 2100, 8.0),
            self._row('01_mid', 25, 2000, 4.0),
            {'name': '02_dgw_off', 'season': 24, 'total_points': 2200},
        ]
        lines = consistency_lines(rows)
        line = self._line_for(lines, '02_dgw_off')
        self.assertEqual(['02_dgw_off', '0', '-', '-', '-', '-', '-', '2200'], line.split())


class H1PoissonPredictTests(unittest.TestCase):
    def tearDown(self):
        apply_results_tag(None)

    def test_tag_nests_logs_and_is_passed_to_workers(self):
        from simulation import sweep

        apply_results_tag('h1_poisson_mw')
        self.assertEqual('h1_poisson_mw', sweep.RESULTS_DIR.name)
        with mock.patch(
            'simulation.sweep._worker_python', return_value='python.exe'
        ):
            cmd = sweep._worker_command(2, 6, 25)
        self.assertEqual(['--tag', 'h1_poisson_mw'], cmd[-2:])

    def test_predict_skips_horizon_redistribute_when_the_model_is_absent(self):
        from simulation.season_model import FixedSeasonModels

        runner = mock.Mock()
        runner.current_season.players = []
        bundle = FixedSeasonModels('team', mock.Mock(), mock.Mock(), None, mock.Mock(), 5)
        bundle.predict(runner, already_prepared=True)
        runner.build_horizon_predict_data.assert_not_called()
        runner.redistribute_horizon_points.assert_not_called()
        runner.sync_predicted_pp90.assert_called_once()


if __name__ == '__main__':
    unittest.main()
