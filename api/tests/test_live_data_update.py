import unittest
from unittest import mock

from premierlearning.runner import Runner
from run_pipeline import _learning_argv


class LiveDataUpdateTests(unittest.TestCase):
    def test_pipeline_always_prepends_update(self):
        # The default model is LightGBM v6 since 2026-11-09.
        lightgbm = ['--learning-v2', '--lightgbm']
        self.assertEqual(
            ['--update'] + lightgbm + ['--horizon-reps', '5'],
            _learning_argv(['--horizon-reps', '5']),
        )
        self.assertEqual(['--update'] + lightgbm, _learning_argv([]))
        self.assertEqual(
            lightgbm + ['--update', '--horizon-reps', '5'],
            _learning_argv(['--update', '--horizon-reps', '5']),
        )

    def test_pipeline_honours_no_update(self):
        self.assertEqual(
            ['--learning-v2', '--lightgbm', '--horizon-reps', '5'],
            _learning_argv(['--horizon-reps', '5', '--no-update']),
        )
        self.assertEqual(['--learning-v2', '--lightgbm'], _learning_argv(['-no-update']))

    def test_other_v2_models_can_still_be_chosen(self):
        self.assertEqual(['--update', '--learning-v2', '--v2-blend'], _learning_argv(['--v2-blend']))
        self.assertEqual(['--update', '--learning-v2', '--v2-blend'],
                         _learning_argv(['--learning-v2', '--v2-blend']))
        self.assertEqual(['--update', '--learning-v2', '--lightgbm', '--lgbm-version', 'v3'],
                         _learning_argv(['--learning-v2', '--lightgbm', '--lgbm-version', 'v3']))

    def test_v1_and_v1_recipes_keep_the_old_model(self):
        self.assertEqual(
            ['--update', '--horizon-reps', '5'],
            _learning_argv(['--v1', '--horizon-reps', '5']),
        )
        self.assertEqual(['--update', '--h1-poisson'], _learning_argv(['--h1-poisson']))
        self.assertEqual(['--update', '--learning-v2'], _learning_argv(['--learning-v2']))

    def test_update_flag_refreshes_even_when_last_event_matches(self):
        runner = Runner(update_data=False)
        api_json = {'events': [{'id': 1, 'data_checked': False}]}
        file_json = {'events': [{'id': 1, 'data_checked': False}]}
        with mock.patch.object(runner, 'read_json', return_value=file_json):
            with mock.patch('os.path.isfile', return_value=True):
                with mock.patch('sys.argv', ['premierlearning']):
                    self.assertFalse(runner.should_update_data(api_json))
                with mock.patch('sys.argv', ['premierlearning', '--update']):
                    self.assertTrue(runner.should_update_data(api_json))
                with mock.patch('sys.argv', ['premierlearning', '--no-update']):
                    self.assertFalse(runner.should_update_data(api_json))
