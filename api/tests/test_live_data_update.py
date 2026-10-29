import unittest
from unittest import mock

from premierlearning.runner import Runner
from run_pipeline import _learning_argv


class LiveDataUpdateTests(unittest.TestCase):
    def test_pipeline_always_prepends_update(self):
        self.assertEqual(
            ['--update', '--horizon-reps', '5'],
            _learning_argv(['--horizon-reps', '5']),
        )
        self.assertEqual(['--update'], _learning_argv([]))
        self.assertEqual(
            ['--update', '--horizon-reps', '5'],
            _learning_argv(['--update', '--horizon-reps', '5']),
        )

    def test_pipeline_honours_no_update(self):
        self.assertEqual(
            ['--horizon-reps', '5'],
            _learning_argv(['--horizon-reps', '5', '--no-update']),
        )
        self.assertEqual([], _learning_argv(['-no-update']))

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
