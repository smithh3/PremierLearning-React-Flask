import os
import unittest
from unittest import mock

from squadtools.__main__ import main, public_manager_id


class PublicManagerIdTests(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get('FPL_MANAGER_ID')
        os.environ.pop('FPL_MANAGER_ID', None)

    def tearDown(self):
        if self._old is None:
            os.environ.pop('FPL_MANAGER_ID', None)
        else:
            os.environ['FPL_MANAGER_ID'] = self._old

    def test_reads_fpl_manager_id(self):
        os.environ['FPL_MANAGER_ID'] = '4733713'
        self.assertEqual(4733713, public_manager_id())

    @mock.patch('squadtools.__main__.send_predictions_email')
    @mock.patch('squadtools.__main__.OptimiseRunner')
    @mock.patch('squadtools.__main__.get_user_jsons_login')
    @mock.patch('squadtools.__main__.load_credentials')
    def test_suggest_with_id_skips_login(
        self, load_credentials, get_user_jsons_login, runner_cls, send_email
    ):
        os.environ['FPL_MANAGER_ID'] = '4733713'
        opt = mock.Mock()
        opt.get_this_week_transfers.return_value = []
        opt.transfers = None
        opt.squad.next_gameweek = 4
        runner_cls.return_value = opt

        main(suggest_transfers=True)

        load_credentials.assert_not_called()
        get_user_jsons_login.assert_not_called()
        runner_cls.assert_called_once_with(4733713)
        opt.run.assert_called_once()
        send_email.assert_called_once_with(opt, applied=False)

    @mock.patch('squadtools.__main__.load_credentials')
    def test_apply_with_id_is_refused(self, load_credentials):
        os.environ['FPL_MANAGER_ID'] = '4733713'
        with self.assertRaises(SystemExit) as raised:
            main(apply_transfers=True)
        self.assertIn('cannot apply', str(raised.exception).lower())
        load_credentials.assert_not_called()
