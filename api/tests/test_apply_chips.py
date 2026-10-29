import unittest
from unittest import mock

from squadtools.optimise_runner import OptimiseRunner
from squadtools.simulation_strategy import ChipCalendar
from squadtools.user_squad import UserSquad


class _Player:
    def __init__(self, player_id, current_cost=55):
        self.id = player_id
        self.current_cost = current_cost

    def format_as_request_pick(self, placement, is_captain, is_vice_captain):
        return {
            'element': self.id,
            'position': placement,
            'is_captain': is_captain,
            'is_vice_captain': is_vice_captain,
        }


class _Wrapped:
    def __init__(self, selling_price):
        self.selling_price = selling_price


class _Transfer:
    def __init__(self, player_out, player_in, selling_price=40, purchase_price=70):
        self.player_out = player_out
        self.player_in = player_in
        self.selling_price = selling_price
        self.purchase_price = purchase_price

    def format_as_request(self):
        return {
            'element_in': self.player_in.id,
            'element_out': self.player_out.id,
            'purchase_price': self.purchase_price,
            'selling_price': self.selling_price,
        }

    def __str__(self):
        return '%s -> %s' % (self.player_out.id, self.player_in.id)


def _squad_with_lineup():
    squad = UserSquad.__new__(UserSquad)
    squad.manager_id = 102483
    squad.next_gameweek = 8
    squad.staged_transfers = []
    squad.wrapped_player_dict = {}
    starters = [_Player(index) for index in range(1, 12)]
    bench = [_Player(index) for index in range(12, 16)]
    squad.future_starting_teams = {
        8: mock.Mock(
            players=starters,
            bench=bench,
            captain=starters[0],
            vice_captain=starters[1],
        )
    }
    return squad


class TransferPayloadTests(unittest.TestCase):
    def test_wildcard_uses_chip_field_and_live_sell_price(self):
        squad = _squad_with_lineup()
        outgoing = _Player(412)
        incoming = _Player(99, current_cost=61)
        squad.wrapped_player_dict[412] = _Wrapped(40)
        squad.staged_transfers = [_Transfer(outgoing, incoming, selling_price=44, purchase_price=99)]

        payload = squad.format_transfer_requests(chip='wildcard')

        self.assertEqual('wildcard', payload['chip'])
        self.assertNotIn('chips', payload)
        self.assertEqual(102483, payload['entry'])
        self.assertEqual(8, payload['event'])
        self.assertEqual(
            [{
                'element_in': 99,
                'element_out': 412,
                'purchase_price': 61,
                'selling_price': 40,
            }],
            payload['transfers'],
        )

    def test_free_hit_chip_name_is_freehit(self):
        squad = _squad_with_lineup()
        payload = squad.format_transfer_requests(chip='free_hit')
        self.assertEqual('freehit', payload['chip'])
        self.assertEqual([], payload['transfers'])

    def test_bench_boost_and_triple_captain_are_team_chips(self):
        squad = _squad_with_lineup()
        self.assertEqual('bboost', squad.pick_team_request_format(chip='bench_boost')['chip'])
        self.assertEqual('3xc', squad.pick_team_request_format(chip='triple_captain')['chip'])
        self.assertIsNone(squad.pick_team_request_format(chip='wildcard')['chip'])
        self.assertIsNone(squad.format_transfer_requests(chip='bench_boost')['chip'])

    def test_lineup_payload_has_fifteen_picks(self):
        picks = _squad_with_lineup().pick_team_request_format()['picks']
        self.assertEqual(15, len(picks))
        self.assertEqual(1, sum(1 for pick in picks if pick['is_captain']))
        self.assertEqual(1, sum(1 for pick in picks if pick['is_vice_captain']))


class ApplyChipTests(unittest.TestCase):
    def _runner(self, chip_week, transfers):
        opt = OptimiseRunner(102483)
        opt.squad = mock.Mock()
        opt.squad.next_gameweek = 8
        opt.squad.is_pre_gw1.return_value = False
        opt.chip_calendar = ChipCalendar(**{chip_week: (8,)}) if chip_week else None
        opt.transfers = {
            'scheme': (len(transfers),),
            'transfer_tuple': tuple(transfers),
        }
        return opt

    def _transfer(self, player_in_id):
        transfer = mock.Mock()
        transfer.player_in.id = player_in_id
        transfer.__str__ = lambda self: 'out -> %s' % player_in_id
        return transfer

    def test_wildcard_submits_the_rebuild_with_the_chip(self):
        transfer = self._transfer(99)
        opt = self._runner('wildcards', [transfer])
        with mock.patch(
            'squadtools.optimise_runner.load_selectable_player_ids',
            return_value={99},
        ):
            opt.apply_this_week_transfers('user', 'secret')

        opt.squad.stage_transfer.assert_called_once_with(transfer)
        opt.squad.log_in_and_make_transfers.assert_called_once_with(
            'user', 'secret', chip='wildcard'
        )
        opt.squad.log_in_and_pick_team.assert_called_once_with(
            'user', 'secret', chip='wildcard'
        )

    def test_free_hit_with_no_transfers_still_plays_the_chip(self):
        opt = self._runner('free_hits', [])
        with mock.patch(
            'squadtools.optimise_runner.load_selectable_player_ids',
            return_value=set(),
        ):
            opt.apply_this_week_transfers('user', 'secret')

        opt.squad.stage_transfer.assert_not_called()
        opt.squad.log_in_and_make_transfers.assert_called_once_with(
            'user', 'secret', chip='free_hit'
        )

    def test_bench_boost_goes_on_the_lineup_not_the_transfers(self):
        transfer = self._transfer(99)
        opt = self._runner('bench_boosts', [transfer])
        with mock.patch(
            'squadtools.optimise_runner.load_selectable_player_ids',
            return_value={99},
        ):
            opt.apply_this_week_transfers('user', 'secret')

        opt.squad.log_in_and_make_transfers.assert_called_once_with(
            'user', 'secret', chip='bench_boost'
        )
        opt.squad.log_in_and_pick_team.assert_called_once_with(
            'user', 'secret', chip='bench_boost'
        )

    def test_unavailable_wildcard_target_is_not_submitted(self):
        transfer = self._transfer(99)
        opt = self._runner('wildcards', [transfer])
        with mock.patch(
            'squadtools.optimise_runner.load_selectable_player_ids',
            return_value={1},
        ):
            with self.assertRaises(RuntimeError) as raised:
                opt.apply_this_week_transfers('user', 'secret')

        self.assertIn('wildcard', str(raised.exception))
        opt.squad.log_in_and_make_transfers.assert_not_called()
        opt.squad.log_in_and_pick_team.assert_not_called()

    def test_ordinary_week_still_submits_without_a_chip(self):
        transfer = self._transfer(99)
        opt = self._runner(None, [transfer])
        with mock.patch(
            'squadtools.optimise_runner.load_selectable_player_ids',
            return_value={99},
        ):
            opt.apply_this_week_transfers('user', 'secret')

        opt.squad.log_in_and_make_transfers.assert_called_once_with(
            'user', 'secret', chip=None
        )
        opt.squad.log_in_and_pick_team.assert_called_once_with(
            'user', 'secret', chip=None
        )
