from .user_squad import AbstractSquad, UserSquad
from .predictions_report import write_predictions_csv
from persistence import DB_Handler
from premierlearning.config_constants import POINTS_HORIZON
from premierlearning.fpl_apis import load_selectable_player_ids

import json

class OptimiseRunner:

    def __init__(self, id, user_json=None):
        self.nope = None
        self.squad = None
        self.transfers = None
        self.csrftoken = None
        self.manager_id = id
        self.player_dict = None
        self.user_json = user_json
    
    def get_ids(self):
        return self.squad.get_player_ids()

    def populate_squad(self):
        with open('data/element_types.json') as f_in:
            elements_dict = json.load(f_in)

        db_handler = DB_Handler()
        self.player_dict = db_handler.get_player_objects()
        db_handler.close_connection()

        # self.squad = UserSquad(self.user_json, self.player_dict, elements_dict)
        if self.user_json:
            self.squad = UserSquad(self.user_json, self.player_dict, elements_dict)
        else:
            self.squad = AbstractSquad(self.player_dict, elements_dict)
            self.squad.populate_from_manager_id(self.manager_id)
    
    def write_top_30_info(self):
        write_predictions_csv(
            self.player_dict,
            self.squad.next_gameweek,
            n=POINTS_HORIZON,
        )
        
    def run(self):
        self.populate_squad()
        self.write_top_30_info()
        if self.squad.built_from_scratch:
            print('Initial squad built from scratch (15 players). Skipping transfer suggestions.')
            print('Squad:', ', '.join(p.get_name() for p in self.squad.squad))
            self.transfers = None
            return
        self.transfers = self.squad.make_sensible_transfers()
    
    def get_transfers(self):
        temp_gameweek = self.squad.next_gameweek
        transfer_index = 0
        transfers_list = []
        for scheme_entry in self.transfers['scheme']:
            transfer_gw_dict = {'gameweek': temp_gameweek, 'transfers': [] }
            for _ in range(scheme_entry):
                transfer_gw_dict['transfers'].append(str(self.transfers['transfer_tuple'][transfer_index]))
                transfer_index += 1
            transfers_list.append(transfer_gw_dict)
            temp_gameweek += 1
        return transfers_list

    def get_transfers_json(self):
        transfers = {}
        temp_gameweek = self.squad.next_gameweek
        transfer_index = 0
        for scheme_entry in self.transfers['scheme']:
            transfers_this_week = []
            for _ in range(scheme_entry):
                transfers_this_week.append(self.transfers['transfer_tuple'][transfer_index].format_as_json())
                transfer_index += 1
            transfers[temp_gameweek] = transfers_this_week
            temp_gameweek += 1
        return transfers

    def get_future_starting_teams(self):
        return self.squad.get_starting_teams_json()

    def get_this_week_transfers(self):
        if not self.transfers:
            return []
        this_week_count = self.transfers['scheme'][0]
        return list(self.transfers['transfer_tuple'][:this_week_count])

    def apply_this_week_transfers(self, email, password):
        this_week = self.get_this_week_transfers()
        selectable_ids = load_selectable_player_ids()
        if this_week and selectable_ids:
            available = []
            for transfer in this_week:
                if transfer.player_in.id not in selectable_ids:
                    print('Skipping %s: player is unavailable' % transfer)
                    continue
                available.append(transfer)
            this_week = available
        if this_week:
            print('Submitting %d transfer(s) for GW %s:' % (len(this_week), self.squad.next_gameweek))
            for transfer in this_week:
                print('  %s' % transfer)
                self.squad.stage_transfer(transfer)
            self.squad.log_in_and_make_transfers(email, password)
            print('Transfers submitted')
        else:
            print('No transfers suggested for this gameweek; skipping transfer submit')
        self.apply_this_week_team_selection(email, password)

    def apply_this_week_team_selection(self, email, password):
        print('Selecting starting XI')
        self.squad.log_in_and_pick_team(email, password)
        print('Starting XI submitted')
