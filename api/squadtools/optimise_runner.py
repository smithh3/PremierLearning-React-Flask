from .user_squad import AbstractSquad
from persistence import DB_Handler

import json

class OptimiseRunner:

    def __init__(self, id):
        self.nope = None
        self.squad = None
        self.transfers = None
        self.csrftoken = None
        self.manager_id = id
        self.player_dict = None
    
    def get_ids(self):
        return self.squad.get_player_ids()

    def populate_squad(self):
        with open('data/element_types.json') as f_in:
            elements_dict = json.load(f_in)

        db_handler = DB_Handler()
        self.player_dict = db_handler.get_player_objects()
        db_handler.close_connection()

        # self.squad = UserSquad(self.user_json, self.player_dict, elements_dict)
        self.squad = AbstractSquad(self.player_dict, elements_dict)
        self.squad.populate_from_manager_id(self.manager_id)
    
    def write_top_30_info(self):
        # players_select = sorted(self.player_dict.values(), reverse=True, key=lambda p : p.future_matches[0].points)
        matches_count = 5
        players_select = sorted(self.player_dict.values(), reverse=True, key=lambda p : p.get_points_within_n_gameweeks(self.squad.next_gameweek, matches_count))
        with open('top_players.csv', 'w', encoding='utf-8') as file:
            for i in range(60):
                file.write(f'{str(players_select[i])},'
                           f'pred:{sum(match.points for match in players_select[i].future_matches[:5]):.1f},'
                           f'prev:{sum(match.points for match in players_select[i].matches[-5:]):.1f},'
                           f'{",".join(f"{match.opponent}:{match.points:.1f}" for match in players_select[i].future_matches[:5])}\n')
        
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
