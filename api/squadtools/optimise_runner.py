from .user_squad import AbstractSquad, UserSquad
from persistence import DB_Handler
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
    
    @staticmethod
    def _match_minutes(match) -> float:
        mins = getattr(match, 'minutes', None)
        if mins is None and getattr(match, 'game_stats', None):
            mins = match.game_stats.get('minutes', 0)
        return float(mins or 0)

    def _last_n_appearance_points(self, player, n: int = 5) -> float:
        """Sum of points from the last n appearances (minutes > 0), same window as model form."""
        played = [m for m in player.matches if self._match_minutes(m) > 0]
        return sum(float(m.points or 0) for m in played[-n:])

    def write_top_30_info(self):
        # players_select = sorted(self.player_dict.values(), reverse=True, key=lambda p : p.future_matches[0].points)
        matches_count = 5
        players_select = sorted(self.player_dict.values(), reverse=True, key=lambda p : p.get_points_within_n_gameweeks(self.squad.next_gameweek, matches_count))
        with open('top_players.csv', 'w', encoding='utf-8') as file:
            for i in range(60):
                player = players_select[i]
                err = 0.0
                if player.future_matches:
                    next_errs = [
                        float(getattr(m, 'points_error', 0.0) or 0.0)
                        for m in player.future_matches[:matches_count]
                    ]
                    err = sum(next_errs) / len(next_errs) if next_errs else 0.0
                prev_pts = self._last_n_appearance_points(player, matches_count)
                file.write(
                    f'{str(player)},'
                    f'pred:{sum(match.points for match in player.future_matches[:5]):.1f},'
                    f'prev:{prev_pts:.1f},'
                    f'err:{err:.2f},'
                    f'{",".join(f"{match.opponent}:{match.points:.1f}" for match in player.future_matches[:5])}\n'
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
