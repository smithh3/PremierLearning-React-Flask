import sys
import requests
import os
import os.path
from premierlearning.clean_match import (
    BUILT_KEYS, CHANCE_TO_PLAY, STAT_KEYS, get_input_keys, get_opponent_feature_vector, OPPONENT_FEATURE_KEYS
)
from premierlearning.config_constants import POINTS_HORIZON, SEASON_YR
import numpy


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from premierlearning.learning import *

from sklearn import linear_model
from .season import *
from persistence import DB_Handler

FANTASY_API = 'https://fantasy.premierleague.com/api/bootstrap-static/'
FIXTURES_API = 'https://fantasy.premierleague.com/api/fixtures/'
PLAYER_API = 'https://fantasy.premierleague.com/api/element-summary/%i/'


class Runner:
    def __init__(self):
        self.squad = None
        self.current_season = None
        self.past_seasons = []

    def run(self):
        self.update_persisted_data_if_outdated()

        print('Setting up players and teams...')
        self.current_season = CurrentSeason()
        self.populate_past_seasons()
        self.process_player_matches()

        # self.calculate_player_ict_and_points_regression()

        print('Preparing data for learning...')

        horizon_only = '--horizon-only' in sys.argv
        self.learn(horizon_only=horizon_only)

        self.current_season.calculate_player_points_over_next_5_gameweeks()

        if horizon_only:
            self.print_top_players_next_horizon(100)

        self.persist_players()

        # self.populate_squad()

        # print()

    def update_persisted_data_if_outdated(self):
        fantasy_json_from_api = self.get_request_json(FANTASY_API)

        if not self.api_matches_configured_season(fantasy_json_from_api):
            api_yr = self.get_api_season_year(fantasy_json_from_api)
            print(
                f'Skipping live FPL update: API is season {api_yr}/{api_yr + 1 if api_yr is not None else "?"} '
                f'but SEASON_YR={SEASON_YR} (season_{SEASON_YR}_{SEASON_YR + 1}). '
                f'Past seasons like 25/26 are loaded from local vaastav data, not the live API.'
            )
            return

        update_all = self.should_update_data(fantasy_json_from_api)

        if update_all:
            print(f'Getting latest data for season_{SEASON_YR}_{SEASON_YR + 1}...')
            self.persist_json(FANTASY_JSON_FILE_LOCATION % (SEASON_YR, SEASON_YR + 1), fantasy_json_from_api)

        if update_all or not os.path.isfile(FIXTURES_JSON_FILE_LOCATION % (SEASON_YR, SEASON_YR + 1)):
            fixtures_json = self.get_request_json(FIXTURES_API)
            if fixtures_json is not None:
                self.persist_json(FIXTURES_JSON_FILE_LOCATION % (SEASON_YR, SEASON_YR + 1), fixtures_json)

        if fantasy_json_from_api is not None:
            for element in fantasy_json_from_api['elements']:
                player_id = element['id']
                player_file_path = PLAYER_JSON_FILE_LOCATION % (SEASON_YR, SEASON_YR + 1) + str(player_id) + '.json'

                if update_all or not os.path.isfile(player_file_path):
                    player_json = self.get_request_json(PLAYER_API % player_id)
                    if player_json is not None:
                        self.persist_json(player_file_path, player_json)

    @staticmethod
    def get_request_json(uri):
        try:
            return requests.get(uri).json()
        except:
            return None

    @staticmethod
    def get_api_season_year(fantasy_json_from_api):
        """Infer season start year (e.g. 26 for 26/27) from GW1 deadline."""
        if not fantasy_json_from_api or 'events' not in fantasy_json_from_api:
            return None
        try:
            gw1 = next(event for event in fantasy_json_from_api['events'] if event.get('id') == 1)
            return int(gw1['deadline_time'][:4]) % 100
        except (StopIteration, KeyError, TypeError, ValueError):
            return None

    def api_matches_configured_season(self, fantasy_json_from_api):
        api_yr = self.get_api_season_year(fantasy_json_from_api)
        return api_yr is not None and api_yr == SEASON_YR

    def should_update_data(self, fantasy_json_from_api):
        if '--update' in sys.argv:
        # if ALWAYS_UPDATE:
            return True
        if fantasy_json_from_api is None:
            return False

        fantasy_json_file = None
        if os.path.isfile(FANTASY_JSON_FILE_LOCATION % (SEASON_YR, SEASON_YR + 1)):
            fantasy_json_file = self.read_json(FANTASY_JSON_FILE_LOCATION % (SEASON_YR, SEASON_YR + 1))

        if fantasy_json_file is None:
            return True

        last_api_match_id = self.get_last_event_checked_id(fantasy_json_from_api)
        last_file_match_id = self.get_last_event_checked_id(fantasy_json_file)
        return last_api_match_id != last_file_match_id

    def populate_past_seasons(self):
        for i in reversed(range(18, SEASON_YR)):
            self.past_seasons.append(PastSeason(i))

        all_seasons = []
        for season in self.past_seasons:
            all_seasons.append(season)
        all_seasons.append(self.current_season)

        for player in self.current_season.players:
            for season in all_seasons:
                if player.code in season.player_code_dict:
                    player_last_season = season.player_code_dict[player.code]
                    if season.years[0] - 1 in player.history_past:
                        player_last_season.points_per_min_last_season = player.history_past[season.years[0] - 1]
                        player_last_season.played_last_season = True
    
    def process_player_matches(self):
        last_szn = next(szn for szn in self.past_seasons if szn.years[0] == self.current_season.years[0] - 1)
        if self.current_season.next_gameweek < 12:
            for player in self.current_season.players:
                if player.code in last_szn.player_code_dict:
                    player.attach_last_szn_matches(last_szn.player_code_dict[player.code])
        
        for player in self.current_season.players:
            player.process_matches()
        for season in self.past_seasons:
            for player in season.players:
                player.process_matches()


    def populate_formatted_match_data(self):
        learner_input = self.get_learning_inputs_for_season(self.current_season)
        for season in self.past_seasons:
            learner_input.extend(self.get_learning_inputs_for_season(season))
        return learner_input

    def get_learning_inputs_for_season(self, season) -> list:
        learner_input = []
        for player in season.players:
            for match in player.matches:
                learner_input.append(match.to_clean_match())
        return learner_input

    def calculate_player_ict_and_points_regression(self):
        ict_values = []
        points = []

        # for season in list(self.past_seasons + [self.current_season]):
        for player in self.current_season.players:
            # for player in season.players:
            for match in player.matches:
                if match.valid_for_learning:  # and match.is_training_data (if not then its testing data)
                    ict_values.append([match.past_stats['ict_index']['avg']])
                    points.append([match.game_stats['total_points']])

        # for player in self.past_seasons[0].players:
        #     for match in player.matches:
        #         if match.valid_for_learning:  # and match.is_training_data (if not then its testing data)
        #             ict_values.append([match.average_ict_in_previous_matches])
        #             points.append([match.average_points_in_previous_matches])

        ict_reg = linear_model.LinearRegression()
        ict_reg.fit(ict_values, points)

        # predicted_points = ict_reg.predict(ict_values)

        for player in self.current_season.players:
            for match in player.matches:
                if match.valid_for_learning:  # also fit testing data here using the regression fitted on training data
                    points_predicted_by_ict = ict_reg.predict([[match.past_stats['ict_index']['avg']]])[0][0]
                    match.ict_point_regression_deviation = points_predicted_by_ict - match.past_stats['total_points']['avg']

            player_predicted_points_from_ict = ict_reg.predict([[player.past_stats['ict_index']['avg']]])[0][0]
            player.ict_deviation = player_predicted_points_from_ict - player.past_stats['total_points']['avg']

        # pyplot.scatter(ict_values, points)
        # pyplot.plot(ict_values, predicted_points, color='blue', linewidth=3)
        #
        # pyplot.show()


    def learn(self, horizon_only: bool = False):
        clean_matches = self.populate_formatted_match_data()

        valid_clean_matches = [match for match in clean_matches if match['is_valid']]
        count = 0
        for match in valid_clean_matches:
            match['id'] = count
            count += 1
        future_matches = []
        future_clean_matches = []
        for player in self.current_season.players:
            for future_match in player.future_matches:
                future_matches.append(future_match)
                future_clean_matches.append(future_match.to_clean_match())
        
        count = 0
        for match in future_clean_matches:
            match['id'] = count
            count += 1

        if not horizon_only:
            learnt_stats = {}
            # Horizon-1: per-match points (fixture shape) + minutes / chance to play
            learnt_stats['total_points'] = get_avg_predictions(
                valid_clean_matches, future_clean_matches, get_input_keys('total_points'), 'total_points', True, horizon=1
            )
            learnt_stats['minutes'] = get_avg_predictions(
                valid_clean_matches, future_clean_matches, get_input_keys('minutes'), 'minutes', True, horizon=1
            )
            learnt_stats[CHANCE_TO_PLAY] = get_avg_predictions(
                clean_matches, future_clean_matches, get_input_keys(CHANCE_TO_PLAY), CHANCE_TO_PLAY, False, horizon=1
            )

            def set_zero_if_neg(val):
                if val < 0:
                    return 0
                return val

            for i in range(len(future_matches)):
                for stat_key in ['total_points', 'minutes', CHANCE_TO_PLAY]:
                    future_matches[i].game_stats[stat_key] = set_zero_if_neg(learnt_stats[stat_key][i][0])
                future_matches[i].points = future_matches[i].game_stats['total_points']
                future_matches[i].set_minutes_from_stats()

        # Horizon-N: stabler total, then redistribute using horizon-1 week weights
        # (equal split when horizon-only / no h1 weights)
        horizon = POINTS_HORIZON
        horizon_train = self.build_horizon_training_data(horizon)
        horizon_pred = self.build_horizon_predict_data(horizon)
        if horizon_train['player_clean'] and horizon_pred['player_clean']:
            horizon_totals = get_avg_predictions(
                horizon_train['player_clean'],
                horizon_pred['player_clean'],
                get_input_keys('total_points'),
                'total_points',
                True,
                horizon=horizon,
                opponent_learn=horizon_train['opponents'],
                opponent_pred=horizon_pred['opponents'],
                output_learn=horizon_train['labels'],
            )
            self.redistribute_horizon_points(horizon_pred['players'], horizon_totals, horizon)
        else:
            print(f'Skipping horizon={horizon} model (insufficient training or predict rows)')

        if not horizon_only:
            for future_match in future_matches:
                future_match.update_points_based_on_expected_minutes()

    def print_top_players_next_horizon(self, n: int = 100):
        horizon = POINTS_HORIZON
        ranked = sorted(
            self.current_season.players,
            key=lambda p: p.points_over_next_5_gameweeks,
            reverse=True,
        )[:n]
        print(f'\nTop {n} players by predicted points over next {horizon} gameweeks:')
        print(f'{"#":>3}  {"Player":<28}  {"Pos":<4}  {"Team":<16}  {"Pts":>6}')
        for i, player in enumerate(ranked, start=1):
            print(
                f'{i:>3}  {player.get_name():<28}  {str(player.position):<4}  '
                f'{str(player.team_name):<16}  {player.points_over_next_5_gameweeks:>6.1f}'
            )

    def build_horizon_training_data(self, horizon: int):
        player_clean = []
        opponents = []
        labels = []
        seasons = list(self.past_seasons) + [self.current_season]
        n_opp = len(OPPONENT_FEATURE_KEYS)

        for season in seasons:
            for player in season.players:
                matches = player.matches
                for i, match in enumerate(matches):
                    if not match.valid_for_learning:
                        continue
                    if i + horizon > len(matches):
                        continue
                    window = matches[i:i + horizon]
                    clean = match.to_clean_match()
                    opp_matrix = [get_opponent_feature_vector(m.to_clean_match()) for m in window]
                    if any(len(row) != n_opp for row in opp_matrix):
                        continue
                    player_clean.append(clean)
                    opponents.append(opp_matrix)
                    labels.append(sum(float(m.points) for m in window))

        return {
            'player_clean': player_clean,
            'opponents': numpy.array(opponents, dtype=float) if opponents else numpy.empty((0, horizon, n_opp)),
            'labels': numpy.array(labels, dtype=float).reshape(-1, 1) if labels else numpy.empty((0, 1)),
        }

    def build_horizon_predict_data(self, horizon: int):
        player_clean = []
        opponents = []
        players = []
        n_opp = len(OPPONENT_FEATURE_KEYS)
        zero_opp = [0.0] * n_opp

        for player in self.current_season.players:
            futs = player.future_matches[:horizon]
            if not futs:
                continue
            player_clean.append(futs[0].to_clean_match())
            opp_matrix = [get_opponent_feature_vector(m.to_clean_match()) for m in futs]
            while len(opp_matrix) < horizon:
                opp_matrix.append(list(zero_opp))
            opponents.append(opp_matrix)
            players.append(player)

        return {
            'player_clean': player_clean,
            'opponents': numpy.array(opponents, dtype=float) if opponents else numpy.empty((0, horizon, n_opp)),
            'players': players,
        }

    def redistribute_horizon_points(self, players, horizon_totals, horizon: int):
        """Allocate horizon total T across next fixtures: final_i = T * (p_i / sum p)."""
        max_reasonable_total = 12.0 * horizon
        failed_sanity = []
        for player, total_pred in zip(players, horizon_totals):
            T = float(total_pred[0]) if hasattr(total_pred, '__len__') else float(total_pred)
            if T < 0:
                T = 0
            futs = player.future_matches[:horizon]
            if not futs:
                continue
            weights = [max(0.0, float(m.points)) for m in futs]
            weight_sum = sum(weights)

            if T > max_reasonable_total or (weight_sum > 0 and T > weight_sum * 2.5):
                failed_sanity.append(
                    f'{player.get_name()}: T={T:.1f} h1_sum={weight_sum:.1f}'
                )

            if weight_sum <= 0:
                share = T / len(futs)
                for match in futs:
                    match.points = share
                    match.game_stats['total_points'] = share
            else:
                for match, weight in zip(futs, weights):
                    points = T * (weight / weight_sum)
                    match.points = points
                    match.game_stats['total_points'] = points

        if failed_sanity:
            print(
                f'Horizon redistribute sanity check failed for {len(failed_sanity)} players '
                f'(T > {max_reasonable_total:.0f} or T > 2.5× h1 sum); scores still applied:'
            )
            for line in failed_sanity[:20]:
                print(f'  {line}')
            if len(failed_sanity) > 20:
                print(f'  ... and {len(failed_sanity) - 20} more')

    def persist_players(self):
        print('Connecting to database')
        db_handler = DB_Handler()
        db_handler.init_db()
        print('Persisting players')
        db_handler.persist_players(self.current_season.players)
        db_handler.connection.commit()
        db_handler.close_connection()
        print('Run finished')

    def dump_element_types(self):
        with open('element_types.json', 'w') as f:
            json.dump(self.current_season.element_type_dict, f)

    @staticmethod
    def get_last_event_checked_id(fantasy_json):
        last_api_match_id = None
        for event in fantasy_json['events']:
            if event['data_checked']:
                last_api_match_id = event['id']
            else:
                break
        return last_api_match_id

    @staticmethod
    def read_json(file_path):
        with open(file_path) as file:
            return json.load(file)

    @staticmethod
    def persist_json(file_path, json_data):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(json_data, file)
