import sys
import requests
import os
import os.path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from premierlearning.clean_match import (
    BUILT_KEYS, CHANCE_TO_PLAY, STAT_KEYS, get_input_keys,
    get_inputs_from_data_key, use_team_expectancies, apply_feature_interactions,
)
from premierlearning.config_constants import POINTS_HORIZON, SEASON_YR
from premierlearning.helper_functions import effective_minutes_for_points
from premierlearning.player import Player, MIN_APPEARANCES_TO_PREDICT, AVG_WINDOW, LAST_5_WINDOW
from premierlearning.team_expectancy import run_team_expectancies, print_future_scorelines
import numpy

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

        team_only = '--team-only' in sys.argv
        horizon_only = '--horizon-only' in sys.argv
        minutes_only = '--minutes-only' in sys.argv
        expectancies = use_team_expectancies()
        if use_linear_model():
            print('Linear-only mode: RidgeCV for team + player predictions (no neural nets)')

        print('Setting up players and teams...')
        self.current_season = CurrentSeason()
        self.populate_past_seasons()

        if expectancies or team_only:
            print('Running team expectancies...')
            run_team_expectancies(self.past_seasons, self.current_season)

        if team_only:
            print_future_scorelines(self.current_season)
            print('Team-only run finished')
            return

        self.process_player_matches()

        if horizon_only:
            self.print_sample_horizon_feature_columns()
        if minutes_only:
            self.print_sample_minutes_feature_columns()

        print('Preparing data for learning...')
        self.learn(horizon_only=horizon_only, minutes_only=minutes_only)

        self.current_season.calculate_player_points_over_next_5_gameweeks()

        if horizon_only:
            self.print_top_players_next_horizon(100)
        if minutes_only:
            self.print_top_players_next_match_minutes(200)

        if '--no-persist' in sys.argv:
            print('Skipping persist (--no-persist)')
        else:
            self.persist_players()

    def print_sample_horizon_feature_columns(self):
        """Print predict-time inputs and one in-season train row (form = last ≤5 before that GW)."""
        targets = [
            ('Erling', 'Haaland'),
            ('Bruno', 'Fernandes'),  # matches "Bruno Borges Fernandes"
        ]
        found = []
        for first, last in targets:
            player = next(
                (
                    p for p in self.current_season.players
                    if last.lower() in (p.last_name or '').lower()
                    and first.lower() in (p.first_name or '').lower()
                ),
                None,
            )
            if player is None:
                print(f'Could not find player matching {first} {last}')
            else:
                found.append(player)

        if not found:
            return

        points_keys = get_input_keys('total_points')
        horizon = POINTS_HORIZON
        min_played = 3
        # Prefer a completed past season so the TRAIN example has full prior history
        past_by_code = {}
        for season in sorted(self.past_seasons, key=lambda s: s.years[0], reverse=True):
            for p in season.players:
                if p.code not in past_by_code:
                    past_by_code[p.code] = (p, season.years)

        print('\n=== Sample horizon model columns (validation) ===')
        for player in found:
            # Same predict row the horizon model scores
            futs = player.future_matches[:horizon]
            if futs:
                clean = futs[0].to_clean_match()
                xg_sum, xc_sum = self._sum_team_xg_xc(futs)
                clean['team_xg_over_n'] = xg_sum
                clean['team_xc_over_n'] = xc_sum
                apply_feature_interactions(clean)
                feats = get_inputs_from_data_key(
                    [clean], points_keys, True, include_opponent_features=True
                )[0]
                prior_apps = [
                    m for m in player.matches if self._match_minutes(m) > 0
                ]
                print(
                    f'\n{player.get_name()} ({player.team_name}) — PREDICT '
                    f'(exact model input row)'
                )
                self._print_form_appearance_breakdown(prior_apps)
                print('  target: (future — not known)')
                self._print_model_input_features(feats)

            # Same filters as build_horizon_training_data (last eligible row)
            train_entry = past_by_code.get(player.code)
            train_player = train_entry[0] if train_entry else player
            train_season_years = train_entry[1] if train_entry else self.current_season.years
            sample = None
            for i, match in enumerate(train_player.matches):
                if getattr(match, 'is_last_season_carry', False):
                    continue
                if not match.valid_for_learning:
                    continue
                if i + horizon > len(train_player.matches):
                    continue
                window = train_player.matches[i:i + horizon]
                played = self._played_in_window(window)
                if len(played) < min_played:
                    continue
                prior_apps = [
                    m for m in train_player.matches[:i]
                    if self._match_minutes(m) > 0
                ]
                clean = match.to_clean_match()
                xg_sum, xc_sum = self._sum_team_xg_xc(window)
                clean['team_xg_over_n'] = xg_sum
                clean['team_xc_over_n'] = xc_sum
                apply_feature_interactions(clean)
                sample = {
                    'match': match,
                    'clean': clean,
                    'target': self._pooled_points_per_90(played),
                    'played': len(played),
                    'prior_apps': prior_apps,
                    'season': train_season_years,
                }

            if sample is None:
                print(
                    f'\n{player.get_name()}: no eligible TRAIN window '
                    f'(≥{min_played}/{horizon} appearances in-window)'
                )
                continue

            match = sample['match']
            feats = get_inputs_from_data_key(
                [sample['clean']], points_keys, True, include_opponent_features=True
            )[0]
            print(
                f'\n{player.get_name()} — TRAIN {sample["season"]} '
                f'GW{match.gameweek} vs {match.opponent} ({"H" if match.home else "A"}) '
                f'(exact model input row; target window {sample["played"]}/{horizon} played)'
            )
            self._print_form_appearance_breakdown(sample['prior_apps'])
            print(f'  target (points/90 over window): {sample["target"]:.6f}')
            self._print_model_input_features(feats)
        print('=== End sample columns ===\n')

    def _print_model_input_features(self, feats):
        """Print the exact dict that becomes the model feature vector."""
        # Display aliases: form xGI keys are often read as "xg" but are goal involvements
        aliases = {
            'expected_goal_involvements_last_5_avg': 'xgi_last_5_avg',
            'expected_goal_involvements_avg': 'xgi_avg',
        }
        print('  --- model input features (exact) ---')
        for key, val in feats.items():
            label = aliases.get(key, key)
            if isinstance(val, float):
                print(f'  {label}: {val:.6f}')
            else:
                print(f'  {label}: {val}')

    def _print_form_appearance_breakdown(self, prior_apps):
        """Show appearance windows that should match total_points_last_5_avg / avg."""
        def fmt(window):
            if not window:
                return '(none)'
            bits = [
                f'GW{getattr(m, "gameweek", "?")}:'
                f'{float(m.points):.0f}pts/{self._match_minutes(m):.0f}m'
                for m in window
            ]
            pp90 = self._pooled_points_per_90(window)
            return f'{", ".join(bits)} -> pp90={pp90:.4f}'

        print(f'  form appearances available: {len(prior_apps)}')
        print(f'  last_{LAST_5_WINDOW} (should = total_points_last_5_avg): {fmt(prior_apps[-LAST_5_WINDOW:])}')
        print(f'  last_{AVG_WINDOW} (should = total_points_avg): {fmt(prior_apps[-AVG_WINDOW:])}')

    def print_sample_minutes_feature_columns(self):
        """Print one minutes training row per sample player (features + target minutes)."""
        targets = [
            ('Erling', 'Haaland'),
            ('Bruno', 'Fernandes'),
        ]
        found = []
        for first, last in targets:
            player = next(
                (
                    p for p in self.current_season.players
                    if last.lower() in (p.last_name or '').lower()
                    and first.lower() in (p.first_name or '').lower()
                ),
                None,
            )
            if player is None:
                print(f'Could not find player matching {first} {last}')
            else:
                found.append(player)

        if not found:
            return

        minutes_keys = get_input_keys('minutes')
        past_by_code = {}
        for season in sorted(self.past_seasons, key=lambda s: s.years[0], reverse=True):
            for p in season.players:
                if p.code not in past_by_code:
                    past_by_code[p.code] = (p, season.years)

        print('\n=== Sample minutes model columns (validation) ===')
        for player in found:
            futs = player.future_matches[:1]
            if futs:
                clean = futs[0].to_clean_match()
                feats = get_inputs_from_data_key(
                    [clean], minutes_keys, True, include_opponent_features=True
                )[0]
                print(
                    f'\n{player.get_name()} ({player.team_name}) — PREDICT next match'
                )
                print('  target: (future — not known)')
                self._print_model_input_features(feats)

            train_entry = past_by_code.get(player.code)
            train_player = train_entry[0] if train_entry else player
            train_season_years = train_entry[1] if train_entry else self.current_season.years
            sample = None
            for match in train_player.matches:
                if getattr(match, 'is_last_season_carry', False):
                    continue
                if not getattr(match, 'valid_for_minutes', False):
                    continue
                clean = match.to_clean_match()
                sample = {
                    'match': match,
                    'clean': clean,
                    'target': float(match.game_stats.get('minutes', 0) or 0),
                    'season': train_season_years,
                }

            if sample is None:
                print(f'\n{player.get_name()}: no eligible TRAIN minutes row')
                continue

            match = sample['match']
            feats = get_inputs_from_data_key(
                [sample['clean']], minutes_keys, True, include_opponent_features=True
            )[0]
            print(
                f'\n{player.get_name()} — TRAIN {sample["season"]} '
                f'GW{match.gameweek} vs {match.opponent} ({"H" if match.home else "A"})'
            )
            print(f'  target (minutes): {sample["target"]:.6f}')
            self._print_model_input_features(feats)
        print('=== End sample minutes columns ===\n')

    def print_top_players_next_match_minutes(self, n: int = 200):
        ranked = []
        for player in self.current_season.players:
            if player.matches_played < MIN_APPEARANCES_TO_PREDICT:
                continue
            if not player.future_matches:
                continue
            mins = float(player.future_matches[0].game_stats.get('minutes', 0) or 0)
            ranked.append((player, mins))
        ranked.sort(key=lambda x: x[1], reverse=True)
        ranked = ranked[:n]
        header = [
            f'Top {n} players by predicted minutes in next match:',
            (
                'H2/mt = second-half last season mins/fixture (incl. DNPs); '
                '2s/mt = prev2 season + first half of last season; '
                'Apps = H2 appearances/fixtures; Mv = club move.'
            ),
            (
                f'{"#":>3}  {"Player":<28}  {"Pos":<4}  {"Team":<16}  {"Pred":>6}  '
                f'{"H2/mt":>6}  {"2s/mt":>6}  {"Apps":>7}  {"Mv":>3}'
            ),
        ]
        lines = list(header)
        print('\n' + '\n'.join(header))
        for i, (player, mins) in enumerate(ranked, start=1):
            fixtures = int(getattr(player, 'last_season_h2_fixtures', 0) or 0)
            apps = int(getattr(player, 'last_season_h2_appearances', 0) or 0)
            h2_mpm = (
                f'{float(player.last_season_h2_mins_per_match):.1f}' if fixtures > 0 else '-'
            )
            two_s = f'{float(getattr(player, "last_two_seasons_mins_per_match", 0) or 0):.1f}'
            apps_s = f'{apps}/{fixtures}' if fixtures > 0 else '-'
            moved = 'Y' if getattr(player, 'club_changed_vs_prior', False) else '.'
            row = (
                f'{i:>3}  {player.get_name():<28}  {str(player.position):<4}  '
                f'{str(player.team_name):<16}  {mins:>6.1f}  {h2_mpm:>6}  '
                f'{two_s:>6}  {apps_s:>7}  {moved:>3}'
            )
            print(row)
            lines.append(row)
        out_path = os.path.join(os.getcwd(), 'top_player_minutes.txt')
        with open(out_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(lines) + '\n')
        print(f'Wrote {out_path}')

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
                    if season.years[0] - 1 in player.history_past_xgi:
                        player_last_season.xgi_per_min_last_season = player.history_past_xgi[season.years[0] - 1]
                    if season.years[0] - 1 in player.history_past_minutes:
                        player_last_season.minutes_last_season = player.history_past_minutes[season.years[0] - 1]
                    if season.years[0] - 2 in player.history_past:
                        player_last_season.points_per_min_two_seasons_ago = player.history_past[season.years[0] - 2]
    
    def process_player_matches(self):
        # Oldest → newest so each season can carry form from the prior year into early-GW train rows
        # (same prefix predict uses on the current season).
        seasons_chrono = sorted(self.past_seasons, key=lambda s: s.years[0])
        by_year = {s.years[0]: s for s in seasons_chrono}

        for season in seasons_chrono:
            prev = by_year.get(season.years[0] - 1)
            prev2 = by_year.get(season.years[0] - 2)
            prev2_dict = prev2.player_code_dict if prev2 is not None else {}
            if prev is not None:
                for player in season.players:
                    prev2_player = prev2_dict.get(player.code)
                    if player.code in prev.player_code_dict:
                        prev_player = prev.player_code_dict[player.code]
                        player.attach_last_szn_matches(prev_player)
                        Player.apply_prior_season_stats(player, prev_player, prev2_player)
                    else:
                        player.set_changed_club_from_prior(None)
                        Player.apply_prior_season_stats(player, None, prev2_player)
            for player in season.players:
                player.process_matches()

        prev2_now = by_year.get(self.current_season.years[0] - 2)
        prev2_now_dict = prev2_now.player_code_dict if prev2_now is not None else {}
        if self.current_season.next_gameweek < 12:
            last_szn = by_year.get(self.current_season.years[0] - 1)
            if last_szn is None:
                last_szn = next(
                    (s for s in self.past_seasons if s.years[0] == self.current_season.years[0] - 1),
                    None,
                )
            if last_szn is not None:
                for player in self.current_season.players:
                    prev2_player = prev2_now_dict.get(player.code)
                    if player.code in last_szn.player_code_dict:
                        prev_player = last_szn.player_code_dict[player.code]
                        player.attach_last_szn_matches(prev_player)
                        Player.apply_prior_season_stats(player, prev_player, prev2_player)
                    else:
                        player.set_changed_club_from_prior(None)
                        Player.apply_prior_season_stats(player, None, prev2_player)
            else:
                for player in self.current_season.players:
                    player.set_changed_club_from_prior(None)
                    Player.apply_prior_season_stats(
                        player, None, prev2_now_dict.get(player.code)
                    )
        else:
            # Mid-season: still set club-change flag from prior year (no early-GW carry attach)
            last_szn = by_year.get(self.current_season.years[0] - 1)
            if last_szn is not None:
                for player in self.current_season.players:
                    prev2_player = prev2_now_dict.get(player.code)
                    if player.code in last_szn.player_code_dict:
                        prev_player = last_szn.player_code_dict[player.code]
                        player.set_changed_club_from_prior(prev_player)
                        Player.apply_prior_season_stats(player, prev_player, prev2_player)
                    else:
                        player.set_changed_club_from_prior(None)
                        Player.apply_prior_season_stats(player, None, prev2_player)

        for player in self.current_season.players:
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


    def learn(self, horizon_only: bool = False, minutes_only: bool = False):
        clean_matches = self.populate_formatted_match_data()

        valid_clean_matches = [match for match in clean_matches if match['is_valid']]
        minutes_clean_matches = [
            match for match in clean_matches if match.get('is_valid_for_minutes')
        ]
        count = 0
        for match in valid_clean_matches:
            match['id'] = count
            count += 1
        count = 0
        for match in minutes_clean_matches:
            match['id'] = count
            count += 1
        future_matches = []
        future_clean_matches = []
        skipped_low_apps = 0
        for player in self.current_season.players:
            if player.matches_played < MIN_APPEARANCES_TO_PREDICT:
                skipped_low_apps += 1
                for future_match in player.future_matches:
                    future_match.points = 0
                    future_match.game_stats['total_points'] = 0
                    future_match.game_stats['minutes'] = 0
                continue
            for future_match in player.future_matches:
                future_matches.append(future_match)
                future_clean_matches.append(future_match.to_clean_match())
        if skipped_low_apps:
            print(
                f'Skipping predict for {skipped_low_apps} players with '
                f'<{MIN_APPEARANCES_TO_PREDICT} appearances (incl. last-season carry)'
            )

        count = 0
        for match in future_clean_matches:
            match['id'] = count
            count += 1

        if minutes_only:
            minutes_preds = get_avg_predictions(
                minutes_clean_matches, future_clean_matches, get_input_keys('minutes'),
                'minutes', True
            )
            for i, future_match in enumerate(future_matches):
                mins = float(minutes_preds[i][0])
                if mins < 0:
                    mins = 0.0
                mins = min(90.0, mins)
                future_match.game_stats['minutes'] = mins
                future_match.set_minutes_from_stats()
            return

        # Per-match error margins from |residual| Ridge (heteroscedastic)
        points_errors, _ = fit_and_predict_points_error_margins(
            valid_clean_matches, future_clean_matches, get_input_keys('total_points'), True
        )
        for future_match, err in zip(future_matches, points_errors):
            future_match.points_error = float(err)

        if not horizon_only:
            learnt_stats = {}
            # Horizon-1: points/90 per fixture (same construction as h5 / last_5), then × mins/90
            h1_pp90_labels = numpy.array(
                [[self._clean_match_points_per_90(m)] for m in valid_clean_matches],
                dtype=float,
            )
            learnt_stats['total_points'] = get_avg_predictions(
                valid_clean_matches, future_clean_matches, get_input_keys('total_points'),
                'total_points', True, output_learn=h1_pp90_labels, log_targets=True
            )
            learnt_stats['minutes'] = get_avg_predictions(
                minutes_clean_matches, future_clean_matches, get_input_keys('minutes'),
                'minutes', True
            )
            # Unused in points pipeline (old minutes/chance gating commented out)
            # learnt_stats[CHANCE_TO_PLAY] = get_avg_predictions(
            #     clean_matches, future_clean_matches, get_input_keys(CHANCE_TO_PLAY),
            #     CHANCE_TO_PLAY, False
            # )

            def set_zero_if_neg(val):
                if val < 0:
                    return 0
                return val

            for i in range(len(future_matches)):
                pp90 = set_zero_if_neg(learnt_stats['total_points'][i][0])
                mins = min(90.0, set_zero_if_neg(learnt_stats['minutes'][i][0]))
                # chance = set_zero_if_neg(learnt_stats[CHANCE_TO_PLAY][i][0])
                future_matches[i].game_stats['minutes'] = mins
                # future_matches[i].game_stats[CHANCE_TO_PLAY] = chance
                # Expected fixture points = predicted pp90 × eased minutes / 90
                points = pp90 * effective_minutes_for_points(mins) / 90.0
                future_matches[i].game_stats['total_points'] = points
                future_matches[i].points = points
                future_matches[i].set_minutes_from_stats()

        # Next-N GW: label = points/90 over appearances in the exact next-N fixtures
        # (≥3 appearances required). Total T = pp90 × expected_mins/90; split by h1 fixture weights.
        horizon = POINTS_HORIZON
        horizon_train = self.build_horizon_training_data(horizon)
        horizon_pred = self.build_horizon_predict_data(horizon)
        if horizon_train['player_clean'] and horizon_pred['player_clean']:
            horizon_reps = parse_horizon_reps()
            if horizon_reps > 1:
                print(f'Horizon-{horizon} model: {horizon_reps} independent fits ({HORIZON_REPS_FLAG})')
            horizon_pp90 = get_avg_predictions(
                horizon_train['player_clean'],
                horizon_pred['player_clean'],
                get_input_keys('total_points'),
                'total_points',
                True,
                output_learn=horizon_train['labels'],
                log_targets=True,
                reps=horizon_reps,
            )
            self.redistribute_horizon_points(horizon_pred['players'], horizon_pp90, horizon)
        else:
            print(f'Skipping next-{horizon} points model (insufficient training or predict rows)')

        for future_match in future_matches:
            future_match.update_points_based_on_expected_minutes()

    @staticmethod
    def _clean_match_points_per_90(clean_match) -> float:
        pts = float(clean_match['game_stats'].get('total_points', 0) or 0)
        mins = float(clean_match['game_stats'].get('minutes', 0) or 0)
        if mins <= 0:
            return 0.0
        return pts * 90.0 / mins

    def print_top_players_next_horizon(self, n: int = 100):
        horizon = POINTS_HORIZON
        ranked = sorted(
            (
                p for p in self.current_season.players
                if p.matches_played >= MIN_APPEARANCES_TO_PREDICT
            ),
            key=lambda p: p.points_over_next_5_gameweeks,
            reverse=True,
        )[:n]
        print(f'\nTop {n} players by predicted points over next {horizon} gameweeks:')
        print(f'{"#":>3}  {"Player":<28}  {"Pos":<4}  {"Team":<16}  {"Pts":>6}')
        lines = [
            f'Top {n} players by predicted points over next {horizon} gameweeks:',
            f'{"#":>3}  {"Player":<28}  {"Pos":<4}  {"Team":<16}  {"Pts":>6}',
        ]
        for i, player in enumerate(ranked, start=1):
            row = (
                f'{i:>3}  {player.get_name():<28}  {str(player.position):<4}  '
                f'{str(player.team_name):<16}  {player.points_over_next_5_gameweeks:>6.1f}'
            )
            print(row)
            lines.append(row)
        metrics_dir = get_last_horizon_metrics_dir()
        if metrics_dir is not None:
            metrics_dir.mkdir(parents=True, exist_ok=True)
            (metrics_dir / 'top100.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    @staticmethod
    def _sum_team_xg_xc(matches):
        xg = sum(float(getattr(m.team_match, 'predicted_xg', 0.0) or 0.0) for m in matches)
        xc = sum(float(getattr(m.team_match, 'predicted_xc', 0.0) or 0.0) for m in matches)
        return xg, xc

    @staticmethod
    def _match_minutes(match) -> float:
        if getattr(match, 'game_stats', None):
            return float(match.game_stats.get('minutes', 0) or 0)
        return float(getattr(match, 'minutes', 0) or 0)

    @classmethod
    def _played_in_window(cls, window):
        """Appearances inside this exact fixture window only (no extending)."""
        return [m for m in window if cls._match_minutes(m) > 0]

    @classmethod
    def _pooled_points_per_90(cls, played_matches) -> float:
        """Same construction as total_points_last_5_avg: sum(pts)*90 / sum(mins)."""
        pts = sum(float(m.points) for m in played_matches)
        mins = sum(cls._match_minutes(m) for m in played_matches)
        if mins <= 0:
            return 0.0
        return pts * 90.0 / mins

    def build_horizon_training_data(self, horizon: int):
        """Rows at match i: label = points/90 over played games in matches[i:i+horizon].

        Exact window only (no stretching). Requires ≥3 appearances inside those
        `horizon` fixtures. Past form is still the usual last-5 window; future
        difficulty is team xg/xc summed over the same fixture window.
        """
        player_clean = []
        labels = []
        skipped_low_apps = 0
        seasons = list(self.past_seasons) + [self.current_season]
        min_played = 3

        for season in seasons:
            for player in season.players:
                matches = player.matches
                for i, match in enumerate(matches):
                    if not match.valid_for_learning:
                        continue
                    if i + horizon > len(matches):
                        continue
                    window = matches[i:i + horizon]
                    played = self._played_in_window(window)
                    if len(played) < min_played:
                        skipped_low_apps += 1
                        continue
                    clean = match.to_clean_match()
                    xg_sum, xc_sum = self._sum_team_xg_xc(window)
                    clean['team_xg_over_n'] = xg_sum
                    clean['team_xc_over_n'] = xc_sum
                    apply_feature_interactions(clean)
                    player_clean.append(clean)
                    labels.append(self._pooled_points_per_90(played))

        for i, clean in enumerate(player_clean):
            clean['id'] = i

        print(
            f'Horizon-{horizon} train rows: {len(player_clean)} '
            f'(skipped {skipped_low_apps} windows with <{min_played} appearances in-window)'
        )
        return {
            'player_clean': player_clean,
            'labels': numpy.array(labels, dtype=float).reshape(-1, 1) if labels else numpy.empty((0, 1)),
        }

    def build_horizon_predict_data(self, horizon: int):
        """One row per player: last-5 form + team xg/xc summed over next `horizon` fixtures."""
        player_clean = []
        players = []

        for player in self.current_season.players:
            if player.matches_played < MIN_APPEARANCES_TO_PREDICT:
                continue
            futs = player.future_matches[:horizon]
            if not futs:
                continue
            clean = futs[0].to_clean_match()
            xg_sum, xc_sum = self._sum_team_xg_xc(futs)
            clean['team_xg_over_n'] = xg_sum
            clean['team_xc_over_n'] = xc_sum
            apply_feature_interactions(clean)
            player_clean.append(clean)
            players.append(player)

        for i, clean in enumerate(player_clean):
            clean['id'] = i

        return {
            'player_clean': player_clean,
            'players': players,
        }

    def _expected_minutes_over_horizon(self, player, futs, horizon: int) -> float:
        """Minutes scale to turn predicted pp90 into a points total for redistribute."""
        pred_mins = sum(
            effective_minutes_for_points(self._match_minutes(m)) for m in futs
        )
        if pred_mins > 0:
            return pred_mins
        avg_mins = float(player.past_stats.get('minutes', {}).get('last_5_avg', 0) or 0)
        return effective_minutes_for_points(avg_mins) * horizon

    def redistribute_horizon_points(self, players, horizon_pp90_preds, horizon: int):
        """Convert horizon pp90 → window total T via expected minutes; split by h1 fixture weights.

        Sum of next-`horizon` points equals T; each fixture's share follows the ratio of
        h1 expected points (already pp90 × minutes/90 when full learn ran).
        """
        max_reasonable_total = 12.0 * horizon
        failed_sanity = []
        for player, pred in zip(players, horizon_pp90_preds):
            pp90 = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
            if pp90 < 0:
                pp90 = 0.0
            futs = player.future_matches[:horizon]
            if not futs:
                continue
            expected_mins = self._expected_minutes_over_horizon(player, futs, horizon)
            T = pp90 * expected_mins / 90.0 if expected_mins > 0 else 0.0

            weights = [max(0.0, float(m.points)) for m in futs]
            weight_sum = sum(weights)

            if T > max_reasonable_total or (weight_sum > 0 and T > weight_sum * 2.5):
                failed_sanity.append(
                    f'{player.get_name()}: T={T:.1f} pp90={pp90:.1f} h1_sum={weight_sum:.1f}'
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
