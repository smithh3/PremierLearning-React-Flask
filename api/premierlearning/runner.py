import sys
import json
import requests
import os
import os.path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from premierlearning.clean_match import (
    BUILT_KEYS, CHANCE_TO_PLAY, STAT_KEYS, get_input_keys,
    get_inputs_from_data_key, use_team_expectancies, apply_feature_interactions,
    TEAM_XG_NEXT_KEY, TEAM_XC_NEXT_KEY,
)
from premierlearning.config_constants import POINTS_HORIZON, SEASON_YR
from premierlearning.helper_functions import (
    effective_minutes_for_points,
    implied_pp90,
)
from premierlearning.player import (
    Player, MIN_APPEARANCES_TO_PREDICT, AVG_WINDOW, LAST_5_WINDOW,
    is_form_appearance,
)
from premierlearning.team_expectancy import (
    run_team_expectancies, print_future_scorelines, write_future_scorelines,
)
import numpy

from premierlearning.learning import *

from sklearn import linear_model
from .season import *
from persistence import DB_Handler

FANTASY_API = 'https://fantasy.premierleague.com/api/bootstrap-static/'
FIXTURES_API = 'https://fantasy.premierleague.com/api/fixtures/'
PLAYER_API = 'https://fantasy.premierleague.com/api/element-summary/%i/'


def _flag_int(argv, flag):
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            try:
                return int(argv[index + 1])
            except (TypeError, ValueError):
                raise SystemExit('%s needs an integer (got %r)' % (flag, argv[index + 1]))
    return None


def parse_as_of_args(argv=None):
    """Optional --as-of-season YY --as-of-gw N for a historical point-in-time run."""
    argv = list(sys.argv if argv is None else argv)
    season = _flag_int(argv, '--as-of-season')
    gameweek = _flag_int(argv, '--as-of-gw')
    if (season is None) != (gameweek is None):
        raise SystemExit('--as-of-season and --as-of-gw must be used together')
    return season, gameweek


def parse_train_as_of_gw(argv=None):
    """Optional --train-as-of-gw N: freeze labels at that deadline (1 = pre-season)."""
    argv = list(sys.argv if argv is None else argv)
    return _flag_int(argv, '--train-as-of-gw')


class Runner:
    def __init__(
        self,
        current_season=None,
        target_season_year=SEASON_YR,
        as_of_gameweek=None,
        train_as_of_gameweek=None,
        horizon_gameweeks=POINTS_HORIZON,
        update_data=None,
    ):
        argv_season, argv_gw = parse_as_of_args()
        if as_of_gameweek is None:
            as_of_gameweek = argv_gw
        if current_season is None and argv_season is not None:
            target_season_year = argv_season
        self.squad = None
        self.current_season = current_season
        self.past_seasons = []
        self._horizon_only = False
        self.target_season_year = int(
            current_season.years[0] if current_season is not None else target_season_year
        )
        self.as_of_gameweek = (
            int(as_of_gameweek)
            if as_of_gameweek is not None
            else getattr(current_season, 'as_of_gameweek', None)
        )
        if train_as_of_gameweek is None:
            train_as_of_gameweek = parse_train_as_of_gw()
        if train_as_of_gameweek is not None:
            if self.as_of_gameweek is None:
                raise SystemExit('--train-as-of-gw requires --as-of-season and --as-of-gw')
            train_as_of_gameweek = int(train_as_of_gameweek)
            if train_as_of_gameweek < 1 or train_as_of_gameweek > int(self.as_of_gameweek):
                raise SystemExit(
                    '--train-as-of-gw must be between 1 and --as-of-gw '
                    '(1 = no current-season labels)'
                )
            self.train_as_of_gameweek = train_as_of_gameweek
        else:
            self.train_as_of_gameweek = (
                int(self.as_of_gameweek) if self.as_of_gameweek is not None else None
            )
        self.horizon_gameweeks = int(horizon_gameweeks)
        self.historical_mode = self.as_of_gameweek is not None
        self.update_data = (
            not self.historical_mode if update_data is None else bool(update_data)
        )

    def run(self, persist=None):
        if self.update_data:
            self.update_persisted_data_if_outdated()

        team_only = '--team-only' in sys.argv
        horizon_only = '--horizon-only' in sys.argv
        minutes_only = '--minutes-only' in sys.argv
        expectancies = use_team_expectancies() and '--learning-v2' not in sys.argv
        if use_linear_model():
            print('Linear-only mode: RidgeCV for team + player predictions (no neural nets)')

        print('Setting up players and teams...')
        if self.current_season is None:
            if self.historical_mode:
                from simulation.historical_data import VaastavSeasonData
                trained_through = self._trained_through_gameweek()
                if trained_through <= 0:
                    train_note = (
                        'Training uses prior seasons only (train-as-of GW%s); '
                        'this season is features/predict.'
                        % self.train_as_of_gameweek
                    )
                else:
                    train_note = (
                        'Training uses completed GWs 1–%s of this season plus '
                        'prior seasons; later GWs are prediction only.'
                        % trained_through
                    )
                print(
                    'Point-in-time %s/%s GW%s. %s'
                    % (
                        self.target_season_year,
                        self.target_season_year + 1,
                        self.as_of_gameweek,
                        train_note,
                    )
                )
                self.current_season = VaastavSeasonData(
                    self.target_season_year
                ).build_current_season(
                    self.as_of_gameweek,
                    self.horizon_gameweeks,
                )
                self.current_season.training_complete_through_gw = trained_through
            else:
                self.current_season = CurrentSeason(self.target_season_year)
        self.past_seasons = []
        self.populate_past_seasons()
        self.validate_temporal_cutoff()

        if expectancies or team_only:
            print('Running team expectancies...')
            if run_team_expectancies(self.past_seasons, self.current_season):
                if not self.historical_mode:
                    write_future_scorelines(self.current_season)

        if team_only:
            print_future_scorelines(self.current_season)
            print('Team-only run finished')
            return

        self.process_player_matches()

        if minutes_only:
            self.print_sample_minutes_feature_columns()

        print('Preparing data for learning...')
        self.learn(horizon_only=horizon_only, minutes_only=minutes_only)

        self.current_season.calculate_player_points_over_next_5_gameweeks()

        if horizon_only:
            self.print_top_players_next_horizon()
            self.print_sample_horizon_feature_columns()
        if minutes_only:
            self.print_top_players_next_match_minutes(200)

        if persist is None:
            persist = '--no-persist' not in sys.argv and not self.historical_mode
        if not persist:
            print('Skipping live snapshot persistence')
        else:
            self.persist_players()
        return self.current_season

    def validate_temporal_cutoff(self):
        """Fail closed if a historical prediction can see future training rows."""
        if not self.historical_mode:
            return
        invalid_seasons = [
            season.years
            for season in self.past_seasons
            if int(season.years[0]) >= self.target_season_year
        ]
        if invalid_seasons:
            raise ValueError(
                f'Historical training includes non-prior seasons: {invalid_seasons}'
            )
        leaked = []
        for player in self.current_season.players:
            for match in player.matches:
                if getattr(match, 'is_last_season_carry', False):
                    continue
                if (
                    match.gameweek is not None
                    and int(match.gameweek) >= int(self.as_of_gameweek)
                ):
                    leaked.append((player.id, int(match.gameweek)))
                    if len(leaked) >= 10:
                        break
            if len(leaked) >= 10:
                break
        if leaked:
            raise ValueError(
                f'Point-in-time cutoff GW{self.as_of_gameweek} leaked training matches: '
                f'{leaked}'
            )

    def _trained_through_gameweek(self):
        if self.train_as_of_gameweek is None:
            return None
        return max(0, int(self.train_as_of_gameweek) - 1)

    def apply_training_label_cutoff(self):
        """Keep this-season history for features; drop it from training labels."""
        cutoff = self._trained_through_gameweek()
        if cutoff is None or self.current_season is None:
            return
        setattr(self.current_season, 'training_complete_through_gw', cutoff)
        for player in getattr(self.current_season, 'players', []) or []:
            for match in getattr(player, 'matches', []) or []:
                if getattr(match, 'is_last_season_carry', False):
                    continue
                gameweek = getattr(match, 'gameweek', None)
                if gameweek is not None and int(gameweek) > cutoff:
                    match.valid_for_learning = False
                    match.valid_for_minutes = False

    def _default_sample_players(self):
        """Haaland / Fernandes used by --horizon-only and --minutes-only."""
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
                self._safe_print(f'Could not find player matching {first} {last}')
            else:
                found.append(player)
        return found

    def print_sample_horizon_feature_columns(self, players=None):
        """Print predict-time inputs and one in-season train row (form = last ≤5 before that GW)."""
        found = list(players) if players is not None else self._default_sample_players()
        if not found:
            return

        points_keys = get_input_keys('total_points')
        horizon = self.horizon_gameweeks
        min_played = 3
        # Prefer a completed past season so the TRAIN example has full prior history
        past_by_code = {}
        for season in sorted(self.past_seasons, key=lambda s: s.years[0], reverse=True):
            for p in season.players:
                if p.code not in past_by_code:
                    past_by_code[p.code] = (p, season.years)

        self._safe_print('\n=== Sample horizon model columns (Haaland / Fernandes) ===')
        for player in found:
            # Same predict row the horizon model scores
            futs = player.future_matches[:horizon]
            if futs:
                clean = futs[0].to_clean_match()
                xg_sum, xc_sum = self._sum_team_xg_xc(futs)
                clean[TEAM_XG_NEXT_KEY] = xg_sum
                clean[TEAM_XC_NEXT_KEY] = xc_sum
                apply_feature_interactions(clean)
                feats = get_inputs_from_data_key(
                    [clean], points_keys, True, include_opponent_features=True
                )[0]
                prior_apps = [
                    m for m in player.matches
                    if is_form_appearance(self._match_minutes(m))
                ]
                pred_total = getattr(player, 'points_over_next_5_gameweeks', None)
                pred_bit = ''
                if pred_total is not None:
                    pp90 = float(pred_total) / horizon if horizon else 0.0
                    pred_bit = f'; predicted {horizon}/{horizon} pp90={pp90:.2f}'
                self._safe_print(
                    f'\n{player.get_name()} ({player.team_name}) - PREDICT '
                    f'(exact model input row{pred_bit})'
                )
                self._print_form_appearance_breakdown(prior_apps)
                self._safe_print('  target: (future - not known)')
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
                    if is_form_appearance(self._match_minutes(m))
                ]
                clean = match.to_clean_match()
                xg_sum, xc_sum = self._sum_team_xg_xc(window)
                clean[TEAM_XG_NEXT_KEY] = xg_sum
                clean[TEAM_XC_NEXT_KEY] = xc_sum
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
                self._safe_print(
                    f'\n{player.get_name()}: no eligible TRAIN window '
                    f'(>={min_played}/{horizon} appearances in-window)'
                )
                continue

            match = sample['match']
            feats = get_inputs_from_data_key(
                [sample['clean']], points_keys, True, include_opponent_features=True
            )[0]
            self._safe_print(
                f'\n{player.get_name()} - TRAIN {sample["season"]} '
                f'GW{match.gameweek} vs {match.opponent} ({"H" if match.home else "A"}) '
                f'(exact model input row; target window {sample["played"]}/{horizon} played)'
            )
            self._print_form_appearance_breakdown(sample['prior_apps'])
            self._safe_print(f'  target (points/90 over window): {sample["target"]:.6f}')
            self._print_model_input_features(feats)
        self._safe_print('=== End sample columns ===\n')

    def _print_model_input_features(self, feats):
        """Print each model feature once, in the order the vector is built."""
        self._safe_print('  --- model input features (exact) ---')
        seen = set()
        for key, val in feats.items():
            if key in seen:
                continue
            seen.add(key)
            if isinstance(val, float):
                self._safe_print(f'  {key}: {val:.6f}')
            else:
                self._safe_print(f'  {key}: {val}')

    def _print_form_appearance_breakdown(self, prior_apps):
        """Show appearance windows that should match points_last_3 / last_7 per-90."""
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

        self._safe_print(f'  form appearances available: {len(prior_apps)}')
        last7 = prior_apps[-AVG_WINDOW:]
        mins7 = ','.join('%.0f' % self._match_minutes(m) for m in last7) if last7 else '(none)'
        self._safe_print(f'  last_{AVG_WINDOW} minutes played: {mins7}')
        self._safe_print(
            f'  last_{LAST_5_WINDOW} (should = points_last_{LAST_5_WINDOW}_per_90_excl_null): '
            f'{fmt(prior_apps[-LAST_5_WINDOW:])}'
        )
        self._safe_print(
            f'  last_{AVG_WINDOW} (should = total_points_avg): '
            f'{fmt(prior_apps[-AVG_WINDOW:])}'
        )

    def print_sample_minutes_feature_columns(self, players=None):
        """Print one minutes training row per sample player (features + target minutes)."""
        found = list(players) if players is not None else self._default_sample_players()
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
                'Sz/mt = this-season mins/fixture (incl. DNPs, default minutes rate); '
                'LS/mt = full last-season mins/fixture; '
                '2s/mt = full two-seasons-ago mins/fixture; '
                'Apps = last-season appearances/fixtures; Mv = club move. '
                'Predict last-season / 2s drop 3+ DNP streaks if the last two fixtures were appearances.'
            ),
            (
                f'{"#":>3}  {"Player":<28}  {"Pos":<4}  {"Team":<16}  {"Pred":>6}  '
                f'{"Sz/mt":>6}  {"LS/mt":>6}  {"2s/mt":>6}  {"Apps":>7}  {"Mv":>3}'
            ),
        ]
        lines = list(header)
        print('\n' + '\n'.join(header))
        for i, (player, mins) in enumerate(ranked, start=1):
            fixtures = int(getattr(player, 'last_season_fixtures', 0) or 0)
            apps = int(getattr(player, 'last_season_appearances', 0) or 0)
            clean = player.future_matches[0].to_clean_match()
            sz_val = float(clean.get('season_mins_per_match') or 0)
            ls_val = float(clean.get('last_season_mins_per_match') or 0)
            two_val = float(clean.get('two_season_prior_mins_per_match') or 0)
            sz_mpm = f'{sz_val:.1f}'
            ls_mpm = f'{ls_val:.1f}' if fixtures > 0 else '-'
            two_s = f'{two_val:.1f}'
            apps_s = f'{apps}/{fixtures}' if fixtures > 0 else '-'
            moved = 'Y' if getattr(player, 'club_changed_vs_prior', False) else '.'
            row = (
                f'{i:>3}  {player.get_name():<28}  {str(player.position):<4}  '
                f'{str(player.team_name):<16}  {mins:>6.1f}  {sz_mpm:>6}  '
                f'{ls_mpm:>6}  {two_s:>6}  {apps_s:>7}  {moved:>3}'
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
                f'but configured season is {self.target_season_year}/'
                f'{self.target_season_year + 1}. '
                f'Past seasons like 25/26 are loaded from local vaastav data, not the live API.'
            )
            return

        # Keep this deadline's chance-of-playing flags for future replays and
        # v2 training; the live API only ever shows the current ones.
        from squadtools.chance_to_play import archive_live_chances
        archived = archive_live_chances(self.target_season_year, fantasy_json_from_api)
        if archived is not None:
            print(f'Saved chance-of-playing flags to {archived}')
        # FPL's full player news (LightGBM v6, premierlearning/lgbm/news.py): this
        # deadline's from the live API, passed deadlines from fplcache.
        try:
            from premierlearning.lgbm.news import archive_live_news, backfill_live_news
            news_path = archive_live_news(self.target_season_year, fantasy_json_from_api)
            if news_path is not None:
                print(f'Saved FPL player news to {news_path}')
            backfill_live_news(self.target_season_year, fantasy_json_from_api.get('events'))
        except Exception as error:  # noqa: BLE001 - news is extra; never stop the run
            print(f'Could not save FPL player news: {error}')

        update_all = self.should_update_data(fantasy_json_from_api)

        if update_all:
            print(
                f'Getting latest data for season_{self.target_season_year}_'
                f'{self.target_season_year + 1}...'
            )
            self.persist_json(
                FANTASY_JSON_FILE_LOCATION
                % (self.target_season_year, self.target_season_year + 1),
                fantasy_json_from_api,
            )
        else:
            print(
                'Keeping local FPL files (last finished GW unchanged). '
                'Injury and chance-of-playing news are not refreshed unless '
                'you pass --update.'
            )

        fixture_path = FIXTURES_JSON_FILE_LOCATION % (
            self.target_season_year,
            self.target_season_year + 1,
        )
        if update_all or not os.path.isfile(fixture_path):
            fixtures_json = self.get_request_json(FIXTURES_API)
            if fixtures_json is not None:
                self.persist_json(fixture_path, fixtures_json)

        if fantasy_json_from_api is not None:
            for element in fantasy_json_from_api['elements']:
                player_id = element['id']
                player_file_path = (
                    PLAYER_JSON_FILE_LOCATION
                    % (self.target_season_year, self.target_season_year + 1)
                    + str(player_id)
                    + '.json'
                )

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
        return api_yr is not None and api_yr == self.target_season_year

    def should_update_data(self, fantasy_json_from_api):
        if '--no-update' in sys.argv or '-no-update' in sys.argv:
            return False
        if '--update' in sys.argv:
            return True
        if fantasy_json_from_api is None:
            return False

        fantasy_json_file = None
        fantasy_path = FANTASY_JSON_FILE_LOCATION % (
            self.target_season_year,
            self.target_season_year + 1,
        )
        if os.path.isfile(fantasy_path):
            fantasy_json_file = self.read_json(fantasy_path)

        if fantasy_json_file is None:
            return True

        last_api_match_id = self.get_last_event_checked_id(fantasy_json_from_api)
        last_file_match_id = self.get_last_event_checked_id(fantasy_json_file)
        return last_api_match_id != last_file_match_id

    def populate_past_seasons(self):
        for i in reversed(range(18, self.target_season_year)):
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
                    if getattr(player, 'skip_historical_predict', False):
                        continue
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
                    if getattr(player, 'skip_historical_predict', False):
                        continue
                    player.set_changed_club_from_prior(None)
                    Player.apply_prior_season_stats(
                        player, None, prev2_now_dict.get(player.code)
                    )
        else:
            # Mid-season: still set club-change flag from prior year (no early-GW carry attach)
            last_szn = by_year.get(self.current_season.years[0] - 1)
            if last_szn is not None:
                for player in self.current_season.players:
                    if getattr(player, 'skip_historical_predict', False):
                        continue
                    prev2_player = prev2_now_dict.get(player.code)
                    if player.code in last_szn.player_code_dict:
                        prev_player = last_szn.player_code_dict[player.code]
                        player.set_changed_club_from_prior(prev_player)
                        Player.apply_prior_season_stats(player, prev_player, prev2_player)
                    else:
                        player.set_changed_club_from_prior(None)
                        Player.apply_prior_season_stats(player, None, prev2_player)

        for player in self.current_season.players:
            if getattr(player, 'skip_historical_predict', False):
                continue
            player.process_matches()
        self.apply_training_label_cutoff()


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
        if '--learning-v2' in sys.argv and '--lightgbm' in sys.argv:
            # LightGBM v3 live (premierlearning/lgbm/live.py); same raw-points output as v2.
            from premierlearning.lgbm.live import learn_lightgbm
            learn_lightgbm(self)
            return
        if '--learning-v2' in sys.argv:
            from premierlearning.learning_v2.learn import BLEND_FLAG, learn_blend, learn_h1
            if BLEND_FLAG in sys.argv:
                learn_blend(self)
            else:
                learn_h1(self)
            return
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

        self._horizon_only = horizon_only
        if horizon_only:
            print('Horizon-only: skipping minutes model; assuming 90 minutes per fixture')
            for future_match in future_matches:
                future_match.game_stats['minutes'] = 90.0
                future_match.set_minutes_from_stats()

        if minutes_only:
            minutes_reps = parse_minutes_reps()
            if minutes_reps > 1:
                print(
                    f'Minutes model: {minutes_reps} independent fits '
                    f'({MINUTES_REPS_FLAG}); drop val MAE > 8% over best'
                )
            minutes_preds = get_avg_predictions(
                minutes_clean_matches, future_clean_matches, get_input_keys('minutes'),
                'minutes', True, reps=minutes_reps
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
            points_loss = parse_points_loss()
            poisson = points_loss == 'poisson'
            minutes_weight = parse_points_minutes_weight()
            points_reps = parse_points_reps()
            sample_weight = None
            if minutes_weight:
                sample_weight = numpy.array(
                    [
                        float(m.get('game_stats', {}).get('minutes', 0) or 0) / 90.0
                        for m in valid_clean_matches
                    ],
                    dtype=float,
                )
                print(
                    'h1 points: minutes/90 sample weights on %s rows'
                    % len(valid_clean_matches)
                )
            if poisson:
                print(
                    'h1 points: Poisson loss, %s fit(s), log1p off'
                    % points_reps
                )
            # Horizon-1: points/90 per fixture (same construction as h5 / last_5), then × mins/90
            h1_pp90_labels = numpy.array(
                [[self._clean_match_points_per_90(m)] for m in valid_clean_matches],
                dtype=float,
            )
            learnt_stats['total_points'] = get_avg_predictions(
                valid_clean_matches, future_clean_matches, get_input_keys('total_points'),
                'total_points', True, output_learn=h1_pp90_labels,
                log_targets=not poisson, reps=points_reps,
                sample_weight=sample_weight,
                loss=points_loss if poisson else None,
            )
            minutes_reps = parse_minutes_reps()
            if minutes_reps > 1:
                print(
                    f'Minutes model: {minutes_reps} independent fits '
                    f'({MINUTES_REPS_FLAG}); drop val MAE > 8% over best'
                )
            learnt_stats['minutes'] = get_avg_predictions(
                minutes_clean_matches, future_clean_matches, get_input_keys('minutes'),
                'minutes', True, reps=minutes_reps
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
                future_matches[i].predicted_pp90 = pp90
                future_matches[i].game_stats['minutes'] = mins
                points = pp90 * effective_minutes_for_points(mins) / 90.0
                future_matches[i].game_stats['total_points'] = points
                future_matches[i].points = points
                future_matches[i].set_minutes_from_stats()

        # Next-N GW: label = points/90 over appearances in fixtures assigned to
        # the exact next N gameweeks. Blanks and doubles therefore retain their
        # real gameweek meaning.
        if parse_no_horizon_model():
            print(
                'Skipping horizon model (--h1-poisson / --no-horizon-model); '
                'h1 fixture scores stay unscaled'
            )
        else:
            horizon = self.horizon_gameweeks
            horizon_train = self.build_horizon_training_data(horizon)
            horizon_pred = self.build_horizon_predict_data(horizon)
            if horizon_train['player_clean'] and horizon_pred['player_clean']:
                horizon_reps = parse_horizon_reps()
                if horizon_reps > 1:
                    print(
                        f'Horizon-{horizon} model: {horizon_reps} independent fits '
                        f'({HORIZON_REPS_FLAG}); h1 points stay 1 fit'
                    )
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
                self.redistribute_horizon_points(
                    horizon_pred['players'],
                    horizon_pred['fixture_windows'],
                    horizon_pp90,
                    horizon,
                )
            else:
                print(
                    f'Skipping next-{horizon} points model '
                    '(insufficient training or predict rows)'
                )

        self.sync_predicted_pp90(future_matches)

    def sync_predicted_pp90(self, future_matches=None):
        """Store implied pp90 after models; chance is applied later in squadtools."""
        matches = future_matches
        if matches is None:
            matches = [
                match
                for player in self.current_season.players
                for match in player.future_matches
            ]
        for match in matches:
            minutes = self._match_minutes(match)
            match.predicted_pp90 = implied_pp90(match.points, minutes)

    @staticmethod
    def _clean_match_points_per_90(clean_match) -> float:
        pts = float(clean_match['game_stats'].get('total_points', 0) or 0)
        mins = float(clean_match['game_stats'].get('minutes', 0) or 0)
        if mins <= 0:
            return 0.0
        return pts * 90.0 / mins

    @staticmethod
    def _safe_print(text):
        """Windows consoles are often cp1252; don't crash on player names."""
        try:
            print(text)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, 'encoding', None) or 'ascii'
            print(text.encode(encoding, errors='replace').decode(encoding, errors='replace'))

    def _last_appearance_minutes(self, player, n: int):
        played = [
            m for m in (getattr(player, 'matches', None) or [])
            if self._match_minutes(m) > 0
        ][-n:]
        return [int(round(self._match_minutes(m))) for m in played]

    def print_top_players_next_horizon(self, n=None):
        """Compact post-run table: player, next-horizon pp90, team, price, last-7 mins."""
        raw = os.environ.get('PL_EMAIL_TOP_N')
        if n is None:
            n = int(raw) if raw not in (None, '') else 100
        if n <= 0:
            return
        horizon = self.horizon_gameweeks
        when = ''
        if self.historical_mode:
            trained_through = self._trained_through_gameweek()
            if trained_through is None:
                trained_through = max(0, int(self.as_of_gameweek) - 1)
            when = (
                ' — %s/%s GW%s (trained through GW%s)'
                % (
                    self.target_season_year,
                    self.target_season_year + 1,
                    self.as_of_gameweek,
                    trained_through,
                )
            )
        ranked = sorted(
            (
                p for p in self.current_season.players
                if p.matches_played >= MIN_APPEARANCES_TO_PREDICT
            ),
            key=lambda p: p.points_over_next_5_gameweeks,
            reverse=True,
        )[:n]
        title = (
            f'Top {n} by predicted points/{horizon} GWs '
            f'(pp90, assuming 90 minutes){when}'
        )
        mins_h = 'last%dm' % AVG_WINDOW
        header = (
            f'{"#":>3}  {"Player":<32}  {"Team":<16}  {"Price":>5}  {"pp90":>5}  '
            f'{"ls_pp90":>7}  {"ls_mpm":>6}  {mins_h:<27}'
        )
        lines = [title, header]
        self._safe_print('\n' + title)
        self._safe_print(header)
        payload = []
        for i, player in enumerate(ranked, start=1):
            total = float(player.points_over_next_5_gameweeks or 0)
            pp90 = total / horizon if horizon else 0.0
            price = float(getattr(player, 'current_cost', 0) or 0) / 10.0
            last_mins = self._last_appearance_minutes(player, AVG_WINDOW)
            mins_s = ','.join(str(m) for m in last_mins) if last_mins else '-'
            ls_pp90 = float(getattr(player, 'points_per_min_last_season', 0) or 0) * 90.0
            ls_mpm = float(getattr(player, 'last_season_mins_per_match', 0) or 0)
            row = (
                f'{i:>3}  {player.get_name():<32}  {str(player.team_name):<16}  '
                f'{price:>5.1f}  {pp90:>5.1f}  {ls_pp90:>7.1f}  {ls_mpm:>6.1f}  '
                f'{mins_s:<27}'
            )
            self._safe_print(row)
            lines.append(row)
            payload.append({
                'rank': i,
                'player': player.get_name(),
                'team': str(player.team_name),
                'price': '%.1f' % price,
                'pred': '%.1f' % pp90,
                'ls_pp90': '%.1f' % ls_pp90,
                'ls_mpm': '%.1f' % ls_mpm,
                'last7_minutes': last_mins,
            })
        metrics_dir = get_last_horizon_metrics_dir()
        if metrics_dir is not None:
            metrics_dir.mkdir(parents=True, exist_ok=True)
            body = '\n'.join(lines) + '\n'
            (metrics_dir / 'top100.txt').write_text(body, encoding='utf-8')
            (metrics_dir / 'email_top.txt').write_text(body, encoding='utf-8')
            (metrics_dir / 'email_top.json').write_text(
                json.dumps(payload, indent=2), encoding='utf-8'
            )

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
        """Pooled FPL pp90 over appearances in the window (skip 0-minute games)."""
        pts = 0.0
        mins = 0.0
        for match in played_matches:
            match_mins = cls._match_minutes(match)
            pts += float(getattr(match, 'points', 0) or 0)
            mins += match_mins
        if mins <= 0:
            return 0.0
        return pts * 90.0 / mins

    def build_horizon_training_data(self, horizon: int):
        """Build one row per player/GW over an exact gameweek window.

        A five-GW row includes every fixture assigned to those five events, so a
        double counts twice and a blank contributes no fixture.  The full window
        must be completed before the season's point-in-time training cutoff.
        """
        player_clean = []
        labels = []
        skipped_low_apps = 0
        seasons = list(self.past_seasons) + [self.current_season]
        min_played = 3

        for season in seasons:
            complete_through = int(
                getattr(season, 'training_complete_through_gw', 38) or 0
            )
            for player in season.players:
                matches = [
                    match
                    for match in player.matches
                    if not getattr(match, 'is_last_season_carry', False)
                    and getattr(match, 'gameweek', None) is not None
                ]
                seen_gameweeks = set()
                for match in matches:
                    if not match.valid_for_learning:
                        continue
                    anchor_gw = int(match.gameweek)
                    if anchor_gw in seen_gameweeks:
                        continue
                    seen_gameweeks.add(anchor_gw)
                    if anchor_gw + horizon - 1 > complete_through:
                        continue
                    window = [
                        candidate
                        for candidate in matches
                        if anchor_gw
                        <= int(candidate.gameweek)
                        < anchor_gw + horizon
                    ]
                    played = self._played_in_window(window)
                    if len(played) < min_played:
                        skipped_low_apps += 1
                        continue
                    clean = match.to_clean_match()
                    xg_sum, xc_sum = self._sum_team_xg_xc(window)
                    clean[TEAM_XG_NEXT_KEY] = xg_sum
                    clean[TEAM_XC_NEXT_KEY] = xc_sum
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
        """One row per player over fixtures in the next ``horizon`` GWs."""
        player_clean = []
        players = []
        fixture_windows = []
        start_gw = int(getattr(self.current_season, 'next_gameweek', 0) or 0)

        for player in self.current_season.players:
            if getattr(player, 'skip_historical_predict', False):
                continue
            if player.matches_played < MIN_APPEARANCES_TO_PREDICT:
                continue
            futs = [
                match
                for match in player.future_matches
                if (
                    match.gameweek is not None
                    and start_gw <= int(match.gameweek) < start_gw + horizon
                )
            ]
            if not futs:
                continue
            clean = futs[0].to_clean_match()
            xg_sum, xc_sum = self._sum_team_xg_xc(futs)
            clean[TEAM_XG_NEXT_KEY] = xg_sum
            clean[TEAM_XC_NEXT_KEY] = xc_sum
            apply_feature_interactions(clean)
            player_clean.append(clean)
            players.append(player)
            fixture_windows.append(futs)

        for i, clean in enumerate(player_clean):
            clean['id'] = i

        return {
            'player_clean': player_clean,
            'players': players,
            'fixture_windows': fixture_windows,
        }

    def _expected_minutes_over_horizon(self, player, futs, horizon: int) -> float:
        """Minutes scale to turn predicted pp90 into a points total for redistribute."""
        if getattr(self, '_horizon_only', False):
            return 90.0 * len(futs)
        pred_mins = sum(
            effective_minutes_for_points(self._match_minutes(m)) for m in futs
        )
        if pred_mins > 0:
            return pred_mins
        avg_mins = float(player.past_stats.get('minutes', {}).get('last_5_avg', 0) or 0)
        return effective_minutes_for_points(avg_mins) * len(futs)

    def redistribute_horizon_points(
        self,
        players,
        fixture_windows,
        horizon_pp90_preds,
        horizon: int,
    ):
        """Convert horizon pp90 → window total T via expected minutes; split by h1 fixture weights.

        Sum of next-`horizon` points equals T; each fixture's share follows the ratio of
        h1 expected points (already pp90 × minutes/90 when full learn ran).

        `--horizon-only` never trains minutes: assume 90 per fixture so
        T = pp90 × n.
        """
        max_reasonable_total = 12.0 * horizon
        failed_sanity = []
        for player, futs, pred in zip(players, fixture_windows, horizon_pp90_preds):
            pp90 = max(0.0, float(pred[0]) if hasattr(pred, '__len__') else float(pred))
            if not futs:
                continue
            expected_mins = self._expected_minutes_over_horizon(player, futs, horizon)
            T = pp90 * expected_mins / 90.0 if expected_mins > 0 else 0.0
            per_mins = expected_mins / len(futs) if expected_mins > 0 else 0.0

            weights = [max(0.0, float(m.points)) for m in futs]
            weight_sum = sum(weights)
            # These points are about to be overwritten, so keep them: the ratio
            # between the two is the only record of what the horizon model did.
            for match, weight in zip(futs, weights):
                match.h1_points = weight

            if T > max_reasonable_total or (weight_sum > 0 and T > weight_sum * 2.5):
                failed_sanity.append(
                    f'{player.get_name()}: T={T:.1f} pp90={pp90:.1f} h1_sum={weight_sum:.1f}'
                )

            if weight_sum <= 0:
                share = T / len(futs)
                for match in futs:
                    # An even split rescales nothing, so leave the factor unset
                    # rather than quote one the fixtures did not go through.
                    match.horizon_scale = None
                    match.points = share
                    match.game_stats['total_points'] = share
                    if per_mins > 0 and self._match_minutes(match) <= 0:
                        match.game_stats['minutes'] = per_mins
                        match.set_minutes_from_stats()
            else:
                scale = T / weight_sum
                for match, weight in zip(futs, weights):
                    points = T * (weight / weight_sum)
                    match.horizon_scale = scale
                    match.points = points
                    match.game_stats['total_points'] = points
                    if per_mins > 0 and self._match_minutes(match) <= 0:
                        match.game_stats['minutes'] = per_mins
                        match.set_minutes_from_stats()

        if failed_sanity:
            print(
                f'Horizon redistribute sanity check failed for {len(failed_sanity)} players '
                f'(T > {max_reasonable_total:.0f} or T > 2.5× h1 sum); scores still applied:'
            )
            if not os.environ.get('PL_SIMULATION_BUILD'):
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
