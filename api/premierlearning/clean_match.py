from typing import List, TypedDict
import math
import sys

from premierlearning.config_constants import PRIOR_EARLY_TAU


class StatWindow(TypedDict):
    # last: int
    # last_3_avg: float
    last_5_avg: float
    avg: float
    # deviation: float


class PastStats(TypedDict):
    total_points: StatWindow
    goals_scored: StatWindow
    assists: StatWindow
    yellow_cards: StatWindow
    red_cards: StatWindow
    clean_sheets: StatWindow
    goals_conceded: StatWindow
    bonus: StatWindow
    saves: StatWindow
    # ict_index: StatWindow
    minutes: StatWindow
    conceded_2_plus: StatWindow
    conceded_4_plus: StatWindow
    minutes_60_plus: StatWindow
    chance_to_play: StatWindow


class GameStats(TypedDict):
    total_points: float
    goals_scored: float
    assists: float
    yellow_cards: float
    red_cards: float
    clean_sheets: float
    goals_conceded: float
    bonus: float
    saves: float
    # ict_index: float
    minutes: float
    conceded_2_plus: float
    conceded_4_plus: float
    minutes_60_plus: float
    chance_to_play: float


# These keys directly access data from the fpl api
STAT_KEYS = [
    'total_points',
    'goals_scored',
    'assists',
    # 'yellow_cards',
    # 'red_cards',
    'clean_sheets',
    'goals_conceded',
    # 'bonus',
    'minutes',
    # 'penalties_saved',
    # 'penalties_missed',
    # 'own_goals'
    'saves',
    # 'ict_index',
    'expected_goal_involvements',
    'expected_goals_conceded',
    'defensive_contribution'
]

# These keys are for data that needs to be processed from the fpl api
BUILT_KEYS = [
    'conceded_2_plus',
    'conceded_4_plus',
    'minutes_60_plus'
]

CHANCE_TO_PLAY = 'chance_to_play'

# Past-stat keys stored as per-minute rates (not raw totals)
PER_MINUTE_STAT_KEYS = set(STAT_KEYS + BUILT_KEYS) - {'minutes'}


def use_team_expectancies() -> bool:
    return '--no-team-expectancies' not in sys.argv


def get_default_past_stats() -> PastStats:
    stats: PastStats = {}
    for key in STAT_KEYS + BUILT_KEYS + [CHANCE_TO_PLAY]:
        stats[key] = get_default_stat_window()
    return stats

def get_default_game_stats() -> GameStats:
    stats: GameStats = {}
    for key in STAT_KEYS + BUILT_KEYS + [CHANCE_TO_PLAY]:
        stats[key] = 0
    return stats

def get_default_stat_window() -> StatWindow:
    stat_window: StatWindow = {
        # 'last': 0,
        # 'last_3_avg': 0,
        'last_5_avg': 0,
        'avg': 0,
        # 'deviation': 0
    }
    return stat_window


class CleanMatch(TypedDict):
    is_valid: bool
    is_valid_for_minutes: bool
    matches_played: int
    past_stats: PastStats
    is_gkp: int
    is_def: int
    is_mid: int
    is_att: int
    is_home: int
    team_average_points: float
    team_average_scored: float
    team_average_conceded: float
    opponent_average_points: float
    opponent_average_scored: float
    opponent_average_conceded: float
    team_xg_over_n: float
    team_xc_over_n: float
    team_xg_last_5: float
    team_xc_last_5: float
    played_last_season: int
    last_season_points_per_90: float
    last_season_xgi_per_90: float
    last_season_minutes: float
    last_season_mins_per_match: float
    last_season_fixtures: int
    last_7_mins_per_match: float
    two_season_prior_points_per_90: float
    two_season_prior_mins_per_match: float
    last_season_h2_mins_per_match: float
    last_two_seasons_mins_per_match: float
    last_season_h2_fixture_frac: float
    changed_club: int
    club_move_team_diff: float
    new_stats_included: int
    # Position-gated interactions
    defcon_avg_x_new_stats_x_non_att: float
    team_xg_over_n_x_outfield: float
    early_x_last_season_mins: float
    ls_mpm_x_new_club: float
    mins_last_1: float
    mins_last_2_avg: float
    mins_last_5_avg: float
    last_7_pts_per_appearance: float
    # last_season_prior_x_early: float
    game_stats: GameStats


# Base player / context features (no ict; fixture difficulty via team xg/xc).
#
# Season-start experiment notes (pair with player.py form comment):
#   - matches_played was briefly commented out when it mismatched train/serve;
#     blanking it made zero-form players worse — keep it on for now.
#   - played_last_season / two_season_prior / last_season_prior_x_early were
#     tried as early-season priors; left off while we rely on carry form +
#     last_season_*_per_90 instead. Re-enable and retrain to A/B.
_BASE_STANDARD = [
    'matches_played',
    'is_gkp',
    'is_def',
    'is_mid',
    'is_att',
    # 'played_last_season',  # temporarily disabled — early-season prior experiment
    # last_season_*_per_90 are points-rate priors; excluded from the minutes net
    # (see MINUTES_EXCLUDE_STANDARD_KEYS) so super-subs don't look like starters.
    'last_season_points_per_90',
    'last_season_xgi_per_90',
    # 'two_season_prior_points_per_90',  # temporarily disabled — see PRIOR_* in config
    # changed_club temporarily disabled — binary 0/1 was boosting movers (e.g. Anderson)
    # instead of punishing early minutes/points. Detection still runs on the player;
    # re-enable here once encoded differently (see chat notes).
    # 'changed_club',
    'new_stats_included',
]

_INTERACTION_KEYS = [
    # defcon only meaningful for non-attackers, and only when new stats exist
    'defcon_avg_x_new_stats_x_non_att',
    # team attack xG for outfield only (not GK)
    'team_xg_over_n_x_outfield',
    # OLD early-season prior interaction (needs PRIOR_EARLY_TAU import):
    # # prior fades as this-season appearances grow: (last + 0.5*two_ago) * exp(-apps/tau)
    # 'last_season_prior_x_early',
]

LAST_SEASON_MINUTES_CAP = 38.0 * 90.0  # scale last-season minutes to ~[0, 1]

# Minutes-only root features (not in points / horizon nets).
# H2 = second half of last season; 2s = prev2 season + H1 of last season.
# Recency stretches into H2 and blends toward H2 mins/match near GW0.
# Predict (not train): if the last two fixtures were appearances, 3+ consecutive
# DNPs are dropped from H2 / 2s / recency (injury, not availability).
# last_season_h2_fixture_frac stays raw coverage so dropped holes do not look
# like a short FPL history.
MINUTES_ONLY_ROOT_KEYS = [
    'early_x_last_season_mins',
    'last_season_h2_mins_per_match',
    'last_two_seasons_mins_per_match',
    'last_season_h2_fixture_frac',
    'ls_mpm_x_new_club',
    'mins_last_1',
    'mins_last_2_avg',
    'mins_last_5_avg',
    'last_7_pts_per_appearance',
]
MINUTES_ONLY_ROOT_DEFAULTS = {
    'early_x_last_season_mins': 0.0,
    'last_season_h2_mins_per_match': 0.0,
    'last_two_seasons_mins_per_match': 0.0,
    'last_season_h2_fixture_frac': 0.0,
    'ls_mpm_x_new_club': 0.0,
    'mins_last_1': 0.0,
    'mins_last_2_avg': 0.0,
    'mins_last_5_avg': 0.0,
    'last_7_pts_per_appearance': 0.0,
}

# Dropped from the minutes net: points-rate priors, fixture xG/xC, and the
# new-stats flag. Appearance-only past_stats (minutes_last_5_avg / chance_to_play)
# are also skipped in get_inputs_from_data_key — they exclude DNPs via carry.
MINUTES_EXCLUDE_STANDARD_KEYS = {
    'last_season_points_per_90',
    'last_season_xgi_per_90',
    'defcon_avg_x_new_stats_x_non_att',
    'team_xg_over_n_x_outfield',
    'new_stats_included',
    'team_xg_over_n',
    'team_xc_over_n',
    'team_xg_last_5',
    'team_xc_last_5',
}

_LEGACY_FIXTURE_KEYS = [
    'is_home',
    'team_average_points',
    'team_average_scored',
    'team_average_conceded',
    'opponent_average_points',
    'opponent_average_scored',
    'opponent_average_conceded',
]

_EXPECTANCY_FIXTURE_KEYS = [
    'team_xg_over_n',
    'team_xc_over_n',
    'team_xg_last_5',
    'team_xc_last_5',
]

LEGACY_OPPONENT_FEATURE_KEYS = [
    'is_home',
    'opponent_average_points',
    'opponent_average_scored',
    'opponent_average_conceded',
]

# Flat expectancy model: team xg/xc live in _EXPECTANCY_FIXTURE_KEYS only (no separate opp tower)
EXPECTANCY_OPPONENT_FEATURE_KEYS = []


def apply_feature_interactions(clean: dict) -> None:
    """Position-gated products; call after team_xg_over_n is finalized."""
    non_att = int(bool(clean.get('is_gkp') or clean.get('is_def') or clean.get('is_mid')))
    outfield = int(bool(clean.get('is_att') or clean.get('is_mid') or clean.get('is_def')))
    new_stats = float(clean.get('new_stats_included') or 0)
    past = clean.get('past_stats') or {}
    defcon_avg = float((past.get('defensive_contribution') or {}).get('avg') or 0)
    team_xg_n = float(clean.get('team_xg_over_n') or 0)

    # OLD early-season prior (pair with 'last_season_prior_x_early' in _INTERACTION_KEYS):
    # last_pp90 = float(clean.get('last_season_points_per_90') or 0)
    # two_pp90 = float(clean.get('two_season_prior_points_per_90') or 0)
    # apps = float(clean.get('matches_played') or 0)
    # prior_blend = last_pp90 + TWO_SEASON_PRIOR_SCALE * two_pp90
    # early_weight = math.exp(-apps / PRIOR_EARLY_TAU)  # 1.0 at 0 apps

    clean['defcon_avg_x_new_stats_x_non_att'] = new_stats * defcon_avg * non_att
    clean['team_xg_over_n_x_outfield'] = team_xg_n * outfield
    # Minutes only (included via MINUTES_ONLY_ROOT_KEYS): early-season weight
    # times last-season minutes volume. No raw GW 0-3 flag.
    apps = float(clean.get('matches_played') or 0)
    last_mins = float(clean.get('last_season_minutes') or 0)
    early_weight = math.exp(-apps / PRIOR_EARLY_TAU)
    volume = min(1.0, last_mins / LAST_SEASON_MINUTES_CAP) if LAST_SEASON_MINUTES_CAP else 0.0
    clean['early_x_last_season_mins'] = early_weight * volume
    h2_mpm = float(clean.get('last_season_h2_mins_per_match') or 0)
    # H2 mins/match × new-club factor (1.0 if same club / new-to-PL).
    club_factor = float(clean.get('club_move_team_diff') or 1.0)
    clean['ls_mpm_x_new_club'] = h2_mpm * club_factor
    # clean['last_season_prior_x_early'] = prior_blend * early_weight


def get_standard_stat_keys() -> List[str]:
    if use_team_expectancies():
        return list(_BASE_STANDARD) + list(_EXPECTANCY_FIXTURE_KEYS) + list(_INTERACTION_KEYS)
    return list(_BASE_STANDARD) + list(_LEGACY_FIXTURE_KEYS) + list(_INTERACTION_KEYS)


def get_opponent_feature_keys() -> List[str]:
    if use_team_expectancies():
        return list(EXPECTANCY_OPPONENT_FEATURE_KEYS)
    return list(LEGACY_OPPONENT_FEATURE_KEYS)


def get_player_standard_stat_keys() -> List[str]:
    opp_keys = set(get_opponent_feature_keys())
    return [key for key in get_standard_stat_keys() if key not in opp_keys]


POINTS_KEYS = [
    'total_points',
    # 'ict_index',
    # 'bonus',
    'expected_goal_involvements',
    'expected_goals_conceded',
    'defensive_contribution'
]

NEW_STATS = [
    'expected_goal_involvements',
    'expected_goals_conceded',
    'defensive_contribution'
]

GOALS_KEYS = [
    'goals_scored',
    'assists',
    'bonus',
    'ict_index',
    'expected_goal_involvements'
]

DEF_KEYS = [
    'clean_sheets',
    'goals_conceded',
    'saves',
    'bonus',
    'expected_goals_conceded',
    'defensive_contribution'
]

OTHER_KEYS = [
    'bonus',
    'ict_index',
    'goals_scored',
    'assists',
    'saves',
    'clean_sheets',
    'expected_goal_involvements',
    'expected_goals_conceded',
    'defensive_contribution'
]

def get_input_keys(key) -> List[str]:
    if key in ('bonus', 'ict_index'):
        return OTHER_KEYS
    elif key in ('goals_scored', 'assists'):
        return GOALS_KEYS
    elif key == 'total_points':
        return POINTS_KEYS
    elif key in ('clean_sheets', 'goals_conceded', 'saves', 'conceded_2_plus', 'conceded_4_plus'):
        return DEF_KEYS
    elif key in ('minutes_60_plus', 'minutes'):
        # Minutes net does not use appearance-only past_stats (minutes last_5 / avg
        # or chance_to_play). Recency + priors live in MINUTES_ONLY_ROOT_KEYS.
        # Keep 'minutes' in the list so get_inputs_from_data_key enables minutes_mode.
        return ['minutes']
    return [key]


def get_inputs_from_data_key(clean_matches: List[CleanMatch], special_keys: List[str], include_standard_stats,
                             include_opponent_features: bool = True) -> list:
    inputs = []
    standard_keys = get_standard_stat_keys()
    if not include_opponent_features:
        opp_keys = set(get_opponent_feature_keys())
        standard_keys = [k for k in standard_keys if k not in opp_keys]
    for clean_match in clean_matches:
        clean_match_array = {}
        minutes_mode = any(k in special_keys for k in ('minutes', 'minutes_60_plus'))
        if include_standard_stats:
            for data_key in standard_keys:
                if minutes_mode and data_key in MINUTES_EXCLUDE_STANDARD_KEYS:
                    continue
                clean_match_array[data_key] = clean_match[data_key]
            if minutes_mode:
                for data_key in MINUTES_ONLY_ROOT_KEYS:
                    val = clean_match.get(data_key)
                    if val is None:
                        val = MINUTES_ONLY_ROOT_DEFAULTS.get(data_key, 0.0)
                    clean_match_array[data_key] = float(val)
        for data_key in special_keys:
            if minutes_mode:
                # Skip past_stats minutes / chance_to_play: carry is appearance-only
                # (DNPs dropped), which is the opposite of what the minutes net needs.
                continue
            last_n_stats = clean_match['past_stats'][data_key]
            last_n_stats_with_name = {}
            for last_n_key, val in last_n_stats.items():
                last_n_stats_with_name[f'{data_key}_{last_n_key}'] = val
            clean_match_array.update(last_n_stats_with_name)
        inputs.append(clean_match_array)
    return inputs


def get_opponent_feature_vector(clean_match: CleanMatch) -> list:
    return [float(clean_match[key]) for key in get_opponent_feature_keys()]


def get_outputs_from_data_key(clean_matches: List[CleanMatch], key_name: str) -> list:
    outputs = []
    for clean_match in clean_matches:
        outputs.append({ key_name: clean_match['game_stats'][key_name] })
    return outputs
