from typing import List, TypedDict
import math
import os
import sys

from premierlearning.config_constants import (
    FORM_AVG_WINDOW,
    POINTS_HORIZON,
    PRIOR_EARLY_TAU,
    SHORT_FORM_WINDOW,
)

TEAM_XG_NEXT_KEY = 'team_xg_next_%s' % POINTS_HORIZON
TEAM_XC_NEXT_KEY = 'team_xc_next_%s' % POINTS_HORIZON
TEAM_XG_NEXT_X_OUTFIELD_KEY = '%s_x_outfield' % TEAM_XG_NEXT_KEY
LAST_SEASON_POINTS_PER_90_KEY = 'last_season_points_per_90_excl_null'
LAST_SEASON_XGI_PER_90_KEY = 'last_season_xgi_per_90_excl_null'
TWO_SEASON_POINTS_PER_90_KEY = 'two_season_prior_points_per_90_excl_null'
TWO_SEASON_XGI_PER_90_KEY = 'two_season_prior_xgi_per_90_excl_null'
LAST_SEASON_PP90_X_EARLY_KEY = 'last_season_points_per_90_excl_null_x_early'
TWO_SEASON_PP90_X_EARLY_KEY = 'two_season_prior_points_per_90_excl_null_x_early'
MOST_EXPENSIVE_POS_X_GKP_X_EARLY_KEY = 'most_expensive_in_pos_x_gkp_x_early'
DEFCON_X_NON_ATT_KEY = 'defcon_last_%s_per_90_excl_null_x_non_att' % FORM_AVG_WINDOW
SEASON_POINTS_PER_90_KEY = 'season_points_per_90_excl_null'
SEASON_XGI_PER_90_KEY = 'season_xgi_per_90_excl_null'
SEASON_XGC_PER_90_KEY = 'season_xgc_per_90_excl_null'
SEASON_DEFCON_PER_90_KEY = 'season_defcon_per_90_excl_null'
SEASON_MINS_PER_MATCH_KEY = 'season_mins_per_match'

_PAST_STAT_NAMES = {
    'total_points': 'points',
    'expected_goal_involvements': 'xgi',
    'expected_goals_conceded': 'xgc',
    'defensive_contribution': 'defcon',
}
_PAST_WINDOW_NAMES = {
    'last_5_avg': 'last_%s_per_90_excl_null' % SHORT_FORM_WINDOW,
    'avg': 'last_%s_per_90_excl_null' % FORM_AVG_WINDOW,
}


def past_stat_feature_name(stat_key, window_key) -> str:
    """Model column for a rolling appearance window (minutes >= FORM_MIN_MINUTES)."""
    stat = _PAST_STAT_NAMES.get(stat_key, stat_key)
    window = _PAST_WINDOW_NAMES.get(window_key, window_key)
    return '%s_%s' % (stat, window)


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


def use_ls_pp90_x_mins() -> bool:
    """Last-season pp90 × mins/match. Off by default; on via --ls-pp90-x-mins or PL_LS_PP90_X_MINS=1."""
    if '--no-ls-pp90-x-mins' in sys.argv:
        return False
    if '--ls-pp90-x-mins' in sys.argv:
        return True
    raw = os.environ.get('PL_LS_PP90_X_MINS', '0')
    return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')


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
    team_xg_next_5: float
    team_xc_next_5: float
    team_xg_last_5: float
    team_xc_last_5: float
    played_last_season: int
    last_season_points_per_90_excl_null: float
    last_season_xgi_per_90_excl_null: float
    two_season_prior_points_per_90_excl_null: float
    two_season_prior_xgi_per_90_excl_null: float
    last_season_minutes: float
    last_season_mins_per_match: float
    season_mins_per_match: float
    last_season_fixtures: int
    last_season_fixture_frac: float
    last_7_mins_per_match: float
    two_season_prior_mins_per_match: float
    last_season_pp90_x_mins: float
    season_points_per_90_excl_null: float
    season_xgi_per_90_excl_null: float
    season_xgc_per_90_excl_null: float
    season_defcon_per_90_excl_null: float
    changed_club: int
    club_move_team_diff: float
    is_most_expensive_in_pos: int
    new_stats_included: int
    defcon_last_7_per_90_excl_null_x_non_att: float
    team_xg_next_5_x_outfield: float
    last_season_points_per_90_excl_null_x_early: float
    two_season_prior_points_per_90_excl_null_x_early: float
    early_x_last_season_mins: float
    most_expensive_in_pos_x_gkp_x_early: float
    ls_mpm_x_new_club: float
    mins_last_1: float
    mins_last_2: float
    mins_last_5: float
    last_7_pts_per_appearance: float
    game_stats: GameStats


# Base player / context features (no ict; fixture difficulty via team xg/xc).
#
# Season-start experiment notes (pair with player.py form comment):
#   - matches_played was briefly commented out when it mismatched train/serve;
#     blanking it made zero-form players worse — keep it on for now.
# last_season_* / two_season_prior_* are full-season pp90 rates (DNP skipped —
# same excl_null as form last-3/7 and season_*).
# Early-season fade is last_season_points_per_90_excl_null_x_early.
_BASE_STANDARD = [
    'matches_played',
    'is_gkp',
    'is_def',
    'is_mid',
    'is_att',
    LAST_SEASON_POINTS_PER_90_KEY,
    LAST_SEASON_XGI_PER_90_KEY,
    TWO_SEASON_POINTS_PER_90_KEY,
    # TWO_SEASON_XGI_PER_90_KEY,
    # 'changed_club',
    'new_stats_included',
    SEASON_POINTS_PER_90_KEY,
    SEASON_XGI_PER_90_KEY,
    SEASON_XGC_PER_90_KEY,
    SEASON_DEFCON_PER_90_KEY,
]

_INTERACTION_KEYS = [
    DEFCON_X_NON_ATT_KEY,
    TEAM_XG_NEXT_X_OUTFIELD_KEY,
    LAST_SEASON_PP90_X_EARLY_KEY,
    # TWO_SEASON_PP90_X_EARLY_KEY,
]

# Shrinks last-season pp90 when the player did not play full matches.
# pp90 * (mins_per_match / 90): a 90-minute starter keeps full rate; a 30-minute
# super-sub is scaled to a third. Gated by use_ls_pp90_x_mins().
LS_PP90_X_MINS_KEY = 'last_season_pp90_x_mins'

LAST_SEASON_MINUTES_CAP = 38.0 * 90.0  # scale last-season minutes to ~[0, 1]

# Minutes-only root features (not in points / horizon nets).
# season_mins_per_match is the default rate: this-season fixtures including DNPs.
# last_season / two_season_prior are full seasons, DNP-inclusive (early-GW prior).
# Recency last-1/2/5 still fills from the end of last season.
MINUTES_ONLY_ROOT_KEYS = [
    SEASON_MINS_PER_MATCH_KEY,
    'early_x_last_season_mins',
    'last_season_mins_per_match',
    'two_season_prior_mins_per_match',
    'last_season_fixture_frac',
    'ls_mpm_x_new_club',
    'mins_last_1',
    'mins_last_2',
    'mins_last_5',
    'last_7_pts_per_appearance',
    MOST_EXPENSIVE_POS_X_GKP_X_EARLY_KEY,
]
MINUTES_ONLY_ROOT_DEFAULTS = {
    SEASON_MINS_PER_MATCH_KEY: 0.0,
    'early_x_last_season_mins': 0.0,
    'last_season_mins_per_match': 0.0,
    'two_season_prior_mins_per_match': 0.0,
    'last_season_fixture_frac': 0.0,
    'ls_mpm_x_new_club': 0.0,
    'mins_last_1': 0.0,
    'mins_last_2': 0.0,
    'mins_last_5': 0.0,
    'last_7_pts_per_appearance': 0.0,
    MOST_EXPENSIVE_POS_X_GKP_X_EARLY_KEY: 0.0,
}

# Dropped from the minutes net: points-rate priors, fixture xG/xC, and the
# new-stats flag. Appearance-only past_stats (minutes_last_5_avg / chance_to_play)
# are also skipped in get_inputs_from_data_key — they exclude DNPs via carry.
MINUTES_EXCLUDE_STANDARD_KEYS = {
    LAST_SEASON_POINTS_PER_90_KEY,
    LAST_SEASON_XGI_PER_90_KEY,
    TWO_SEASON_POINTS_PER_90_KEY,
    TWO_SEASON_XGI_PER_90_KEY,
    'last_season_pp90_x_mins',
    LAST_SEASON_PP90_X_EARLY_KEY,
    TWO_SEASON_PP90_X_EARLY_KEY,
    SEASON_POINTS_PER_90_KEY,
    SEASON_XGI_PER_90_KEY,
    SEASON_XGC_PER_90_KEY,
    SEASON_DEFCON_PER_90_KEY,
    DEFCON_X_NON_ATT_KEY,
    TEAM_XG_NEXT_X_OUTFIELD_KEY,
    'new_stats_included',
    TEAM_XG_NEXT_KEY,
    TEAM_XC_NEXT_KEY,
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
    TEAM_XG_NEXT_KEY,
    TEAM_XC_NEXT_KEY,
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
    """Position-gated products; call after team_xg_next_N is finalized."""
    non_att = int(bool(clean.get('is_gkp') or clean.get('is_def') or clean.get('is_mid')))
    outfield = int(bool(clean.get('is_att') or clean.get('is_mid') or clean.get('is_def')))
    new_stats = float(clean.get('new_stats_included') or 0)
    past = clean.get('past_stats') or {}
    defcon_avg = float((past.get('defensive_contribution') or {}).get('avg') or 0)
    team_xg_n = float(clean.get(TEAM_XG_NEXT_KEY) or 0)
    apps = float(clean.get('matches_played') or 0)
    early_weight = math.exp(-apps / PRIOR_EARLY_TAU)
    last_pp90 = float(clean.get(LAST_SEASON_POINTS_PER_90_KEY) or 0)
    two_pp90 = float(clean.get(TWO_SEASON_POINTS_PER_90_KEY) or 0)

    clean[DEFCON_X_NON_ATT_KEY] = new_stats * defcon_avg * non_att
    clean[TEAM_XG_NEXT_X_OUTFIELD_KEY] = team_xg_n * outfield
    clean[LAST_SEASON_PP90_X_EARLY_KEY] = last_pp90 * early_weight
    clean[TWO_SEASON_PP90_X_EARLY_KEY] = two_pp90 * early_weight
    last_mins = float(clean.get('last_season_minutes') or 0)
    volume = min(1.0, last_mins / LAST_SEASON_MINUTES_CAP) if LAST_SEASON_MINUTES_CAP else 0.0
    clean['early_x_last_season_mins'] = early_weight * volume
    ls_mpm = float(clean.get('last_season_mins_per_match') or 0)
    club_factor = float(clean.get('club_move_team_diff') or 1.0)
    clean['ls_mpm_x_new_club'] = ls_mpm * club_factor
    ls_pp90 = float(clean.get(LAST_SEASON_POINTS_PER_90_KEY) or 0)
    clean[LS_PP90_X_MINS_KEY] = ls_pp90 * (ls_mpm / 90.0)
    most_exp = float(clean.get('is_most_expensive_in_pos') or 0)
    is_gkp = float(clean.get('is_gkp') or 0)
    clean[MOST_EXPENSIVE_POS_X_GKP_X_EARLY_KEY] = most_exp * is_gkp * early_weight


def get_interaction_keys() -> List[str]:
    keys = list(_INTERACTION_KEYS)
    if use_ls_pp90_x_mins():
        keys.append(LS_PP90_X_MINS_KEY)
    return keys


def get_standard_stat_keys() -> List[str]:
    if use_team_expectancies():
        keys = list(_BASE_STANDARD) + list(_EXPECTANCY_FIXTURE_KEYS) + get_interaction_keys()
    else:
        keys = list(_BASE_STANDARD) + list(_LEGACY_FIXTURE_KEYS) + get_interaction_keys()
    seen = set()
    unique = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


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
                last_n_stats_with_name[past_stat_feature_name(data_key, last_n_key)] = val
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
