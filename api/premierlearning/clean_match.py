from typing import List, TypedDict
import sys

# from premierlearning.config_constants import PRIOR_EARLY_TAU, TWO_SEASON_PRIOR_SCALE


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
    two_season_prior_points_per_90: float
    new_stats_included: int
    # Position-gated interactions
    defcon_avg_x_new_stats_x_non_att: float
    team_xg_over_n_x_outfield: float
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
    'last_season_points_per_90',
    'last_season_xgi_per_90',
    # 'two_season_prior_points_per_90',  # temporarily disabled — see PRIOR_* in config
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
        # Same form towers as points, plus minutes history and chance-to-play
        return list(POINTS_KEYS) + ['minutes', CHANCE_TO_PLAY]
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
        if include_standard_stats:
            for data_key in standard_keys:
                clean_match_array[data_key] = clean_match[data_key]
        for data_key in special_keys:
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
