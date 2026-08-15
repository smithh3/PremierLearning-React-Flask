from typing import List, TypedDict


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
    'expected_goals',
    'expected_assists',
    'expected_goal_involvements',
    'expected_goals_conceded'
]

# These keys are for data that needs to be processed from the fpl api
BUILT_KEYS = [
    'conceded_2_plus',
    'conceded_4_plus',
    'minutes_60_plus'
]

CHANCE_TO_PLAY = 'chance_to_play'


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
    matches_played: int
    past_stats: PastStats
    ict_point_deviation: float
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
    team_last_season_finish: int
    opponent_last_season_finish: int
    played_last_season: int
    last_season_points_per_min: float
    game_stats: GameStats


STANDARD_STAT_KEYS = [
    'matches_played',
    'ict_point_deviation',
    'is_gkp',
    'is_def',
    'is_mid',
    'is_att',
    'is_home',
    'team_average_points',
    'team_average_scored',
    'team_average_conceded',
    'opponent_average_points',
    'opponent_average_scored',
    'opponent_average_conceded',
    'team_last_season_finish',
    'opponent_last_season_finish',
    'played_last_season',
    'last_season_points_per_min',
    'new_stats_included'
]

POINTS_KEYS = [
    'total_points',
    # 'ict_index',
    # 'bonus',
    'expected_goals',
    'expected_assists',
    'expected_goal_involvements',
    'expected_goals_conceded'
]

NEW_STATS = [
    'expected_goals',
    'expected_assists',
    'expected_goal_involvements',
    'expected_goals_conceded'
]

GOALS_KEYS = [
    'goals_scored',
    'assists',
    'bonus',
    'ict_index',
    'expected_goals',
    'expected_assists',
    'expected_goal_involvements'
]

DEF_KEYS = [
    'clean_sheets',
    'goals_conceded',
    'saves',
    'bonus',
    'expected_goals_conceded'
]

OTHER_KEYS = [
    'bonus',
    'ict_index',
    'goals_scored',
    'assists',
    'saves',
    'clean_sheets',
    'expected_goals',
    'expected_assists',
    'expected_goal_involvements',
    'expected_goals_conceded'
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
        return ['minutes']
    return [key]


def get_inputs_from_data_key(clean_matches: List[CleanMatch], special_keys: List[str], include_standard_stats) -> list:
    inputs = []
    for clean_match in clean_matches:
        clean_match_array = {}
        if include_standard_stats:
            for data_key in STANDARD_STAT_KEYS:
                clean_match_array[data_key] = clean_match[data_key]
        for data_key in special_keys:
            last_n_stats = clean_match['past_stats'][data_key]
            last_n_stats_with_name = {}
            for last_n_key, val in last_n_stats.items():
                last_n_stats_with_name[f'{data_key}_{last_n_key}'] = val
            clean_match_array.update(last_n_stats_with_name)
        inputs.append(clean_match_array)
    return inputs


def get_outputs_from_data_key(clean_matches: List[CleanMatch], key_name: str) -> list:
    outputs = []
    for clean_match in clean_matches:
        outputs.append({ key_name: clean_match['game_stats'][key_name] })
    return outputs
