from datetime import datetime
from abc import ABC, abstractmethod
import math

from premierlearning.clean_match import (
    STAT_KEYS, NEW_STATS, CleanMatch, GameStats, PastStats,
    get_default_game_stats, get_default_past_stats, get_default_stat_window,
    apply_feature_interactions,
)
from premierlearning.config_constants import (
    PRIOR_EARLY_TAU,
    HALF_SEASON_FIXTURES,
    INJURY_DNP_STREAK,
    RETURN_CONSECUTIVE_APPS,
)
from premierlearning.helper_functions import coerce_bool, coerce_optional_int


class RawMatch(ABC):

    def __init__(self):
        self.player_id = None
        self.minutes = None
        self.points = None
        self.gameweek = None
        self.opponent = None
        self.points_error = 0.0

    def format_as_db_insert(self):
        return (
            self.player_id,
            float(self.minutes),
            float(self.points),
            self.gameweek,
            self.opponent
        )

    def format_as_future_db_insert(self):
        return (
            self.player_id,
            float(self.minutes),
            float(self.points),
            self.gameweek,
            self.opponent,
            float(getattr(self, 'points_error', 0.0) or 0.0),
        )
    
    def within_n_gameweeks(self, current_gameweek, n):
        if self.gameweek is None or current_gameweek is None:
            return False
        return int(self.gameweek) - int(current_gameweek) <= n


def string_to_bool(true_or_false):
    """Back-compat alias; prefer coerce_bool."""
    return coerce_bool(true_or_false)


class AbstractMatch(RawMatch):
    
    def __init__(self, player, match_json: dict, teams: dict):
        super().__init__()

        self.player = player
        self.player_id = self.player.id
        self.gameweek = self.get_gw(match_json)

        self.is_gkp = int(player.position == 1)
        self.is_def = int(player.position == 2)
        self.is_mid = int(player.position == 3)
        self.is_att = int(player.position == 4)

        self.minutes = 0
        self.home = self.get_home(match_json)
        self.points = 0
        self.fixture_id = self.get_fixture_id(match_json)
        self.opponent_team = teams[self.get_opponent_id(match_json)]
        self.opponent_team_match = self.get_opponent_team_match()
        self.team = self.get_team(teams)
        self.team_match = self.get_team_match()
        self.opponent = self.opponent_team.name

        self.ict_point_regression_deviation = 0
        self.matches_played = 0
        self.matches_available = 0
        self.valid_for_learning = False
        self.valid_for_minutes = False

        self.past_stats: PastStats = get_default_past_stats()
        self.game_stats: GameStats = get_default_game_stats()
        self.new_stats_included = True

    
    @abstractmethod
    def get_gw(self, match_json: dict):
        pass

    @abstractmethod
    def get_home(self, match_json: dict):
        pass

    @abstractmethod
    def get_fixture_id(self, match_json: dict):
        pass

    @abstractmethod
    def get_opponent_id(self, match_json: dict):
        pass

    @abstractmethod
    def get_opponent_team_match(self):
        pass

    @abstractmethod
    def get_team(self, teams):
        pass

    @abstractmethod
    def get_team_match(self):
        pass
    
    def to_clean_match(self) -> CleanMatch:
        clean_match: CleanMatch = {
            'player_name': self.player.get_name(),
            'player_id': self.player.id,
            'is_valid': self.valid_for_learning,
            'is_valid_for_minutes': getattr(self, 'valid_for_minutes', False),
            'matches_played': self.matches_played,
            'past_stats': self.past_stats,
            'is_gkp': self.is_gkp,
            'is_def': self.is_def,
            'is_mid': self.is_mid,
            'is_att': self.is_att,
            'is_home': int(coerce_bool(self.home)),
            'team_average_points': self.team_match.average_points_in_previous_matches,
            'team_average_scored': self.team_match.average_scored_in_previous_matches,
            'team_average_conceded': self.team_match.average_conceded_in_previous_matches,
            'opponent_average_points': self.opponent_team_match.average_points_in_previous_matches,
            'opponent_average_scored': self.opponent_team_match.average_scored_in_previous_matches,
            'opponent_average_conceded': self.opponent_team_match.average_conceded_in_previous_matches,
            'team_xg_over_n': float(getattr(self.team_match, 'predicted_xg', 0.0) or 0.0),
            'team_xc_over_n': float(getattr(self.team_match, 'predicted_xc', 0.0) or 0.0),
            'team_xg_last_5': 0.0,
            'team_xc_last_5': 0.0,
            'played_last_season': int(self.player.played_last_season),
            'last_season_points_per_90': self.player.points_per_min_last_season * 90.0,
            'last_season_xgi_per_90': (
                getattr(self.player, 'xgi_per_min_last_season', 0) or 0
            ) * 90.0,
            'last_season_minutes': float(getattr(self.player, 'minutes_last_season', 0) or 0),
            'last_season_mins_per_match': float(
                getattr(self.player, 'last_season_mins_per_match', 0) or 0
            ),
            'last_season_fixtures': int(
                getattr(self.player, 'last_season_fixtures', 0) or 0
            ),
            'last_7_mins_per_match': float(
                getattr(self.player, 'last_7_mins_per_match', 0) or 0
            ),
            'two_season_prior_points_per_90': (
                getattr(self.player, 'points_per_min_two_seasons_ago', 0) or 0
            ) * 90.0,
            'two_season_prior_mins_per_match': float(
                getattr(self.player, 'two_season_prior_mins_per_match', 0) or 0
            ),
            'last_season_h2_mins_per_match': float(
                getattr(self.player, 'last_season_h2_mins_per_match', 0) or 0
            ),
            'last_two_seasons_mins_per_match': float(
                getattr(self.player, 'last_two_seasons_mins_per_match', 0) or 0
            ),
            'last_season_h2_fixture_frac': float(
                getattr(self.player, 'last_season_h2_fixture_frac', 0) or 0
            ),
            'changed_club': self.player.changed_club_feature(self.matches_played),
            'club_move_team_diff': self.player.club_move_team_diff_feature(),
            'game_stats': self.game_stats,
            'new_stats_included': 1 if self.new_stats_included else 0
        }
        injury_adj = self._predict_injury_adjusted_minutes()
        if injury_adj is not None:
            # Frac stays raw coverage (n/19). Shrinking it would look like a
            # mid-season arrival rather than a thinner available-minutes sample.
            clean_match['last_season_h2_mins_per_match'] = injury_adj['h2_mpm']
            clean_match['last_two_seasons_mins_per_match'] = injury_adj['two_s_mpm']
        mins_last_1, mins_last_2, mins_last_5 = self._mins_recency_windows(
            injury_adj
        )
        clean_match['mins_last_1'] = mins_last_1
        clean_match['mins_last_2_avg'] = mins_last_2
        clean_match['mins_last_5_avg'] = mins_last_5
        clean_match['last_7_pts_per_appearance'] = self._last_n_pts_per_appearance(7)
        xg5, xc5 = self._past_team_xg_xc(5)
        clean_match['team_xg_last_5'] = xg5
        clean_match['team_xc_last_5'] = xc5
        apply_feature_interactions(clean_match)
        return clean_match

    def _prior_matches(self):
        """Fixtures strictly before this row (all player.matches for a FutureMatch)."""
        prior = []
        for match in self.player.matches:
            if match is self:
                break
            prior.append(match)
        else:
            prior = list(self.player.matches)
        return prior

    @staticmethod
    def _mean_or_zero(values):
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _last_season_fixture_mins(self):
        """Last-season fixtures including DNPs (not appearance-only carry)."""
        mins = []
        for match in getattr(self.player, 'last_szn_matches', None) or []:
            if getattr(match, 'is_last_season_carry', False):
                continue
            mins.append(float(match.game_stats.get('minutes') or 0))
        return mins

    def _h2_fixture_mins(self):
        """Second-half last-season fixtures including DNPs (short seasons: all rows)."""
        ls_mins = self._last_season_fixture_mins()
        return ls_mins[-HALF_SEASON_FIXTURES:]

    def _this_season_fixture_mins(self):
        this_mins = []
        for match in self._prior_matches():
            if getattr(match, 'is_last_season_carry', False):
                continue
            this_mins.append(float(match.game_stats.get('minutes') or 0))
        return this_mins

    def _use_injury_dnp_skip(self):
        """Train rows keep real DNP streaks. Predict overrides in FutureMatch."""
        return False

    @staticmethod
    def _recent_n_played(mins, n=RETURN_CONSECUTIVE_APPS):
        if len(mins) < n:
            return False
        return all(m > 0 for m in mins[-n:])

    @staticmethod
    def _drop_long_dnp_streaks_tagged(tagged, min_streak=INJURY_DNP_STREAK):
        """Drop contiguous DNP runs of length >= min_streak. Keep 1–2 (rotation)."""
        out = []
        i = 0
        n = len(tagged)
        while i < n:
            if tagged[i][1] <= 0:
                j = i
                while j < n and tagged[j][1] <= 0:
                    j += 1
                if j - i < min_streak:
                    out.extend(tagged[i:j])
                i = j
            else:
                out.append(tagged[i])
                i += 1
        return out

    def _minutes_timeline_tagged(self):
        """Chronological (bucket, minutes) for prev2, last-season H1/H2, this season."""
        tagged = []
        for match in getattr(self.player, 'prev2_szn_matches', None) or []:
            tagged.append((
                'prev2',
                float(match.game_stats.get('minutes') or getattr(match, 'minutes', 0) or 0),
            ))
        ls_mins = self._last_season_fixture_mins()
        if len(ls_mins) <= HALF_SEASON_FIXTURES:
            h1, h2 = [], ls_mins
        else:
            h1 = ls_mins[:-HALF_SEASON_FIXTURES]
            h2 = ls_mins[-HALF_SEASON_FIXTURES:]
        tagged.extend(('h1', m) for m in h1)
        tagged.extend(('h2', m) for m in h2)
        tagged.extend(('this', m) for m in self._this_season_fixture_mins())
        return tagged

    def _predict_injury_adjusted_minutes(self):
        """If predict and last two fixtures were appearances, drop 3+ DNP streaks.

        Returns None when the skip does not apply (train, or not yet back).
        last_7_pts_per_appearance is already appearance-only; last-season minutes
        volume is a sum so dropped zeros do not change it.
        """
        if not self._use_injury_dnp_skip():
            return None
        tagged = self._minutes_timeline_tagged()
        gate_mins = [m for bucket, m in tagged if bucket != 'prev2']
        if not AbstractMatch._recent_n_played(gate_mins):
            return None
        filtered = AbstractMatch._drop_long_dnp_streaks_tagged(tagged)

        def bucket_mins(name, rows):
            return [m for b, m in rows if b == name]

        this_mins = bucket_mins('this', filtered)
        h2_mins = bucket_mins('h2', filtered)
        two_s_mins = bucket_mins('prev2', filtered) + bucket_mins('h1', filtered)
        h2_raw = bucket_mins('h2', tagged)
        h2_mpm = AbstractMatch._mean_or_zero(h2_mins or h2_raw)
        two_s_mpm = AbstractMatch._mean_or_zero(two_s_mins)
        if not two_s_mins:
            two_s_mpm = float(
                getattr(self.player, 'last_two_seasons_mins_per_match', 0) or 0
            )
        return {
            'this_mins': this_mins,
            'h2_mins': h2_mins,
            'h2_mpm': h2_mpm,
            'two_s_mpm': two_s_mpm,
        }

    def _mins_recency_windows(self, injury_adj=None):
        """Last 1/2/5 fixtures including DNPs, stretching into last-season H2.

        This-season games are used as-is. Missing slots are H2 fixtures, blended
        toward H2 mins/match with weight exp(-this_season_fixtures / tau).
        """
        if injury_adj is not None:
            this_mins = list(injury_adj['this_mins'])
            h2_mins = list(injury_adj['h2_mins'])
            h2_mpm = float(injury_adj['h2_mpm'] or 0)
        else:
            this_mins = self._this_season_fixture_mins()
            h2_mins = self._h2_fixture_mins()
            h2_mpm = float(getattr(self.player, 'last_season_h2_mins_per_match', 0) or 0)
        early_w = math.exp(-len(this_mins) / PRIOR_EARLY_TAU)

        def window_n(n):
            this_part = this_mins[-n:]
            need = n - len(this_part)
            if need <= 0:
                return AbstractMatch._mean_or_zero(this_part)
            fill = h2_mins[-need:] if h2_mins else []
            if len(fill) < need:
                fill = [h2_mpm] * (need - len(fill)) + list(fill)
            blended_fill = [early_w * h2_mpm + (1.0 - early_w) * x for x in fill]
            return AbstractMatch._mean_or_zero(blended_fill + this_part)

        return window_n(1), window_n(2), window_n(5)

    def _last_n_pts_per_appearance(self, n: int = 7) -> float:
        """Mean FPL points over the last n appearances (minutes > 0). Form, not availability."""
        pts = []
        for match in self._prior_matches():
            minutes = float(match.game_stats.get('minutes') or 0)
            if minutes <= 0:
                continue
            pts.append(float(match.game_stats.get('total_points') or 0))
        return AbstractMatch._mean_or_zero(pts[-n:])

    def _past_team_xg_xc(self, n: int = 5):
        """Sum of team predicted xg/xc over the previous n appearances."""
        prior = self._prior_matches()
        window = prior[-n:]
        xg = sum(float(getattr(m.team_match, 'predicted_xg', 0.0) or 0.0) for m in window)
        xc = sum(float(getattr(m.team_match, 'predicted_xc', 0.0) or 0.0) for m in window)
        return xg, xc


class PastMatch(AbstractMatch):

    def __init__(self, player, match_json, teams):
        super().__init__(player, match_json, teams)

        self.minutes = float(match_json['minutes'] or 0)

        self.points = int(float(match_json['total_points']))
        # self.ict = float(match_json['ict_index'])
        
        # self.average_ict_in_previous_matches = 0

        self.new_stats_included = NEW_STATS[0] in match_json

        for key in STAT_KEYS:
            # Optional era-specific fields (xGI, defcon, …): leave default 0 if absent
            if key in NEW_STATS and key not in match_json:
                continue
            self.game_stats[key] = float(match_json[key])
        
        if self.game_stats['goals_conceded'] >= 2:
            self.game_stats['conceded_2_plus'] = 1
            if self.game_stats['goals_conceded'] >= 4:
                self.game_stats['conceded_4_plus'] = 1
            else:
                self.game_stats['conceded_4_plus'] = 0
        else:
            self.game_stats['conceded_4_plus'] = 0
        
        if self.game_stats['minutes'] >= 60:
            self.game_stats['minutes_60_plus'] = 1
        else:
            self.game_stats['minutes_60_plus'] = 0
        
        if self.game_stats['minutes'] > 0:
            self.game_stats['chance_to_play'] = 1
        else:
            self.game_stats['chance_to_play'] = 0
        

        # self.goals_scored = match_json['goals_scored']
        # self.assists = match_json['assists']
    
    def get_gw(self, match_json: dict):
        return coerce_optional_int(match_json['round'])
    
    def get_home(self, match_json: dict):
        return coerce_bool(match_json['was_home'])
    
    def get_fixture_id(self, match_json: dict):
        return int(match_json['fixture'])
    
    def get_opponent_id(self, match_json: dict):
        return int(match_json['opponent_team'])
    
    def get_team(self, teams):
        # Cannot do player.team because player may play for new team
        if self.home:
            team_id = int(self.opponent_team_match.fixture_json['team_h'])
        else:
            team_id = int(self.opponent_team_match.fixture_json['team_a'])
        return teams[team_id]

    def get_team_match(self):
        if self.fixture_id in self.team.team_match_dict:
            return self.team.team_match_dict[self.fixture_id]
        return self.team.future_team_match_dict[self.fixture_id]
    
    def get_opponent_team_match(self):
        if self.fixture_id in self.opponent_team.team_match_dict:
            return self.opponent_team.team_match_dict[self.fixture_id]
        return self.opponent_team.future_team_match_dict[self.fixture_id]

    def __str__(self):
        return '%i - %i %s (%i)' % (self.opponent_team_match.conceded, self.opponent_team_match.scored,
                                    self.opponent_team.name, self.points)


class FutureMatch(AbstractMatch):

    def __init__(self, player, future_match_json, teams):
        super().__init__(player, future_match_json, teams)

        self.chance_of_playing = player.element_dict['chance_of_playing_next_round']

        # self.average_points_in_previous_matches = player.average_points_so_far

        self.ict_point_regression_deviation = player.ict_deviation
        self.matches_played = getattr(player, 'this_season_matches_played', player.matches_played)
        # self.player_point_deviation_per_game = player.point_deviation_per_game

        self.fixture_date = self.get_fixture_date(future_match_json['kickoff_time'])

    def _use_injury_dnp_skip(self):
        return True
    
    def get_gw(self, match_json: dict):
        return coerce_optional_int(match_json.get('event'))
    
    def get_home(self, match_json: dict):
        return coerce_bool(match_json['is_home'])
    
    def get_fixture_id(self, match_json: dict):
        return int(match_json['id'])
    
    def get_opponent_team_match(self):
        if self.fixture_id in self.opponent_team.future_team_match_dict:
            return self.opponent_team.future_team_match_dict[self.fixture_id]
        return self.opponent_team.team_match_dict[self.fixture_id]
    
    def get_team(self, teams):
        return self.player.team
    
    def get_team_match(self):
        if self.fixture_id in self.team.future_team_match_dict:
            return self.team.future_team_match_dict[self.fixture_id]
        return self.team.team_match_dict[self.fixture_id]

    def update_points_based_on_expected_minutes(self):
        # Scale predicted points by FPL chance_of_playing_next_round.
        #
        # OLD setup (threshold — only dampen when chance < 65%):
        # if self.chance_of_playing is not None and self.chance_of_playing < 0.65:
        #     self.points = self.chance_of_playing * self.points / 100
        # (minutes-based zeroing / other gates lived here too; left off for now)
        #
        # CURRENT: always multiply when chance is known (so 0% injured → 0 points).
        if self.chance_of_playing is None:
            return
        self.points = self.chance_of_playing * self.points / 100.0
        self.game_stats['total_points'] = self.points

    def get_opponent_id(self, match_json):
        if self.home:
            return match_json['team_a']
        return match_json['team_h']

    @staticmethod
    def get_fixture_date(kickoff_time):
        try:
            date_segments = kickoff_time.split('T')[0].split('-')
            return datetime(int(date_segments[0]), int(date_segments[1]), int(date_segments[2]))
        except:
            return None

    
    
    def calc_points_from_stats(self):
        self.points = 1 # for assuming they play
        if self.is_gkp:
            self.points += sum([
                self.game_stats['goals_scored'] * 6,
                self.game_stats['saves'] / 3,
                self.game_stats['clean_sheets'] * 4,
                self.game_stats['conceded_2_plus'] * -1,
                self.game_stats['conceded_4_plus'] * -1
            ])
        elif self.is_def:
            self.points += sum([
                self.game_stats['goals_scored'] * 6,
                self.game_stats['clean_sheets'] * 4,
                self.game_stats['conceded_2_plus'] * -1,
                self.game_stats['conceded_4_plus'] * -1
            ])
        elif self.is_mid:
            self.points += sum([
                self.game_stats['goals_scored'] * 5,
                self.game_stats['clean_sheets']
            ])
        else:
            self.points += self.game_stats['goals_scored'] * 4
        
        self.points += sum([
            self.game_stats['assists'] * 3,
            self.game_stats['bonus'],
            self.game_stats['yellow_cards'] * -1,
            self.game_stats['minutes_60_plus']
        ])
    
    def set_minutes_from_stats(self):
        self.minutes = self.game_stats['minutes']

    def __str__(self):
        return 'gameweek: %i points: %.2f vs: %s' % (self.gameweek, self.points, self.opponent_team.name)
