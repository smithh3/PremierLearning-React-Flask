import copy
import numpy

from abc import ABC

from premierlearning.clean_match import (
    BUILT_KEYS, STAT_KEYS, PER_MINUTE_STAT_KEYS, PastStats, StatWindow, get_default_past_stats,
    get_default_stat_window,
)
from .match import PastMatch, FutureMatch
from .config_constants import (
    SEASON_YR, PLAYER_JSON_FILE_LOCATION, HALF_SEASON_FIXTURES,
    LAST_SEASON_FIXTURES, FORM_AVG_WINDOW, FORM_MIN_MINUTES, SHORT_FORM_WINDOW,
)
from .helper_functions import (
    coerce_bool,
    read_json,
    score_is_present,
)

# -----------------------------------------------------------------------------
# Form / appearance features — EXPERIMENT (Aug 2026)
#
# Problem we were hitting at season start:
#   - Blank / low-history players (zeros + team xG) outranked stars like Haaland
#   - Train/serve mismatch when last-season carry filled last_5 but matches_played
#     was this-season-only (0), a combo the net almost never saw in training
#   - Blanking matches_played entirely made blank-player extrapolation worse
#
# OLD setup (kept commented in get_stat_window / build_snapshot_stats):
#   - last_5_avg = mean of last 5 per-game values (raw totals, not per-90)
#   - avg       = mean of ALL appearances so far (full season to date)
#   - matches_played = rolling count of appearances with minutes > 0 (no carry
#     distinction; carry didn't exist in the old pipeline the same way)
#   - No current-vs-prior season weighting; no pooled per-90
#
# CURRENT experiment (what this file does now):
#   - Prepend last AVG_WINDOW prior-season appearances for early-GW form (carry)
#   - last_5_avg = last SHORT_FORM_WINDOW (3) appearances; for rate stats, pooled per-90
#     (sum(raw) * 90 / sum(mins)) so partial minutes don't dilute like a mean
#   - Appearances under FORM_MIN_MINUTES (20) are dropped from those windows
#     (1-minute 1-point cameos must not explode last-5 / last-7 pp90)
#   - avg = last AVG_WINDOW appearances (not full season), with this-season
#     rows weighted CURRENT_SEASON_STAT_WEIGHT× vs carry LAST_SEASON_STAT_WEIGHT×
#   - matches_played feature = this-season appearances only (capped), so early
#     GW predict is 0 even when carry seeds form — revisit if rankings go bad
#   - Skip predict when rolling appearances < MIN_APPEARANCES_TO_PREDICT
#   - changed_club (0/1): club move vs prior season (stable team.code). Season-level
#     club_changed_vs_prior is set for current + past seasons; the model feature is 1
#     only for the first CHANGED_CLUB_APPS this-season appearances, then 0 once we have
#     enough new-club data. Binary flag is off in _BASE_STANDARD (was boosting movers).
#     Minutes net uses ls_mpm_x_new_club (last-season mins/match × new-club factor).
#   - Minutes model: full last-season mins/match vs full two-seasons-ago,
#     both including DNPs. Recency last-1/2/5 still fills from the end of last
#     season near GW0. Carry is still appearance-only for points. Predict rows
#     (not train) drop 3+ consecutive DNPs from those windows when the last two
#     fixtures were appearances — treat as injury, not availability. Isolated
#     1–2 DNPs stay (rotation).
#
# Flip back: restore the old get_stat_window body below and drop weights / per_90
# / AVG_WINDOW slicing; set matches_played from self.matches_played if you want
# rolling-including-carry again.
# -----------------------------------------------------------------------------
CURRENT_SEASON_STAT_WEIGHT = 4.0
LAST_SEASON_STAT_WEIGHT = 1.0
MAX_MATCHES_PLAYED_FEATURE = 38
MIN_APPEARANCES_TO_PREDICT = 3
# Form windows (appearances with minutes >= FORM_MIN_MINUTES)
LAST_5_WINDOW = SHORT_FORM_WINDOW
AVG_WINDOW = FORM_AVG_WINDOW
# changed_club model feature stays 1 for this many this-season appearances after a move
CHANGED_CLUB_APPS = 5


def is_form_appearance(minutes) -> bool:
    """True when an appearance counts in last-3 / last-7 per-90 form windows."""
    try:
        return float(minutes or 0) >= FORM_MIN_MINUTES
    except (TypeError, ValueError):
        return False


def _team_by_code(teams, code):
    if not teams or code is None:
        return None
    values = teams.values() if hasattr(teams, 'values') else teams
    try:
        code_i = int(code)
    except (TypeError, ValueError):
        return None
    for team in values:
        other = getattr(team, 'code', None)
        if other is None:
            continue
        try:
            if int(other) == code_i:
                return team
        except (TypeError, ValueError):
            continue
    return None


def _h2h_predicted_score(origin, dest):
    """Average predicted scoreline origin vs dest from dest's fixtures this season."""
    try:
        origin_id = int(origin.id)
        dest_id = int(dest.id)
    except (TypeError, ValueError, AttributeError):
        return None
    pairs = []
    matches = list(getattr(dest, 'team_match_dict', {}).values()) + list(
        getattr(dest, 'future_team_match_dict', {}).values()
    )
    for match in matches:
        fx = getattr(match, 'fixture_json', None) or {}
        try:
            home_id = int(fx['team_h'])
            away_id = int(fx['team_a'])
        except (KeyError, TypeError, ValueError):
            continue
        if {home_id, away_id} != {origin_id, dest_id}:
            continue
        dest_xg = float(getattr(match, 'predicted_xg', 0) or 0)
        origin_xg = float(getattr(match, 'predicted_xc', 0) or 0)
        if dest_xg <= 0 and origin_xg <= 0:
            continue
        pairs.append((origin_xg, dest_xg))
    if not pairs:
        return None
    n = float(len(pairs))
    return sum(p[0] for p in pairs) / n, sum(p[1] for p in pairs) / n


def _form_expected_score(origin, dest):
    """Neutral expected goals: each side's attack vs the other's defence."""
    o_s = float(getattr(origin, 'average_scored_so_far', 0) or 0)
    o_c = float(getattr(origin, 'average_conceded_so_far', 0) or 0)
    d_s = float(getattr(dest, 'average_scored_so_far', 0) or 0)
    d_c = float(getattr(dest, 'average_conceded_so_far', 0) or 0)
    return 0.5 * (o_s + d_c), 0.5 * (d_s + o_c)


class RawPlayer(ABC):

    def __init__(self):
        self.id = None
        self.first_name = None
        self.last_name = None
        self.team_id = None
        self.team_name = None
        self.current_cost = None
        self.position = None
        self.is_most_expensive_in_pos = 0
        self.matches = None
        self.future_matches = None,
        self.code = None
    
    def __repr__(self):
        from squadtools.availability import match_optimiser_points

        points = 0
        matches = [match for match in (self.future_matches or []) if match is not None]
        if matches:
            points = match_optimiser_points(matches[0])
        return '%s %s: %.1f (£%.1fm)' % (self.first_name, self.last_name, points, self.current_cost)
    
    def format_as_db_insert(self):
        return (
            self.id,
            self.first_name,
            self.last_name,
            self.team_id,
            self.team_name,
            self.current_cost,
            self.position,
            self.code
        )
    

    def get_points_between_gameweeks(self, gameweek_from, gameweek_to, bench_multiplier, bench_boost_gw, free_hit_gw):
        return sum(self.get_points_in_gameweek_n(n, bench_multiplier, bench_boost_gw, free_hit_gw) for n in range(gameweek_from, gameweek_to + 1))


    def get_points_in_gameweek_n(self, n, bench_multiplier, bench_boost_gw, free_hit_gw):
        # bench_multiplier = 1
        if n is None:
            return 0
        if n == bench_boost_gw:
            bench_multiplier = 1
        if n == free_hit_gw:
            bench_multiplier = 0
        n = int(n)
        # Upcoming captain/XI picks: only future fixtures for that GW (not past/carry rows).
        from squadtools.availability import match_optimiser_points

        return sum(
            match_optimiser_points(fixture)
            for fixture in self.future_matches
            if fixture.gameweek is not None and int(fixture.gameweek) == n
        ) * bench_multiplier

    def get_price_at_gw(self, gw):
        player_file_path = PLAYER_JSON_FILE_LOCATION % (SEASON_YR, SEASON_YR + 1) + str(self.id) + '.json'
        player_json = read_json(player_file_path) or {}
        history = player_json.get('history') or []
        for past_gw in history:
            if past_gw['round'] == gw:
                return past_gw['value']
        if history:
            return history[0]['value']
        return self.current_cost
    
    def get_name(self):
        return f'{self.first_name} {self.last_name}'

    def get_points_within_n_gameweeks(self, current_gameweek, n):
        from squadtools.availability import match_optimiser_points

        return sum(
            match_optimiser_points(fixture)
            for fixture in self.future_matches
            if fixture.within_n_gameweeks(current_gameweek, n)
        )

    def format_as_request_pick(self, placement, is_captain, is_vice_captain):
        return {
            'element': self.id,
            'position': placement,
            'is_captain': is_captain,
            'is_vice_captain': is_vice_captain
        }
    
    def __str__(self):
        return self.__repr__()


class Player(RawPlayer):

    def __init__(
        self, fantasy_json, element_dict, player_json, teams, current_gameweek,
        next_gameweek=None,
    ):
        super().__init__()
        
        self.played_last_season = False
        self.points_per_min_last_season = 0
        self.points_per_min_two_seasons_ago = 0
        self.xgi_per_min_last_season = 0
        self.xgi_per_min_two_seasons_ago = 0
        self.minutes_last_season = 0
        # Last-season availability including DNPs (not just appearances).
        self.last_season_mins_per_match = 0.0
        self.last_7_mins_per_match = 0.0
        self.two_season_prior_mins_per_match = 0.0
        self.last_season_h2_mins_per_match = 0.0
        self.last_season_h2_fixtures = 0
        self.last_season_h2_appearances = 0
        self.last_season_appearances = 0
        self.last_season_fixtures = 0
        self.last_season_fixture_frac = 0.0
        # Season-level: transferred vs prior season (stable team.code). New-to-PL → False.
        # Model feature changed_club is 1 only for first CHANGED_CLUB_APPS this-season apps.
        self.club_changed_vs_prior = False
        self.prior_team = None
        self.current_gameweek = current_gameweek
        self.next_gameweek = int(next_gameweek or 0)

        # self.average_points_so_far = 0
        # self.average_ict_so_far = 0
        self.ict_deviation = 0
        self.matches_played = 0
        self.this_season_matches_played = 0
        # self.point_deviation_per_game = 0

        self.points_over_next_5_gameweeks = 0

        # self.points_so_far = []
        self.past_stats: PastStats = get_default_past_stats()
        self.minutes_so_far = []

        self.element_dict = element_dict
        # self.player_dict = requests.get(PLAYER_API % element_dict['id']).json()
        self.player_json = player_json
        self.id = int(self.element_dict['id'])
        self.code = int(self.element_dict['code'])
        self.history_past = {}
        self.history_past_xgi = {}
        self.history_past_minutes = {}
        self.populate_history_past()

        self.position = int(self.element_dict['element_type'])

        self.first_name = self.element_dict['first_name']
        self.last_name = self.element_dict['second_name']

        self.points_per_game = float(self.element_dict['points_per_game'])

        self.current_cost = float(self.element_dict['now_cost'])
        self.is_most_expensive_in_pos = 0

        self.last_szn_matches = []
        self.prev2_szn_matches = []
        self.last_szn_avg_mins = 0
        self.last_szn_avg_points = 0

        self.matches = []
        self.future_matches = []
        
        self.teams = teams
        self.team = teams[int(self.element_dict['team'])]
        self.team_name = self.team.name
        self.team_id = self.team.id

        self.populate_matches(teams)

        self.last_opponent_strength = self.get_last_opponent_strength(teams)

        self.last_points = float(self.get_last_points())

        # self.attach_last_szn_matches()

        # self.build_snapshot_stats()

        # self.populate_future_matches(teams)

    def __str__(self):
        from squadtools.availability import match_optimiser_points

        points = 0
        if self.future_matches:
            points = match_optimiser_points(self.future_matches[0])
        return '%s %s (%s): %.1f (%s)' % (self.first_name, self.last_name, self.team.name, points, self.current_cost)

    def get_last_points(self):
        if self.last_opponent_strength is not None:
            return self.player_json['history'][-2]['total_points']
        return 0

    def populate_history_past(self):
        if 'history_past' in self.player_json:
            for past_season in self.player_json['history_past']:
                years = past_season['season_name']
                start_year = int(years.split('/')[0][2:])
                total_points = float(past_season['total_points'])
                minutes_played = float(past_season['minutes'])
                xgi = float(past_season.get('expected_goal_involvements') or 0)
                self.history_past_minutes[start_year] = minutes_played
                if minutes_played > 200:
                    self.history_past[start_year] = total_points / minutes_played
                    self.history_past_xgi[start_year] = xgi / minutes_played
                else:
                    self.history_past[start_year] = 0
                    self.history_past_xgi[start_year] = 0

    @staticmethod
    def _stat_value(match, stat_key) -> float:
        if stat_key == 'total_points':
            return float(
                getattr(match, 'points', 0)
                or (getattr(match, 'game_stats', None) or {}).get('total_points')
                or 0
            )
        return float((getattr(match, 'game_stats', None) or {}).get(stat_key) or 0)

    @staticmethod
    def per_min_from_match_list(matches, stat_key='total_points') -> float:
        """Appearance-only per-minute rate (skip DNP / 0 minutes). 0 if <200 minutes."""
        total = 0.0
        total_mins = 0.0
        for match in matches or ():
            if getattr(match, 'is_last_season_carry', False):
                continue
            mins = Player._match_minutes(match)
            if mins <= 0:
                continue
            total_mins += mins
            total += Player._stat_value(match, stat_key)
        if total_mins > 200:
            return total / total_mins
        return 0.0

    @staticmethod
    def per_min_from_matches(player, stat_key='total_points') -> float:
        """Season rate / minutes from non-carry appearances (0 if <200 minutes)."""
        if player is None:
            return 0.0
        return Player.per_min_from_match_list(
            Player.non_carry_matches(player), stat_key
        )

    @staticmethod
    def xgi_per_min_from_matches(player) -> float:
        """Season xGI / minutes from non-carry appearances (0 if <200 minutes)."""
        return Player.per_min_from_matches(player, 'expected_goal_involvements')

    @staticmethod
    def minutes_from_matches(player) -> float:
        """Season minutes from non-carry appearances."""
        total_mins = 0.0
        for match in player.matches:
            if getattr(match, 'is_last_season_carry', False):
                continue
            total_mins += float(match.game_stats.get('minutes') or getattr(match, 'minutes', 0) or 0)
        return total_mins

    @staticmethod
    def _match_minutes(match) -> float:
        return float(match.game_stats.get('minutes') or getattr(match, 'minutes', 0) or 0)

    @staticmethod
    def non_carry_matches(player):
        return [m for m in player.matches if not getattr(m, 'is_last_season_carry', False)]

    @staticmethod
    def mins_per_match_from_matches(player, last_n=None) -> float:
        """Mean minutes per fixture, including 0-minute DNPs (not appearance-only)."""
        matches = Player.non_carry_matches(player)
        if last_n is not None:
            matches = matches[-last_n:]
        if not matches:
            return 0.0
        return sum(Player._match_minutes(m) for m in matches) / len(matches)

    @staticmethod
    def mins_per_match_list(matches) -> float:
        if not matches:
            return 0.0
        return sum(Player._match_minutes(m) for m in matches) / len(matches)

    @staticmethod
    def split_season_halves(player):
        """(H1 fixtures, H2 fixtures), DNP-inclusive. Short seasons are all H2."""
        matches = Player.non_carry_matches(player)
        if len(matches) <= HALF_SEASON_FIXTURES:
            return [], matches
        return matches[:-HALF_SEASON_FIXTURES], matches[-HALF_SEASON_FIXTURES:]

    @staticmethod
    def apply_prior_season_stats(player, prev_player, prev2_player=None):
        """Copy last-season and two-seasons-ago full-season rates onto player."""
        player.last_season_mins_per_match = 0.0
        player.last_7_mins_per_match = 0.0
        player.two_season_prior_mins_per_match = 0.0
        player.last_season_h2_mins_per_match = 0.0
        player.last_season_h2_fixtures = 0
        player.last_season_h2_appearances = 0
        player.last_season_appearances = 0
        player.last_season_fixtures = 0
        player.last_season_fixture_frac = 0.0
        prev2_matches = Player.non_carry_matches(prev2_player) if prev2_player else []
        player.prev2_szn_matches = list(prev2_matches)
        if prev2_player:
            player.two_season_prior_mins_per_match = Player.mins_per_match_from_matches(
                prev2_player
            )
            ppm2 = Player.per_min_from_matches(prev2_player, 'total_points')
            if Player.minutes_from_matches(prev2_player) > 200:
                player.points_per_min_two_seasons_ago = ppm2
            xgi2 = Player.xgi_per_min_from_matches(prev2_player)
            if xgi2 > 0:
                player.xgi_per_min_two_seasons_ago = xgi2
        if prev_player:
            ppm = Player.per_min_from_matches(prev_player, 'total_points')
            last_mins = Player.minutes_from_matches(prev_player)
            if last_mins > 200:
                player.points_per_min_last_season = ppm
                player.played_last_season = True
            xgi_pm = Player.xgi_per_min_from_matches(prev_player)
            if xgi_pm > 0:
                player.xgi_per_min_last_season = xgi_pm
            if last_mins > 0:
                player.minutes_last_season = last_mins
            matches = Player.non_carry_matches(prev_player)
            player.last_season_fixtures = len(matches)
            player.last_season_appearances = sum(
                1 for m in matches if Player._match_minutes(m) > 0
            )
            player.last_season_mins_per_match = Player.mins_per_match_from_matches(prev_player)
            player.last_season_fixture_frac = min(
                1.0, len(matches) / float(LAST_SEASON_FIXTURES)
            )
            player.last_7_mins_per_match = Player.mins_per_match_from_matches(
                prev_player, last_n=AVG_WINDOW
            )
            _, h2 = Player.split_season_halves(prev_player)
            player.last_season_h2_mins_per_match = Player.mins_per_match_list(h2)
            player.last_season_h2_fixtures = len(h2)
            player.last_season_h2_appearances = sum(
                1 for m in h2 if Player._match_minutes(m) > 0
            )

    def get_last_opponent_strength(self, teams):
        history = self.player_json['history']
        if len(history) > 1:
            last_match = history[-1]

            last_opponent = teams[int(last_match['opponent_team'])]
            strength_key = 'strength_attack'
            if coerce_bool(last_match['was_home']):
                strength_key += '_home'
            else:
                strength_key += '_away'

            return last_opponent.strength[strength_key]
        return None

    def populate_matches(self, teams):
        for match_json in self.player_json['history']:
            # JSON: null when unplayed; CSV: '' / 'None' / missing
            if score_is_present(match_json.get('team_h_score')):
                self.matches.append(PastMatch(self, match_json, teams))
    

    def set_changed_club_from_prior(self, past_szn_player):
        """Record club move vs prior season using stable FPL team.code (not season team id)."""
        self.club_changed_vs_prior = False
        self.prior_team = None
        if not past_szn_player:
            return
        self.prior_team = getattr(past_szn_player, 'team', None)
        prev_code = getattr(self.prior_team, 'code', None)
        curr_code = getattr(self.team, 'code', None)
        if prev_code is None or curr_code is None:
            return
        self.club_changed_vs_prior = int(prev_code) != int(curr_code)

    def changed_club_feature(self, this_season_apps: int) -> int:
        """Old binary 0/1 (disabled in _BASE_STANDARD). Kept for debug / re-enable."""
        if not self.club_changed_vs_prior:
            return 0
        return 1 if int(this_season_apps or 0) < CHANGED_CLUB_APPS else 0

    def club_move_team_diff_feature(self) -> float:
        """Minutes-only: 1.0 if new-to-PL or same club; 1 + (old_xg - new_xg) after a move.

        old/new xG is the predicted score if those two sides played each other
        (H2H team-expectancy average when both are in this season, else attack vs
        defence form). Move to a weaker side => >1 (more minutes); stronger => <1.
        """
        if not self.club_changed_vs_prior:
            return 1.0
        prior = getattr(self, 'prior_team', None)
        dest = getattr(self, 'team', None)
        if prior is None or dest is None:
            return 1.0
        origin_now = _team_by_code(getattr(self, 'teams', None), getattr(prior, 'code', None))
        score = None
        if origin_now is not None and origin_now is not dest:
            score = _h2h_predicted_score(origin_now, dest)
        if score is None:
            old_for_form = origin_now if origin_now is not None else prior
            score = _form_expected_score(old_for_form, dest)
        old_xg, new_xg = score
        return 1.0 + (old_xg - new_xg)

    def attach_last_szn_matches(self, past_szn_player):
        """Prepend last 7 prior-season *form* appearances (minutes >= FORM_MIN_MINUTES).

        New-to-league players (no past_szn_player / no appearances) get nothing — form
        stays 0 until they play this season.
        """
        self.set_changed_club_from_prior(past_szn_player)
        self.last_szn_matches = past_szn_player.matches if past_szn_player else []
        if not past_szn_player or not past_szn_player.matches:
            return
        if self.matches and getattr(self.matches[0], 'is_last_season_carry', False):
            return

        played = [
            m for m in past_szn_player.matches
            if is_form_appearance(
                m.game_stats.get('minutes', getattr(m, 'minutes', 0))
            )
        ]
        if not played:
            return

        n = min(AVG_WINDOW, len(played))
        matches_to_insert = []
        for m in played[-n:]:
            # Shallow copy alone shares past_stats with the prior-season match; re-processing
            # carry would then overwrite that season's training snapshots. Give each carry
            # row its own past_stats / game_stats before this season rebuilds form.
            # Rebind player_id so Fargate persist/reload keeps carry on this season's id
            # (FPL element ids change every season; the copy still has last season's id).
            cm = copy.copy(m)
            cm.past_stats = get_default_past_stats()
            cm.game_stats = copy.deepcopy(m.game_stats)
            cm.is_last_season_carry = True
            cm.valid_for_learning = False
            cm.valid_for_minutes = False
            cm.player_id = self.id
            matches_to_insert.append(cm)
        self.matches = matches_to_insert + list(self.matches)


    def process_matches(self):
        self.build_snapshot_stats()
        self.populate_future_matches()


    def populate_future_matches(self):
        if getattr(self, 'skip_historical_predict', False):
            return
        for future_match_json in self.player_json['fixtures']:
            self.future_matches.append(FutureMatch(self, future_match_json, self.teams))

        # minutes_greater_than_0 = list(mins for mins in self.minutes_so_far if mins > 0)
        # average_minutes_when_played = 0
        # if len(minutes_greater_than_0) > 0:
        #     average_minutes_when_played = sum(minutes_greater_than_0) / len(minutes_greater_than_0)

        for match in self.future_matches:
            # if len(self.minutes_so_far) > 0:
            #     # if len is up to 6, include some proportion of stats from last szn :)
            #     match.minutes_last = self.minutes_so_far[-1]
            #     match.average_minutes_last_3 = sum(self.minutes_so_far[-3:]) / len(self.minutes_so_far[-3:])
            #     match.average_minutes_last_5 = sum(self.minutes_so_far[-5:]) / len(self.minutes_so_far[-5:])
            #     match.average_minutes = sum(self.minutes_so_far) / len(self.minutes_so_far)
            #     match.matches_available = len(self.minutes_so_far)
            #     match.average_minutes_when_played = average_minutes_when_played

            # elif len(self.last_szn_matches) > 0:
            #     match.minutes_last = self.last_szn_avg_mins
            #     match.average_minutes_last_3 = self.last_szn_avg_mins
            #     match.average_minutes_last_5 = self.last_szn_avg_mins
            #     match.average_minutes = self.last_szn_avg_mins
            #     match.matches_available = 0
            #     match.average_minutes_when_played = self.last_szn_avg_mins # this is not right, should calculate

            # match.points_last = self.final_stats['points']['last']
            # match.average_points_last_3 = self.final_stats['points']['last_3_avg']
            # match.average_points_last_5 = self.final_stats['points']['last_5_avg']
            match.past_stats = self.past_stats
            # CURRENT: this-season only (not carry) — early GW predict should be 0
            match.matches_played = min(MAX_MATCHES_PLAYED_FEATURE, self.this_season_matches_played)
            # OLD / alt: rolling appearances including carry (same train + predict):
            # match.matches_played = min(MAX_MATCHES_PLAYED_FEATURE, self.matches_played)

            # if len(self.points_so_far) > 0:
            #     match.points_last = self.points_so_far[-1]
            #     match.average_points_last_3 = sum(self.points_so_far[-3:]) / len(self.points_so_far[-3:])
            #     match.average_points_last_5 = sum(self.points_so_far[-5:]) / len(self.points_so_far[-5:])
            # # elif len(self.last_szn_matches) > 0:
            # #     match.points_last = self.last_szn_avg_points
            # #     match.average_points_last_3 = self.last_szn_avg_points
            # #     match.average_points_last_5 = self.last_szn_avg_points

    def build_snapshot_stats(self):
        # Per-minute stats: store (raw_total, minutes) so last_5 can be pooled per-90.
        # minutes / chance: store plain floats.
        stats_so_far = {}
        for key in STAT_KEYS + BUILT_KEYS:
            stats_so_far[key] = []

        appearance_weights = []
        chance_to_play_list = []
        chance_weights = []

        for match in self.matches:
            is_carry = getattr(match, 'is_last_season_carry', False)
            weight = LAST_SEASON_STAT_WEIGHT if is_carry else CURRENT_SEASON_STAT_WEIGHT

            minutes = float(match.game_stats['minutes'] or 0)
            # Blanks: still minutes-train targets (0); form from prior appearances only.
            # Also append 0 into minutes history so DNPs affect future minutes features.
            if minutes <= 0:
                if len(chance_to_play_list) > 0:
                    match.past_stats['chance_to_play'] = self.get_stat_window(
                        chance_to_play_list, chance_weights, per_90=False
                    )
                if self.matches_played >= 1:
                    for key, stats_list in stats_so_far.items():
                        match.past_stats[key] = self._past_stat_window(
                            key, stats_list, appearance_weights
                        )
                # CURRENT: this-season only (see module comment). OLD/alt: self.matches_played
                match.matches_played = min(MAX_MATCHES_PLAYED_FEATURE, self.this_season_matches_played)
                if (
                    not is_carry
                    and self.matches_played >= MIN_APPEARANCES_TO_PREDICT
                ):
                    match.valid_for_minutes = True
                chance_to_play_list.append(0)
                chance_weights.append(weight)
                stats_so_far['minutes'].append(0.0)
                continue

            if len(chance_to_play_list) > 0:
                match.past_stats['chance_to_play'] = self.get_stat_window(
                    chance_to_play_list, chance_weights, per_90=False
                )
            chance_to_play_list.append(1)
            chance_weights.append(weight)

            if self.matches_played >= 1:
                for key, stats_list in stats_so_far.items():
                    match.past_stats[key] = self._past_stat_window(
                        key, stats_list, appearance_weights
                    )

            # CURRENT: this-season only (see module comment). OLD/alt: self.matches_played
            match.matches_played = min(MAX_MATCHES_PLAYED_FEATURE, self.this_season_matches_played)

            # Carry fixtures only seed form for early this-season rows — never train targets.
            # valid_for_learning still uses rolling matches_played (incl. carry) for min appearances.
            if (
                not is_carry
                and self.matches_played >= MIN_APPEARANCES_TO_PREDICT
            ):
                match.valid_for_learning = True
                match.valid_for_minutes = True

            self.matches_played += 1

            for key, stats_list in stats_so_far.items():
                raw = float(match.game_stats[key])
                if key in PER_MINUTE_STAT_KEYS:
                    stats_list.append((raw, minutes))
                else:
                    stats_list.append(raw)
            appearance_weights.append(weight)

            if not is_carry:
                self.this_season_matches_played += 1

        if self.matches_played >= 1:
            for key, stats_list in stats_so_far.items():
                self.past_stats[key] = self._past_stat_window(
                    key, stats_list, appearance_weights
                )
            self.past_stats['chance_to_play'] = self.get_stat_window(
                chance_to_play_list, chance_weights, per_90=False
            )

    def _past_stat_window(self, key, stats_list, appearance_weights):
        return self.get_stat_window(
            stats_list,
            appearance_weights,
            per_90=(key in PER_MINUTE_STAT_KEYS),
        )

    def get_stat_window(
        self,
        stats_list,
        weights=None,
        per_90: bool = False,
    ) -> StatWindow:
        # --- OLD setup (raw per-game means; avg = full history to date) ---------
        # if not stats_list:
        #     return get_default_stat_window()
        # return {
        #     # 'last': stats_list[-1],
        #     # 'last_3_avg': sum(stats_list[-3:]) / len(stats_list[-3:]),
        #     'last_5_avg': sum(stats_list[-5:]) / len(stats_list[-5:]),
        #     'avg': sum(stats_list) / len(stats_list),
        #     # 'deviation': numpy.std(numpy.array(stats_list))
        # }
        # --- CURRENT experiment (see module comment at top) --------------------
        if not stats_list:
            return get_default_stat_window()

        if per_90:
            # last_5_avg / avg: pooled per-90 over last SHORT_FORM_WINDOW / AVG_WINDOW
            # appearances with minutes >= FORM_MIN_MINUTES.
            def pooled_pp90(window):
                raw_sum = 0.0
                mins_sum = 0.0
                for raw, mins in window:
                    raw_sum += float(raw)
                    mins_sum += float(mins)
                return (raw_sum * 90.0 / mins_sum) if mins_sum > 0 else 0.0

            qualifying = [
                row for row in stats_list
                if is_form_appearance(row[1])
            ]
            if not qualifying:
                return get_default_stat_window()
            last_5_avg = pooled_pp90(qualifying[-LAST_5_WINDOW:])
            avg = pooled_pp90(qualifying[-AVG_WINDOW:])
            return {'last_5_avg': last_5_avg, 'avg': avg}

        last_5 = stats_list[-LAST_5_WINDOW:]
        last_5_avg = sum(last_5) / len(last_5)
        last_avg = stats_list[-AVG_WINDOW:]
        if weights is None or len(weights) != len(stats_list):
            avg = sum(last_avg) / len(last_avg)
        else:
            # 4× this-season vs 1× carry inside the AVG_WINDOW slice
            w_slice = weights[-len(last_avg):]
            w_sum = sum(w_slice)
            avg = (
                sum(v * w for v, w in zip(last_avg, w_slice)) / w_sum
                if w_sum else 0.0
            )
        return {
            'last_5_avg': last_5_avg,
            'avg': avg,
        }
    
    # def get_final_stats_initial_values(self):
    #     stats = {}
    #     for key in LEARNING_KEYS:
    #         stats[key] = get_default_stat_window()
    #     return stats
    #     # return {
    #     #     'points': get_default_stat_window(),
    #     #     'goals_scored': get_default_stat_window()
    #     # }

    # def build_snapshot_minutes(self):
    #     minutes_so_far = []
    #     for match in self.matches:
    #         if len(minutes_so_far) > 0:
    #             match.minutes_last = minutes_so_far[-1]
    #             match.average_minutes_last_3 = sum(minutes_so_far[-3:]) / len(minutes_so_far[-3:])
    #             match.average_minutes_last_5 = sum(minutes_so_far[-5:]) / len(minutes_so_far[-5:])
    #             match.average_minutes = sum(minutes_so_far) / len(minutes_so_far)
    #             match.matches_available = len(minutes_so_far)

    #         minutes_so_far.append(int(match.minutes))
    #     self.minutes_so_far = minutes_so_far

    def calculate_points_over_next_5_gameweeks(self):
        self.points_over_next_5_gameweeks = self.get_points_within_n_gameweeks(self.current_gameweek, 5)

    

    # def get_recent_game_minutes(self):
    #     minutes = 0
    #     n = len(self.matches)
    #     if n > 3:
    #         n = 3
    #     for match in self.matches[-n:]:
    #         minutes += match.minutes
    #     return minutes

    def get_price_at_gw(self, gw):
        for past_gw in self.player_json['history']:
            if past_gw['round'] == gw:
                return past_gw['value']

        return self.player_json['history'][0]['value']


def mark_most_expensive_in_position(players):
    """Flag the highest now_cost in each club × position (ties all count)."""
    maxima = {}
    for player in players or ():
        team_id = int(getattr(player, 'team_id', 0) or 0)
        position = int(getattr(player, 'position', 0) or 0)
        cost = float(getattr(player, 'current_cost', 0) or 0)
        if team_id <= 0 or position not in (1, 2, 3, 4) or cost <= 0:
            player.is_most_expensive_in_pos = 0
            continue
        key = (team_id, position)
        prev = maxima.get(key)
        if prev is None or cost > prev:
            maxima[key] = cost
    for player in players or ():
        team_id = int(getattr(player, 'team_id', 0) or 0)
        position = int(getattr(player, 'position', 0) or 0)
        cost = float(getattr(player, 'current_cost', 0) or 0)
        peak = maxima.get((team_id, position))
        player.is_most_expensive_in_pos = int(
            peak is not None and cost >= peak and cost > 0
        )
