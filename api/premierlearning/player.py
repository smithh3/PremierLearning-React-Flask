import copy
import numpy

from abc import ABC

from premierlearning.clean_match import (
    BUILT_KEYS, STAT_KEYS, PER_MINUTE_STAT_KEYS, PastStats, StatWindow, get_default_past_stats,
    get_default_stat_window,
)
from .match import PastMatch, FutureMatch
from .config_constants import SEASON_YR, PLAYER_JSON_FILE_LOCATION
from .helper_functions import read_json, coerce_bool, score_is_present

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
#   - last_5_avg = last 5 appearances only; for rate stats, pooled per-90
#     (sum(raw) * 90 / sum(mins)) so partial minutes don't dilute like a mean
#   - avg = last AVG_WINDOW appearances (not full season), with this-season
#     rows weighted CURRENT_SEASON_STAT_WEIGHT× vs carry LAST_SEASON_STAT_WEIGHT×
#   - matches_played feature = this-season appearances only (capped), so early
#     GW predict is 0 even when carry seeds form — revisit if rankings go bad
#   - Skip predict when rolling appearances < MIN_APPEARANCES_TO_PREDICT
#
# Flip back: restore the old get_stat_window body below and drop weights / per_90
# / AVG_WINDOW slicing; set matches_played from self.matches_played if you want
# rolling-including-carry again.
# -----------------------------------------------------------------------------
CURRENT_SEASON_STAT_WEIGHT = 4.0
LAST_SEASON_STAT_WEIGHT = 1.0
MAX_MATCHES_PLAYED_FEATURE = 38
MIN_APPEARANCES_TO_PREDICT = 3
# Form windows (appearances with minutes > 0 only)
LAST_5_WINDOW = 5
AVG_WINDOW = 7  # same length as last-season carry; OLD used full-season for avg


class RawPlayer(ABC):

    def __init__(self):
        self.id = None
        self.first_name = None
        self.last_name = None
        self.team_id = None
        self.team_name = None
        self.current_cost = None
        self.position = None
        self.matches = None
        self.future_matches = None,
        self.code = None
    
    def __repr__(self):
        points = 0
        if len(self.future_matches) > 0:
            points = self.future_matches[0].points
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
        return sum(
            float(fixture.points or 0)
            for fixture in self.future_matches
            if fixture.gameweek is not None and int(fixture.gameweek) == n
        ) * bench_multiplier

    def get_price_at_gw(self, gw):
        player_file_path = PLAYER_JSON_FILE_LOCATION % (SEASON_YR, SEASON_YR + 1) + str(self.id) + '.json'
        player_json = read_json(player_file_path)
        for past_gw in player_json['history']:
            if past_gw['round'] == gw:
                return past_gw['value']
        return player_json['history'][0]['value']
    
    def get_name(self):
        return f'{self.first_name} {self.last_name}'

    def get_points_within_n_gameweeks(self, current_gameweek, n):
        return sum(fixture.points for fixture in self.future_matches
                   if fixture.within_n_gameweeks(current_gameweek, n))

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

    def __init__(self, fantasy_json, element_dict, player_json, teams, current_gameweek):
        super().__init__()
        
        self.played_last_season = False
        self.points_per_min_last_season = 0
        self.points_per_min_two_seasons_ago = 0
        self.xgi_per_min_last_season = 0
        self.current_gameweek = current_gameweek

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
        self.populate_history_past()

        self.position = int(self.element_dict['element_type'])

        self.first_name = self.element_dict['first_name']
        self.last_name = self.element_dict['second_name']

        self.points_per_game = float(self.element_dict['points_per_game'])

        self.current_cost = float(self.element_dict['now_cost'])

        self.last_szn_matches = []
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
        points = 0
        if len(self.future_matches) > 0:
            points = self.future_matches[0].points
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
                if minutes_played > 200:
                    self.history_past[start_year] = total_points / minutes_played
                    self.history_past_xgi[start_year] = xgi / minutes_played
                else:
                    self.history_past[start_year] = 0
                    self.history_past_xgi[start_year] = 0

    @staticmethod
    def xgi_per_min_from_matches(player) -> float:
        """Season xGI / minutes from non-carry appearances (0 if <200 minutes)."""
        total_xgi = 0.0
        total_mins = 0.0
        for match in player.matches:
            if getattr(match, 'is_last_season_carry', False):
                continue
            mins = float(match.game_stats.get('minutes') or 0)
            if mins <= 0:
                continue
            total_mins += mins
            total_xgi += float(match.game_stats.get('expected_goal_involvements') or 0)
        if total_mins > 200:
            return total_xgi / total_mins
        return 0.0

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
    

    def attach_last_szn_matches(self, past_szn_player):
        """Prepend last 7 prior-season *appearances* (minutes > 0) for early-GW form.

        New-to-league players (no past_szn_player / no appearances) get nothing — form
        stays 0 until they play this season.
        """
        self.last_szn_matches = past_szn_player.matches if past_szn_player else []
        if not past_szn_player or not past_szn_player.matches:
            return
        if self.matches and getattr(self.matches[0], 'is_last_season_carry', False):
            return

        played = [
            m for m in past_szn_player.matches
            if float(m.game_stats.get('minutes', getattr(m, 'minutes', 0)) or 0) > 0
        ]
        if not played:
            return

        n = min(AVG_WINDOW, len(played))
        matches_to_insert = []
        for m in played[-n:]:
            # Shallow copy alone shares past_stats with the prior-season match; re-processing
            # carry would then overwrite that season's training snapshots. Give each carry
            # row its own past_stats / game_stats before this season rebuilds form.
            cm = copy.copy(m)
            cm.past_stats = get_default_past_stats()
            cm.game_stats = copy.deepcopy(m.game_stats)
            cm.is_last_season_carry = True
            cm.valid_for_learning = False
            cm.valid_for_minutes = False
            matches_to_insert.append(cm)
        self.matches = matches_to_insert + list(self.matches)


    def process_matches(self):
        self.build_snapshot_stats()
        self.populate_future_matches()


    def populate_future_matches(self):
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
                        match.past_stats[key] = self.get_stat_window(
                            stats_list,
                            appearance_weights,
                            per_90=(key in PER_MINUTE_STAT_KEYS),
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
                    match.past_stats[key] = self.get_stat_window(
                        stats_list,
                        appearance_weights,
                        per_90=(key in PER_MINUTE_STAT_KEYS),
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
                self.past_stats[key] = self.get_stat_window(
                    stats_list,
                    appearance_weights,
                    per_90=(key in PER_MINUTE_STAT_KEYS),
                )
            self.past_stats['chance_to_play'] = self.get_stat_window(
                chance_to_play_list, chance_weights, per_90=False
            )

    def get_stat_window(self, stats_list, weights=None, per_90: bool = False) -> StatWindow:
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
            # last_5 / avg: pooled per-90 over last 5 / last AVG_WINDOW appearances
            def pooled_pp90(window):
                raw_sum = sum(raw for raw, _mins in window)
                mins_sum = sum(mins for _raw, mins in window)
                return (raw_sum * 90.0 / mins_sum) if mins_sum > 0 else 0.0

            last_5_avg = pooled_pp90(stats_list[-LAST_5_WINDOW:])
            avg = pooled_pp90(stats_list[-AVG_WINDOW:])
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
