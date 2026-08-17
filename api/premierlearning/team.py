from .team_match import TeamMatch
from .helper_functions import coerce_bool


class Team:

    def __init__(self, team_dict, fixtures_json, last_season_team_standings):
        self.team_dict = team_dict
        self.id = int(team_dict['id'])
        self.name = team_dict['short_name']

        self.last_season_finish = self.find_last_season_finish(last_season_team_standings)

        self.average_points_so_far = 0
        self.average_scored_so_far = 0
        self.average_conceded_so_far = 0

        self.team_match_dict = {}
        self.future_team_match_dict = {}
        self.populate_fixtures(fixtures_json)

        self.strength = {}

        for position in ['attack', 'defence']:
            for location in ['home', 'away']:
                strength_key = 'strength_%s_%s' % (position, location)
                raw = team_dict.get(strength_key, 0)
                try:
                    self.strength[strength_key] = float(raw) if raw not in ('', None) else 0.0
                except (TypeError, ValueError):
                    self.strength[strength_key] = 0.0

    def find_last_season_finish(self, last_season_team_standings):
        try:
            place = last_season_team_standings[self.name]
        except Exception:
            place = 21
        return place

    def populate_fixtures(self, fixtures_json):
        for fixture_json in fixtures_json:
            played_home = int(fixture_json['team_h']) == self.id
            if played_home or int(fixture_json['team_a']) == self.id:
                if coerce_bool(fixture_json.get('finished')):
                    self.team_match_dict[int(fixture_json['id'])] = TeamMatch(fixture_json, played_home)
                else:
                    self.future_team_match_dict[int(fixture_json['id'])] = TeamMatch(fixture_json, played_home)

        self.build_snapshot_stats()
        for team_match in self.future_team_match_dict.values():
            team_match.average_scored_in_previous_matches = self.average_scored_so_far
            team_match.average_conceded_in_previous_matches = self.average_conceded_so_far
            team_match.average_points_in_previous_matches = self.average_points_so_far
            team_match.scored_season_avg = self.average_scored_so_far
            team_match.conceded_season_avg = self.average_conceded_so_far
            # last-5 from end of finished matches (set in build_snapshot_stats)
            team_match.scored_last_5_avg = getattr(self, '_final_scored_last_5', self.average_scored_so_far)
            team_match.conceded_last_5_avg = getattr(self, '_final_conceded_last_5', self.average_conceded_so_far)

    def build_snapshot_stats(self):
        match_count = 0
        point_sum_so_far = 0
        scored_list = []
        conceded_list = []

        finished = sorted(self.team_match_dict.values(), key=lambda m: (m.event or 0, m.id))
        for match in finished:
            match.average_points_in_previous_matches = self.average_points_so_far
            match.average_scored_in_previous_matches = self.average_scored_so_far
            match.average_conceded_in_previous_matches = self.average_conceded_so_far
            match.set_form_windows(scored_list, conceded_list)

            match_count += 1
            point_sum_so_far += match.points
            self.average_points_so_far = point_sum_so_far / match_count
            scored_list.append(match.scored)
            conceded_list.append(match.conceded)
            self.average_scored_so_far = sum(scored_list) / len(scored_list)
            self.average_conceded_so_far = sum(conceded_list) / len(conceded_list)

        if scored_list:
            self._final_scored_last_5 = sum(scored_list[-5:]) / len(scored_list[-5:])
            self._final_conceded_last_5 = sum(conceded_list[-5:]) / len(conceded_list[-5:])
        else:
            self._final_scored_last_5 = 0.0
            self._final_conceded_last_5 = 0.0

    def __str__(self):
        return self.name
