class TeamMatch:

    def __init__(self, fixture_json, home):
        self.fixture_json = fixture_json
        self.id = int(self.fixture_json['id'])
        self.home = home
        self.event = fixture_json.get('event')
        if self.event is not None:
            self.event = int(self.event)
        home_away_code = self.HomeAwayChar(home)
        try:
            self.scored = int(self.get_goals_scored(home_away_code.code))
            self.conceded = int(self.get_goals_scored(home_away_code.opponent))
            self.won = self.scored > self.conceded
            self.lost = self.conceded > self.scored
            self.drew = self.scored == self.conceded
            self.points = self.get_points()
        except Exception:
            self.scored = 0
            self.conceded = 0
            self.won = False
            self.lost = False
            self.drew = False
            self.points = 0

        self.average_scored_in_previous_matches = 0
        self.average_conceded_in_previous_matches = 0
        self.average_points_in_previous_matches = 0
        self.scored_last_5_avg = 0.0
        self.scored_season_avg = 0.0
        self.conceded_last_5_avg = 0.0
        self.conceded_season_avg = 0.0
        self.predicted_xg = 0.0
        self.predicted_xc = 0.0
        # FPL fixture difficulty rating (1 easy – 5 hard) for this side and the opponent.
        self.fixture_difficulty = self._side_difficulty(home)
        self.opp_fixture_difficulty = self._side_difficulty(not home)

    def _side_difficulty(self, home):
        key = 'team_h_difficulty' if home else 'team_a_difficulty'
        raw = self.fixture_json.get(key)
        try:
            val = float(raw)
            if 1.0 <= val <= 5.0:
                return val
        except (TypeError, ValueError):
            pass
        return 3.0

    def get_goals_scored(self, h_or_a):
        return self.fixture_json['team_%s_score' % h_or_a]

    def get_points(self):
        points = 0
        if self.drew:
            points = 1
        elif self.won:
            points = 3
        return points

    def set_form_windows(self, scored_list, conceded_list):
        if scored_list:
            self.scored_season_avg = sum(scored_list) / len(scored_list)
            self.scored_last_5_avg = sum(scored_list[-5:]) / len(scored_list[-5:])
            self.average_scored_in_previous_matches = self.scored_season_avg
        if conceded_list:
            self.conceded_season_avg = sum(conceded_list) / len(conceded_list)
            self.conceded_last_5_avg = sum(conceded_list[-5:]) / len(conceded_list[-5:])
            self.average_conceded_in_previous_matches = self.conceded_season_avg

    class HomeAwayChar:
        def __init__(self, home):
            if home:
                self.code = 'h'
                self.opponent = 'a'
            else:
                self.code = 'a'
                self.opponent = 'h'
