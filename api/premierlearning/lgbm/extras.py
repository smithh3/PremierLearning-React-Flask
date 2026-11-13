"""LightGBM feature set 3 columns that need the seasons, not just the batch.

Two column groups, both as of each row's kickoff (matches that kicked off on an
earlier calendar day only), built from FPL's own match data:

1. Team ratings: an attack and a defence rating per club plus home advantage,
   fitted to every Premier League result of the last three years with older
   matches fading (half-life HALF_LIFE_DAYS), the Dixon-Coles time decay. The
   expected goals of a fixture are exp(base + home + attack + opposing defence),
   so a team that ran up goals against weak sides rates lower than one that did
   it against strong ones, which plain recent form cannot tell apart. Fitted to
   goals (every season) and to FPL xG (seasons that have it). A club with no
   league match in the last NEWCOMER_DAYS days (promoted) starts from a weak
   prior rather than the league average. The Dixon-Coles low-score correction
   is left out: it changes 0-0 and 1-0 odds a little, and the clean-sheet stage
   of lgbm_v5 learns its own corrections.
2. What the opponent concedes to each position: per opponent match, the FPL
   points, xG and xA its opponents' goalkeepers, defenders, midfielders and
   forwards (those who played) scored, over its last 5 / 10 / 38 matches, and
   the same relative to the league average for that position over the past
   year (so seasons with different scoring rules, e.g. defcon from 2025/26,
   compare fairly).
"""
import bisect
import math
from collections import defaultdict
from datetime import datetime

import numpy

from premierlearning.helper_functions import coerce_bool
from premierlearning.learning_v2.dataset import game_stat
from premierlearning.learning_v2.kickoff import kickoff_from_match
from premierlearning.learning_v2.teams import TeamTimelines, time_key

EPOCH = datetime(2000, 1, 1)
HALF_LIFE_DAYS = 240.0
WINDOW_DAYS = 3 * 365.0
RIDGE = 3.0                 # pull of each rating toward its prior, in matches' worth of weight
NEWCOMER_DAYS = 300.0
LEAGUE_GOALS = 1.40         # goals per team per match, the promoted prior's reference
NEWCOMER_PRIOR = (math.log(1.05 / LEAGUE_GOALS), math.log(1.75 / LEAGUE_GOALS))   # attack, defence
NEWTON_STEPS = 25
MIN_XG_WEIGHT = 50.0        # weighted xG observations needed for an xG fit
RATING_TARGETS = ('goals', 'xg')
CONCEDED_WINDOWS = (5, 10, 38)
BASELINE_DAYS = 365.0
POSITIONS = 5               # 0 unknown, 1 GK, 2 DEF, 3 MID, 4 FWD
CONCEDED_STATS = ('points', 'xg', 'xa')


def day_number(time):
    return (time - EPOCH).total_seconds() / 86400.0


def row_day(match, year):
    """The calendar day a row kicks off; it sees matches from earlier days only."""
    time = time_key(kickoff_from_match(match), year, int(getattr(match, 'gameweek', 0) or 0))
    return math.floor(day_number(time))


def _code(team):
    try:
        code = int(getattr(team, 'code', None))
    except (TypeError, ValueError):
        return None
    return code if code > 0 else None


def _position(player):
    try:
        position = int(getattr(player, 'position', 0) or 0)
    except (TypeError, ValueError):
        return 0
    return position if 1 <= position < POSITIONS else 0


class TeamRatings:
    """Time-decayed Poisson attack / defence ratings, refitted as of any day."""

    def __init__(self, timelines, target='goals'):
        if target not in RATING_TARGETS:
            raise ValueError('target must be one of %s' % (RATING_TARGETS,))
        rows = []
        for code, records in timelines.records.items():
            for record in records:
                if not record.opponent:
                    continue
                if target == 'xg':
                    if not record.xg_known:
                        continue
                    value = record.xg_for
                else:
                    value = record.goals_for
                rows.append((day_number(record.time), int(code), int(record.opponent), float(record.home),
                             float(value)))
        rows.sort()
        self.target = target
        self.days = numpy.array([row[0] for row in rows], dtype=numpy.float64)
        teams = sorted({row[1] for row in rows} | {row[2] for row in rows})
        self.teams = {code: index for index, code in enumerate(teams)}
        self.team = numpy.array([self.teams[row[1]] for row in rows], dtype=numpy.int64)
        self.opp = numpy.array([self.teams[row[2]] for row in rows], dtype=numpy.int64)
        self.home = numpy.array([row[3] for row in rows], dtype=numpy.float64)
        self.goals = numpy.array([row[4] for row in rows], dtype=numpy.float64)
        self._fits = {}
        self._last = None

    def _priors(self, cut, day):
        """Prior (attack, defence) per team: 0, or the newcomer prior with no recent league match."""
        attack = numpy.zeros(len(self.teams))
        defence = numpy.zeros(len(self.teams))
        start = bisect.bisect_left(self.days, day - NEWCOMER_DAYS)
        recent = numpy.zeros(len(self.teams), dtype=bool)
        recent[self.team[start:cut]] = True
        recent[self.opp[start:cut]] = True
        attack[~recent] = NEWCOMER_PRIOR[0]
        defence[~recent] = NEWCOMER_PRIOR[1]
        return attack, defence

    def at(self, day):
        """(base, home, attack[team index], defence[team index]) from matches before `day`, or None."""
        cut = bisect.bisect_left(self.days, day)
        fit = self._fits.get(cut)
        if fit is not None or cut in self._fits:
            return fit
        start = bisect.bisect_left(self.days, day - WINDOW_DAYS)
        weights = 0.5 ** ((day - self.days[start:cut]) / HALF_LIFE_DAYS)
        if weights.sum() < (MIN_XG_WEIGHT if self.target == 'xg' else 1.0):
            self._fits[cut] = None
            return None
        fit = self._solve(start, cut, weights, *self._priors(cut, day))
        self._fits[cut] = fit
        return fit

    def _solve(self, start, cut, weights, prior_attack, prior_defence):
        teams = len(self.teams)
        team = self.team[start:cut]
        opp = self.opp[start:cut]
        home = self.home[start:cut]
        goals = self.goals[start:cut]
        size = 2 + 2 * teams
        rows = numpy.arange(len(team))
        design = numpy.zeros((len(team), size))
        design[:, 0] = 1.0
        design[:, 1] = home
        design[rows, 2 + team] = 1.0
        design[rows, 2 + teams + opp] = 1.0
        prior = numpy.concatenate([[0.0, 0.0], prior_attack, prior_defence])
        penalty = numpy.full(size, RIDGE)
        penalty[:2] = 1e-6
        if self._last is not None and len(self._last) == size:
            params = self._last.copy()
        else:
            params = prior.copy()
            params[0] = math.log(max(1e-3, float((weights * goals).sum() / weights.sum())))
        for _step in range(NEWTON_STEPS):
            rate = numpy.exp(numpy.clip(design @ params, -10.0, 5.0))
            gradient = design.T @ (weights * (goals - rate)) - penalty * (params - prior)
            hessian = (design * (weights * rate)[:, None]).T @ design + numpy.diag(penalty)
            step = numpy.linalg.solve(hessian, gradient)
            largest = numpy.abs(step).max()
            if largest > 1.0:
                step = step / largest   # damped: a far start cannot overshoot
            params = params + step
            if numpy.abs(step).max() < 1e-6:
                break
        self._last = params
        return params[0], params[1], params[2:2 + teams], params[2 + teams:]

    def ratings(self, code, fit, day):
        """(attack, defence) of one club in a fit; a club the fit never saw is a newcomer."""
        index = self.teams.get(code)
        if fit is None:
            return numpy.nan, numpy.nan
        if index is None:
            return NEWCOMER_PRIOR
        return fit[2][index], fit[3][index]


def rating_columns(rows, timelines):
    """{column: values} for the team-rating columns. rows: (team, opp, home, day) per row."""
    out = {}
    for target in RATING_TARGETS:
        model = TeamRatings(timelines, target)
        names = ['exp_for', 'exp_against', 'cs', 'team_attack', 'team_defence', 'opp_attack', 'opp_defence']
        values = {name: numpy.full(len(rows), numpy.nan) for name in names}
        for day in sorted({row[3] for row in rows}):
            model.at(day)   # oldest first, so each fit starts from the day before's
        cache = {}
        for index, (team, opp, home, day) in enumerate(rows):
            key = (team, opp, home, day)
            result = cache.get(key)
            if result is None:
                result = _fixture_rating(model, team, opp, home, day)
                cache[key] = result
            for name, value in zip(names, result):
                values[name][index] = value
        for name in names:
            out['dc.%s.%s' % (target, name)] = values[name]
    return out


def _fixture_rating(model, team, opp, home, day):
    nan = (numpy.nan,) * 7
    if team is None or opp is None:
        return nan
    fit = model.at(day)
    if fit is None:
        return nan
    base, advantage = fit[0], fit[1]
    team_attack, team_defence = model.ratings(team, fit, day)
    opp_attack, opp_defence = model.ratings(opp, fit, day)
    expected_for = math.exp(base + advantage * home + team_attack + opp_defence)
    expected_against = math.exp(base + advantage * (1.0 - home) + opp_attack + team_defence)
    return (expected_for, expected_against, math.exp(-expected_against),
            team_attack, team_defence, opp_attack, opp_defence)


class Conceded:
    """Per club, oldest first: what each position scored against it in each league match."""

    def __init__(self, seasons, timelines):
        times = {}
        for code, records in timelines.records.items():
            for record in records:
                times[(record.year, record.fixture_id, int(code))] = day_number(record.time)
        totals = defaultdict(lambda: numpy.zeros((POSITIONS, len(CONCEDED_STATS) + 1)))
        for season in seasons:
            year = int(season.years[0])
            for player in getattr(season, 'players', []) or []:
                position = _position(player)
                if not position:
                    continue
                for match in getattr(player, 'matches', []) or []:
                    if getattr(match, 'is_last_season_carry', False):
                        continue
                    fixture_id = getattr(match, 'fixture_id', None)
                    against = _code(getattr(match, 'opponent_team', None))
                    if fixture_id is None or against is None or game_stat(match, 'minutes') <= 0:
                        continue
                    cell = totals[(year, int(fixture_id), against)][position]
                    cell[0] += game_stat(match, 'total_points')
                    cell[1] += game_stat(match, 'expected_goals')
                    cell[2] += game_stat(match, 'expected_assists')
                    if getattr(match, 'new_stats_included', False):
                        cell[3] = 1.0
        by_code = defaultdict(list)
        every = []
        for key, cells in totals.items():
            day = times.get(key)
            if day is None:
                continue
            by_code[key[2]].append((day, cells))
            every.append((day, cells))
        self.days = {}
        self.cells = {}
        for code, items in by_code.items():
            items.sort(key=lambda item: item[0])
            self.days[code] = numpy.array([item[0] for item in items])
            self.cells[code] = numpy.stack([item[1] for item in items])
        every.sort(key=lambda item: item[0])
        self.all_days = numpy.array([item[0] for item in every])
        cells = numpy.stack([item[1] for item in every]) if every else numpy.zeros((0, POSITIONS, 4))
        known = cells[:, :, 3:4]
        # Running sums for the league baseline: points over all, xG / xA over known matches.
        stacked = numpy.concatenate([cells[:, :, :1], cells[:, :, 1:3] * known, known,
                                     numpy.ones_like(known)], axis=2)
        self.sums = numpy.concatenate([numpy.zeros((1,) + stacked.shape[1:]), numpy.cumsum(stacked, axis=0)])

    def baseline(self, day, position):
        """League mean per opponent match for this position over the past year: (points, xg, xa)."""
        hi = bisect.bisect_left(self.all_days, day)
        lo = bisect.bisect_left(self.all_days, day - BASELINE_DAYS)
        total = self.sums[hi, position] - self.sums[lo, position]
        count, known = total[4], total[3]
        points = total[0] / count if count > 0 else numpy.nan
        xg = total[1] / known if known > 0 else numpy.nan
        xa = total[2] / known if known > 0 else numpy.nan
        return points, xg, xa

    def columns(self, against, day, position):
        names = []
        values = []
        days = self.days.get(against)
        cut = 0 if days is None else bisect.bisect_left(days, day)
        cells = None if days is None else self.cells[against][:cut, position]
        base = self.baseline(day, position)
        means = {}
        for window in CONCEDED_WINDOWS:
            recent = None if cells is None else cells[-window:]
            count = 0 if recent is None else len(recent)
            points = recent[:, 0].mean() if count else numpy.nan
            known = 0.0 if recent is None else recent[:, 3].sum()
            xg = (recent[:, 1] * recent[:, 3]).sum() / known if known > 0 else numpy.nan
            xa = (recent[:, 2] * recent[:, 3]).sum() / known if known > 0 else numpy.nan
            means[window] = (points, xg, xa)
            for stat, value in zip(CONCEDED_STATS, (points, xg, xa)):
                names.append('opp_conceded.%s.mean%s' % (stat, window))
                values.append(value)
            names.append('opp_conceded.points.rel%s' % window)
            values.append(_ratio(points, base[0]))
            names.append('opp_conceded.xgi.rel%s' % window)
            values.append(_ratio(xg + xa, base[1] + base[2]))
        names.append('opp_conceded.points.league')
        values.append(base[0])
        return names, values


def _ratio(top, bottom):
    if not (numpy.isfinite(top) and numpy.isfinite(bottom)) or bottom <= 1e-6:
        return numpy.nan
    return top / bottom


def conceded_names():
    names = []
    for window in CONCEDED_WINDOWS:
        names.extend('opp_conceded.%s.mean%s' % (stat, window) for stat in CONCEDED_STATS)
        names.extend(['opp_conceded.points.rel%s' % window, 'opp_conceded.xgi.rel%s' % window])
    return names + ['opp_conceded.points.league']


def conceded_columns(rows, conceded):
    """{column: values}, the same columns for any batch. rows: (opponent, day, position) per row."""
    names = conceded_names()
    out = {name: numpy.full(len(rows), numpy.nan) for name in names}
    cache = {}
    for index, key in enumerate(rows):
        if key not in cache:
            against, day, position = key
            cache[key] = None if against is None else conceded.columns(against, day, position)
        result = cache[key]
        if result is None:
            continue
        for name, value in zip(*result):
            out[name][index] = value
    return out


def season_column_names():
    names = ['dc.%s.%s' % (target, name) for target in RATING_TARGETS for name in (
        'exp_for', 'exp_against', 'cs', 'team_attack', 'team_defence', 'opp_attack', 'opp_defence')]
    return names + conceded_names()


def season_columns(batch, seasons, positions):
    """Every feature-set-3 season column for the batch rows, as {name: values}.

    positions: each row's position (1-4; the table's position column).
    """
    timelines = TeamTimelines(seasons)
    years = list(batch.years) or [0] * len(batch.matches)
    ratings, conceded = [], []
    for match, year, position in zip(batch.matches, years, positions):
        day = row_day(match, year)
        opp = _code(getattr(match, 'opponent_team', None))
        home = 1.0 if coerce_bool(getattr(match, 'home', False)) else 0.0
        ratings.append((_code(getattr(match, 'team', None)), opp, home, day))
        position = int(position) if numpy.isfinite(position) else 0
        conceded.append((opp, day, position if 1 <= position < POSITIONS else 0))
    out = rating_columns(ratings, timelines)
    out.update(conceded_columns(conceded, Conceded(seasons, timelines)))
    return out
