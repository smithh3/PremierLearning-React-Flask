"""Redraw every match of a copy from xG (plan section 2c).

1. Team xG = half the real match's team xG, half the team-rating expectation
   for that pairing, times the team's attack form and the opponent's defence
   form, times a lognormal draw.
2. Split across players on the pitch by their season xG per 90 x minutes.
3. Goals ~ Poisson(player xG x finishing x the opponent's conversion):
   strong sides concede a little less than xG (group_effects); own goals ~
   Poisson per team.
4. Each goal gets a minute inside the scorer's time on the pitch; it is
   assisted at the season's real rate, the assister drawn by (xA per 90 x
   assisting)^ASSISTER_POWER among teammates on the pitch at that minute.
5. Goals conceded and clean sheets from the opponent's goals while on the pitch.
6. Saves scaled by the opponent's drawn xG over its real xG.
7. BPS: the real BPS minus its goal/assist/clean-sheet/conceded/save/minutes
   parts, plus the same parts for the drawn events; bonus 3/2/1 by FPL's rules.
8. Points: FPL scoring of the drawn events; cards, penalties and defensive
   contributions as they happened (they move with the minutes in a swap).

Seasons without FPL xG (before 2022/23) use xG and xA estimated from FPL's
ICT threat and creativity, fitted on the seasons that have both.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np

from .source import DATA_ROOT, bonus_from_bps, event_bps, event_points, load_source

FORM_SD = 0.08
FORM_DRIFT_SD = 0.01
MATCH_SD = 0.25
REAL_WEIGHT = 0.5
RATE_PRIOR_MATCHES = 5.0
FINISHING_PRIOR_XG = 30.0
XG_SEASONS = (22, 23, 24, 25)
INJURY_SEASONS = (21, 22, 23, 24, 25)   # seasons with FPL flags
ICT_PRIOR_THREAT = 1000.0               # about 11 xG of shots: weight of the league threat->xG ratio
ICT_PRIOR_CREATIVITY = 1500.0
SAVES_SMOOTHING = 0.5                   # xG added to both sides of the saves ratio
SQUAD_RATE_SHARE = 0.5                 # little-used players are pulled toward half the position rate
OWN_GOAL_WEIGHT = {1: 0.5, 2: 3.0, 3: 1.0, 4: 0.5}
CALIBRATION_FILE = 'calibration.json'
TEAM_CONVERSION_LIMITS = (0.85, 1.15)
# Assists go to the team's creators more than in proportion to their xA rate
# (a striker's goal is usually laid on by the side's creator, not whoever is
# on). Handing the real goals of 2022/23-2025/26 out in proportion gave the 20
# dearest players 0.83-0.92 of their real assists every season; this power
# gives 0.93-1.01, the next 40 0.99-1.05 and the rest 0.99-1.01.
ASSISTER_POWER = 1.25


# ----- once-only numbers from the seasons with FPL xG -----

def calibration(data_root: Path = None, refresh: bool = False) -> dict:
    """Threat->xG and creativity->xA slopes, goals and xG per player code, and the group effects."""
    root = Path(data_root) if data_root else DATA_ROOT
    path = root / 'synthetic' / CALIBRATION_FILE
    if path.is_file() and not refresh:
        result = json.loads(path.read_text(encoding='utf-8'))
        if 'team_conversion' not in result:
            result.update(group_effects(root, result))
            path.write_text(json.dumps(result), encoding='utf-8')
        return result
    result = _base_calibration(root)
    result.update(group_effects(root, result))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding='utf-8')
    return result


def group_effects(root: Path, calib: dict) -> dict:
    """Team-level conversion: do strong sides beat the copies' goal draw as a group?

    Per team-season, goals over the copies' draw (xG x each player's
    finishing), for and against, regressed on the side's xG difference per
    match in league standard deviations. A slope is used only when it has the
    same sign in every season with FPL xG (2022/23-2025/26); otherwise it is
    0 and a few sides having good years does not move every copy. Measured
    2026-11-10: conceding -0.025/-0.047/-0.020/-0.024 (kept: strong sides
    concede less than xG); scoring +0.048/+0.011/+0.005/-0.010 (dropped: the
    players' own finishing already covers it).

    Also checked and left out (not the same every season, within noise):
    finishing of the 20 dearest players (goals/xG 1.04, 1.09, 1.02, 1.00) and
    their FPL assists per xA over their position's rate (1.09, 1.00, 1.06, 1.02,
    each +- 0.09).
    """
    sources = {year: load_source(year, root) for year in XG_SEASONS}
    partial = dict(calib, team_conversion={'for': 0.0, 'against': 0.0})
    points = {'for': [], 'against': []}
    per_season = {}
    for year, source in sources.items():
        model = SeasonModel(source, partial)
        drawn = defaultdict(float)
        for row in source.rows:
            if row['minutes'] > 0:
                drawn[(row['fixture'], row['_team'])] += row['_xg_real'] * model.finishing[row['element']]
        season = {'for': [], 'against': []}
        for team in model.teams:
            gf = ga = draw_for = draw_against = 0.0
            for f in source.fixtures.values():
                if f.gameweek is None or team not in (f.home, f.away):
                    continue
                opponent = f.away if team == f.home else f.home
                gf += f.home_goals if team == f.home else f.away_goals
                ga += f.away_goals if team == f.home else f.home_goals
                draw_for += drawn[(f.id, team)]
                draw_against += drawn[(f.id, opponent)]
            season['for'].append((model.strength[team], gf / draw_for, draw_for))
            season['against'].append((model.strength[team], ga / draw_against, draw_against))
        per_season[str(year)] = {kind: round(_slope(values), 4) for kind, values in season.items()}
        for kind in points:
            points[kind].extend(season[kind])
    conversion = {}
    for kind, values in points.items():
        signs = {np.sign(season[kind]) for season in per_season.values()}
        conversion[kind] = round(_slope(values), 4) if len(signs) == 1 else 0.0
    return {
        'team_conversion': conversion,
        'team_conversion_pooled': {kind: round(_slope(values), 4) for kind, values in points.items()},
        'team_conversion_by_season': per_season,
    }


def _slope(points) -> float:
    """Weighted least-squares slope of ratio on strength."""
    z = np.array([p[0] for p in points]); ratio = np.array([p[1] for p in points]); w = np.array([p[2] for p in points])
    zm, rm = np.average(z, weights=w), np.average(ratio, weights=w)
    return float(np.sum(w * (z - zm) * (ratio - rm)) / np.sum(w * (z - zm) ** 2))


def _base_calibration(root: Path) -> dict:
    from .injuries import Squads, find_spells, injury_targets
    sums = defaultdict(float)
    by_code = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    ict_by_code = defaultdict(lambda: [0.0, 0.0])        # threat, creativity in xG seasons
    injuries = defaultdict(lambda: [0, 0])               # code -> [spells, seasons as a regular]
    residual, expected = [], []
    for year in INJURY_SEASONS:
        source = load_source(year, root)
        squads = Squads(source.rows, source)
        spells = find_spells(squads)
        # Seasons as a starter when fit (the players who can get a new spell).
        for keys in injury_targets(squads, spells).values():
            for element, _team in keys:
                if element in source.codes:
                    injuries[source.codes[element]][1] += 1
        for spell in spells:
            if spell.element in source.codes:
                injuries[source.codes[spell.element]][0] += 1
        if year not in XG_SEASONS:
            continue
        match_xg, match_goals = defaultdict(float), defaultdict(int)
        for row in source.rows:
            if row['minutes'] <= 0:
                continue
            match_xg[(row['fixture'], row['_team'])] += row['expected_goals']
            sums['tx'] += row['threat'] * row['expected_goals']
            sums['tt'] += row['threat'] ** 2
            sums['cx'] += row['creativity'] * row['expected_assists']
            sums['cc'] += row['creativity'] ** 2
            code = source.codes.get(row['element'])
            if code is not None:
                by_code[code][0] += row['goals_scored']
                by_code[code][1] += row['expected_goals']
                by_code[code][2] += row['assists']
                by_code[code][3] += row['expected_assists']
                ict_by_code[code][0] += row['threat']
                ict_by_code[code][1] += row['creativity']
        for f in source.fixtures.values():
            if f.gameweek is None:
                continue
            for team, goals in ((f.home, f.home_goals), (f.away, f.away_goals)):
                residual.append(goals - match_xg[(f.id, team)])
                expected.append(match_xg[(f.id, team)])
    goals = sum(value[0] for value in by_code.values())
    xg = sum(value[1] for value in by_code.values())
    result = {
        'seasons': list(XG_SEASONS),
        'xg_per_threat': sums['tx'] / sums['tt'],
        'xa_per_creativity': sums['cx'] / sums['cc'],
        'goals_per_xg': goals / xg,
        'assists_per_xa': sum(v[2] for v in by_code.values()) / sum(v[3] for v in by_code.values()),
        # Team goals vary less around match xG than a Poisson draw (a match is a
        # set of shots): variance / mean of goals - xG.
        'goal_variance_ratio': float(np.var(residual) / np.mean(expected)),
        'players': {str(code): [round(g, 0), round(x, 3), round(a, 0), round(xa, 3),
                                round(ict_by_code[code][0], 1), round(ict_by_code[code][1], 1)]
                    for code, (g, x, a, xa) in by_code.items()},
        # Injury spells per player over 2021/22-2025/26, and seasons he was a regular.
        'injuries': {str(code): value for code, value in injuries.items()},
    }
    return result


class SeasonModel:
    """Everything about the real season that every copy shares."""

    def __init__(self, source, calib: dict):
        self.source = source
        rows = source.rows
        per_threat, per_creativity = self._ict_ratios(calib)
        for row in rows:
            if source.has_xg:
                row['_xg_real'] = row.get('expected_goals', 0.0)
                row['_xa_real'] = row.get('expected_assists', 0.0)
            else:
                row['_xg_real'] = per_threat.get(row['element'], calib['xg_per_threat']) * row.get('threat', 0.0)
                row['_xa_real'] = per_creativity.get(row['element'], calib['xa_per_creativity']) * row.get('creativity', 0.0)
        own_goals = sum(row['own_goals'] for row in rows)
        player_goals = sum(row['goals_scored'] for row in rows)
        if not source.has_xg:
            # Put the estimated xG on the season's own scale.
            total = sum(row['_xg_real'] for row in rows if row['minutes'] > 0)
            scale = player_goals / calib['goals_per_xg'] / total
            for row in rows:
                row['_xg_real'] *= scale
        self.finishing_centre = player_goals / max(1e-9, sum(r['_xg_real'] for r in rows if r['minutes'] > 0))
        self.assist_share = sum(row['assists'] for row in rows) / max(1, player_goals)
        team_matches = 2 * sum(1 for f in source.fixtures.values() if f.gameweek is not None)
        self.own_goal_rate = own_goals / max(1, team_matches)
        # Team xG for and against in each real match.
        self.match_xg = defaultdict(float)
        for row in rows:
            if row['minutes'] > 0:
                self.match_xg[(row['fixture'], row['_team'])] += row['_xg_real']
        self._ratings()
        self._player_rates(calib)
        self._team_conversion(calib)
        for row in rows:
            row['_bps_rest'] = row['bps'] - event_bps(row, row['_position'])
            row['_points_rest'] = row['total_points'] - event_points(row, row['_position'], source.has_dc)
            row['_saves_real'] = row.get('saves', 0)

    def _ict_ratios(self, calib):
        """Each player's own xG per threat and xA per creativity in the xG seasons, shrunk."""
        per_threat, per_creativity = {}, {}
        for element, code in self.source.codes.items():
            values = calib['players'].get(str(code))
            if not values:
                continue
            _g, xg, _a, xa, threat, creativity = values
            per_threat[element] = (xg + ICT_PRIOR_THREAT * calib['xg_per_threat']) / (threat + ICT_PRIOR_THREAT)
            per_creativity[element] = (
                (xa + ICT_PRIOR_CREATIVITY * calib['xa_per_creativity']) / (creativity + ICT_PRIOR_CREATIVITY))
        return per_threat, per_creativity

    def _ratings(self):
        fixtures = [f for f in self.source.fixtures.values() if f.gameweek is not None]
        self.home_mean = np.mean([self.match_xg[(f.id, f.home)] for f in fixtures])
        self.away_mean = np.mean([self.match_xg[(f.id, f.away)] for f in fixtures])
        made, allowed, count = defaultdict(float), defaultdict(float), defaultdict(int)
        for f in fixtures:
            for team, opponent, mean in ((f.home, f.away, self.home_mean), (f.away, f.home, self.away_mean)):
                made[team] += self.match_xg[(f.id, team)] / mean
                allowed[opponent] += self.match_xg[(f.id, team)] / mean
                count[team] += 1
        self.attack = {team: made[team] / count[team] for team in count}
        self.defence = {team: allowed[team] / count[team] for team in count}
        self.teams = sorted(count)
        # xG difference per match, in standard deviations across the league.
        difference = {}
        for team in self.teams:
            matches = [f for f in fixtures if team in (f.home, f.away)]
            difference[team] = sum(
                self.match_xg[(f.id, team)] - self.match_xg[(f.id, f.away if team == f.home else f.home)]
                for f in matches) / max(1, len(matches))
        spread = float(np.std(list(difference.values()))) or 1.0
        centre = float(np.mean(list(difference.values())))
        self.strength = {team: (value - centre) / spread for team, value in difference.items()}

    def _player_rates(self, calib):
        minutes, xg, xa = defaultdict(float), defaultdict(float), defaultdict(float)
        position = {}
        for row in self.source.rows:
            element = row['element']
            minutes[element] += row['minutes']
            xg[element] += row['_xg_real'] if row['minutes'] > 0 else 0.0
            xa[element] += row['_xa_real'] if row['minutes'] > 0 else 0.0
            position[element] = row['_position']
        pos_rate = {}
        for pos in (1, 2, 3, 4):
            m = sum(minutes[e] for e in minutes if position[e] == pos)
            pos_rate[pos] = (
                90.0 * sum(xg[e] for e in xg if position[e] == pos) / max(1.0, m),
                90.0 * sum(xa[e] for e in xa if position[e] == pos) / max(1.0, m),
            )
        # The prior is half the position's rate: players with few minutes are
        # mostly squad players, and a full position rate took scoring share
        # from the regulars.
        k, share = RATE_PRIOR_MATCHES, SQUAD_RATE_SHARE
        self.xg90 = {e: (xg[e] + k * share * pos_rate[position[e]][0]) / (minutes[e] / 90.0 + k) for e in minutes}
        self.xa90 = {e: (xa[e] + k * share * pos_rate[position[e]][1]) / (minutes[e] / 90.0 + k) for e in minutes}
        players = calib['players']
        # Goals per xG and FPL assists per xA for each position (forwards get
        # far more FPL assists than xA: flick-ons, penalties won, rebounds).
        totals = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        for element, code in self.source.codes.items():
            values = players.get(str(code))
            if values and element in position:
                for index in range(4):
                    totals[position[element]][index] += values[index]
        centres = {pos: (t[0] / t[1] if t[1] > 0 else 1.0, t[2] / t[3] if t[3] > 0 else calib['assists_per_xa'])
                   for pos, t in totals.items()}
        self.finishing, self.assisting = {}, {}
        for element in minutes:
            goals, expected, assists, expected_assists = players.get(
                str(self.source.codes.get(element)), (0.0, 0.0, 0.0, 0.0))[:4]
            goal_centre, assist_centre = centres.get(position[element], (1.0, calib['assists_per_xa']))
            self.finishing[element] = (
                (goals + FINISHING_PRIOR_XG * goal_centre) / (expected + FINISHING_PRIOR_XG)
            )
            # FPL assists include penalties won and rebounds, so some players
            # get many more than their xA; same shrinkage as finishing.
            self.assisting[element] = (
                (assists + FINISHING_PRIOR_XG * assist_centre) / (expected_assists + FINISHING_PRIOR_XG)
            )
        # Keep each player's finishing relative to others, but on the source
        # season's own level, so a copy scores as many goals as the real season.
        drawn = sum(row['_xg_real'] * self.finishing[row['element']]
                    for row in self.source.rows if row['minutes'] > 0)
        real_goals = sum(row['goals_scored'] for row in self.source.rows)
        level = real_goals / max(1e-9, drawn)
        self.finishing = {element: value * level for element, value in self.finishing.items()}
        self.shot_chance = min(0.45, max(0.0, 1.0 - calib['goal_variance_ratio']))
        # How injury-prone each player is: (spells + 1) / (seasons as a starter
        # when fit + 1) over 2021/22-2025/26, so new injuries land on the players who get them.
        self.proneness = {}
        for element, code in self.source.codes.items():
            spells, seasons = calib.get('injuries', {}).get(str(code), (0, 0))
            self.proneness[element] = (spells + 1.0) / (seasons + 1.0)

    def _team_conversion(self, calib):
        """Goal-draw multipliers for strong sides scoring more and conceding less (group_effects)."""
        slopes = calib.get('team_conversion') or {}
        low, high = TEAM_CONVERSION_LIMITS
        self.convert_for = {team: min(high, max(low, 1.0 + slopes.get('for', 0.0) * self.strength[team]))
                            for team in self.teams}
        self.convert_against = {team: min(high, max(low, 1.0 + slopes.get('against', 0.0) * self.strength[team]))
                                for team in self.teams}
        # Keep the season's goal total: rescale over the real matches.
        drawn = defaultdict(float)
        for row in self.source.rows:
            if row['minutes'] > 0:
                drawn[(row['fixture'], row['_team'])] += row['_xg_real'] * self.finishing[row['element']]
        base = adjusted = 0.0
        for f in self.source.fixtures.values():
            if f.gameweek is None:
                continue
            for team, opponent in ((f.home, f.away), (f.away, f.home)):
                base += drawn[(f.id, team)]
                adjusted += drawn[(f.id, team)] * self.convert_for[team] * self.convert_against[opponent]
        scale = base / adjusted if adjusted > 0 else 1.0
        self.convert_for = {team: value * scale for team, value in self.convert_for.items()}


def team_form(model: SeasonModel, rng) -> Dict[str, Dict[int, np.ndarray]]:
    """Attack and defence multipliers per team per gameweek (index 1..38)."""
    form = {'attack': {}, 'defence': {}}
    for kind in form:
        for team in model.teams:
            walk = 1.0 + rng.normal(0.0, FORM_SD) + np.cumsum(rng.normal(0.0, FORM_DRIFT_SD, 39))
            form[kind][team] = np.clip(walk, 0.6, 1.6)
    return form


def _interval(row):
    minutes = min(90, int(row['minutes']))
    if row['_started']:
        return 0.0, float(minutes)
    return float(90 - minutes), 90.0


def draw_matches(model: SeasonModel, squads, form, rng, fixtures=None) -> dict:
    """Redraw every fixture in place on the copy's rows. Returns {fixture: (home, away)}.

    fixtures: the copy's fixtures (schedule.py), default the real ones. A
    fixture keeps its real match xG wherever it is played.
    """
    source = model.source
    scores = {}
    fixtures = source.fixtures if fixtures is None else fixtures
    for fixture in sorted(fixtures.values(), key=lambda f: (f.gameweek or 99, f.id)):
        if fixture.gameweek is None:
            continue
        gw = fixture.gameweek
        expected = {}
        for team, opponent, mean in ((fixture.home, fixture.away, model.home_mean),
                                     (fixture.away, fixture.home, model.away_mean)):
            rating = mean * model.attack[team] * model.defence[opponent]
            blend = REAL_WEIGHT * model.match_xg[(fixture.id, team)] + (1 - REAL_WEIGHT) * rating
            blend *= form['attack'][team][gw] * form['defence'][opponent][gw]
            expected[team] = blend * math.exp(rng.normal(-MATCH_SD ** 2 / 2, MATCH_SD))
        lines = {team: [r for r in squads.team_match[(fixture.id, team)] if r['minutes'] > 0]
                 for team in (fixture.home, fixture.away)}
        goals = {team: [] for team in lines}       # (minute, scorer or None, own goal by)
        for team, opponent in ((fixture.home, fixture.away), (fixture.away, fixture.home)):
            on = lines[team]
            weights = np.array([model.xg90.get(r['element'], 0.0) * r['minutes'] for r in on])
            if weights.sum() <= 0:
                weights = np.array([float(r['minutes']) for r in on])
            shares = weights / weights.sum() if weights.sum() > 0 else weights
            xa_weights = np.array([model.xa90.get(r['element'], 0.0) * r['minutes'] for r in on])
            xa_shares = xa_weights / xa_weights.sum() if xa_weights.sum() > 0 else xa_weights
            scoring = np.zeros(len(on))
            for index, row in enumerate(on):
                row['_xg'] = expected[team] * shares[index]
                row['_xa'] = expected[team] * model.assist_share * xa_shares[index]
                scoring[index] = (row['_xg'] * model.finishing.get(row['element'], 1.0)
                                  * model.convert_for[team] * model.convert_against[opponent])
            if on and scoring.sum() > 0:
                count = team_goal_count(rng, float(scoring.sum()), model.shot_chance)
                for scorer in rng.choice(len(on), size=count, p=scoring / scoring.sum()):
                    row = on[int(scorer)]
                    start, end = _interval(row)
                    goals[team].append((rng.uniform(start, end), row, None))
            own_goals = int(rng.poisson(model.own_goal_rate))
            outfield = [r for r in lines[opponent]]
            if own_goals and outfield:
                w = np.array([OWN_GOAL_WEIGHT[r['_position']] * r['minutes'] for r in outfield], dtype=float)
                for _goal in range(own_goals):
                    culprit = outfield[int(rng.choice(len(outfield), p=w / w.sum()))]
                    start, end = _interval(culprit)
                    goals[team].append((rng.uniform(start, end), None, culprit))
        for team in lines:
            for row in squads.team_match[(fixture.id, team)]:
                for name in ('goals_scored', 'assists', 'own_goals', 'goals_conceded', 'clean_sheets'):
                    row[name] = 0
                if row['minutes'] <= 0:
                    row['_xg'] = row['_xa'] = 0.0
        for team, opponent in ((fixture.home, fixture.away), (fixture.away, fixture.home)):
            for minute, scorer, culprit in goals[team]:
                if culprit is not None:
                    culprit['own_goals'] += 1
                    continue
                scorer['goals_scored'] += 1
                if rng.random() < model.assist_share:
                    mates = [r for r in lines[team] if r is not scorer and _on_at(r, minute)]
                    w = np.array([max(0.002, model.xa90.get(r['element'], 0.0) * model.assisting.get(r['element'], 1.0))
                                  for r in mates]) ** ASSISTER_POWER
                    if mates:
                        mates[int(rng.choice(len(mates), p=w / w.sum()))]['assists'] += 1
            for row in lines[opponent]:
                row['goals_conceded'] = sum(1 for minute, _s, _c in goals[team] if _on_at(row, minute))
        for team, opponent in ((fixture.home, fixture.away), (fixture.away, fixture.home)):
            for row in squads.team_match[(fixture.id, team)]:
                row['clean_sheets'] = int(row['minutes'] >= 60 and row['goals_conceded'] == 0)
                if row['_position'] == 1 and row['minutes'] > 0:
                    # The real saves were made against the opponent really
                    # faced on this date (a different one after a reshuffle).
                    # Smoothed so a match with tiny real xG does not multiply them.
                    real = row.get('_opp_xg_real', model.match_xg[(fixture.id, opponent)])
                    ratio = min(2.5, max(0.4, (expected[opponent] + SAVES_SMOOTHING) / (real + SAVES_SMOOTHING)))
                    value = row.get('_saves_real', 0) * ratio
                    row['saves'] = int(math.floor(value) + (rng.random() < value - math.floor(value)))
                row['_xgc'] = expected[opponent] * min(90, row['minutes']) / 90.0
        home_goals = len(goals[fixture.home])
        away_goals = len(goals[fixture.away])
        scores[fixture.id] = (home_goals, away_goals)
        everyone = squads.team_match[(fixture.id, fixture.home)] + squads.team_match[(fixture.id, fixture.away)]
        bps = {}
        for row in everyone:
            row['team_h_score'], row['team_a_score'] = home_goals, away_goals
            if row['minutes'] > 0:
                row['bps'] = row['_bps_rest'] + event_bps(row, row['_position'])
                bps[row['element']] = row['bps']
            else:
                row['bps'] = 0
        bonus = bonus_from_bps(bps)
        for row in everyone:
            row['bonus'] = bonus.get(row['element'], 0)
            # Without minutes only the leftover counts (a booking on the bench).
            row['total_points'] = row['_points_rest'] + event_points(row, row['_position'], source.has_dc)
    return scores


def team_goal_count(rng, mean: float, shot_chance: float) -> int:
    """Goals with this mean but less spread than Poisson, like a set of shots.

    Binomial(n, q) with n * q = mean has variance mean * (1 - q); q comes from
    how much real goals vary around match xG. The fractional part of n is one
    extra shot taken with the leftover probability.
    """
    if shot_chance <= 0:
        return int(rng.poisson(mean))
    shots = mean / shot_chance
    whole = int(shots)
    goals = int(rng.binomial(whole, shot_chance)) if whole else 0
    return goals + int(rng.random() < (shots - whole) * shot_chance)


def _on_at(row, minute) -> bool:
    start, end = _interval(row)
    return start <= minute <= end
