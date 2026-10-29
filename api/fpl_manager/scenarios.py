"""On-demand match scores. Nothing is written unless the caller keeps the object.

Team goals are independent Poisson draws around the fixture's recorded team
xG, the sum of player expected_goals. That figure is the xG from the match
file, not a pre-match forecast. If every player on a team is missing the
field, the fixture fails. A stored 0 is a real rate and is used as it is.

Scorers are drawn one goal at a time, with probability proportional to xG.
A player who has minutes but no xG value gets a documented fallback weight of
0.01 so they can still be picked. Players with a stored 0 do not. The draw
is with replacement, so one player can score more than once. Nothing forces
the season share to equal xG.

Assists, for each goal: teammates with minutes, other than the scorer, are
weighted by xA. A missing xA contributes 0 when any teammate has an xA value.
If nobody has an xA value, the goal is unassisted. An extra "no assist"
outcome has weight max(0, team xG - sum of those xA weights), so assists are
not invented to fill every goal.

Points start from the historical total. Goals, assists, clean sheets and
goals conceded are removed and replaced with the standard table. Cards,
appearance points, saves, bonus and defensive contributions stay as they
happened. Bonus in particular still belongs to the real scoreline. Minutes
stay historical, including the 60-minute clean-sheet gate. Goals conceded are
the simulated opponent total for anyone who played, which ignores the minute
they came on.

Two episodes that share one ScenarioSeason see the same draws. A new
ScenarioSeason with the same seed and the same fixtures replays those draws,
because each gameweek is sampled once, fixtures in id order, and earlier
weeks are sampled before a later one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

GOAL_POINTS = {1: 6, 2: 6, 3: 5, 4: 4}
CLEAN_SHEET_POINTS = {1: 4, 2: 4, 3: 1, 4: 0}
ASSIST_POINTS = 3
FALLBACK_XG = 0.01


class MissingMatchXG(LookupError):
    """The fixture has no expected-goals values to centre a score on."""


@dataclass(frozen=True)
class PlayerFixture:
    player_id: int
    team_id: int
    position: int
    minutes: float
    total_points: float
    goals: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    expected_goals: Optional[float] = None
    expected_assists: Optional[float] = None


@dataclass(frozen=True)
class FixtureSpec:
    fixture_id: int
    gameweek: int
    home_team_id: int
    away_team_id: int
    players: tuple


def component_points(position, minutes, goals, assists, clean_sheets, goals_conceded) -> float:
    """Goal, assist, clean sheet and goals-conceded points. Nothing else."""
    position = int(position)
    minutes = float(minutes)
    points = GOAL_POINTS[position] * int(goals) + ASSIST_POINTS * int(assists)
    if minutes >= 60 and int(clean_sheets):
        points += CLEAN_SHEET_POINTS[position]
    if minutes > 0 and position in (1, 2):
        points -= int(goals_conceded) // 2
    return float(points)


def adjust_points(player: PlayerFixture, goals: int, assists: int, opponent_goals: int) -> float:
    """Historical total, with the score-dependent parts swapped for the simulation."""
    historical = component_points(
        player.position,
        player.minutes,
        player.goals,
        player.assists,
        player.clean_sheets,
        player.goals_conceded,
    )
    clean = 1 if int(opponent_goals) == 0 and float(player.minutes) >= 60 else 0
    conceded = int(opponent_goals) if float(player.minutes) > 0 else 0
    simulated = component_points(
        player.position,
        player.minutes,
        goals,
        assists,
        clean,
        conceded,
    )
    return float(player.total_points) - historical + simulated


class ScenarioSeason:
    def __init__(self, fixtures: Sequence[FixtureSpec], seed: int):
        self.fixtures = tuple(sorted(fixtures, key=lambda fixture: (fixture.gameweek, fixture.fixture_id)))
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.scores = {}
        self._points = {}
        self._sampled = set()

    def outcomes(self, player_id, gameweek) -> float:
        self._ensure(int(gameweek))
        return float(self._points.get(int(gameweek), {}).get(int(player_id), 0.0))

    def _ensure(self, gameweek: int):
        pending = sorted({
            fixture.gameweek
            for fixture in self.fixtures
            if fixture.gameweek <= int(gameweek) and fixture.gameweek not in self._sampled
        })
        for week in pending:
            self._sample_week(week)

    def _sample_week(self, gameweek: int):
        week = [fixture for fixture in self.fixtures if fixture.gameweek == int(gameweek)]
        for fixture in week:
            _require_xg(fixture)
        points = {}
        for fixture in week:
            scored, score = sample_fixture(fixture, self.rng)
            self.scores[fixture.fixture_id] = score
            for player_id, value in scored.items():
                points[player_id] = points.get(player_id, 0.0) + value
        self._points[int(gameweek)] = points
        self._sampled.add(int(gameweek))


def sample_fixture(fixture: FixtureSpec, rng) -> tuple:
    home = [player for player in fixture.players if player.team_id == fixture.home_team_id]
    away = [player for player in fixture.players if player.team_id == fixture.away_team_id]
    home_xg = _team_xg(home, fixture.fixture_id)
    away_xg = _team_xg(away, fixture.fixture_id)
    home_goals = int(rng.poisson(home_xg))
    away_goals = int(rng.poisson(away_xg))
    home_scorers = [_draw_scorer(rng, home) for _goal in range(home_goals)]
    away_scorers = [_draw_scorer(rng, away) for _goal in range(away_goals)]
    home_assists = [
        _draw_assist(rng, home, scorer, home_xg) for scorer in home_scorers
    ]
    away_assists = [
        _draw_assist(rng, away, scorer, away_xg) for scorer in away_scorers
    ]
    home_points = _player_points(home, home_scorers, home_assists, away_goals)
    away_points = _player_points(away, away_scorers, away_assists, home_goals)
    home_points.update(away_points)
    return home_points, (home_goals, away_goals)


def _require_xg(fixture: FixtureSpec):
    home = [player for player in fixture.players if player.team_id == fixture.home_team_id]
    away = [player for player in fixture.players if player.team_id == fixture.away_team_id]
    _team_xg(home, fixture.fixture_id)
    _team_xg(away, fixture.fixture_id)


def _team_xg(players, fixture_id) -> float:
    if not players or all(player.expected_goals is None for player in players):
        raise MissingMatchXG(
            'Fixture %s has no expected goals to sample from' % fixture_id
        )
    return float(sum(
        0.0 if player.expected_goals is None else max(0.0, float(player.expected_goals))
        for player in players
    ))


def _draw_scorer(rng, players) -> int:
    weights = []
    for player in players:
        if float(player.minutes) <= 0:
            weights.append(0.0)
        elif player.expected_goals is None:
            weights.append(FALLBACK_XG)
        else:
            weights.append(max(0.0, float(player.expected_goals)))
    if sum(weights) <= 0:
        weights = [1.0 if float(player.minutes) > 0 else 0.0 for player in players]
    if sum(weights) <= 0:
        raise MissingMatchXG('No player with minutes is available to score')
    index = int(rng.choice(len(players), p=_normalise(weights)))
    return int(players[index].player_id)


def _draw_assist(rng, players, scorer_id: int, team_xg: float):
    candidates = [
        player for player in players
        if int(player.player_id) != int(scorer_id) and float(player.minutes) > 0
    ]
    if not candidates:
        return None
    known = [player.expected_assists is not None for player in candidates]
    if not any(known):
        return None
    weights = [
        0.0 if player.expected_assists is None else max(0.0, float(player.expected_assists))
        for player in candidates
    ]
    weights.append(max(0.0, float(team_xg) - sum(weights)))
    if sum(weights) <= 0:
        return None
    index = int(rng.choice(len(weights), p=_normalise(weights)))
    if index == len(candidates):
        return None
    return int(candidates[index].player_id)


def _player_points(players, scorers, assists, opponent_goals) -> dict:
    goal_counts = {}
    assist_counts = {}
    for player_id in scorers:
        goal_counts[player_id] = goal_counts.get(player_id, 0) + 1
    for player_id in assists:
        if player_id is None:
            continue
        assist_counts[player_id] = assist_counts.get(player_id, 0) + 1
    points = {}
    for player in players:
        points[int(player.player_id)] = adjust_points(
            player,
            goal_counts.get(player.player_id, 0),
            assist_counts.get(player.player_id, 0),
            opponent_goals,
        )
    return points


def _normalise(weights):
    total = float(sum(weights))
    return [float(weight) / total for weight in weights]
