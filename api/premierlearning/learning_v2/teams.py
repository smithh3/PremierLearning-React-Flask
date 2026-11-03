"""Team match histories across seasons, and time blocks for identity embeddings.

Each club (stable FPL team code) gets one timeline of its finished league
matches from every season loaded: goals for and against, real xG for and
against (summed from its players' rows, where the season has xG), home or away,
and the opponent. A row sees only the matches that kicked off before it.

Every player in the same team and fixture shares that history, so it is stored
once in a table and each row keeps an index into it (see TeamTables).

Blocks split an identity embedding by time: block 0 is the last
BLOCK_FIRST_GAMEWEEKS gameweeks before the training cutoff, then
BLOCK_GAMEWEEKS-gameweek blocks back from there.
"""
from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy

from premierlearning.helper_functions import coerce_bool
from premierlearning.learning_v2.kickoff import days_between, kickoff_from_match, parse_kickoff

TEAM_HISTORY_PATHS = ('current_team.fixture_history', 'current_opposition.fixture_history')
TEAM_HISTORY_TABLE = 'team_history'
# Leaves a team-history token can carry.
TEAM_HISTORY_FIELDS = (
    'opposition_team_id_embedding',
    'home_away',
    'days_before_current',
    'goals_for',
    'goals_against',
    'xg_for',
    'xg_against',
    'xg_known',
)
# Fading averages over the same history, as plain team_features inputs.
TEAM_DECAY_FIELDS = (
    'goals_for_decay',
    'goals_against_decay',
    'xg_for_decay',
    'xg_against_decay',
    'xg_weight',
)
# A side with no league history (promoted) reads as a weak team, not an average one.
PROMOTED_PRIOR = {'goals_for': 1.05, 'goals_against': 1.75, 'xg_for': 1.10, 'xg_against': 1.75}
# The prior counts as this many matches' worth of weight in the fading average.
PRIOR_WEIGHT = 2.0
GAMEWEEKS_PER_SEASON = 38


def ref_key(side_path: str) -> str:
    """Row array holding each row's index into the team-history table."""
    return '__ref__' + side_path.replace('.', '__')


def is_team_history_path(path: str) -> bool:
    return any(path.startswith(prefix + '.') for prefix in TEAM_HISTORY_PATHS)


def team_history_side(path: str):
    """('current_team.fixture_history', field) for a team-history leaf or mask."""
    for prefix in TEAM_HISTORY_PATHS:
        if path.startswith(prefix + '.'):
            return prefix, path[len(prefix) + 1:].split('.')[-1]
    raise KeyError(path)


def gameweek_index(year, gameweek) -> int:
    """Gameweeks since a fixed origin: one season is 38."""
    return int(year) * GAMEWEEKS_PER_SEASON + int(gameweek or 0)


def block_reference(year, cutoff_gameweek) -> int:
    """The first gameweek the fit predicts. A match there or later is age 0."""
    return gameweek_index(year, int(cutoff_gameweek or 0) + 1)


def block_of(age: int, blocks: int, first: int, size: int) -> int:
    """0 for the newest `first` gameweeks (and the future), then `size`-gameweek blocks."""
    if blocks <= 1 or age <= first:
        return 0
    return min(blocks - 1, 1 + int((age - first - 1) // max(1, size)))


def encode_block(index: int, block: int, blocks: int) -> int:
    """One int input carries the id and its block: id * blocks + block."""
    if blocks <= 1:
        return int(index)
    return int(index) * int(blocks) + int(block)


def _synthetic_kickoff(year, gameweek):
    """A stand-in time for a match with no kickoff: weekly from 1 August."""
    return datetime(2000 + int(year), 8, 1) + timedelta(days=7 * max(0, int(gameweek or 1) - 1))


def time_key(kickoff, year, gameweek) -> datetime:
    return kickoff if kickoff is not None else _synthetic_kickoff(year, gameweek)


@dataclass
class TeamRecord:
    """One finished league match from one club's side."""
    time: datetime
    year: int
    gameweek: int
    fixture_id: int
    code: int
    opponent: int = 0
    home: float = 0.0
    goals_for: float = 0.0
    goals_against: float = 0.0
    xg_for: float = 0.0
    xg_against: float = 0.0
    xg_known: float = 0.0
    kickoff_known: bool = False


def _code(team):
    try:
        code = int(getattr(team, 'code', None))
    except (TypeError, ValueError):
        return None
    return code if code > 0 else None


def _season_year(season) -> int:
    return int(season.years[0])


def _expected_goals(match) -> float:
    stats = getattr(match, 'game_stats', None) or {}
    try:
        return float(stats.get('expected_goals') or 0)
    except (TypeError, ValueError):
        return 0.0


class TeamTimelines:
    """Every club's finished league matches, oldest first, across the seasons given."""

    def __init__(self, seasons):
        records = {}
        for season in seasons:
            year = _season_year(season)
            teams = getattr(season, 'teams', None) or {}
            ids = {}
            if hasattr(teams, 'items'):
                for team_id, team in teams.items():
                    code = _code(team)
                    if code is not None:
                        ids[int(getattr(team, 'id', team_id))] = code
            for team in (teams.values() if hasattr(teams, 'values') else []):
                code = _code(team)
                if code is None:
                    continue
                for team_match in (getattr(team, 'team_match_dict', None) or {}).values():
                    self._from_team_match(records, year, code, team_match, ids)
            for player in getattr(season, 'players', []) or []:
                for match in getattr(player, 'matches', []) or []:
                    if getattr(match, 'is_last_season_carry', False):
                        continue
                    self._from_player_match(records, year, match)
        by_code = defaultdict(list)
        for (year, fixture_id, code), record in records.items():
            other = records.get((year, fixture_id, record.opponent))
            if other is not None:
                record.xg_against = other.xg_for
                record.xg_known = 1.0 if (record.xg_known and other.xg_known) else 0.0
            else:
                record.xg_known = 0.0
            by_code[code].append(record)
        self.records = {}
        self.times = {}
        for code, items in by_code.items():
            items.sort(key=lambda record: (record.time, record.fixture_id))
            self.records[code] = items
            self.times[code] = [record.time for record in items]
        self._decay = {}

    @staticmethod
    def _from_team_match(records, year, code, team_match, ids):
        fixture_id = getattr(team_match, 'id', None)
        if fixture_id is None:
            return
        key = (year, int(fixture_id), code)
        fixture = getattr(team_match, 'fixture_json', None) or {}
        home = getattr(team_match, 'home', None)
        opponent = 0
        if isinstance(fixture, dict) and fixture.get('team_h') is not None and home is not None:
            other = fixture.get('team_a') if coerce_bool(home) else fixture.get('team_h')
            try:
                opponent = ids.get(int(other), 0)
            except (TypeError, ValueError):
                opponent = 0
        kickoff = parse_kickoff(fixture.get('kickoff_time')) if isinstance(fixture, dict) else None
        gameweek = int(getattr(team_match, 'event', 0) or 0)
        records[key] = TeamRecord(
            time=time_key(kickoff, year, gameweek),
            year=year,
            gameweek=gameweek,
            fixture_id=int(fixture_id),
            code=code,
            opponent=int(opponent or 0),
            home=1.0 if coerce_bool(home) else 0.0,
            goals_for=float(getattr(team_match, 'scored', 0) or 0),
            goals_against=float(getattr(team_match, 'conceded', 0) or 0),
            kickoff_known=kickoff is not None,
        )

    @staticmethod
    def _from_player_match(records, year, match):
        fixture_id = getattr(match, 'fixture_id', None)
        code = _code(getattr(match, 'team', None))
        if fixture_id is None or code is None:
            return
        key = (year, int(fixture_id), code)
        record = records.get(key)
        if record is None:
            team_match = getattr(match, 'team_match', None)
            if team_match is None:
                return
            gameweek = int(getattr(match, 'gameweek', 0) or getattr(team_match, 'event', 0) or 0)
            kickoff = kickoff_from_match(match)
            record = TeamRecord(
                time=time_key(kickoff, year, gameweek),
                year=year,
                gameweek=gameweek,
                fixture_id=int(fixture_id),
                code=code,
                home=1.0 if coerce_bool(getattr(match, 'home', False)) else 0.0,
                goals_for=float(getattr(team_match, 'scored', 0) or 0),
                goals_against=float(getattr(team_match, 'conceded', 0) or 0),
                kickoff_known=kickoff is not None,
            )
            records[key] = record
        if not record.kickoff_known:
            kickoff = kickoff_from_match(match)
            if kickoff is not None:
                record.time = kickoff
                record.kickoff_known = True
        if not record.opponent:
            record.opponent = _code(getattr(match, 'opponent_team', None)) or 0
            record.home = 1.0 if coerce_bool(getattr(match, 'home', False)) else 0.0
        if not record.gameweek:
            record.gameweek = int(getattr(match, 'gameweek', 0) or 0)
        record.xg_for += _expected_goals(match)
        if getattr(match, 'new_stats_included', False):
            record.xg_known = 1.0

    def cut(self, code, time) -> int:
        """How many of the club's matches kicked off strictly before `time`."""
        times = self.times.get(code)
        if not times or time is None:
            return 0
        return bisect.bisect_left(times, time)

    def before(self, code, cut, length):
        records = self.records.get(code, ())
        return records[max(0, cut - length):cut]

    def decay(self, code, cut, time, length, half_life_days):
        """Fading averages of the last `length` matches, shrunk toward the promoted prior."""
        key = (code, cut, time, length, half_life_days)
        cached = self._decay.get(key)
        if cached is not None:
            return cached
        totals = defaultdict(float)
        weight = 0.0
        xg_weight = 0.0
        lam = math.log(2.0) / max(1.0, float(half_life_days))
        for record in self.before(code, cut, length):
            w = math.exp(-lam * days_between(time, record.time))
            weight += w
            totals['goals_for'] += w * record.goals_for
            totals['goals_against'] += w * record.goals_against
            if record.xg_known:
                xg_weight += w
                totals['xg_for'] += w * record.xg_for
                totals['xg_against'] += w * record.xg_against
        prior = PROMOTED_PRIOR
        out = {
            'goals_for_decay': (totals['goals_for'] + PRIOR_WEIGHT * prior['goals_for']) / (weight + PRIOR_WEIGHT),
            'goals_against_decay': (
                (totals['goals_against'] + PRIOR_WEIGHT * prior['goals_against']) / (weight + PRIOR_WEIGHT)
            ),
            'xg_weight': xg_weight / weight if weight > 0 else 0.0,
        }
        if xg_weight > 0:
            out['xg_for_decay'] = (totals['xg_for'] + PRIOR_WEIGHT * prior['xg_for']) / (xg_weight + PRIOR_WEIGHT)
            out['xg_against_decay'] = (
                (totals['xg_against'] + PRIOR_WEIGHT * prior['xg_against']) / (xg_weight + PRIOR_WEIGHT)
            )
        else:
            # No xG in these seasons: goals stand in, and xg_weight 0 says so.
            out['xg_for_decay'] = out['goals_for_decay']
            out['xg_against_decay'] = out['goals_against_decay']
        self._decay[key] = out
        return out


class TeamTables:
    """One stored history per (club, cut, kickoff), shared by every row that needs it.

    Entry 0 is an empty history, for rows whose club is unknown.
    """

    def __init__(self, timelines: TeamTimelines, length: int, team_id):
        self.timelines = timelines
        self.length = int(length)
        self.team_id = team_id          # (opponent code, record) -> embedding index
        self._index = {}
        self._entries = [None]

    def ref(self, code, time) -> int:
        if code is None or time is None:
            return 0
        cut = self.timelines.cut(code, time)
        key = (code, cut, time)
        found = self._index.get(key)
        if found is not None:
            return found
        self._index[key] = len(self._entries)
        self._entries.append((code, cut, time))
        return self._index[key]

    def finalize(self, fields) -> dict:
        """{'mask': [K, L], field: [K, L]} for the fields a plan uses."""
        count = len(self._entries)
        arrays = {'mask': numpy.zeros((count, self.length), dtype=numpy.float32)}
        for field in fields:
            dtype = numpy.int32 if field == 'opposition_team_id_embedding' else numpy.float32
            arrays[field] = numpy.zeros((count, self.length), dtype=dtype)
        for index, entry in enumerate(self._entries):
            if entry is None:
                continue
            code, cut, time = entry
            records = self.timelines.before(code, cut, self.length)
            if not records:
                continue
            offset = self.length - len(records)
            arrays['mask'][index, offset:] = 1.0
            for step, record in enumerate(records):
                slot = offset + step
                for field in fields:
                    arrays[field][index, slot] = self._value(field, record, time)
        return arrays

    def _value(self, field, record, time):
        if field == 'opposition_team_id_embedding':
            return self.team_id(record.opponent, record)
        if field == 'home_away':
            return record.home
        if field == 'days_before_current':
            return days_between(time, record.time)
        return float(getattr(record, field))

    def __len__(self):
        return len(self._entries)
