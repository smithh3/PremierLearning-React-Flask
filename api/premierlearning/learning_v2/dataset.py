"""Build simple_h1 tensors from season objects.

Each training row is one finished fixture, including did-not-plays. History is
the last history_len played matches for that player code, from any season
loaded, strictly before that kickoff. The seven-game carry copy is left out so
those fixtures are not stored twice. Unplayed fixtures stay out of that
history, and a fixture's own outcome never appears in its features.
Season history is a separate padded list, one token per earlier season.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import bisect

import numpy

from premierlearning.helper_functions import coerce_bool
from premierlearning.learning_v2.external_hook import (
    can_fill_external,
    external_key,
    external_value,
    load_external,
    plan_uses_external,
)
from premierlearning.learning_v2.kickoff import days_between, kickoff_from_match
from premierlearning.learning_v2.plan import input_specs, tensor_key
from premierlearning.learning_v2.teams import (
    TEAM_DECAY_FIELDS,
    TEAM_HISTORY_FIELDS,
    TEAM_HISTORY_PATHS,
    TEAM_HISTORY_TABLE,
    TeamTables,
    TeamTimelines,
    block_of,
    block_reference,
    encode_block,
    gameweek_index,
    is_team_history_path,
    ref_key,
    team_history_side,
    time_key,
)

PLAYER_LAST_N = {
    'minutes_last_n': 'minutes',
    'points_last_n': 'total_points',
    'xG_last_n': 'expected_goals',
    'xA_last_n': 'expected_assists',
    'goals_last_n': 'goals_scored',
    'assists_last_n': 'assists',
    'xGC_last_n': 'expected_goals_conceded',
    'defcon_last_n': 'defensive_contribution',
}
SQUAD_FIELDS = {
    'player_id_embedding',
    'position',
    'xG_last_n',
    'xA_last_n',
    'goals_last_n',
    'assists_last_n',
    'defcon_last_n',
    'xGC_last_n',
    'minutes_last_n',
}
TEAM_LAST_N = {
    'xG_last_n': 'xG',
    'xGC_last_n': 'xGC',
    'goals_last_n': 'goals',
    'goals_conceded_last_n': 'goals_conceded',
}
PREMATCH_FIELDS = ('xG', 'xGC', 'goals', 'goals_conceded')
PERFORMANCE = {
    'points_scored': 'total_points',
    'minutes_played': 'minutes',
    'xG': 'expected_goals',
    'xA': 'expected_assists',
    'xGC': 'expected_goals_conceded',
    'goals': 'goals_scored',
    'assists': 'assists',
    'defcon': 'defensive_contribution',
}
HISTORY_FIELDS = set(PERFORMANCE) | set(PREMATCH_FIELDS) | {
    'team_id_embedding',
    'opposition_team_id_embedding',
    'home_away',
    'days_since_previous_match',
    'days_before_current',
}
ZERO_PRIOR = {'xG': 0.0, 'xGC': 0.0, 'goals': 0.0, 'goals_conceded': 0.0}
_ZERO_FORM = {name: 0.0 for name in PLAYER_LAST_N}
_KICKOFFS = {}
_SORT_KEYS = {}


@dataclass
class ExampleBatch:
    """Row arrays, labels and the matches they came from.

    tables holds inputs shared by many rows, stored once ({table: {field:
    array}}); table_inputs maps a model input key to (table, field, row array
    of indexes into that table). Plans without team histories have neither.
    """
    arrays: dict
    labels: numpy.ndarray
    matches: list
    owners: list = field(default_factory=list)
    years: list = field(default_factory=list)
    tables: dict = field(default_factory=dict)
    table_inputs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _SeasonSummary:
    year: int
    minutes: float
    points: float
    xg: float
    xa: float


@dataclass
class _Row:
    player: object
    match: object
    prior: tuple
    teammates: list
    opponents: list
    is_future: bool
    year: int
    timeline: object


class Vocab:
    """Stable codes mapped to embedding indexes. 0 is pad / unknown."""

    def __init__(self):
        self._player = {}
        self._team = {}
        self.frozen = False
        # Gameweek index the time blocks count back from (teams.block_reference).
        # Set by the fit and kept when frozen, so later snapshots use the same blocks.
        self.block_reference = None

    def freeze(self):
        """Stop growing, so fitted embeddings keep their rows. New codes read as unknown."""
        self.frozen = True
        return self

    def add_player(self, code):
        if not self.frozen:
            self._index(self._player, code)

    def add_team(self, code):
        if not self.frozen:
            self._index(self._team, code)

    def player_index(self, code) -> int:
        return self._lookup(self._player, code)

    def team_index(self, code) -> int:
        return self._lookup(self._team, code)

    @property
    def player_size(self) -> int:
        return len(self._player) + 1

    @property
    def team_size(self) -> int:
        return len(self._team) + 1

    @property
    def position_size(self) -> int:
        return 5

    @staticmethod
    def _index(table, code):
        try:
            code = int(code)
        except (TypeError, ValueError):
            return
        if code <= 0 or code in table:
            return
        table[code] = len(table) + 1

    @staticmethod
    def _lookup(table, code) -> int:
        try:
            return int(table.get(int(code), 0))
        except (TypeError, ValueError):
            return 0


def can_fill(path: str) -> bool:
    """True when the simple dataset knows how to produce this leaf."""
    if external_key(path) is not None:
        return can_fill_external(path)
    if path.startswith('current_player_features.'):
        return path.split('.')[-1] in PLAYER_LAST_N
    if path.startswith('current_team.team_features.') or path.startswith('current_opposition.team_features.'):
        name = path.split('.')[-1]
        return name in TEAM_LAST_N or name in TEAM_DECAY_FIELDS
    if is_team_history_path(path):
        _side, name = team_history_side(path)
        return name in TEAM_HISTORY_FIELDS
    if path.startswith('current_team.teammates.'):
        return path.split('.')[-1] in SQUAD_FIELDS
    if path.startswith('current_opposition.players.'):
        return path.split('.')[-1] in SQUAD_FIELDS
    if path.startswith('player_fixture_history.'):
        return _history_field(path) in HISTORY_FIELDS
    return path in {
        'player_id_embedding',
        'position',
        'current_team.team_id_embedding',
        'current_opposition.opposition_team_id_embedding',
        'match_facts.home_away',
        'match_facts.days_since_previous_match',
        'match_facts.team_xg',
        'match_facts.team_xgc',
        'context.new_stats_included',
        'context.changed_club',
        'context.chance_of_playing',
        'context.chance_known',
        ORACLE_MINUTES_PATH,
        'last_season.seasons_before',
        'last_season.played',
        'last_season.points_per_90',
        'last_season.xgi_per_90',
        'last_season.minutes',
    }


def unsupported_leaves(plan) -> list:
    return [leaf.path for leaf in plan.leaves() if not can_fill(leaf.path)]


def _history_field(path: str) -> str:
    rest = path[len('player_fixture_history.'):]
    if rest.startswith('team_features_pre_match.') or rest.startswith('opposition_features_pre_match.'):
        return rest.split('.')[-1]
    if rest.startswith('player_performance.') or rest.startswith('match_facts.'):
        return rest.split('.')[-1]
    return rest


def game_stat(match, key: str) -> float:
    stats = getattr(match, 'game_stats', None) or {}
    if key == 'minutes' and (key not in stats or stats.get(key) is None):
        return float(getattr(match, 'minutes', 0) or 0)
    if key == 'total_points' and (key not in stats or stats.get(key) is None):
        return float(getattr(match, 'points', 0) or 0)
    try:
        return float(stats.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


# Cheats on purpose: the fixture's real minutes, to measure what perfect minutes
# knowledge would be worth. Only the *_oracle_* plans have this leaf.
ORACLE_MINUTES_PATH = 'context.actual_minutes_oracle'

# What the training-only minutes head predicts.
MINUTES_TARGET_BANDS = 'bands'
MINUTES_TARGET_LOG = 'log'
MINUTES_TARGETS = (MINUTES_TARGET_BANDS, MINUTES_TARGET_LOG)
# 0, 1-59 and 60+ minutes: the bands FPL's appearance points use.
MINUTES_BAND_EDGES = (1.0, 60.0)
FULL_MATCH_MINUTES = 90.0


def minutes_targets(matches, target=MINUTES_TARGET_BANDS) -> numpy.ndarray:
    """One row per match: its minutes band, or log1p minutes scaled so 90 is 1."""
    minutes = numpy.array([game_stat(match, 'minutes') for match in matches], dtype=numpy.float32)
    if target == MINUTES_TARGET_LOG:
        scaled = numpy.log1p(numpy.minimum(minutes, FULL_MATCH_MINUTES)) / numpy.log1p(FULL_MATCH_MINUTES)
        return scaled.astype(numpy.float32).reshape(-1, 1)
    bands = numpy.searchsorted(MINUTES_BAND_EDGES, minutes, side='right')
    return numpy.asarray(bands, dtype=numpy.int32).reshape(-1, 1)


def score_targets(matches, weighted=False) -> numpy.ndarray:
    """One row per match: [team goals for, team goals against], or -1s when unknown.

    The score head's loss skips negative targets. weighted adds a third column
    so each team's fixture counts once however many player rows it has: the
    rows of one team fixture share 1 / their count, scaled so known rows
    average 1.
    """
    out = numpy.full((len(matches), 2), -1.0, dtype=numpy.float32)
    out = _fill_scores(matches, out)
    if not weighted:
        return out
    weights = numpy.zeros((len(matches), 1), dtype=numpy.float32)
    groups = defaultdict(list)
    for index, match in enumerate(matches):
        if out[index, 0] >= 0:
            groups[id(getattr(match, 'team_match', None))].append(index)
    known = sum(len(rows) for rows in groups.values())
    if groups:
        scale = known / float(len(groups))
        for rows in groups.values():
            weights[rows, 0] = scale / len(rows)
    return numpy.concatenate([out, weights], axis=1)


def _fill_scores(matches, out):
    for index, match in enumerate(matches):
        team_match = getattr(match, 'team_match', None)
        if team_match is None:
            continue
        scored = getattr(team_match, 'scored', None)
        conceded = getattr(team_match, 'conceded', None)
        if scored is None or conceded is None:
            continue
        try:
            out[index] = (float(scored), float(conceded))
        except (TypeError, ValueError):
            continue
    return out


def _team_code(match):
    team = getattr(match, 'team', None)
    code = getattr(team, 'code', None)
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _opp_code(match):
    team = getattr(match, 'opponent_team', None)
    code = getattr(team, 'code', None)
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _season_year(season) -> int:
    return int(season.years[0])


def _clear_match_caches():
    _KICKOFFS.clear()
    _SORT_KEYS.clear()


def _cached_kickoff(match):
    if match is None:
        return None
    key = id(match)
    try:
        return _KICKOFFS[key]
    except KeyError:
        parsed = kickoff_from_match(match)
        _KICKOFFS[key] = parsed
        return parsed


def match_sort_key(match):
    key = id(match)
    cached = _SORT_KEYS.get(key)
    if cached is not None:
        return cached
    kick = _cached_kickoff(match)
    fixture_id = int(getattr(match, 'fixture_id', 0) or 0)
    if kick is not None:
        value = (0, kick.toordinal() * 1440 + kick.hour * 60 + kick.minute, fixture_id)
    else:
        carry = 0 if getattr(match, 'is_last_season_carry', False) else 1
        gameweek = int(getattr(match, 'gameweek', 0) or 0)
        value = (1, carry * 1000 + gameweek, fixture_id)
    _SORT_KEYS[key] = value
    return value


def is_before(match, target) -> bool:
    """True when match is strictly earlier than the row being scored."""
    if match is target:
        return False
    match_kick = _cached_kickoff(match)
    target_kick = _cached_kickoff(target)
    if match_kick is not None and target_kick is not None:
        return match_kick < target_kick
    same_fixture = int(getattr(match, 'fixture_id', -1) or -1) == int(
        getattr(target, 'fixture_id', -2) or -2
    )
    if same_fixture:
        return False
    return match_sort_key(match) < match_sort_key(target)


class _Timeline:
    """One player's matches, oldest first, with prefix totals reused across rows."""

    def __init__(self, matches):
        self.matches = tuple(matches)
        self.kicks = tuple(_cached_kickoff(match) for match in self.matches)
        self.all_kicks = all(kick is not None for kick in self.kicks)
        self._cut = {}
        self._form = {}
        self.values = {
            stat: [game_stat(match, stat) for match in self.matches]
            for stat in dict.fromkeys(PLAYER_LAST_N.values())
        }
        total = 0.0
        totals = [0.0]
        for minutes in self.values['minutes']:
            total += minutes
            totals.append(total)
        self.cum_minutes = tuple(totals)
        self.columns = {}

    def cutoff(self, target) -> int:
        """Number of matches strictly before target. Equal kickoffs are excluded."""
        kick = _cached_kickoff(target)
        if kick is not None and self.all_kicks:
            cached = self._cut.get(kick)
            if cached is None:
                cached = bisect.bisect_left(self.kicks, kick)
                self._cut[kick] = cached
            return cached
        cached = self._cut.get(id(target))
        if cached is None:
            cached = 0
            while cached < len(self.matches) and is_before(self.matches[cached], target):
                cached += 1
            self._cut[id(target)] = cached
        return cached

    def prior(self, target) -> tuple:
        return self.matches[:self.cutoff(target)]

    def minutes_before(self, target) -> float:
        return self.cum_minutes[self.cutoff(target)]

    def form(self, target, n: int) -> dict:
        end = self.cutoff(target)
        cached = self._form.get((end, n))
        if cached is not None:
            return cached
        start = end - n if end > n else 0
        span = end - start
        if span <= 0:
            means = _ZERO_FORM
        else:
            scale = 1.0 / span
            means = {
                name: sum(self.values[stat][start:end]) * scale
                for name, stat in PLAYER_LAST_N.items()
            }
        self._form[(end, n)] = means
        return means


_EMPTY_TIMELINE = _Timeline(())


def _unique(items):
    seen = set()
    out = []
    for item in items:
        key = id(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _teams_in(season):
    found = []
    teams = getattr(season, 'teams', None) or {}
    if hasattr(teams, 'values'):
        found.extend(teams.values())
    for player in getattr(season, 'players', []) or []:
        for match in list(getattr(player, 'matches', []) or []) + list(
            getattr(player, 'future_matches', []) or []
        ):
            for side in (getattr(match, 'team', None), getattr(match, 'opponent_team', None)):
                if side is not None:
                    found.append(side)
    return _unique(found)


def _mean_window(values) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def team_priors(seasons, window: int) -> dict:
    """Pre-match team form keyed by TeamMatch identity. The current match is excluded."""
    priors = {}
    for season in seasons:
        for team in _teams_in(season):
            finished = sorted(
                getattr(team, 'team_match_dict', {}).values(),
                key=lambda match: (getattr(match, 'event', None) or 0, getattr(match, 'id', 0)),
            )
            previous = {'xG': [], 'xGC': [], 'goals': [], 'goals_conceded': []}
            for match in finished:
                priors[id(match)] = {
                    'xG': _mean_window(previous['xG'][-window:]),
                    'xGC': _mean_window(previous['xGC'][-window:]),
                    'goals': _mean_window(previous['goals'][-window:]),
                    'goals_conceded': _mean_window(previous['goals_conceded'][-window:]),
                }
                previous['xG'].append(float(getattr(match, 'predicted_xg', 0) or 0))
                previous['xGC'].append(float(getattr(match, 'predicted_xc', 0) or 0))
                previous['goals'].append(float(getattr(match, 'scored', 0) or 0))
                previous['goals_conceded'].append(float(getattr(match, 'conceded', 0) or 0))
            final = {
                'xG': _mean_window(previous['xG'][-window:]),
                'xGC': _mean_window(previous['xGC'][-window:]),
                'goals': _mean_window(previous['goals'][-window:]),
                'goals_conceded': _mean_window(previous['goals_conceded'][-window:]),
            }
            for match in getattr(team, 'future_team_match_dict', {}).values():
                priors[id(match)] = dict(final)
    return priors


def _player_code(player):
    try:
        code = int(getattr(player, 'code', None))
    except (TypeError, ValueError):
        return None
    return code if code > 0 else None


def _real_matches(player):
    """Played fixtures only. Carry rows are copies of the previous season."""
    return [
        match for match in (getattr(player, 'matches', []) or [])
        if not getattr(match, 'is_last_season_carry', False)
    ]


def _build_indexes(seasons, vocab: Vocab):
    chrono = {}
    squads = defaultdict(list)
    played_by_code = defaultdict(list)
    season_matches = defaultdict(lambda: defaultdict(list))
    players = []
    for season in seasons:
        year = _season_year(season)
        for team in _teams_in(season):
            vocab.add_team(getattr(team, 'code', None))
        for player in getattr(season, 'players', []) or []:
            players.append((player, year))
            vocab.add_player(getattr(player, 'code', None))
            code = _player_code(player)
            played = _real_matches(player)
            key = ('code', code) if code is not None else ('player', id(player))
            played_by_code[key].extend(played)
            if played:
                season_matches[key][year].extend(played)
            upcoming = list(getattr(player, 'future_matches', []) or [])
            # Unplayed fixtures are stored as 0 minutes and 0 points. Keeping them
            # on the timeline makes a later gameweek look like a recent blank.
            for match in played + upcoming:
                if getattr(match, 'is_last_season_carry', False):
                    continue
                fixture_id = getattr(match, 'fixture_id', None)
                code = _team_code(match)
                if fixture_id is None or code is None:
                    continue
                vocab.add_team(code)
                vocab.add_team(_opp_code(match))
                squads[(year, int(fixture_id), int(code))].append(player)
    timelines = {
        key: _Timeline(sorted(_unique(matches), key=match_sort_key))
        for key, matches in played_by_code.items()
    }
    for player, _year in players:
        code = _player_code(player)
        key = ('code', code) if code is not None else ('player', id(player))
        chrono[id(player)] = timelines[key]
    for key, squad in list(squads.items()):
        squads[key] = _unique(squad)
    seasons = {}
    for key, by_year in season_matches.items():
        summaries = []
        for season_year, matches in sorted(by_year.items()):
            minutes = points = xg = xa = 0.0
            for match in matches:
                minutes += game_stat(match, 'minutes')
                points += game_stat(match, 'total_points')
                xg += game_stat(match, 'expected_goals')
                xa += game_stat(match, 'expected_assists')
            summaries.append(_SeasonSummary(int(season_year), minutes, points, xg, xa))
        seasons[key] = tuple(summaries)
    return chrono, squads, seasons


def _ranked_squad(players, target_match, chrono) -> list:
    ranked = []
    for player in players:
        timeline = chrono.get(id(player), _EMPTY_TIMELINE)
        ranked.append((
            -timeline.minutes_before(target_match),
            int(getattr(player, 'code', 0) or 0),
            id(player),
            player,
        ))
    ranked.sort()
    return ranked


def _squad_from_ranked(ranked, target_player, limit: int) -> list:
    target_code = getattr(target_player, 'code', None)
    chosen = []
    for _, _, _, player in ranked:
        if player is target_player:
            continue
        if target_code is not None and getattr(player, 'code', None) == target_code:
            continue
        chosen.append(player)
        if len(chosen) == limit:
            break
    return chosen


def _label(match) -> float:
    return game_stat(match, 'total_points')


def _position(player) -> int:
    try:
        position = int(getattr(player, 'position', 0) or 0)
    except (TypeError, ValueError):
        return 0
    return position if position in (1, 2, 3, 4) else 0


def _home(match) -> float:
    return 1.0 if coerce_bool(getattr(match, 'home', False)) else 0.0


def _changed_club(player, match) -> float:
    feature = getattr(player, 'changed_club_feature', None)
    if feature is None:
        return 0.0
    apps = getattr(match, 'matches_played', 0)
    try:
        return float(feature(apps) or 0)
    except TypeError:
        return 0.0


# (season start year, gameweek) -> {player_id: chance} from the cached fplcache
# files, or {} when that deadline has none (seasons before 2021/22, live season).
_DEADLINE_CHANCES = {}


def _deadline_chances(year, gameweek) -> dict:
    key = (int(year), int(gameweek))
    if key not in _DEADLINE_CHANCES:
        from squadtools.chance_to_play import ensure_gameweek_chances
        try:
            _DEADLINE_CHANCES[key] = ensure_gameweek_chances(
                key[0], key[1], allow_download=False,
            )
        except (OSError, ValueError):
            _DEADLINE_CHANCES[key] = {}
    return _DEADLINE_CHANCES[key]


def _scale_chance(raw) -> float:
    if raw is None or raw == '':
        return 1.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if value > 1.0:
        value = value / 100.0
    return min(1.0, max(0.0, value))


# season start year -> {(player_id, fixture_id): minutes}
_ACTUAL_MINUTES = {}


def _actual_minutes(year) -> dict:
    key = int(year)
    if key not in _ACTUAL_MINUTES:
        from simulation.historical_data import VaastavSeasonData
        minutes = {}
        for player_id, rows in VaastavSeasonData(key).player_rows.items():
            for game in rows:
                if game.get('fixture') not in (None, ''):
                    fixture_id = int(float(game['fixture']))
                    minutes[(int(player_id), fixture_id)] = float(game.get('minutes') or 0)
        _ACTUAL_MINUTES[key] = minutes
    return _ACTUAL_MINUTES[key]


def _oracle_minutes(row: _Row) -> float:
    """The fixture's real minutes. Upcoming fixtures carry none, so read the results."""
    if not row.is_future:
        return game_stat(row.match, 'minutes')
    key = (int(row.player.id), int(row.match.fixture_id))
    return _actual_minutes(row.year).get(key, 0.0)


def _chance_and_known(row: _Row):
    """(chance, 1) from the FPL flag known at the deadline, or (1.0, 0) if unknown.

    Upcoming fixtures carry the flag on the match. A finished fixture uses the
    flag published before its own gameweek's deadline, so training sees the
    same signal prediction does. No flag for a listed player means no doubt.
    """
    if row.is_future:
        return _scale_chance(getattr(row.match, 'chance_of_playing', None)), 1.0
    gameweek = getattr(row.match, 'gameweek', None)
    player_id = getattr(row.player, 'id', None)
    if gameweek is None or player_id is None:
        return 1.0, 0.0
    chances = _deadline_chances(row.year, gameweek)
    if not chances:
        return 1.0, 0.0
    return _scale_chance(chances.get(int(player_id))), 1.0


def _chance(row: _Row) -> float:
    return _chance_and_known(row)[0]


def _prior_block(priors, team_match) -> dict:
    if team_match is None:
        return ZERO_PRIOR
    return priors.get(id(team_match), ZERO_PRIOR)


def _this_xg(match, attr) -> float:
    team_match = getattr(match, 'team_match', None)
    return float(getattr(team_match, attr, 0) or 0)


def _plan_uses_teams(plan) -> bool:
    """True when a leaf needs the cross-season team timelines."""
    for leaf in plan.leaves():
        if is_team_history_path(leaf.path):
            return True
        if '.team_features.' in leaf.path and leaf.path.split('.')[-1] in TEAM_DECAY_FIELDS:
            return True
    return False


def _last_played_gameweek(season) -> int:
    weeks = [
        int(getattr(match, 'gameweek', 0) or 0)
        for player in getattr(season, 'players', []) or []
        for match in getattr(player, 'matches', []) or []
        if not getattr(match, 'is_last_season_carry', False)
    ]
    return max(weeks, default=0)


def _block_reference(vocab, predict_season, cutoff_gameweek) -> int:
    """The gameweek blocks count back from: the fit's, once its vocab is frozen."""
    if vocab.frozen and getattr(vocab, 'block_reference', None) is not None:
        return int(vocab.block_reference)
    if predict_season is None:
        return 0
    if cutoff_gameweek is None:
        cutoff_gameweek = _last_played_gameweek(predict_season)
    reference = block_reference(_season_year(predict_season), cutoff_gameweek)
    if not vocab.frozen:
        vocab.block_reference = reference
    return reference


class _Catalog:
    def __init__(self, seasons, plan, vocab, block_ref=0):
        self.plan = plan
        self.vocab = vocab
        self.block_ref = int(block_ref or 0)
        self.chrono, self.squads, self.seasons = _build_indexes(seasons, vocab)
        self.match_years = {}
        for season in seasons:
            year = _season_year(season)
            for player in getattr(season, 'players', []) or []:
                for match in list(getattr(player, 'matches', []) or []) + list(
                    getattr(player, 'future_matches', []) or []
                ):
                    self.match_years.setdefault(id(match), year)
        self.priors = team_priors(seasons, plan.config.team_form_n)
        self.external = load_external() if plan_uses_external(plan) else None
        self.team_timelines = TeamTimelines(seasons) if _plan_uses_teams(plan) else None
        self.team_tables = None
        if self.team_timelines is not None:
            self.team_tables = TeamTables(
                self.team_timelines, plan.config.team_history_len, self._record_team_id,
            )
        self._ranked = {}
        self.static = {}
        self._history_ready = False
        for timeline in self.chrono.values():
            previous = None
            for match, kick in zip(timeline.matches, timeline.kicks):
                identity = id(match)
                if identity not in self.static:
                    self.static[identity] = _history_static(self, match, kick, previous)
                previous = kick

    def block(self, year, gameweek, blocks) -> int:
        """Time block of a match in (year, gameweek), counted back from the fit's cutoff."""
        if blocks <= 1 or not gameweek or year is None:
            return 0
        age = self.block_ref - gameweek_index(year, gameweek)
        return block_of(age, blocks, self.plan.config.block_first, self.plan.config.block_size)

    def match_block(self, match, blocks, year=None) -> int:
        if blocks <= 1:
            return 0
        if year is None:
            year = self.match_years.get(id(match))
        return self.block(year, getattr(match, 'gameweek', None), blocks)

    def team_id(self, code, block=0) -> int:
        blocks = self.plan.config.team_blocks
        return encode_block(self.vocab.team_index(code), block, blocks)

    def player_id(self, player, block=0) -> int:
        blocks = self.plan.config.player_blocks
        return encode_block(self.vocab.player_index(getattr(player, 'code', None)), block, blocks)

    def _record_team_id(self, code, record) -> int:
        """Embedding index of the opponent in one team-history token."""
        block = self.block(record.year, record.gameweek, self.plan.config.team_blocks)
        return self.team_id(code, block)

    def row_time(self, row):
        kick = _cached_kickoff(row.match)
        return time_key(kick, row.year, getattr(row.match, 'gameweek', None))

    def team_ref(self, code, row) -> int:
        if self.team_tables is None or code is None:
            return 0
        return self.team_tables.ref(code, self.row_time(row))

    def team_decay(self, code, row, field) -> float:
        timelines = self.team_timelines
        time = self.row_time(row)
        cut = timelines.cut(code, time) if code is not None else 0
        values = timelines.decay(
            code, cut, time, self.plan.config.team_history_len, self.plan.config.team_decay_half_life,
        )
        return float(values[field])

    def timeline(self, player) -> _Timeline:
        return self.chrono.get(id(player), _EMPTY_TIMELINE)

    def season_history(self, player) -> tuple:
        code = _player_code(player)
        key = ('code', code) if code is not None else ('player', id(player))
        return self.seasons.get(key, ())

    def ranked(self, players, target_match) -> list:
        if not players:
            return []
        key = (id(players), _cached_kickoff(target_match))
        cached = self._ranked.get(key)
        if cached is None:
            cached = _ranked_squad(players, target_match, self.chrono)
            self._ranked[key] = cached
        return cached

    def prepare_history(self, specs):
        if self._history_ready:
            return
        for timeline in self.chrono.values():
            count = len(timeline.matches)
            columns = {}
            for spec in specs:
                rest = spec.path[len('player_fixture_history.'):]
                if rest == 'match_facts.days_before_current':
                    continue
                columns[rest] = numpy.zeros(count, dtype=spec.dtype)
            for index, match in enumerate(timeline.matches):
                static = self.static[id(match)]
                for rest, column in columns.items():
                    column[index] = static[rest]
            timeline.columns = columns
        self._history_ready = True

    def row(self, player, match, year, is_future) -> _Row:
        timeline = self.timeline(player)
        prior = timeline.prior(match)
        fixture_id = getattr(match, 'fixture_id', None)
        limit = self.plan.config.max_squad
        teammates = []
        opponents = []
        if fixture_id is not None:
            team_code = _team_code(match)
            opp_code = _opp_code(match)
            if team_code is not None:
                teammates = _squad_from_ranked(
                    self.ranked(self.squads.get((year, int(fixture_id), int(team_code)), []), match),
                    player, limit,
                )
            if opp_code is not None:
                opponents = _squad_from_ranked(
                    self.ranked(self.squads.get((year, int(fixture_id), int(opp_code)), []), match),
                    player, limit,
                )
        return _Row(player, match, prior, teammates, opponents, is_future, year, timeline)


def _history_static(catalog: _Catalog, match, kick, previous_kick) -> dict:
    """Fields of one history token that do not depend on the row being scored."""
    team_block = _prior_block(catalog.priors, getattr(match, 'team_match', None))
    opp_block = _prior_block(catalog.priors, getattr(match, 'opponent_team_match', None))
    time_block = catalog.match_block(match, catalog.plan.config.team_blocks)
    static = {
        'team_id_embedding': catalog.team_id(_team_code(match), time_block),
        'opposition_team_id_embedding': catalog.team_id(_opp_code(match), time_block),
        'match_facts.home_away': _home(match),
        'match_facts.days_since_previous_match': days_between(kick, previous_kick),
    }
    for field in PREMATCH_FIELDS:
        static['team_features_pre_match.' + field] = float(team_block.get(field, 0.0))
        static['opposition_features_pre_match.' + field] = float(opp_block.get(field, 0.0))
    for field, stat in PERFORMANCE.items():
        static['player_performance.' + field] = game_stat(match, stat)
    return static


def _scalar(catalog: _Catalog, row: _Row, path: str):
    player = row.player
    match = row.match
    config = catalog.plan.config
    if path == 'player_id_embedding':
        return catalog.player_id(player, catalog.match_block(match, config.player_blocks, row.year))
    if path == 'position':
        return _position(player)
    if path == 'current_team.team_id_embedding':
        return catalog.team_id(_team_code(match), catalog.match_block(match, config.team_blocks, row.year))
    if path == 'current_opposition.opposition_team_id_embedding':
        return catalog.team_id(_opp_code(match), catalog.match_block(match, config.team_blocks, row.year))
    if path == 'match_facts.home_away':
        return _home(match)
    if path == 'match_facts.days_since_previous_match':
        previous = row.prior[-1] if row.prior else None
        return days_between(_cached_kickoff(match), _cached_kickoff(previous))
    if path == 'match_facts.team_xg':
        return _this_xg(match, 'predicted_xg')
    if path == 'match_facts.team_xgc':
        return _this_xg(match, 'predicted_xc')
    if path == 'context.new_stats_included':
        return 1.0 if getattr(match, 'new_stats_included', 0) else 0.0
    if path == 'context.changed_club':
        return _changed_club(player, match)
    if path == 'context.chance_of_playing':
        return _chance(row)
    if path == 'context.chance_known':
        return _chance_and_known(row)[1]
    if path == ORACLE_MINUTES_PATH:
        return _oracle_minutes(row)
    if path.startswith('current_player_features.'):
        field = path.split('.')[-1]
        return row.timeline.form(match, catalog.plan.config.team_form_n)[field]
    if path.startswith('current_team.team_features.'):
        name = path.split('.')[-1]
        if name in TEAM_DECAY_FIELDS:
            return catalog.team_decay(_team_code(match), row, name)
        block = _prior_block(catalog.priors, getattr(match, 'team_match', None))
        return float(block[TEAM_LAST_N[name]])
    if path.startswith('current_opposition.team_features.'):
        name = path.split('.')[-1]
        if name in TEAM_DECAY_FIELDS:
            return catalog.team_decay(_opp_code(match), row, name)
        block = _prior_block(catalog.priors, getattr(match, 'opponent_team_match', None))
        return float(block[TEAM_LAST_N[name]])
    if catalog.external is not None and external_key(path) is not None:
        fixture_id = getattr(match, 'fixture_id', None)
        return external_value(
            catalog.external, path, row.year, None if fixture_id is None else int(fixture_id),
            _team_code(match), _opp_code(match), _player_code(player),
        )
    raise KeyError(path)


def _empty(specs, count: int) -> dict:
    arrays = {}
    for spec in specs:
        arrays[spec.key] = numpy.zeros((count, *spec.shape), dtype=spec.dtype)
    return arrays


def _group_specs(specs):
    history = []
    seasons = []
    team_squad = []
    opp_squad = []
    scalars = []
    history_mask = None
    season_mask = None
    team_mask = None
    opp_mask = None
    prefix = 'player_fixture_history.'
    season_prefix = 'last_season.'
    for spec in specs:
        path = spec.path
        if path == 'player_fixture_history.mask':
            history_mask = spec
        elif path == 'last_season.mask':
            season_mask = spec
        elif path == 'current_team.teammates.mask':
            team_mask = spec
        elif path == 'current_opposition.players.mask':
            opp_mask = spec
        elif path.startswith(prefix):
            history.append((spec, path[len(prefix):]))
        elif path.startswith(season_prefix):
            seasons.append((spec, path[len(season_prefix):]))
        elif path.startswith('current_team.teammates.'):
            team_squad.append((spec, path.split('.')[-1]))
        elif path.startswith('current_opposition.players.'):
            opp_squad.append((spec, path.split('.')[-1]))
        elif spec.shape == (1,):
            scalars.append(spec)
        else:
            raise KeyError(path)
    return (
        history, history_mask, seasons, season_mask,
        team_squad, team_mask, opp_squad, opp_mask, scalars,
    )


def _write_history(arrays, index, row: _Row, history, history_mask):
    length = history_mask.shape[0] if history_mask is not None else 0
    if history and history[0][0].shape:
        length = history[0][0].shape[0]
    end = len(row.prior)
    count = min(end, length)
    if history_mask is not None and count:
        arrays[history_mask.key][index, -count:] = 1.0
    if not count:
        return
    offset = length - count
    source = end - count
    target_kick = _cached_kickoff(row.match)
    kicks = row.timeline.kicks
    columns = row.timeline.columns
    for spec, rest in history:
        dest = arrays[spec.key][index]
        if rest == 'match_facts.days_before_current':
            for step in range(count):
                dest[offset + step] = days_between(target_kick, kicks[source + step])
        else:
            dest[offset:offset + count] = columns[rest][source:end]


def _season_value(summary: _SeasonSummary, row_year: int, field: str) -> float:
    if field == 'seasons_before':
        return float(row_year - summary.year)
    if field == 'played':
        return 1.0 if summary.minutes > 0 else 0.0
    if field == 'minutes':
        return float(summary.minutes)
    if summary.minutes <= 0:
        return 0.0
    per_min = 90.0 / summary.minutes
    if field == 'points_per_90':
        return summary.points * per_min
    if field == 'xgi_per_90':
        return (summary.xg + summary.xa) * per_min
    raise KeyError(field)


def _write_seasons(arrays, index, row: _Row, seasons, season_mask, catalog: _Catalog):
    """Earlier seasons only, oldest at the left of the real tokens, padded in front."""
    if season_mask is None:
        return
    length = season_mask.shape[0]
    prior = [
        summary for summary in catalog.season_history(row.player)
        if summary.year < row.year
    ][-length:]
    count = len(prior)
    if count:
        arrays[season_mask.key][index, -count:] = 1.0
    offset = length - count
    for spec, field in seasons:
        dest = arrays[spec.key][index]
        for step, summary in enumerate(prior):
            dest[offset + step] = _season_value(summary, row.year, field)


def _write_squad(arrays, index, row: _Row, players, columns, mask, catalog: _Catalog):
    if mask is not None:
        count = min(len(players), mask.shape[0])
        if count:
            arrays[mask.key][index, :count] = 1.0
    if not players or not columns:
        return
    window = catalog.plan.config.team_form_n
    limit = columns[0][0].shape[0]
    block = catalog.match_block(row.match, catalog.plan.config.player_blocks, row.year)
    for slot, player in enumerate(players[:limit]):
        timeline = catalog.timeline(player)
        form = timeline.form(row.match, window)
        for spec, field in columns:
            if field == 'player_id_embedding':
                value = catalog.player_id(player, block)
            elif field == 'position':
                value = _position(player)
            else:
                value = form[field]
            arrays[spec.key][index, slot] = value


def _team_history_sides(specs) -> list:
    """The team-history nodes a plan uses: their leaves live in a shared table."""
    return sorted({team_history_side(spec.path)[0] for spec in specs if is_team_history_path(spec.path)})


def _write_rows(catalog: _Catalog, rows: list, specs) -> ExampleBatch:
    sides = _team_history_sides(specs)
    specs = [spec for spec in specs if not is_team_history_path(spec.path)]
    arrays = _empty(specs, len(rows))
    for side in sides:
        arrays[ref_key(side)] = numpy.zeros(len(rows), dtype=numpy.int64)
    (
        history, history_mask, seasons, season_mask,
        team_squad, team_mask, opp_squad, opp_mask, scalars,
    ) = _group_specs(specs)
    catalog.prepare_history([spec for spec, _rest in history])
    labels = numpy.zeros((len(rows), 1), dtype=numpy.float32)
    matches = []
    owners = []
    years = []
    for index, row in enumerate(rows):
        matches.append(row.match)
        owners.append(row.player)
        years.append(row.year)
        labels[index, 0] = _label(row.match)
        for spec in scalars:
            arrays[spec.key][index, 0] = _scalar(catalog, row, spec.path)
        _write_history(arrays, index, row, history, history_mask)
        _write_seasons(arrays, index, row, seasons, season_mask, catalog)
        _write_squad(arrays, index, row, row.teammates, team_squad, team_mask, catalog)
        _write_squad(arrays, index, row, row.opponents, opp_squad, opp_mask, catalog)
        for side in sides:
            code = _team_code(row.match) if side == TEAM_HISTORY_PATHS[0] else _opp_code(row.match)
            arrays[ref_key(side)][index] = catalog.team_ref(code, row)
    return ExampleBatch(arrays, labels, matches, owners, years)


def _attach_team_tables(catalog: _Catalog, specs, batches):
    """Build the shared team-history table once, after every batch has its refs."""
    table_specs = [spec for spec in specs if is_team_history_path(spec.path)]
    if not table_specs or catalog.team_tables is None:
        return
    fields = sorted({team_history_side(spec.path)[1] for spec in table_specs} - {'mask'})
    table = catalog.team_tables.finalize(fields)
    tables = {TEAM_HISTORY_TABLE: table}
    table_inputs = {}
    for spec in table_specs:
        side, name = team_history_side(spec.path)
        table_inputs[spec.key] = (TEAM_HISTORY_TABLE, name, ref_key(side))
    for batch in batches:
        batch.tables = tables
        batch.table_inputs = table_inputs


def window_rate_batch(batch: ExampleBatch, horizon: int):
    """One row per player-GW, labelled with points per fixture over `horizon` GWs.

    The v2 counterpart of v1's window pp90, except that did-not-plays count, so
    the rate carries minutes risk the way v2's fixture points do. Only windows
    complete in the batch are kept. Returns the batch and each row's player-season
    group: overlapping windows share labels, so splits must keep a group together.
    """
    by_gameweek = defaultdict(lambda: defaultdict(list))
    last_gameweek = {}
    for owner, year, match, points in zip(batch.owners, batch.years, batch.matches, batch.labels.reshape(-1)):
        gameweek = getattr(match, 'gameweek', None)
        if gameweek is None:
            continue
        by_gameweek[id(owner)][int(gameweek)].append(float(points))
        last_gameweek[year] = max(last_gameweek.get(year, 0), int(gameweek))
    keep, rates, groups = [], [], []
    seen = set()
    group_ids = {}
    for index, (owner, year, match) in enumerate(zip(batch.owners, batch.years, batch.matches)):
        gameweek = getattr(match, 'gameweek', None)
        if gameweek is None:
            continue
        gameweek = int(gameweek)
        key = (id(owner), gameweek)
        if key in seen or gameweek + horizon - 1 > last_gameweek[year]:
            continue
        seen.add(key)
        window = [
            points
            for week in range(gameweek, gameweek + horizon)
            for points in by_gameweek[id(owner)].get(week, ())
        ]
        keep.append(index)
        rates.append(sum(window) / len(window))
        groups.append(group_ids.setdefault(id(owner), len(group_ids)))
    keep = numpy.asarray(keep, dtype=numpy.int64)
    return ExampleBatch(
        {key: values[keep] for key, values in batch.arrays.items()},
        numpy.asarray(rates, dtype=numpy.float32).reshape(-1, 1),
        [batch.matches[index] for index in keep],
        [batch.owners[index] for index in keep],
        [batch.years[index] for index in keep],
        tables=getattr(batch, 'tables', None) or {},
        table_inputs=getattr(batch, 'table_inputs', None) or {},
    ), numpy.asarray(groups)


def _label_rows(season, cutoff, catalog: _Catalog):
    year = _season_year(season)
    rows = []
    for player in getattr(season, 'players', []) or []:
        if getattr(player, 'skip_historical_predict', False):
            continue
        for match in getattr(player, 'matches', []) or []:
            if getattr(match, 'is_last_season_carry', False):
                continue
            if cutoff is not None:
                gameweek = getattr(match, 'gameweek', None)
                if gameweek is not None and int(gameweek) > int(cutoff):
                    continue
            rows.append(catalog.row(player, match, year, False))
    return rows


def _predict_rows(season, catalog: _Catalog):
    year = _season_year(season)
    rows = []
    for player in getattr(season, 'players', []) or []:
        if getattr(player, 'skip_historical_predict', False):
            continue
        for match in getattr(player, 'future_matches', []) or []:
            rows.append(catalog.row(player, match, year, True))
    return rows


def _played_rows(season, gameweeks, catalog: _Catalog):
    """Played fixtures in the given gameweeks, for a look back at the season start."""
    wanted = {int(gameweek) for gameweek in gameweeks}
    year = _season_year(season)
    rows = []
    for player in getattr(season, 'players', []) or []:
        if getattr(player, 'skip_historical_predict', False):
            continue
        for match in getattr(player, 'matches', []) or []:
            if getattr(match, 'is_last_season_carry', False):
                continue
            gameweek = getattr(match, 'gameweek', None)
            if gameweek is None or int(gameweek) not in wanted:
                continue
            rows.append(catalog.row(player, match, year, False))
    return rows


def build_dataset(seasons, plan, predict_season=None, cutoff_gameweek=None, report_gameweeks=None,
                  vocab=None, include_train=True):
    """Return train batch, predict batch, and the vocab fit on every season.

    Pass a frozen vocab from an earlier fit to predict with that fit's weights;
    include_train=False then skips writing the training rows.
    """
    missing = unsupported_leaves(plan)
    if missing:
        raise SystemExit(
            'learning_v2 dataset does not fill this plan yet: %s' % ', '.join(missing[:12])
        )
    seasons = list(seasons)
    if predict_season is None:
        predict_season = seasons[-1]
    _clear_match_caches()
    vocab = Vocab() if vocab is None else vocab
    block_ref = 0
    if plan.config.team_blocks > 1 or plan.config.player_blocks > 1:
        block_ref = _block_reference(vocab, predict_season, cutoff_gameweek)
    catalog = _Catalog(seasons, plan, vocab, block_ref=block_ref)
    specs = input_specs(plan)
    train_rows = []
    for season in seasons if include_train else ():
        cutoff = cutoff_gameweek if season is predict_season else None
        train_rows.extend(_label_rows(season, cutoff, catalog))
    predict_rows = _predict_rows(predict_season, catalog)
    train_batch = _write_rows(catalog, train_rows, specs)
    predict_batch = _write_rows(catalog, predict_rows, specs)
    if report_gameweeks is None:
        _attach_team_tables(catalog, specs, (train_batch, predict_batch))
        return train_batch, predict_batch, vocab
    report_batch = _write_rows(
        catalog, _played_rows(predict_season, report_gameweeks, catalog), specs,
    )
    _attach_team_tables(catalog, specs, (train_batch, predict_batch, report_batch))
    return train_batch, predict_batch, vocab, report_batch


def mask_array(path: str, arrays: dict):
    """Sequence mask that belongs to a leaf, or None for a per-row float."""
    if path.startswith('player_fixture_history.'):
        key = tensor_key('player_fixture_history.mask')
    elif path.startswith('last_season.'):
        key = tensor_key('last_season.mask')
    elif path.startswith('current_team.teammates.'):
        key = tensor_key('current_team.teammates.mask')
    elif path.startswith('current_opposition.players.'):
        key = tensor_key('current_opposition.players.mask')
    else:
        return None
    return arrays.get(key)
