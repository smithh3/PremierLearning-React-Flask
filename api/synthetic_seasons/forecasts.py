"""LightGBM v6 forecasts on the synthetic copies, and the error check against real seasons.

The v6 points model (prediction set 97: --lightgbm --train-once-per-season,
plan simple_h1_b1, feature set 4, no team ids, 3 seeds) is fitted once per
season at GW1 on the real seasons before it, so one fit serves every copy of
that season. Each gameweek's inputs are then rebuilt from the copy instead of
the real season:
- matches, minutes, xG, points and fixture scores: the copy's files;
- FPL flags and news: the real files with the copy's flags.json on top;
- all-competition load (minutes in the last 7/14/28 days, days since he last
  played or started, in the squad for the club's last match) and each club's
  schedule (days since its last match, matches in the last 14 days...):
  rebuilt with the copy's Premier League kick-offs and minutes in place of
  the real ones, and no cup or international minutes inside a copy injury;
- the fixture order: the copy's (schedule.py), with FPL's calendar showing
  each moved fixture in its copy gameweek (manager_check.CopyManagedSeason).

Forecasts are saved per copy as forecasts.csv.gz (every player-fixture from
each deadline through the next FORECAST_GWS gameweeks).
"""
from __future__ import annotations

import bisect
import csv
import gzip
import json
import math
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault('PL_SIMULATION_BUILD', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from simulation.historical_data import HistoricalCurrentSeason, VaastavSeasonData, _read_csv  # noqa: E402
from squadtools.chance_to_play import _players_to_chance_map, ensure_gameweek_chances  # noqa: E402

from .generate import copy_dir  # noqa: E402
from .schedule import load_reshuffle  # noqa: E402
from .source import DATA_ROOT, season_name  # noqa: E402

# The prediction set 97 build (simulation/model_eval/experiments.py, lgbm_v6).
V6_SETTINGS = dict(
    lightgbm=True, train_once_per_season=True, lgbm_plan='simple_h1_b1', lgbm_features=4,
    lgbm_ids=False, lgbm_seeds=3, horizon_gameweeks=5, seed=42,
)
# Every remaining fixture, as the database snapshots hold: the manager's chip
# planning looks over the whole rest of the season.
FORECAST_GWS = 38
FORECAST_COLUMNS = ('as_of_gameweek', 'target_gameweek', 'fixture_id', 'fixture_order', 'player_id', 'player_code',
                    'player_name', 'position', 'team_id', 'opponent_team_id', 'is_home', 'price', 'selectable',
                    'predicted_minutes', 'predicted_pp90', 'predicted_points', 'chance_of_playing')


# ----------------------------------------------------------------- the copy as a season source

class CopySeasonData(VaastavSeasonData):
    """A real season's files with the copy's matches, scores and flags in place."""

    def __init__(self, year: int, index: int, data_root: Path = None):
        super().__init__(year, data_root)
        self.copy_index = int(index)
        self.copy_directory = copy_dir(year, index, data_root)
        self.fixtures = _read_csv(self.copy_directory / 'fixtures.csv')
        self.fixtures_by_id = {int(float(row['id'])): row for row in self.fixtures}
        rows = defaultdict(list)
        with gzip.open(self.copy_directory / 'player_matches.csv.gz', 'rt', newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                rows[int(float(row['element']))].append(row)
        for player_rows in rows.values():
            player_rows.sort(key=lambda row: (int(float(row.get('round') or 0)), int(float(row.get('fixture') or 0))))
        self.player_rows = dict(rows)
        flags = json.loads((self.copy_directory / 'flags.json').read_text(encoding='utf-8'))
        self.flags = {int(gw): {int(element): entry for element, entry in players.items()}
                      for gw, players in flags.items()}
        self.flagged_elements = {element for players in self.flags.values() for element in players}
        self.meta = json.loads((self.copy_directory / 'meta.json').read_text(encoding='utf-8'))
        schedule = load_reshuffle(self.copy_directory)
        # (real fixture, team id) -> copy fixture on that date; fixture -> copy gameweek if moved.
        self.slot_fixture = schedule.get('slot_fixture', {})
        self.moved_fixtures = {fid: gw for fid, (gw, _kickoff) in schedule.get('moved', {}).items()}
        self._season_usage_totals = None
        self._predict_skips = {}

    @property
    def label(self) -> str:
        return '%s copy %03d' % (season_name(self.season_start_year), self.copy_index)

    def chances(self, gameweek: int, deadline=None):
        chances = dict(ensure_gameweek_chances(self.season_start_year, gameweek, deadline=deadline,
                                               data_root=self.data_root, allow_download=False))
        for element, entry in self.flags.get(int(gameweek), {}).items():
            chances[element] = _players_to_chance_map({str(element): entry}).get(element)
        return chances

    def known_unavailable(self, raw: dict, gameweek: int) -> bool:
        # The end-of-season news in players_raw.csv belongs to the real season;
        # for players whose flags the copy changed, only the copy's chance counts.
        if int(float(raw['id'])) in self.flagged_elements:
            return False
        return super().known_unavailable(raw, gameweek)

    def build_current_season(self, as_of_gameweek, horizon_gameweeks=5, predict_through_gameweek=38):
        return CopyCurrentSeason(self, as_of_gameweek, horizon_gameweeks, predict_through_gameweek)


class CopyCurrentSeason(HistoricalCurrentSeason):
    def _load_chance_by_player(self):
        return self.source.chances(self.as_of_gameweek, self.source.deadline_for_gameweek(self.as_of_gameweek))


# ----------------------------------------------------------------- news and all-competition load

def copy_news_store(source: CopySeasonData):
    from premierlearning.lgbm.news import NewsStore

    class CopyNewsStore(NewsStore):
        def snapshot(self, year, gameweek):
            key = (int(year), int(gameweek))
            if key in self._files:
                return self._files[key]
            payload = super().snapshot(year, gameweek)
            overrides = source.flags.get(key[1]) if key[0] == source.season_start_year else None
            if payload is not None and overrides:
                players = dict(payload.get('players') or {})
                players.update({str(element): entry for element, entry in overrides.items()})
                payload = dict(payload, players=players)
                self._files[key] = payload
            return payload

    return CopyNewsStore(source.data_root)


class LoadTables:
    """The all-competition appearances and club matches, loaded once (about 20 s)."""

    def __init__(self):
        from premierlearning.external_data import features

        self.features = features
        ctx = features.Context.__new__(features.Context)
        ctx.espn_ids = defaultdict(set)
        for row in features._rows('espn_team_map.csv'):
            ctx.espn_ids[int(row['fpl_team_code'])].add(row['espn_team_id'])
        by_team = defaultdict(list)
        for row in features._rows('team_matches.csv'):
            match = features.TeamMatch(row)
            if match.kickoff is not None and match.known_from is not None:
                by_team[row['team_id']].append(match)
        ctx.team_matches = {}
        for team_id, items in by_team.items():
            items.sort(key=lambda m: m.kickoff)
            ctx.team_matches[team_id] = (items, [m.kickoff for m in items])
        self.ctx = ctx
        self.real_team_matches = ctx.team_matches
        # code -> [(kickoff, minutes, starter, event_id, team_id, category, fpl_season, fpl_fixture, fpl_team_code)]
        self.apps = defaultdict(list)
        self.events = {}   # (fpl_season, fixture, fpl_team_code) -> (event_id, team_id)
        for row in features._rows('appearances.csv'):
            season = int(row['fpl_season']) if row['fpl_season'] else None
            fixture = int(row['fpl_fixture_id']) if row['fpl_fixture_id'] else None
            side = int(row['fpl_team_code']) if row['fpl_team_code'] else None
            self.apps[int(row['fpl_code'])].append((
                features._t(row['kickoff_utc']), int(row['minutes'] or 0), int(row['starter'] or 0),
                row['event_id'], row['team_id'], row['category'], season, fixture, side))
            if fixture is not None and side is not None:
                self.events[(season, fixture, side)] = (row['event_id'], row['team_id'])
        for items in self.apps.values():
            items.sort(key=lambda a: a[0])
        self.feature_rows = defaultdict(list)   # fpl_season -> [(fixture, code, team_code, kickoff, as_of)]
        for row in features._rows(os.path.join('features', 'player_fixture_features.csv')):
            self.feature_rows[int(row['fpl_season'])].append((
                int(row['fpl_fixture_id']), int(row['fpl_code']), int(row['team_code']),
                features._t(row['kickoff_utc']), features._t(row['as_of_utc'])))
        self.team_rows = defaultdict(list)      # fpl_season -> [(fixture, team_code, kickoff, as_of)]
        for row in features._rows(os.path.join('features', 'team_fixture_features.csv')):
            self.team_rows[int(row['fpl_season'])].append((
                int(row['fpl_fixture_id']), int(row['team_code']),
                features._t(row['kickoff_utc']), features._t(row['as_of_utc'])))

    def copy_loads(self, source: CopySeasonData):
        """The external inputs the copy changes: {'player': {key: PLAYER_SCALARS}, 'team': {key: TEAM_SCALARS}}.

        Keys are (fpl_season, copy fixture, code). Rebuilt with:
        - each team's Premier League matches at the copy's kick-off (a
          reshuffled date can move by a day or two: schedule.py);
        - the copy's Premier League minutes where they differ from the real
          ones, no league or cup minutes inside a copy injury.
        Player load is rebuilt for every player of the season (dates move for
        all of them after a reshuffle); team schedule inputs (days since the
        last match, matches in the last 14 days, days to the next, matches in
        the next 7) for every team-fixture. The double/blank inputs are the
        real ones for that date: only clean gameweeks move.
        """
        from premierlearning.learning_v2.external_hook import load_external
        from premierlearning.learning_v2.external_inputs import PLAYER_SCALARS, TEAM_SCALARS

        features = self.features
        fpl_season = 2000 + source.season_start_year
        code_of = {player_id: int(float(raw['code'])) for player_id, raw in source.players_raw.items()}
        team_code = {int(float(row['id'])): int(float(row['code'])) for row in source.teams_rows}
        kickoff = {fid: features._t((row.get('kickoff_time') or '').replace('Z', '')[:16] or None)
                   for fid, row in source.fixtures_by_id.items()}
        # (real fixture, team code) -> copy fixture on that date, and back.
        slot = {(real_fid, team_code[team]): new for (real_fid, team), new in source.slot_fixture.items()}
        back = {(new, code): real_fid for (real_fid, code), new in slot.items()}
        real = VaastavSeasonData(source.season_start_year, source.data_root)
        real_minutes = {}
        for element, rows in real.player_rows.items():
            for row in rows:
                real_minutes[(element, int(float(row['fixture'])))] = int(float(row.get('minutes') or 0))
        copy_rows = defaultdict(dict)   # code -> {copy fixture: (minutes, starter, team_code, minutes changed?)}
        for element, rows in source.player_rows.items():
            code = code_of.get(element)
            if code is None:
                continue
            for row in rows:
                fixture = int(float(row['fixture']))
                minutes = int(float(row.get('minutes') or 0))
                if 'starts' in row and row['starts'] not in ('', None):
                    starter = int(float(row['starts']))
                else:
                    starter = int(minutes > 45)
                fx = source.fixtures_by_id[fixture]
                home = str(row.get('was_home')).strip().lower() == 'true'
                side = team_code.get(int(float(fx['team_h'] if home else fx['team_a'])))
                real_fid = back.get((fixture, side), fixture)
                changed = real_minutes.get((element, real_fid)) != minutes
                copy_rows[code][fixture] = (minutes, starter, side, changed)
        injured = defaultdict(set)   # code -> copy fixtures he misses in a copy injury
        for spell in source.meta.get('spells', ()):
            if spell['kind'] not in ('moved', 'added'):
                continue
            code = code_of.get(int(spell['element']))
            for row in source.player_rows.get(int(spell['element']), ()):
                gameweek = int(float(row.get('round') or 0))
                if spell['to_gw'] <= gameweek <= spell['to_last_gw'] and int(float(row.get('minutes') or 0)) == 0:
                    injured[code].add(int(float(row['fixture'])))

        ctx = self.ctx
        ctx.team_matches = self._shifted_club_matches(fpl_season, slot, kickoff, real)
        new_apps = {}
        for code in set(code_of.values()):
            fixtures = copy_rows.get(code, {})
            windows = _windows(sorted(kickoff[f] for f in injured.get(code, ()) if kickoff.get(f)))
            items, seen = [], set()
            for app in self.apps.get(code, ()):
                kick, minutes, starter, event, team, category, season, fixture, side = app
                if season == fpl_season and fixture is not None:
                    fixture = slot.get((fixture, side), fixture)
                    seen.add(fixture)
                    if fixture in injured.get(code, ()):
                        continue
                    if fixtures.get(fixture, (0, 0, 0, False))[3]:
                        minutes, starter = fixtures[fixture][0], fixtures[fixture][1]
                    if kickoff.get(fixture) is not None:
                        kick = kickoff[fixture]
                    items.append((kick, minutes, starter, event, team, category))
                    continue
                if category != 'premier_league' and any(lo <= kick <= hi for lo, hi in windows):
                    continue
                items.append((kick, minutes, starter, event, team, category))
            for fixture, (minutes, starter, side_code, _changed) in fixtures.items():
                if fixture in seen or minutes <= 0 or kickoff.get(fixture) is None:
                    continue
                real_fid = back.get((fixture, side_code), fixture)
                event, team = self.events.get((fpl_season, real_fid, side_code), ('fpl-%d' % fixture, ''))
                items.append((kickoff[fixture], minutes, starter, event, team, 'premier_league'))
            items.sort(key=lambda a: a[0])
            new_apps[code] = (items, [a[0] for a in items], {a[3] for a in items})
        ctx.apps = new_apps
        try:
            names = [column for column, _default in PLAYER_SCALARS.values()]
            players = {}
            for fixture, code, side_code, kick, as_of in self.feature_rows.get(fpl_season, ()):
                fixture = slot.get((fixture, side_code), fixture)
                load = features.player_load(ctx, code, side_code, kickoff.get(fixture) or kick, as_of)
                players[(fpl_season, fixture, code)] = tuple(float(load[name]) for name in names)
            columns = [column for column, _default in TEAM_SCALARS.values()]
            real_team = load_external()._team
            teams = {}
            for fixture, side_code, kick, as_of in self.team_rows.get(fpl_season, ()):
                values = real_team.get((fpl_season, fixture, side_code))
                if values is None:
                    continue
                new = slot.get((fixture, side_code), fixture)
                schedule, _upcoming = features.team_schedule(ctx, side_code, kickoff.get(new) or kick, as_of)
                values = list(values)
                for name, value in schedule.items():
                    if name in columns:
                        values[columns.index(name)] = float(value)
                teams[(fpl_season, new, side_code)] = tuple(values)
        finally:
            ctx.team_matches = self.real_team_matches
        return {'player': players, 'team': teams}

    def _shifted_club_matches(self, fpl_season, slot, kickoff, real):
        """ctx.team_matches with each club's Premier League match on a moved date at the copy's kick-off."""
        features = self.features
        if not slot:
            return self.real_team_matches
        real_kick = {fid: features._t((row.get('kickoff_time') or '').replace('Z', '')[:16] or None)
                     for fid, row in real.fixtures_by_id.items()}
        shifts = defaultdict(dict)   # espn team id -> {(event id or real kickoff): new kickoff}
        for (real_fid, code), new in slot.items():
            new_kick = kickoff.get(new)
            if new_kick is None:
                continue
            event = self.events.get((fpl_season, real_fid, code))
            espn_ids = [event[1]] if event and event[1] else list(self.ctx.espn_ids.get(code, ()))
            for espn_id in espn_ids:
                shifts[espn_id][event[0] if event else real_kick.get(real_fid)] = new_kick
        out = dict(self.real_team_matches)
        for espn_id, moves in shifts.items():
            items = []
            for match in self.real_team_matches.get(espn_id, ([], []))[0]:
                key = match.event_id if match.event_id in moves else (
                    match.kickoff if match.category == 'premier_league' and match.kickoff in moves else None)
                if key is not None and match.kickoff != moves[key]:
                    moved = object.__new__(type(match))
                    for name in type(match).__slots__:
                        setattr(moved, name, getattr(match, name))
                    moved.kickoff = moves[key]
                    match = moved
                items.append(match)
            items.sort(key=lambda m: m.kickoff)
            out[espn_id] = (items, [m.kickoff for m in items])
        return out


def _windows(kicks, gap_days=10):
    """Runs of a copy injury's missed kickoffs, one day either side."""
    import datetime as dt

    windows = []
    for kick in kicks:
        if windows and (kick - windows[-1][1]).days <= gap_days:
            windows[-1][1] = kick
        else:
            windows.append([kick, kick])
    return [(lo - dt.timedelta(days=1), hi + dt.timedelta(days=1)) for lo, hi in windows]


@contextmanager
def copy_inputs(source, loads):
    """Point the news and external player-load and team-schedule lookups at the copy while predicting.

    loads: LoadTables.copy_loads(source), {'player': {...}, 'team': {...}}.
    """
    from premierlearning.learning_v2.external_hook import load_external
    from premierlearning.lgbm import news

    if not isinstance(source, CopySeasonData):
        yield
        return
    external = load_external()
    tables = [(external._player, (loads or {}).get('player', {})), (external._team, (loads or {}).get('team', {}))]
    saved = [{key: table.get(key) for key in values} for table, values in tables]
    for table, values in tables:
        table.update(values)
    store = copy_news_store(source)
    original_store = news.NewsStore
    news.NewsStore = lambda *args, **kwargs: store
    try:
        yield
    finally:
        news.NewsStore = original_store
        for (table, _values), old in zip(tables, saved):
            for key, value in old.items():
                if value is None:
                    table.pop(key, None)
                else:
                    table[key] = value


# ----------------------------------------------------------------- past seasons, built once

class PastSeasonCache:
    """The real seasons before the target one, read and processed once per forecaster.

    Every deadline's runner otherwise re-reads seasons 2018/19 onwards from disk
    and re-runs their match processing (about 20 s of a 47 s gameweek). Those
    seasons are real data from before the target season, so a copy cannot
    change them; everything that does come from the copy (this season's
    players, matches, flags, news and all-competition load) is still rebuilt
    at every deadline. Runner.process_player_matches only reads the past
    seasons while processing the current one, and re-processing a past season
    gives the same result, so after the first deadline the past players skip
    process_matches(). check_cache.py (scratch) compared the model inputs with
    and without the cache.
    """

    def __init__(self, year: int):
        self.year = int(year)
        self.seasons = None
        self.processed = False

    def build(self):
        from premierlearning.season import PastSeason

        if self.seasons is None:
            self.seasons = {i: PastSeason(i) for i in range(18, self.year)}
        return self.seasons

    def attach(self, runner):
        """runner.populate_past_seasons(), with the cached seasons in place of new ones."""
        from premierlearning import runner as runner_module

        seasons = self.build()
        original = runner_module.PastSeason
        runner_module.PastSeason = lambda year_from: seasons[year_from]
        try:
            runner.populate_past_seasons()
        finally:
            runner_module.PastSeason = original

    def mark_processed(self):
        """After a first process_player_matches(): later ones skip the past players' match rebuild."""
        if self.processed:
            return
        for season in self.seasons.values():
            for player in season.players:
                player.process_matches = _already_processed
        self.processed = True


def _already_processed():
    return None


# ----------------------------------------------------------------- fitting and predicting

class SeasonForecaster:
    """One v6 fit for a season (real seasons before it), then forecasts for its copies."""

    def __init__(self, year: int, log=print):
        from simulation.prediction_builder import HistoricalPredictionBuilder

        self.year = int(year)
        self.log = log
        self.builder = HistoricalPredictionBuilder(None, seasons=(self.year,), **V6_SETTINGS)
        self.real = self.builder.source(self.year)
        self.fixed = None
        self._loads = None
        self.past = PastSeasonCache(self.year)

    def fit(self):
        from simulation.prediction_builder import _learning_v2_argv, _set_random_seed, _simulation_environment

        began = time.time()
        _set_random_seed(self.builder.seed + self.year * 100)
        self.builder.sources[self.year] = self.real
        runner = self.builder._new_runner(self.year, 1)
        with _simulation_environment(), _learning_v2_argv(True):
            self.fixed = self.builder._fit_fixed_models(runner)
        del runner
        self.log('%s: v6 fitted in %.0f s' % (season_name(self.year), time.time() - began))

    def load_tables(self):
        if self._loads is None:
            began = time.time()
            self._loads = LoadTables()
            self.log('all-competition tables loaded in %.0f s' % (time.time() - began))
        return self._loads

    def predict(self, source, gameweek, loads=None):
        """Prediction rows (prediction_builder format) for one deadline of `source`."""
        from simulation.prediction_builder import _learning_v2_argv, _prediction_rows, _simulation_environment

        self.builder.sources[self.year] = source
        try:
            runner = self._runner(source, gameweek)
            with _simulation_environment(), _learning_v2_argv(True), copy_inputs(source, loads or {}):
                self.fixed.predict(runner, already_prepared=False)
                self.past.mark_processed()
            return _prediction_rows(runner.current_season, self.year, gameweek)
        finally:
            self.builder.sources[self.year] = self.real

    def _runner(self, source, gameweek):
        """HistoricalPredictionBuilder._new_runner, with the past seasons from the cache."""
        from premierlearning.runner import Runner

        builder = self.builder
        current = source.build_current_season(gameweek, builder.horizon_gameweeks, predict_through_gameweek=38)
        runner = Runner(current_season=current, target_season_year=self.year, as_of_gameweek=gameweek,
                        horizon_gameweeks=builder.horizon_gameweeks, update_data=False)
        self.past.attach(runner)
        runner.validate_temporal_cutoff()
        if builder.learning_v2:
            runner.v2_model_plan = builder.model_plan
            runner.v2_weighting = builder.v2_weighting
            runner.v2_points_reps = builder.points_reps
        return runner

    def copy_source(self, index):
        source = CopySeasonData(self.year, index)
        loads = self.load_tables().copy_loads(source)
        return source, loads

    def forecast_copy(self, index, gameweeks=range(1, 39)):
        """Forecasts for every deadline of one copy, saved to its forecasts.csv.gz."""
        source, loads = self.copy_source(index)
        path = source.copy_directory / 'forecasts.csv.gz'
        began = time.time()
        rows = []
        for gameweek in gameweeks:
            if gameweek not in source.gameweeks:
                continue
            for row in self.predict(source, gameweek, loads):
                if row['target_gameweek'] < gameweek + FORECAST_GWS:
                    rows.append(row)
        temporary = path.with_suffix('.tmp')
        with gzip.open(temporary, 'wt', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=FORECAST_COLUMNS, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow({name: _cell(row.get(name)) for name in FORECAST_COLUMNS})
        os.replace(temporary, path)
        self.log('%s: %d forecast rows in %.0f min -> %s' % (source.label, len(rows), (time.time() - began) / 60, path))
        return path


INT_COLUMNS = ('as_of_gameweek', 'target_gameweek', 'fixture_id', 'fixture_order', 'player_id', 'player_code',
               'position', 'team_id', 'opponent_team_id', 'price')
FLOAT_COLUMNS = ('predicted_minutes', 'predicted_pp90', 'predicted_points', 'chance_of_playing')


def read_forecasts(path):
    """{as_of_gameweek: [row]} from a forecasts.csv.gz, typed like the database rows."""
    by_week = defaultdict(list)
    with gzip.open(path, 'rt', newline='', encoding='utf-8') as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for name in INT_COLUMNS:
                row[name] = int(float(row[name])) if row.get(name) not in (None, '') else None
            for name in FLOAT_COLUMNS:
                row[name] = float(row[name]) if row.get(name) not in (None, '') else None
            row['selectable'] = row.get('selectable') in ('1', 'True', 'true')
            row['is_home'] = row.get('is_home') in ('1', 'True', 'true')
            by_week[row['as_of_gameweek']].append(row)
    return dict(by_week)


def _dump_rows(rows, path):
    """One real deadline's rows, to compare with prediction set 97 in the database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FORECAST_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _cell(row.get(name)) for name in FORECAST_COLUMNS})


def _cell(value):
    if value is None:
        return ''
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return round(value, 4)
    return value


# ----------------------------------------------------------------- error check

CHECK_ORIGINS = tuple(range(2, 38, 4))   # 2, 6, ..., 34


def season_error(forecaster, source, origins=CHECK_ORIGINS, loads=None, log=print):
    """The model-eval metrics (simulation/model_eval/metrics.py) of v6 on one season or copy."""
    from simulation.model_eval.config import load_config
    from simulation.model_eval.metrics import compute
    from simulation.model_eval.panel import SCORE_OPTIMISER, SeasonFacts, attach_predictions

    config = load_config()
    try:
        previous = VaastavSeasonData(source.season_start_year - 1, source.data_root)
    except (FileNotFoundError, ValueError):
        previous = None
    facts = SeasonFacts(source, previous)
    units = []
    for origin in origins:
        if origin not in source.gameweeks:
            continue
        began = time.time()
        unit = facts.build_unit(origin, config['horizon'], config['eligibility'])
        if not len(unit.player_ids):
            continue
        rows = forecaster.predict(source, origin, loads)
        if not isinstance(source, CopySeasonData) and origin == origins[1]:
            _dump_rows(rows, DATA_ROOT / 'synthetic' / 'check_rows' / ('%s_real_gw%d.csv.gz' % (
                season_name(source.season_start_year), origin)))
        attach_predictions(unit, 'v6', rows, SCORE_OPTIMISER)
        units.append(unit)
        log('  GW%d: %d players, %.0f s' % (origin, len(unit.player_ids), time.time() - began))
    metrics = compute(units, 'v6', config)
    # Points per player-gameweek, so a copy that is simply higher scoring shows up.
    actual = [unit.actual[unit.has_fixture[:, 0], 0] for unit in units]
    total = sum(len(values) for values in actual)
    metrics['mean_actual'] = float(sum(values.sum() for values in actual) / total) if total else None
    return metrics


CHECK_KEYS = ('overall_n', 'overall_rmse', 'overall_mae', 'overall_bias', 'mean_actual', 'spearman_single_gw',
              'spearman_five_gw', 'calibration_slope', 'calibration_gap')


def error_check(years, copies=(1, 2), origins=CHECK_ORIGINS, out_path=None, log=print):
    """v6 error on each real season and its copies, at the same deadlines. Writes a markdown table."""
    out_path = Path(out_path or DATA_ROOT / 'synthetic' / 'POINTS_MODEL_CHECK.md')
    results = []
    for year in years:
        forecaster = SeasonForecaster(year, log=log)
        forecaster.fit()
        log('%s real season' % season_name(year))
        results.append(_result(year, 'real', season_error(forecaster, forecaster.real, origins, log=log)))
        _save(results, out_path)
        for index in copies:
            source, loads = forecaster.copy_source(index)
            log('%s (%d player-fixture loads and %d team-fixture schedules rebuilt)' % (
                source.label, len(loads['player']), len(loads['team'])))
            results.append(_result(year, 'copy %03d' % index, season_error(forecaster, source, origins, loads, log)))
            _save(results, out_path)
        del forecaster
    return results


def _result(year, kind, metrics):
    row = {'season': season_name(year), 'kind': kind}
    row.update({key: metrics.get(key) for key in CHECK_KEYS})
    print(json.dumps(row), flush=True)
    return row


def _save(results, path):
    lines = ['# v6 points-model error: real seasons vs synthetic copies', '',
             'Same deadlines in every row (GW %s), next-gameweek points, the model-eval eligibility rules.'
             % ', '.join(str(gw) for gw in CHECK_ORIGINS), '',
             '| Season | Data | Rows | RMSE | MAE | Bias | Points per row | Rank corr. 1 GW | Rank corr. 5 GW '
             '| Calib. slope |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for row in results:
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            row['season'], row['kind'], row['overall_n'], _fmt(row['overall_rmse']), _fmt(row['overall_mae']),
            _fmt(row['overall_bias']), _fmt(row['mean_actual']), _fmt(row['spearman_single_gw']),
            _fmt(row['spearman_five_gw']), _fmt(row['calibration_slope'])))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    return '%.3f' % value
