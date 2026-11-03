"""Read just what the ESPN matching needs from each local FPL season folder.

Handles both layouts in api/data:
  CSV  (vaastav): teams.csv or raw.json, players_raw.csv, fixtures.csv, gws/merged_gw.csv
  JSON (live API): fantasy.json, fixtures.json, players/player_<id>.json
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from dataclasses import dataclass, field

from premierlearning.external_data.cache import API_ROOT

DATA_DIR = os.path.join(API_ROOT, 'data')
SEASON_DIR = re.compile(r'^season_(\d\d)_(\d\d)$')


@dataclass
class FplSeason:
    year: int                      # start year, e.g. 2025 for 2025/26
    teams: dict = field(default_factory=dict)      # id -> {code, name, short_name}
    players: dict = field(default_factory=dict)    # id -> {code, first_name, second_name, web_name, element_type}
    fixtures: dict = field(default_factory=dict)   # id -> {kickoff, team_h, team_a, event}
    appearances: list = field(default_factory=list)  # (element id, fixture id, minutes, was_home)
    future: list = field(default_factory=list)       # (element id, fixture id, was_home), unplayed (live season)
    deadlines: dict = field(default_factory=dict)    # gameweek -> deadline ISO time (UTC)


def season_folders():
    out = []
    for name in sorted(os.listdir(DATA_DIR)):
        match = SEASON_DIR.match(name)
        if match:
            out.append((2000 + int(match.group(1)), os.path.join(DATA_DIR, name)))
    return out


def _read_csv(path):
    with open(path, encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _bool(value):
    return str(value).strip().lower() in ('true', '1')


def _event(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _deadlines_from_events(season, events):
    for ev in events or []:
        if ev.get('deadline_time') and ev.get('id') is not None:
            season.deadlines[int(ev['id'])] = ev['deadline_time']


def _from_bootstrap(season, bootstrap):
    _deadlines_from_events(season, bootstrap.get('events'))
    for team in bootstrap['teams']:
        season.teams[int(team['id'])] = {'code': int(team['code']), 'name': team['name'], 'short_name': team.get('short_name', '')}
    for el in bootstrap['elements']:
        season.players[int(el['id'])] = {
            'code': int(el['code']), 'first_name': el.get('first_name', ''), 'second_name': el.get('second_name', ''),
            'web_name': el.get('web_name', ''), 'element_type': int(el.get('element_type', 0)),
        }


def _fixture_rows(season, rows):
    for fx in rows:
        if not fx.get('kickoff_time') or fx.get('team_h') in (None, ''):
            continue
        season.fixtures[int(fx['id'])] = {'kickoff': fx['kickoff_time'], 'team_h': int(fx['team_h']), 'team_a': int(fx['team_a']),
                                          'event': _event(fx.get('event'))}
        gw = _event(fx.get('event'))
        if gw is not None and fx.get('deadline_time') and gw not in season.deadlines:
            season.deadlines[gw] = fx['deadline_time']  # old vaastav fixtures.csv carries it


def load_season(year: int, path: str) -> FplSeason:
    season = FplSeason(year=year)
    if os.path.exists(os.path.join(path, 'fantasy.json')):
        with open(os.path.join(path, 'fantasy.json'), encoding='utf-8') as handle:
            _from_bootstrap(season, json.load(handle))
        with open(os.path.join(path, 'fixtures.json'), encoding='utf-8') as handle:
            _fixture_rows(season, json.load(handle))
        for file in glob.glob(os.path.join(path, 'players', 'player_*.json')):
            element = int(os.path.basename(file)[len('player_'):-len('.json')])
            with open(file, encoding='utf-8') as handle:
                data = json.load(handle)
            for row in data.get('history', []):
                season.appearances.append((int(row['element']), int(row['fixture']), int(row['minutes']), bool(row['was_home'])))
            played = {row['fixture'] for row in data.get('history', [])}
            for row in data.get('fixtures', []):
                if row.get('id') is not None and row['id'] not in played and not row.get('finished'):
                    season.future.append((element, int(row['id']), bool(row.get('is_home'))))
        return season

    if os.path.exists(os.path.join(path, 'teams.csv')):
        for team in _read_csv(os.path.join(path, 'teams.csv')):
            season.teams[int(team['id'])] = {'code': int(team['code']), 'name': team['name'], 'short_name': team.get('short_name', '')}
    elif os.path.exists(os.path.join(path, 'raw.json')):
        with open(os.path.join(path, 'raw.json'), encoding='utf-8') as handle:
            raw = json.load(handle)
        for team in raw['teams']:
            season.teams[int(team['id'])] = {'code': int(team['code']), 'name': team['name'], 'short_name': team.get('short_name', '')}
        _deadlines_from_events(season, raw.get('events'))
    for el in _read_csv(os.path.join(path, 'players_raw.csv')):
        season.players[int(el['id'])] = {
            'code': int(el['code']), 'first_name': el.get('first_name', ''), 'second_name': el.get('second_name', ''),
            'web_name': el.get('web_name', ''), 'element_type': int(el.get('element_type') or 0),
        }
    _fixture_rows(season, _read_csv(os.path.join(path, 'fixtures.csv')))
    merged = os.path.join(path, 'gws', 'merged_gw.csv')
    files = [merged] if os.path.exists(merged) else glob.glob(os.path.join(path, 'gws', 'gw*.csv'))
    for file in files:
        with open(file, encoding='utf-8', errors='replace', newline='') as handle:
            for row in csv.DictReader(handle):
                try:
                    season.appearances.append((int(row['element']), int(row['fixture']), int(float(row['minutes'])), _bool(row['was_home'])))
                except (KeyError, ValueError):
                    continue
    return season


def load_all() -> dict:
    seasons = {}
    for year, path in season_folders():
        try:
            seasons[year] = load_season(year, path)
        except FileNotFoundError as error:
            print(f'skip FPL season {year}: {error}', flush=True)
            continue
        s = seasons[year]
        print(f'FPL {year}: {len(s.teams)} teams, {len(s.players)} players, {len(s.fixtures)} fixtures, {len(s.appearances)} appearance rows', flush=True)
    return seasons
