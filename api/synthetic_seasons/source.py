"""One real season, read from data/season_XX_YY in the vaastav layout.

Only FPL's own numbers are used: the per-player match rows
(players/<name>_<id>/gw.csv), fixtures.csv, players_raw.csv (identity and
position only) and the saved FPL flags at each deadline
(chance_of_playing/gwN.json and fpl_news/gwN.json, from Randdalf/fplcache).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

DATA_ROOT = Path(__file__).resolve().parents[1] / 'data'
GOAL_POINTS = {1: 6, 2: 6, 3: 5, 4: 4}
CLEAN_SHEET_POINTS = {1: 4, 2: 4, 3: 1, 4: 0}
# BPS parts that depend on the drawn events (FPL's published table).
BPS_GOAL = {1: 12, 2: 12, 3: 18, 4: 24}
BPS_ASSIST = 9
BPS_CLEAN_SHEET = 12          # goalkeepers and defenders, 60+ minutes
BPS_CONCEDED = -4             # goalkeepers and defenders, per goal
BPS_SAVE = 2
BPS_OWN_GOAL = -6
# Defensive contribution points, from 2025/26: 2 points at 10 CBIT (defenders)
# or 12 CBIRT (midfielders and forwards).
DC_THRESHOLD = {2: 10, 3: 12, 4: 12}
DC_POINTS = 2
INT_FIELDS = (
    'minutes', 'starts', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
    'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards', 'red_cards',
    'saves', 'bonus', 'bps', 'total_points', 'defensive_contribution', 'round', 'fixture',
    'element', 'team_h_score', 'team_a_score',
)
FLOAT_FIELDS = (
    'expected_goals', 'expected_assists', 'expected_goal_involvements',
    'expected_goals_conceded', 'threat', 'creativity', 'influence', 'ict_index',
)


def season_name(year: int) -> str:
    year = int(year) % 100
    return 'season_%02d_%02d' % (year, year + 1)


def _num(value, kind=float):
    if value in (None, '', 'None', 'null'):
        return kind(0)
    try:
        return kind(float(value))
    except (TypeError, ValueError):
        return kind(0)


def _read_csv(path: Path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def parse_time(value) -> Optional[datetime]:
    if value in (None, '', 'None'):
        return None
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass
class Fixture:
    id: int
    gameweek: Optional[int]
    home: int
    away: int
    home_goals: int
    away_goals: int
    kickoff: Optional[datetime]
    raw: dict = field(repr=False, default_factory=dict)


@dataclass
class SourceSeason:
    year: int
    directory: Path
    row_columns: List[str]
    rows: List[dict]                  # one per player per team fixture, parsed
    fixtures: Dict[int, Fixture]
    fixture_columns: List[str]
    positions: Dict[int, int]         # element -> 1..4
    codes: Dict[int, int]             # element -> FPL code (same player across seasons)
    chance: Dict[int, dict]           # gw -> chance_of_playing payload
    news: Dict[int, dict]             # gw -> fpl_news payload
    has_xg: bool
    has_starts: bool
    has_dc: bool

    @property
    def name(self) -> str:
        return season_name(self.year)

    def deadline(self, gameweek: int) -> Optional[datetime]:
        for payload in (self.news.get(int(gameweek)), self.chance.get(int(gameweek))):
            if payload and payload.get('deadline'):
                return parse_time(payload['deadline'])
        return None

    def flag(self, gameweek: int, element: int) -> dict:
        """FPL's news entry for a player at a deadline ({} when unknown)."""
        payload = self.news.get(int(gameweek)) or self.chance.get(int(gameweek)) or {}
        return dict((payload.get('players') or {}).get(str(int(element))) or {})


def load_source(year: int, data_root: Path = None) -> SourceSeason:
    root = Path(data_root) if data_root else DATA_ROOT
    directory = root / season_name(year)
    fixture_columns, fixture_rows = _read_csv(directory / 'fixtures.csv')
    fixtures = parse_fixtures(fixture_rows)
    _cols, raw_players = _read_csv(directory / 'players_raw.csv')
    positions = {int(float(r['id'])): int(float(r['element_type'])) for r in raw_players}
    codes = {int(float(r['id'])): int(float(r['code'])) for r in raw_players}
    rows, row_columns = [], None
    for path in _player_files(directory / 'players'):
        columns, player_rows = _read_csv(path)
        if row_columns is None or len(columns) > len(row_columns):
            row_columns = columns
        for raw in player_rows:
            rows.append(_parse_row(raw, fixtures, positions))
    rows = [row for row in rows if row is not None]
    rows.sort(key=lambda r: (r['_kickoff'], r['fixture'], r['element']))
    chance, news = {}, {}
    for gameweek in range(1, 39):
        for folder, store in (('chance_of_playing', chance), ('fpl_news', news)):
            path = directory / folder / ('gw%d.json' % gameweek)
            if path.is_file():
                store[gameweek] = json.loads(path.read_text(encoding='utf-8'))
    return SourceSeason(
        year=int(year), directory=directory, row_columns=row_columns, rows=rows,
        fixtures=fixtures, fixture_columns=fixture_columns, positions=positions, codes=codes,
        chance=chance, news=news,
        has_xg='expected_goals' in row_columns, has_starts='starts' in row_columns,
        has_dc='defensive_contribution' in row_columns and int(year) % 100 >= 25,
    )


def _player_files(players_dir: Path) -> List[Path]:
    """One gw.csv per element id, the same folder the forecasts read.

    A renamed player can have two folders with the same id; reading both gave
    the copies duplicate match rows (premierlearning.season.player_folders).
    """
    from premierlearning.season import player_folders

    return [players_dir / name / 'gw.csv' for name in sorted(player_folders(players_dir).values())]


def parse_fixtures(fixture_rows) -> Dict[int, Fixture]:
    fixtures = {}
    for raw in fixture_rows:
        fixture = Fixture(
            id=int(float(raw['id'])),
            gameweek=None if raw.get('event') in (None, '', 'None') else int(float(raw['event'])),
            home=int(float(raw['team_h'])), away=int(float(raw['team_a'])),
            home_goals=_num(raw.get('team_h_score'), int), away_goals=_num(raw.get('team_a_score'), int),
            kickoff=parse_time(raw.get('kickoff_time')), raw=raw,
        )
        fixtures[fixture.id] = fixture
    return fixtures


def _parse_row(raw: dict, fixtures, positions) -> Optional[dict]:
    row = dict(raw)
    for name in INT_FIELDS:
        if name in row:
            row[name] = _num(row[name], int)
    for name in FLOAT_FIELDS:
        if name in row:
            row[name] = _num(row[name], float)
    fixture = fixtures.get(row.get('fixture'))
    if fixture is None or fixture.gameweek is None:
        return None
    home = str(raw.get('was_home')).strip().lower() == 'true'
    row['_team'] = fixture.home if home else fixture.away
    row['_opponent'] = fixture.away if home else fixture.home
    row['_home'] = home
    row['_gw'] = fixture.gameweek
    row['_kickoff'] = fixture.kickoff or datetime.min.replace(tzinfo=timezone.utc)
    row['_position'] = positions.get(row['element'], 3)
    if row['_position'] not in (1, 2, 3, 4):
        # 2024/25 Assistant Managers (element type 5): no forecasts, not in the copies.
        return None
    if 'starts' in raw:
        row['_started'] = bool(row['starts'])
    else:
        row['_started'] = row['minutes'] > 45
    return row


# ----- FPL scoring, used both to check the real rows and to score the copies -----

def event_points(row: dict, position: int, with_dc: bool) -> int:
    """Every scoring rule we know. Real rows: total_points minus this is the leftover."""
    minutes = int(row.get('minutes') or 0)
    if minutes <= 0:
        return 0
    points = 2 if minutes >= 60 else 1
    points += GOAL_POINTS[position] * int(row.get('goals_scored') or 0)
    points += 3 * int(row.get('assists') or 0)
    if minutes >= 60 and int(row.get('clean_sheets') or 0):
        points += CLEAN_SHEET_POINTS[position]
    if position in (1, 2):
        points -= int(row.get('goals_conceded') or 0) // 2
    if position == 1:
        points += int(row.get('saves') or 0) // 3
    points += 5 * int(row.get('penalties_saved') or 0)
    points -= 2 * int(row.get('penalties_missed') or 0)
    points -= int(row.get('yellow_cards') or 0)
    points -= 3 * int(row.get('red_cards') or 0)
    points -= 2 * int(row.get('own_goals') or 0)
    points += int(row.get('bonus') or 0)
    if with_dc and position in DC_THRESHOLD:
        if int(row.get('defensive_contribution') or 0) >= DC_THRESHOLD[position]:
            points += DC_POINTS
    return points


def event_bps(row: dict, position: int) -> int:
    """The BPS parts that the drawn events change."""
    minutes = int(row.get('minutes') or 0)
    if minutes <= 0:
        return 0
    bps = 6 if minutes > 60 else 3
    bps += BPS_GOAL[position] * int(row.get('goals_scored') or 0)
    bps += BPS_ASSIST * int(row.get('assists') or 0)
    if position in (1, 2):
        if minutes >= 60 and int(row.get('clean_sheets') or 0):
            bps += BPS_CLEAN_SHEET
        bps += BPS_CONCEDED * int(row.get('goals_conceded') or 0)
    bps += BPS_SAVE * int(row.get('saves') or 0)
    bps += BPS_OWN_GOAL * int(row.get('own_goals') or 0)
    return bps


def bonus_from_bps(bps_by_player: Dict[int, int]) -> Dict[int, int]:
    """FPL's 3/2/1 with its tie rules: rank = 1 + players with more BPS."""
    values = sorted(bps_by_player.values(), reverse=True)
    result = {}
    for player, bps in bps_by_player.items():
        rank = 1 + sum(1 for value in values if value > bps)
        result[player] = {1: 3, 2: 2, 3: 1}.get(rank, 0)
    return result
