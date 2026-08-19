"""Point-in-time adapters for historical FPL seasons.

The vaastav data is stored as an end-of-season export.  This module deliberately
uses only identity fields from ``players_raw.csv`` and only pre-cutoff match rows
for model features.  Price, team and position are rebuilt from the cumulative
``gws/gwN.csv`` snapshots.  Historical injury percentages are not available, so
the prediction pipeline leaves them unknown and relies on its minutes model.
"""

import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from premierlearning.helper_functions import coerce_bool, coerce_optional_int
from premierlearning.player import Player
from premierlearning.season import Season
from premierlearning.team import Team


POSITION_IDS = {
    'GK': 1,
    'GKP': 1,
    'DEF': 2,
    'MID': 3,
    'FWD': 4,
}
UNFIT_STATUSES = {'u', 'n', 'i', 's'}


def _read_csv(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8-sig') as csv_file:
        return list(csv.DictReader(csv_file))


def _parse_datetime(value: object) -> Optional[datetime]:
    if value in (None, '', 'None', 'null'):
        return None
    text = str(value).strip().replace('Z', '+00:00')
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, '', 'None', 'null'):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class HistoricalActual:
    player_id: int
    gameweek: int
    points: float
    minutes: float
    price: int
    fixture_ids: tuple


class VaastavSeasonData:
    """Parsed, reusable source data for one historical season."""

    def __init__(self, season_start_year: int, data_root: Optional[Path] = None):
        self.season_start_year = int(season_start_year)
        self.years = (self.season_start_year, self.season_start_year + 1)
        self.data_root = (
            Path(data_root)
            if data_root is not None
            else Path(__file__).resolve().parents[1] / 'data'
        )
        self.season_dir = self.data_root / (
            f'season_{self.years[0]:02d}_{self.years[1]:02d}'
        )
        if not self.season_dir.is_dir():
            raise FileNotFoundError(f'Historical season directory not found: {self.season_dir}')

        self.fixtures = _read_csv(self.season_dir / 'fixtures.csv')
        self.teams_rows = _read_csv(self.season_dir / 'teams.csv')
        self.players_raw = {
            int(float(row['id'])): row
            for row in _read_csv(self.season_dir / 'players_raw.csv')
            if row.get('id') not in (None, '')
        }
        if not self.fixtures or not self.teams_rows or not self.players_raw:
            raise ValueError(
                f'{self.season_dir} must contain fixtures.csv, teams.csv and players_raw.csv'
            )

        self.fixtures_by_id = {
            int(float(row['id'])): row for row in self.fixtures
        }
        self.team_name_to_id = {}
        for row in self.teams_rows:
            team_id = int(float(row['id']))
            for key in ('name', 'short_name'):
                value = row.get(key)
                if value:
                    self.team_name_to_id[str(value).strip().casefold()] = team_id

        self.player_rows = self._load_player_rows()
        self.gw_updates = self._load_gw_updates()

    def _load_player_rows(self) -> Dict[int, List[dict]]:
        result: Dict[int, List[dict]] = {}
        players_dir = self.season_dir / 'players'
        if not players_dir.is_dir():
            return result
        for path in players_dir.glob('*/gw.csv'):
            suffix = path.parent.name.rsplit('_', 1)[-1]
            try:
                player_id = int(suffix)
            except ValueError:
                continue
            rows = _read_csv(path)
            rows.sort(
                key=lambda row: (
                    coerce_optional_int(row.get('round')) or 0,
                    coerce_optional_int(row.get('fixture')) or 0,
                )
            )
            result[player_id] = rows
        return result

    def _load_gw_updates(self) -> Dict[int, Dict[int, dict]]:
        updates: Dict[int, Dict[int, dict]] = {}
        gws_dir = self.season_dir / 'gws'
        for gw in range(1, 39):
            rows = _read_csv(gws_dir / f'gw{gw}.csv')
            by_player: Dict[int, dict] = {}
            for row in rows:
                raw_id = row.get('element')
                if raw_id in (None, ''):
                    continue
                by_player[int(float(raw_id))] = {
                    'element': int(float(raw_id)),
                    'name': row.get('name') or '',
                    'position': row.get('position') or '',
                    'team': row.get('team') or '',
                    'value': int(round(_float(row.get('value')))),
                }
            if by_player:
                updates[gw] = by_player
        return updates

    @property
    def gameweeks(self) -> List[int]:
        events = {
            coerce_optional_int(row.get('event'))
            for row in self.fixtures
        }
        return sorted(event for event in events if event is not None)

    def state_at_gameweek(self, gameweek: int) -> Dict[int, dict]:
        """Return the registry visible for a GW, carrying blank-GW players forward."""
        state: Dict[int, dict] = {}
        for gw in range(1, int(gameweek) + 1):
            state.update(self.gw_updates.get(gw, {}))
        return state

    def deadline_for_gameweek(self, gameweek: int) -> Optional[datetime]:
        kickoffs = [
            _parse_datetime(row.get('kickoff_time'))
            for row in self.fixtures
            if coerce_optional_int(row.get('event')) == int(gameweek)
        ]
        kickoffs = [kickoff for kickoff in kickoffs if kickoff is not None]
        if not kickoffs:
            return None
        # FPL deadlines in these seasons were 90 minutes before first kick-off.
        return min(kickoffs) - timedelta(minutes=90)

    def team_id_from_snapshot(self, snapshot: dict, raw: dict) -> Optional[int]:
        team_name = str(snapshot.get('team') or '').strip().casefold()
        if team_name in self.team_name_to_id:
            return self.team_name_to_id[team_name]
        try:
            return int(float(raw.get('team')))
        except (TypeError, ValueError):
            return None

    def known_unavailable(self, raw: dict, gameweek: int) -> bool:
        """Use only dated end-state news that was already known at the deadline."""
        news_at = _parse_datetime(raw.get('news_added'))
        deadline = self.deadline_for_gameweek(gameweek)
        if news_at is None or deadline is None or news_at > deadline:
            return False
        if str(raw.get('status') or '').strip().lower() in UNFIT_STATUSES:
            return True
        can_select = raw.get('can_select')
        if can_select not in (None, '') and not coerce_bool(can_select):
            return True
        return False

    def history_before(self, player_id: int, gameweek: int) -> List[dict]:
        return [
            deepcopy(row)
            for row in self.player_rows.get(int(player_id), [])
            if (
                coerce_optional_int(row.get('round')) is not None
                and coerce_optional_int(row.get('round')) < int(gameweek)
            )
        ]

    def fixtures_for_team(
        self, team_id: int, gameweek_from: int, gameweek_to_exclusive: int
    ) -> List[dict]:
        rows = []
        for fixture in self.fixtures:
            event = coerce_optional_int(fixture.get('event'))
            if event is None or not (int(gameweek_from) <= event < int(gameweek_to_exclusive)):
                continue
            if int(float(fixture['team_h'])) != team_id and int(float(fixture['team_a'])) != team_id:
                continue
            row = deepcopy(fixture)
            row['id'] = int(float(row['id']))
            row['event'] = event
            row['team_h'] = int(float(row['team_h']))
            row['team_a'] = int(float(row['team_a']))
            row['is_home'] = row['team_h'] == team_id
            rows.append(row)
        rows.sort(key=lambda row: (row['event'], row.get('kickoff_time') or '', row['id']))
        return rows

    def snapshot_fixtures(self, as_of_gameweek: int) -> List[dict]:
        """Mask post-cutoff results while retaining the final published calendar."""
        rows = []
        for fixture in self.fixtures:
            row = deepcopy(fixture)
            event = coerce_optional_int(row.get('event'))
            row['event'] = event
            row['id'] = int(float(row['id']))
            row['team_h'] = int(float(row['team_h']))
            row['team_a'] = int(float(row['team_a']))
            is_finished = event is not None and event < int(as_of_gameweek)
            row['finished'] = bool(is_finished)
            row['finished_provisional'] = bool(is_finished)
            row['started'] = bool(is_finished)
            if not is_finished:
                row['team_h_score'] = None
                row['team_a_score'] = None
            rows.append(row)
        return rows

    def actuals_for_gameweek(self, gameweek: int) -> Dict[int, HistoricalActual]:
        state = self.state_at_gameweek(gameweek)
        result: Dict[int, HistoricalActual] = {}
        for player_id in state:
            rows = [
                row
                for row in self.player_rows.get(player_id, [])
                if coerce_optional_int(row.get('round')) == int(gameweek)
            ]
            result[player_id] = HistoricalActual(
                player_id=player_id,
                gameweek=int(gameweek),
                points=sum(_float(row.get('total_points')) for row in rows),
                minutes=sum(_float(row.get('minutes')) for row in rows),
                price=int(state[player_id].get('value') or 0),
                fixture_ids=tuple(
                    int(float(row['fixture']))
                    for row in rows
                    if row.get('fixture') not in (None, '')
                ),
            )
        return result

    def build_current_season(
        self,
        as_of_gameweek: int,
        horizon_gameweeks: int = 5,
        predict_through_gameweek: int = 38,
    ):
        return HistoricalCurrentSeason(
            self,
            as_of_gameweek,
            horizon_gameweeks,
            predict_through_gameweek,
        )


class HistoricalCurrentSeason(Season):
    """A ``CurrentSeason``-compatible view of one historical GW deadline."""

    def __init__(
        self,
        source: VaastavSeasonData,
        as_of_gameweek: int,
        horizon_gameweeks: int = 5,
        predict_through_gameweek: int = 38,
    ):
        super().__init__()
        self.source = source
        self.years = source.years
        self.as_of_gameweek = int(as_of_gameweek)
        self.next_gameweek = self.as_of_gameweek
        self.current_gameweek = max(0, self.as_of_gameweek - 1)
        self.training_complete_through_gw = self.current_gameweek
        self.horizon_gameweeks = int(horizon_gameweeks)
        self.predict_through_gameweek = min(
            38,
            max(self.as_of_gameweek, int(predict_through_gameweek)),
        )
        self.fixtures_json = source.snapshot_fixtures(self.as_of_gameweek)
        self.last_season_team_standings = self._load_last_season_standings()
        self.element_type_dict = self._load_element_types()
        self._populate_teams()
        self._populate_players()
        self.populate_player_code_dict()

    def _load_last_season_standings(self):
        path = self.source.season_dir / 'last_season_team_standings.json'
        if not path.is_file():
            return None
        with path.open(encoding='utf-8-sig') as file:
            return json.load(file)

    def _load_element_types(self):
        path = self.source.data_root / 'element_types.json'
        with path.open(encoding='utf-8-sig') as file:
            return json.load(file)

    def _populate_teams(self):
        for raw_row in self.source.teams_rows:
            row = deepcopy(raw_row)
            row['id'] = int(float(row['id']))
            team = Team(row, self.fixtures_json, self.last_season_team_standings)
            self.teams[team.id] = team

    def _populate_players(self):
        registry = self.source.state_at_gameweek(self.as_of_gameweek)
        prediction_end = self.predict_through_gameweek + 1
        for player_id in sorted(registry):
            raw = self.source.players_raw.get(player_id)
            if raw is None:
                continue
            snapshot = registry[player_id]
            team_id = self.source.team_id_from_snapshot(snapshot, raw)
            if team_id not in self.teams:
                continue
            position = POSITION_IDS.get(str(snapshot.get('position') or '').upper())
            if position is None:
                position = coerce_optional_int(raw.get('element_type'))
            if position is None:
                continue

            history = self.source.history_before(player_id, self.as_of_gameweek)
            appearances = sum(1 for row in history if _float(row.get('minutes')) > 0)
            points = sum(_float(row.get('total_points')) for row in history)
            element = {
                'id': player_id,
                'code': int(float(raw['code'])),
                'element_type': int(position),
                'first_name': raw.get('first_name') or '',
                'second_name': raw.get('second_name') or snapshot.get('name') or '',
                'points_per_game': points / appearances if appearances else 0.0,
                'now_cost': int(snapshot.get('value') or round(_float(raw.get('now_cost')))),
                'team': int(team_id),
                # Deliberately unknown: weekly historical injury percentages are absent.
                'chance_of_playing_next_round': None,
                'status': raw.get('status'),
                'can_select': not self.source.known_unavailable(raw, self.as_of_gameweek),
            }
            future_fixtures = self.source.fixtures_for_team(
                int(team_id), self.as_of_gameweek, prediction_end
            )
            player_json = {
                'history': history,
                'fixtures': future_fixtures,
            }
            player = Player(
                None,
                element,
                player_json,
                self.teams,
                self.current_gameweek,
            )
            player.historical_selectable = bool(element['can_select'])
            player.availability_mode = 'minutes_proxy'
            self.players.append(player)

    # Abstract ``Season`` hooks are intentionally unused because this object is
    # assembled from the indexed source in one pass.
    def populate_fixtures_json(self):
        return None

    def populate_players(self):
        return None

    def populate_teams(self):
        return None

