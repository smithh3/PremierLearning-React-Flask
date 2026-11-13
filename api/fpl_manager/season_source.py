"""One historical season: saved forecasts for decisions, real points for the score."""
from __future__ import annotations

import pickle
from pathlib import Path

from simulation.historical_data import VaastavSeasonData
from squadtools.simulation_strategy import (
    INITIAL_BUDGET,
    OwnedPlayer,
    SimulationStrategy,
    SquadState,
)

from fpl_manager.cache import OptimiserCache
from fpl_manager.episode import SeasonEpisode
from fpl_manager.forecasts import load_decision_forecast, load_forecast_pool
from fpl_manager.mip import MipOptimiser

# v3 squad optimiser: HiGHS first, answers stored by money range, and a
# tenth of the bench's predicted points counted (v3 scores autosubs).
V3_BENCH_WEIGHT = 0.1
# Saved forecasts never change, so each week's pool is written here once and
# read back from the file afterwards. Building it from the database took
# about 0.3s a week; keeping all of them in memory would take about 150 MB a
# season.
POOL_FILE_DIR = Path(__file__).resolve().parent.parent / 'fpl_manager_runs' / 'pool_files'

# Which fixture calendar the forecasts are laid out on.
# 'known': each fixture in the gameweek FPL showed at that deadline (blanks and
# doubles appear only once FPL announced them; see simulation.fixture_calendar).
# 'final': every fixture in the week it was finally played, as the prediction
# sets store it. That knows reschedules before they happened.
CALENDARS = ('known', 'final')
# Squad answers and pool files are stored by snapshot id. A known-calendar week
# uses this id plus the snapshot's, so it never reuses a final-calendar answer.
KNOWN_CALENDAR_ID_OFFSET = 1_000_000_000


def default_calendar(observation_version) -> str:
    return 'known' if str(observation_version or 'v1') == 'v3' else 'final'


def cache_snapshot_id(snapshot_id, calendar) -> int:
    snapshot_id = int(snapshot_id)
    if calendar == 'known' and snapshot_id:
        return KNOWN_CALENDAR_ID_OFFSET + snapshot_id
    return snapshot_id


class ManagedSeason:
    """Saved per-gameweek forecasts and the historical points for that season."""

    def __init__(
        self, db, prediction_set_id, season, first, last, observation_version='v1',
        solver=None, ranged=None, pool_files=None, calendar=None,
    ):
        self.db = db
        self.prediction_set_id = int(prediction_set_id)
        self.season = int(season)
        self.first = int(first)
        self.observation_version = str(observation_version or 'v1')
        v3 = self.observation_version == 'v3'
        self.pool_files = v3 if pool_files is None else bool(pool_files)
        self.calendar = default_calendar(self.observation_version) if calendar is None else str(calendar)
        if self.calendar not in CALENDARS:
            raise ValueError('calendar is one of %s, not %r' % (CALENDARS, self.calendar))
        self._actuals = VaastavSeasonData(self.season)
        self.pool = load_forecast_pool(
            db, prediction_set_id, self.season, self.first,
            calendar=self.calendar_at(self.first),
        )
        self.last = _last_predicted_week(self.pool, self.first, int(last))
        snapshot = db.get_prediction_snapshot(prediction_set_id, self.season, self.first)
        snapshot_id = 0 if snapshot is None else cache_snapshot_id(snapshot['id'], self.calendar)
        self.optimiser = MipOptimiser(
            horizon_gameweeks=5 if self.observation_version in ('v2', 'fast', 'v3') else 6,
            cache=OptimiserCache(db),
            snapshot_id=snapshot_id,
            season_start_year=self.season,
            observation_version=self.observation_version,
            solver=solver or ('highs' if v3 else 'cbc'),
            ranged=v3 if ranged is None else bool(ranged),
            bench_weight=V3_BENCH_WEIGHT if v3 else 0.0,
        )
        self._points = {}
        self._minutes = {}
        self._squad = None

    def calendar_at(self, gameweek):
        """fixture id -> gameweek FPL showed at this deadline, or None for 'final'."""
        if self.calendar != 'known':
            return None
        from simulation.fixture_calendar import known_calendar

        deadline = self._actuals.deadline_for_gameweek(int(gameweek))
        if deadline is None:
            raise ValueError('No GW%s deadline for %s/%s' % (gameweek, self.season, self.season + 1))
        return known_calendar(self.season, deadline)

    def _decision_forecast(self, gameweek):
        pool, snapshot_id = load_decision_forecast(
            self.db, self.prediction_set_id, self.season, int(gameweek),
            calendar=self.calendar_at(gameweek),
        )
        return pool, cache_snapshot_id(snapshot_id, self.calendar)

    def forecast_at(self, gameweek):
        if not self.pool_files:
            return self._decision_forecast(gameweek)
        snapshot = self.db.get_prediction_snapshot(
            self.prediction_set_id, self.season, int(gameweek),
        )
        if snapshot is None:
            return self._decision_forecast(gameweek)
        cache_id = cache_snapshot_id(snapshot['id'], self.calendar)
        path = POOL_FILE_DIR / ('snapshot_%s.pkl' % cache_id)
        if path.is_file():
            try:
                with path.open('rb') as handle:
                    return pickle.load(handle), cache_id
            except Exception:  # noqa: BLE001 - a half-written file is rebuilt
                pass
        pool, snapshot_id = self._decision_forecast(gameweek)
        POOL_FILE_DIR.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix('.%s.tmp' % id(self))
        with partial.open('wb') as handle:
            pickle.dump(pool, handle, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            partial.replace(path)
        except OSError:
            # Another worker wrote the same file first; theirs is identical.
            partial.unlink(missing_ok=True)
        return pool, snapshot_id

    def outcomes(self, player_id, gameweek):
        gameweek = int(gameweek)
        week = self._points.get(gameweek)
        if week is None:
            self._load_week(gameweek)
            week = self._points[gameweek]
        return float(week.get(int(player_id), 0.0))

    def appearances(self, player_id, gameweek):
        """True when the player got minutes that gameweek."""
        gameweek = int(gameweek)
        week = self._minutes.get(gameweek)
        if week is None:
            self._load_week(gameweek)
            week = self._minutes[gameweek]
        return float(week.get(int(player_id), 0.0)) > 0.0

    def _load_week(self, gameweek):
        rows = self._actuals.actuals_for_gameweek(gameweek)
        self._points[gameweek] = {int(player): float(row.points) for player, row in rows.items()}
        self._minutes[gameweek] = {int(player): float(row.minutes) for player, row in rows.items()}

    def episode(self):
        if self._squad is None:
            self._squad = _opening_squad(self.pool, self.first)
        return SeasonEpisode(
            self.pool,
            self._squad,
            self.outcomes,
            season_start_year=self.season,
            first_gameweek=self.first,
            last_gameweek=self.last,
            optimiser=self.optimiser,
            forecast_at=self.forecast_at,
            observation_version=self.observation_version,
            appearances=self.appearances if self.observation_version == 'v3' else None,
        )


def is_oracle_set(db, prediction_set_id) -> bool:
    prediction_set = db.get_prediction_set(prediction_set_id) or {}
    return bool((prediction_set.get('config') or {}).get('uses_future_information'))


def load_managed_season(
    db, prediction_set_id, year, first, last, observation_version='v1', allow_oracle=False,
    solver=None, ranged=None, calendar=None,
):
    if prediction_set_id is None:
        prediction_set_id = db.latest_prediction_set_id(year)
    if prediction_set_id is None:
        raise ValueError(
            'No complete prediction snapshot for %s/%s. '
            'Train and save historical predictions before the manager.'
            % (year, year + 1)
        )
    if is_oracle_set(db, prediction_set_id) and not allow_oracle:
        raise ValueError(
            'Prediction set %s sees future results. A manager only uses it with '
            '--allow-oracle, as a cheating upper bound.' % prediction_set_id
        )
    return ManagedSeason(
        db, prediction_set_id, year, first, last, observation_version=observation_version,
        solver=solver, ranged=ranged, calendar=calendar,
    )


def _last_predicted_week(pool, first, last) -> int:
    weeks = [
        int(week)
        for player in pool.players.values()
        for week in player.points_by_gameweek
        if int(first) <= int(week) <= int(last)
    ]
    if not weeks:
        raise ValueError(
            'The GW%s snapshot has no predicted points between GW%s and GW%s'
            % (first, first, last)
        )
    return max(weeks)


def _opening_squad(pool, gameweek) -> SquadState:
    chosen = SimulationStrategy().build_initial_squad(
        pool, as_of_gameweek=int(gameweek), budget=INITIAL_BUDGET,
    )
    spent = sum(int(pool[player_id].price) for player_id in chosen)
    return SquadState(
        owned={
            int(player_id): OwnedPlayer(int(player_id), int(pool[player_id].price))
            for player_id in chosen
        },
        bank=int(INITIAL_BUDGET - spent),
        free_transfers=0,
    )
