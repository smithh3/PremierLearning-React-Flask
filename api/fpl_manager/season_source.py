"""One historical season: saved forecasts for decisions, real points for the score."""
from __future__ import annotations

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


class ManagedSeason:
    """Saved per-gameweek forecasts and the historical points for that season."""

    def __init__(self, db, prediction_set_id, season, first, last):
        self.db = db
        self.prediction_set_id = int(prediction_set_id)
        self.season = int(season)
        self.first = int(first)
        self.pool = load_forecast_pool(db, prediction_set_id, self.season, self.first)
        self.last = _last_predicted_week(self.pool, self.first, int(last))
        snapshot = db.get_prediction_snapshot(prediction_set_id, self.season, self.first)
        snapshot_id = 0 if snapshot is None else int(snapshot['id'])
        self.optimiser = MipOptimiser(
            horizon_gameweeks=6,
            cache=OptimiserCache(db),
            snapshot_id=snapshot_id,
            season_start_year=self.season,
        )
        self._actuals = VaastavSeasonData(self.season)
        self._points = {}
        self._squad = None

    def forecast_at(self, gameweek):
        return load_decision_forecast(
            self.db, self.prediction_set_id, self.season, int(gameweek),
        )

    def outcomes(self, player_id, gameweek):
        gameweek = int(gameweek)
        week = self._points.get(gameweek)
        if week is None:
            rows = self._actuals.actuals_for_gameweek(gameweek)
            week = {int(player): float(row.points) for player, row in rows.items()}
            self._points[gameweek] = week
        return float(week.get(int(player_id), 0.0))

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
        )


def load_managed_season(db, prediction_set_id, year, first, last):
    if prediction_set_id is None:
        prediction_set_id = db.latest_prediction_set_id(year)
    if prediction_set_id is None:
        raise ValueError(
            'No complete prediction snapshot for %s/%s. '
            'Train and save historical predictions before the manager.'
            % (year, year + 1)
        )
    return ManagedSeason(db, prediction_set_id, year, first, last)


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
