"""Decision-time forecasts. Final season points are never used as a stand-in."""
from __future__ import annotations

from squadtools.simulation_strategy import PredictionPool

from fpl_manager.observation import LAST_GAMEWEEK


class MissingPredictionSnapshot(LookupError):
    """No complete historical_player_predictions snapshot for this decision."""


def load_decision_forecast(db, prediction_set_id: int, season_start_year: int, as_of_gameweek: int):
    """The saved snapshot for this gameweek, and its id.

    That snapshot was built with results only through the previous week.
    It already holds a prediction for every remaining fixture. The squad
    optimiser reads the next six of those weeks; this does not retrain the
    points model.
    """
    pool = load_forecast_pool(db, prediction_set_id, season_start_year, as_of_gameweek)
    snapshot = db.get_prediction_snapshot(
        int(prediction_set_id),
        int(season_start_year),
        int(as_of_gameweek),
    )
    if snapshot is None:
        raise MissingPredictionSnapshot(
            'No prediction snapshot for set %s, season %s, GW%s.'
            % (prediction_set_id, season_start_year, as_of_gameweek)
        )
    return pool, int(snapshot['id'])


def load_forecast_pool(
    db,
    prediction_set_id: int,
    season_start_year: int,
    as_of_gameweek: int,
    minutes_scale=None,
) -> PredictionPool:
    rows = db.get_historical_predictions(
        int(prediction_set_id),
        int(season_start_year),
        int(as_of_gameweek),
    )
    if not rows:
        raise MissingPredictionSnapshot(
            'No complete prediction snapshot for set %s, season %s, GW%s. '
            'Final points are not a forecast.'
            % (prediction_set_id, season_start_year, as_of_gameweek)
        )
    return PredictionPool.from_rows(rows, minutes_scale=minutes_scale)


def fixed_payload(pool: PredictionPool, gameweek: int, horizon: int = 8) -> dict:
    """Points and fixture counts that do not change between sampled seasons."""
    points = {}
    fixtures = {}
    for player in pool.players.values():
        code = str(int(player.player_code))
        week_points = {}
        week_fixtures = {}
        for offset in range(int(horizon)):
            week = int(gameweek) + offset
            if week > LAST_GAMEWEEK:
                break
            week_points[str(week)] = float(player.points(week))
            week_fixtures[str(week)] = len(player.fixture_ids_by_gameweek.get(week) or ())
        points[code] = week_points
        fixtures[code] = week_fixtures
    return {'points': points, 'fixture_counts': fixtures}
