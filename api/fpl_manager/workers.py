"""Process entry points for parallel multi-season runs.

The function is importable so Windows can start a fresh interpreter and call
it. Each process opens its own database connection and plays into the one
shared policy.
"""
from __future__ import annotations

from pathlib import Path


def multi_process_env(
    years, from_gw, to_gw, prediction_set_id, checkpoint, observation_version='v1',
    allow_oracle=False, solver=None, ranged=None, calendar=None,
    copies=(), heldout_copies=(),
):
    """Build this process's environment. The database connection stays here.

    v3 plays each season as its own episode (see multi.py); the others play
    a whole run as one.
    """
    from flask import Flask

    from persistence import DB_Handler
    from fpl_manager.env import ManagerEnv, masked_monitor
    from fpl_manager.multi import MultiSeasonRun, SeasonByRun
    from fpl_manager.observation import features_for
    from fpl_manager.season_source import load_managed_season
    from squadtools.simulation_strategy import ensure_quiet_cbc

    ensure_quiet_cbc()

    app = Flask('fpl_manager_worker')
    app.app_context().push()
    db = DB_Handler()
    seasons = [
        load_managed_season(
            db,
            prediction_set_id,
            int(year),
            int(from_gw),
            int(to_gw),
            observation_version=observation_version,
            allow_oracle=allow_oracle,
            solver=solver,
            ranged=ranged,
            calendar=calendar,
        )
        for year in years
    ]
    log_path = Path(checkpoint) / 'runs.csv'

    if copies:
        from synthetic_seasons.training import mixed_factory

        factory = mixed_factory(db, seasons, copies, heldout_copies, log_path)
    elif observation_version == 'v3':
        factory = SeasonByRun(seasons, log_path=log_path)
    else:
        def factory():
            return MultiSeasonRun(seasons, log_path=log_path)

    return masked_monitor(ManagerEnv(factory, features_for(observation_version)))
