"""Process entry points for parallel multi-season runs.

The function is importable so Windows can start a fresh interpreter and call
it. Each process opens its own database connection and plays into the one
shared policy.
"""
from __future__ import annotations

from pathlib import Path


def multi_process_env(years, from_gw, to_gw, prediction_set_id, checkpoint):
    """Build this process's environment. The database connection stays here."""
    from flask import Flask
    from stable_baselines3.common.monitor import Monitor

    from persistence import DB_Handler
    from fpl_manager.env import ManagerEnv
    from fpl_manager.multi import MultiSeasonRun
    from fpl_manager.season_source import load_managed_season

    app = Flask('fpl_manager_worker')
    app.app_context().push()
    db = DB_Handler()
    seasons = [
        load_managed_season(db, prediction_set_id, int(year), int(from_gw), int(to_gw))
        for year in years
    ]
    log_path = Path(checkpoint) / 'runs.csv'

    def factory():
        return MultiSeasonRun(seasons, log_path=log_path)

    class _MaskedMonitor(Monitor):
        def action_masks(self):
            return self.env.action_masks()

    return _MaskedMonitor(ManagerEnv(factory))
