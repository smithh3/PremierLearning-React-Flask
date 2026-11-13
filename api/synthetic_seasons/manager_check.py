"""Does the trained manager do as well on synthetic copies as on the real seasons it trained on?

    python -m synthetic_seasons.manager_check --checkpoint fpl_manager_runs\\v3_lgbm_v6 --copies 1,2

Plays the saved policy's top choice (as eval.csv does) and the rule-based
manager (fpl_manager/rule_baseline.py, default thresholds) over:
- the real 2023/24, 2024/25, 2025/26 (prediction set 97, the forecasts it trained on);
- copy k of each of those seasons, with the copy's v6 forecasts (forecasts.csv.gz)
  and the copy's points.
Copies can be easier or harder than the real season, so the number that
matters is the policy's lead over the rules on each. If the policy learned the
real seasons rather than how to manage, its lead shrinks on the copies.
The policy file is only read; the paused run is not touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from fpl_manager.cache import OptimiserCache
from fpl_manager.mip import MipOptimiser
from fpl_manager.season_source import POOL_FILE_DIR, V3_BENCH_WEIGHT, ManagedSeason, _last_predicted_week
from squadtools.simulation_strategy import PredictionPool

from .forecasts import CopySeasonData, read_forecasts
from .source import DATA_ROOT, season_name

# Squad answers are cached by snapshot id; copies get ids far from the real ones.
COPY_SNAPSHOT_BASE = 2_000_000_000
# Bumped whenever the copies are rebuilt: the cache keys answers by id alone, so
# a rebuilt copy must not reuse the old one's answers. 0: the 2026-11-09 copies
# 1-2 (manager_check.csv); 1: the reshuffled copies 1-10 of 2026-11-10;
# 2: the same copies with the can_select fix (fix_selectable.py, 2026-11-11).
COPY_GENERATION = 2


def copy_snapshot_id(year, index, gameweek) -> int:
    return (COPY_SNAPSHOT_BASE + COPY_GENERATION * 100_000_000 + int(year) % 100 * 1_000_000
            + int(index) * 100 + int(gameweek))


class CopyManagedSeason(ManagedSeason):
    """A ManagedSeason whose forecasts and points come from a synthetic copy."""

    def __init__(self, db, year, index, first=1, last=38):
        self.db = db
        self.prediction_set_id = 97
        self.season = int(year)
        self.copy_index = int(index)
        self.first = int(first)
        self.observation_version = 'v3'
        self.pool_files = False
        self.calendar = 'known'
        self._actuals = CopySeasonData(self.season, self.copy_index)
        self._forecast_path = self._actuals.copy_directory / 'forecasts.csv.gz'
        if not self._forecast_path.is_file():
            raise FileNotFoundError(
                '%s has no forecasts yet (python -m synthetic_seasons forecast)' % self._forecast_path)
        # Read only when a week's pool file is missing (training.prepare_pools writes them all).
        self._rows = None
        self.pool = self._pool(self.first)
        self.last = _last_predicted_week(self.pool, self.first, int(last))
        self.optimiser = MipOptimiser(
            horizon_gameweeks=5, cache=OptimiserCache(db),
            snapshot_id=copy_snapshot_id(self.season, self.copy_index, self.first),
            season_start_year=self.season, observation_version='v3', solver='highs', ranged=True,
            bench_weight=V3_BENCH_WEIGHT,
        )
        self._points = {}
        self._minutes = {}
        self._squad = None

    def _pool(self, gameweek):
        """The week's pool on the known calendar, from its pool file (written on first use)."""
        path = POOL_FILE_DIR / ('snapshot_%s.pkl' % copy_snapshot_id(self.season, self.copy_index, gameweek))
        if path.is_file():
            try:
                with path.open('rb') as handle:
                    return pickle.load(handle)
            except Exception:  # noqa: BLE001 - a half-written file is rebuilt
                pass
        pool = self._pool_from_rows(gameweek)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix('.%s.tmp' % id(self))
        with partial.open('wb') as handle:
            pickle.dump(pool, handle, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            partial.replace(path)
        except OSError:
            # Another worker wrote the same file first; theirs is identical.
            partial.unlink(missing_ok=True)
        return pool

    def _pool_from_rows(self, gameweek):
        from simulation.fixture_calendar import apply_known_calendar

        if self._rows is None:
            self._rows = read_forecasts(self._forecast_path)
        rows = self._rows.get(int(gameweek)) or []
        if not rows:
            raise LookupError('%s copy %s has no GW%s forecasts' % (season_name(self.season), self.copy_index, gameweek))
        rows = apply_known_calendar(rows, self.calendar_at(gameweek), int(gameweek))
        return PredictionPool.from_rows(rows)

    def pool_for(self, gameweek):
        return self._pool(gameweek)

    def release_rows(self):
        self._rows = None

    def forecast_at(self, gameweek):
        return self._pool(gameweek), copy_snapshot_id(self.season, self.copy_index, gameweek)

    def calendar_at(self, gameweek):
        """FPL's real calendar at this deadline, with each reshuffled fixture in its copy gameweek.

        Only fixtures FPL never moved are reshuffled (schedule.py), so the
        real calendar always showed them in their real gameweek; the copy
        shows them in the new one from the start. Deadlines are the real ones.
        """
        calendar = super().calendar_at(gameweek)
        if calendar is None:
            return None
        calendar = dict(calendar)
        for fixture_id, copy_gameweek in self._actuals.moved_fixtures.items():
            if calendar.get(fixture_id) is not None:
                calendar[fixture_id] = copy_gameweek
        return calendar


def play(seasons, choose):
    """One run over the seasons; returns the points per season."""
    from fpl_manager.multi import MultiSeasonRun

    run = MultiSeasonRun(seasons, log_path=None, per_season=True)
    while True:
        action = choose(run.observation(), run.mask())
        _observation, _reward, done, info = run.step(action)
        if not done:
            continue
        if info.get('run_finished'):
            return list(run.points), list(run.percentiles)
        run.start_next_season()


def policy_chooser(checkpoint):
    from fpl_manager.observation import OBSERVATION_NAMES_V3
    from fpl_manager.train import load_checkpoint

    # Read a copy, so nothing in the paused run's folder is opened for long.
    scratch = Path(tempfile.mkdtemp(prefix='manager_check_'))
    for name in ('policy.zip', 'manifest.json', 'rng.pkl'):
        shutil.copy2(Path(checkpoint) / name, scratch / name)
    model, manifest = load_checkpoint(scratch, features=list(OBSERVATION_NAMES_V3))

    def choose(observation, mask):
        action, _state = model.predict(np.asarray(observation, dtype=np.float32)[None], deterministic=True,
                                       action_masks=np.asarray(mask, dtype=bool)[None])
        return int(np.asarray(action).reshape(-1)[0])

    return choose, int(manifest.get('num_timesteps') or 0)


def rules_chooser():
    from fpl_manager.rule_baseline import DEFAULT, choose

    return lambda observation, mask: choose(observation, mask, DEFAULT)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='synthetic_seasons.manager_check')
    parser.add_argument('--checkpoint', default=r'fpl_manager_runs\v3_lgbm_v6')
    parser.add_argument('--seasons', default='23,24,25')
    parser.add_argument('--copies', default='1')
    parser.add_argument('--skip-real', action='store_true')
    # 2026-11-13: set 105 is set 97 with the can_select fix (the real seasons'
    # forecasts the copies_fixed run trains on); 97 kept as the default.
    parser.add_argument('--prediction-set-id', type=int, default=97)
    parser.add_argument('--managers', default='policy,rules', help='policy, rules or both')
    parser.add_argument('--out', default=str(DATA_ROOT / 'synthetic' / 'manager_check.csv'))
    args = parser.parse_args(argv)
    years = [int(value) for value in args.seasons.split(',') if value.strip()]
    copies = [int(value) for value in args.copies.split(',') if value.strip()]
    managers = [value.strip() for value in args.managers.split(',') if value.strip()]

    from flask import Flask
    from fpl_manager.season_source import load_managed_season
    from persistence import DB_Handler

    policy, steps = (policy_chooser(args.checkpoint) if 'policy' in managers else (None, 0))
    print('Policy %s at step %s' % (args.checkpoint, steps), flush=True)
    rules = rules_chooser()
    choosers = [(name, choose) for name, choose in (('policy', policy), ('rules', rules)) if name in managers]
    app = Flask(__name__)
    out = Path(args.out)
    with app.app_context():
        db = DB_Handler()
        try:
            data = [] if args.skip_real else [('real', lambda: [
                load_managed_season(db, args.prediction_set_id, year, 1, 38, observation_version='v3',
                                    calendar='known')
                for year in years])]
            for index in copies:
                data.append(('copy %03d' % index, lambda index=index: [
                    CopyManagedSeason(db, year, index) for year in years]))
            for label, make in data:
                for name, choose in choosers:
                    began = time.time()
                    points, percentiles = play(make(), choose)
                    row = {'data': label, 'manager': name, 'policy_steps': steps,
                           'prediction_set_id': args.prediction_set_id if label == 'real' else 97}
                    for year, value, percentile in zip(years, points, percentiles):
                        row['%s_points' % year] = round(value, 1)
                        row['%s_percentile_on_real_curve' % year] = round(percentile, 1)
                    row['total_points'] = round(sum(points), 1)
                    row['minutes'] = round((time.time() - began) / 60, 1)
                    print(json.dumps(row), flush=True)
                    _append(out, row)
        finally:
            db.close_connection()
    return 0


def _append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


if __name__ == '__main__':
    raise SystemExit(main())
