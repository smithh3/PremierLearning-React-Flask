"""Manager training on a mix of real seasons and synthetic copies.

    python -m fpl_manager --seasons 23,24,25 --prediction-set-id 97 --observation v3 \
        --copies 1-5 --heldout-copies 6-10 --checkpoint fpl_manager_runs\\v3_lgbm_v6_copies

Each training run still plays 2023/24, 2024/25, 2025/26 in order. For each
season it picks one version at random: the real season (prediction set 97)
or one of its training copies, each equally likely. runs.csv records the
version in `<year>_data`.

The evaluation runs (eval.csv, the policy's top choice) go round a cycle:
the real seasons, then held-out copy 6 of every season, copy 7, ... The
held-out copies are never trained on. Before a held-out version's first
evaluation the worker also plays it once with the rule-based manager
(heldout_rules.csv), so the policy's lead over the rules can be read off on
data it never saw. Percentiles on copies are against the real season's
finishing curve.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from fpl_manager.multi import EVALUATION_LOG, MultiSeasonRun

REAL = 'real'
RULES_LOG = 'heldout_rules.csv'
# Copies kept loaded per worker for training draws (one per season fits in memory).
COPIES_KEPT = 3


def copy_label(index) -> str:
    return 'copy %03d' % int(index)


class SeasonMix:
    """The versions each season can be played as, and which ones a run gets."""

    def __init__(self, real_seasons, train_copies, heldout_copies, load_copy, rng=None):
        self.real = list(real_seasons)
        self.train_copies = [int(i) for i in train_copies]
        self.heldout_copies = [int(i) for i in heldout_copies]
        overlap = set(self.train_copies) & set(self.heldout_copies)
        if overlap:
            raise ValueError('copies %s are both trained on and held out' % sorted(overlap))
        self._load_copy = load_copy
        self.rng = rng if rng is not None else np.random.default_rng()
        self._loaded = {}
        for season in self.real:
            season.data_label = REAL

    def training_seasons(self):
        options = [None] + self.train_copies
        chosen = []
        for season in self.real:
            index = options[int(self.rng.integers(len(options)))]
            chosen.append(season if index is None else self._kept(season, index))
        return chosen

    def evaluation_cycle(self):
        return [REAL] + [copy_label(index) for index in self.heldout_copies]

    def evaluation_seasons(self, finished_evaluations):
        """The versions for the next evaluation run, by how many have finished."""
        cycle = self.evaluation_cycle()
        label = cycle[int(finished_evaluations) % len(cycle)]
        # Let go of the training copies, so a worker never holds more than one set.
        self._loaded.clear()
        if label == REAL:
            return label, list(self.real)
        index = self.heldout_copies[cycle.index(label) - 1]
        return label, [self._copy(season, index) for season in self.real]

    def _kept(self, real, index):
        key = (int(real.season), int(index))
        season = self._loaded.pop(key, None)
        if season is None:
            season = self._copy(real, index)
        self._loaded[key] = season          # most recently used last
        while len(self._loaded) > COPIES_KEPT:
            self._loaded.pop(next(iter(self._loaded)))
        return season

    def _copy(self, real, index):
        season = self._load_copy(int(real.season), int(index), int(real.first), int(real.last))
        if (int(season.first), int(season.last)) != (int(real.first), int(real.last)):
            raise ValueError('%s copy %s covers GW%s-%s, the real season GW%s-%s' % (
                real.season, index, season.first, season.last, real.first, real.last))
        season.data_label = copy_label(index)
        return season


class MixedRun(MultiSeasonRun):
    """A per-season run whose seasons were drawn from a SeasonMix."""

    def __init__(self, mix, log_path=None):
        self.mix = mix
        super().__init__(mix.training_seasons(), log_path=log_path, per_season=True)

    def begin_evaluation(self):
        """Called once this run is marked as an evaluation: swap in the cycle's versions."""
        directory = None if self.log_path is None else self.log_path.parent
        label, seasons = self.mix.evaluation_seasons(_finished_evaluations(directory))
        if directory is not None and label != REAL:
            ensure_rules_baseline(directory, label, seasons)
        print('Evaluation run %s plays %s of every season.' % (
            self.run_id, 'the real versions' if label == REAL else label), flush=True)
        self.seasons = seasons
        self.index = 0
        self._begin()


class MixedSeasonByRun:
    """Episode factory like fpl_manager.multi.SeasonByRun, with each run's seasons drawn."""

    def __init__(self, mix, log_path=None):
        self.mix = mix
        self.log_path = log_path
        self.run = None

    def __call__(self):
        run = self.run
        if run is not None and not run.done and run._waiting:
            run.start_next_season()
            return run
        self.run = MixedRun(self.mix, log_path=self.log_path)
        return self.run


def mixed_factory(db, real_seasons, train_copies, heldout_copies, log_path, seed=None):
    from .manager_check import CopyManagedSeason

    def load_copy(year, index, first, last):
        return CopyManagedSeason(db, year, index, first=first, last=last)

    rng = np.random.default_rng(seed)
    mix = SeasonMix(real_seasons, train_copies, heldout_copies, load_copy, rng)
    return MixedSeasonByRun(mix, log_path=log_path)


def describe(train_copies, heldout_copies) -> str:
    return (
        'Synthetic copies: each season of a run is the real one or training copy %s, '
        'picked at random (1 in %s each); runs.csv names it. Evaluation runs cycle '
        'through the real seasons and held-out copies %s, which are never trained on; '
        'the rule-based manager plays each held-out copy once (heldout_rules.csv).'
        % (_span(train_copies), len(train_copies) + 1, _span(heldout_copies))
    )


def _span(numbers):
    numbers = [int(n) for n in numbers]
    if numbers and numbers == list(range(numbers[0], numbers[-1] + 1)) and len(numbers) > 2:
        return '%s-%s' % (numbers[0], numbers[-1])
    return ','.join(str(n) for n in numbers)


def _finished_evaluations(directory) -> int:
    if directory is None:
        return 0
    path = Path(directory) / EVALUATION_LOG
    if not path.is_file():
        return 0
    with path.open(newline='') as handle:
        return sum(1 for _row in csv.DictReader(handle))


def ensure_rules_baseline(directory, label, seasons):
    """Play the rule-based manager once on this version (it never changes, so once is enough)."""
    path = Path(directory) / RULES_LOG
    if path.is_file():
        with path.open(newline='') as handle:
            if any(row.get('data') == label for row in csv.DictReader(handle)):
                return
    from fpl_manager.rule_baseline import DEFAULT, choose

    print('Rules baseline: the rule-based manager plays %s once (not training).' % label, flush=True)
    began = time.time()
    run = MultiSeasonRun(seasons, log_path=None, per_season=True)
    while True:
        _observation, _reward, done, info = run.step(choose(run.observation(), run.mask(), DEFAULT))
        if not done:
            continue
        if info.get('run_finished'):
            break
        run.start_next_season()
    row = {'data': label}
    for season, points, percentile in zip(seasons, run.points, run.percentiles):
        row['%s_points' % season.season] = round(points, 1)
        row['%s_percentile' % season.season] = round(percentile, 1)
    row['total_points'] = round(sum(run.points), 1)
    row['total_percentile'] = round(sum(run.percentiles), 1)
    row['minutes'] = round((time.time() - began) / 60, 1)
    print('Rules baseline %s' % json.dumps(row), flush=True)
    new_file = not path.is_file()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def prepare_pools(years, copies, log=print):
    """Write each copy's weekly forecast pools once, so workers never hold the forecast rows."""
    from flask import Flask

    from persistence import DB_Handler

    from .forecasts import copy_dir
    from .manager_check import CopyManagedSeason

    for year in years:
        for index in copies:
            directory = copy_dir(year, index)
            forecasts = directory / 'forecasts.csv.gz'
            matches = directory / 'player_matches.csv.gz'
            if not forecasts.is_file() or forecasts.stat().st_mtime < matches.stat().st_mtime:
                raise FileNotFoundError('%s is missing or older than the copy it forecasts' % forecasts)
    app = Flask(__name__)
    with app.app_context():
        db = DB_Handler()
        try:
            for year in years:
                for index in copies:
                    began = time.time()
                    season = CopyManagedSeason(db, year, index)
                    for gameweek in range(season.first, season.last + 1):
                        season.pool_for(gameweek)
                    season.release_rows()
                    log('%s/%s %s: GW%s-%s pools ready in %.0f s' % (
                        year, year + 1, copy_label(index), season.first, season.last, time.time() - began))
                    del season
        finally:
            db.close_connection()
