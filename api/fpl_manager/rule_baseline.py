"""A rule-based manager on the v3 inputs: the score the trained policy has to beat.

    python -m fpl_manager.rule_baseline --prediction-set-id 35 --out fpl_manager_runs/baseline_set35.csv

Each threshold set plays one run over the seasons, in this order of priority
each week:

1. Wildcard, free hit, bench boost, triple captain: play the chip when this
   week is within a point of its best week and its gain clears the
   threshold, or when this is the chip's last week and it gains anything.
2. Spend every free transfer when spending now is at least as good as any
   roll and clears the spend threshold.
3. Otherwise make one transfer if that alone clears the one-transfer
   threshold.
4. Otherwise hold.

The rules see only the policy's inputs, so they use no information the
policy lacks. Every result is written; the best is printed at the end.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from flask import Flask

from persistence import DB_Handler

from fpl_manager.actions import (
    ACTIONS,
    BENCH_BOOST,
    FREE_HIT,
    HOLD,
    TRANSFER_1,
    TRIPLE_CAPTAIN,
    WILDCARD,
    transfer_count,
)
from fpl_manager.multi import MultiSeasonRun
from fpl_manager.observation import OBSERVATION_NAMES_V3, POINT_SCALE
from fpl_manager.season_source import load_managed_season

GRID = {
    'spend': (0.0, 2.0, 4.0),
    'one': (1.0, 3.0),
    'wildcard': (10.0, 20.0, 30.0),
    'free_hit': (8.0, 15.0),
    'bench_boost': (6.0, 10.0),
    'triple_captain': (8.0,),
}
DEFAULT = {
    'spend': 2.0,
    'one': 1.0,
    'wildcard': 20.0,
    'free_hit': 8.0,
    'bench_boost': 6.0,
    'triple_captain': 8.0,
}
# A chip week within this many points of the best week counts as the best.
NEAR_BEST = 1.0


def choose(observation, mask, thresholds) -> int:
    """The action index the rules pick for one week."""
    values = dict(zip(OBSERVATION_NAMES_V3, observation))

    def points(name):
        return float(values[name]) * POINT_SCALE

    def legal(action):
        return bool(mask[ACTIONS.index(action)])

    def last_week(name):
        return float(values[name]) <= 0.0

    chips = (
        (WILDCARD, 'wildcard_now_gain_5', 'wildcard_gain_5', 'wildcard',
         'gameweeks_until_wildcard_deadline'),
        (FREE_HIT, 'free_hit_now_gain', 'free_hit_gain_5', 'free_hit',
         'gameweeks_until_free_hit_deadline'),
        (BENCH_BOOST, 'bench_boost_now', 'bench_boost_best', 'bench_boost',
         'gameweeks_until_bench_boost_deadline'),
        (TRIPLE_CAPTAIN, 'triple_captain_now', 'triple_captain_best', 'triple_captain',
         'gameweeks_until_triple_captain_deadline'),
    )
    for action, now_name, best_name, threshold, deadline in chips:
        if not legal(action):
            continue
        now = points(now_name)
        best = points(best_name)
        if now >= thresholds[threshold] and now >= best - NEAR_BEST:
            return ACTIONS.index(action)
        if last_week(deadline) and now > 0:
            return ACTIONS.index(action)

    spend_now = points('spend_now_gain_5')
    if spend_now >= thresholds['spend'] and spend_now >= points('roll_gain_5') - 1e-6:
        counts = [
            action for action in ACTIONS
            if transfer_count(action) and legal(action)
        ]
        if counts:
            return ACTIONS.index(max(counts, key=transfer_count))
    if points('transfer_1_gain_5') >= thresholds['one'] and legal(TRANSFER_1):
        return ACTIONS.index(TRANSFER_1)
    return ACTIONS.index(HOLD)


def play(seasons, thresholds):
    """One run over every season with these thresholds. Returns the run."""
    run = MultiSeasonRun(seasons, log_path=None, per_season=True)
    while True:
        action = choose(run.observation(), run.mask(), thresholds)
        _observation, _reward, done, info = run.step(action)
        if not done:
            continue
        if info.get('run_finished'):
            return run
        run.start_next_season()


def grid(quick=False):
    if quick:
        return [dict(DEFAULT)]
    names = list(GRID)
    return [dict(zip(names, values)) for values in itertools.product(*(GRID[name] for name in names))]


def build_parser():
    parser = argparse.ArgumentParser(description='Rule-based manager baseline on v3 inputs')
    parser.add_argument('--prediction-set-id', type=int, required=True)
    parser.add_argument('--seasons', default='23,24,25')
    parser.add_argument('--out', required=True, help='CSV to append results to')
    parser.add_argument('--quick', action='store_true', help='Only the default thresholds')
    parser.add_argument('--allow-oracle', action='store_true')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    years = [int(part) % 100 for part in str(args.seasons).split(',') if part.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__)
    best = None
    with app.app_context():
        db = DB_Handler()
        try:
            seasons = [
                load_managed_season(
                    db, args.prediction_set_id, year, 1, 38,
                    observation_version='v3', allow_oracle=args.allow_oracle,
                )
                for year in years
            ]
            for thresholds in grid(args.quick):
                run = play(seasons, thresholds)
                row = dict(thresholds)
                for year, points, percentile in zip(years, run.points, run.percentiles):
                    row['%s_points' % year] = round(points, 1)
                    row['%s_percentile' % year] = round(percentile, 1)
                row['total_points'] = round(sum(run.points), 1)
                row['total_percentile'] = round(sum(run.percentiles), 1)
                row['optimisations'] = int(sum(run.solved))
                row['from_cache'] = int(sum(run.cached))
                _append(out, row)
                print('Rules %s: %s points, percentiles %s' % (
                    thresholds, row['total_points'],
                    ' + '.join('%.0f' % value for value in run.percentiles),
                ), flush=True)
                if best is None or row['total_points'] > best['total_points']:
                    best = row
        finally:
            db.close_connection()
    if best is not None:
        print('Best rules: %s' % best, flush=True)
    return 0


def _append(path, row):
    new_file = not path.is_file()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


if __name__ == '__main__':
    raise SystemExit(main())
