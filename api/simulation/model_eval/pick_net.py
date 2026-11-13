"""Helpers for run_three_seasons.cmd (2026-11-07, Hal asked).

  python -m simulation.model_eval.pick_net pick <run_id>
      Prints the net to carry to three seasons: v2_b1_noids (more data: B1's
      all-competition inputs) unless its 2025/26 total is more than MARGIN below
      v2_b0_noids, else v2_b0_noids. Reads run_metrics/model_eval/<run_id>/summary.json.

  python -m simulation.model_eval.pick_net set-id <experiment> [--seasons 23,24,25]
      Prints the experiment's prediction set id (experiments.json) if every
      season has 38 completed snapshots; exits 1 otherwise, so a manager run is
      never started on a partial set.
"""
import json
import sqlite3
import sys

from simulation.model_eval.experiments import RESULTS_ROOT, load_manifest

MORE_DATA = 'v2_b1_noids'
BASE = 'v2_b0_noids'
# Seed-to-seed noise on the total is ~0.02 and the 95% intervals ~+-0.3: within
# 0.10 the net with more data counts as just as good.
MARGIN = 0.10
SNAPSHOTS = 38


def pick(run_id):
    summary = json.loads((RESULTS_ROOT / run_id / 'summary.json').read_text(encoding='utf-8'))
    totals = summary['totals']
    more, base = totals.get(MORE_DATA), totals.get(BASE)
    if more is None:
        choice, why = BASE, '%s was not scored' % MORE_DATA
    elif base is None or more >= base - MARGIN:
        choice, why = MORE_DATA, 'total %+.2f against %+.2f (within %.2f counts as a tie)' % (
            more, base if base is not None else float('nan'), MARGIN)
    else:
        choice, why = BASE, '%s total %+.2f is more than %.2f below %+.2f' % (MORE_DATA, more, MARGIN, base)
    print(choice)
    print('%s: %s' % (choice, why), file=sys.stderr)
    return 0


def _database_file():
    import os

    path = os.environ.get('DATABASE_PATH', '').strip()
    if path.startswith('/') and not path.startswith('//'):
        drive = os.path.splitdrive(os.getcwd())[0] or 'C:'
        path = drive + path
    return path


def set_id(experiment, seasons):
    entry = load_manifest().get(experiment)
    if not entry:
        print('%s has no recorded prediction set' % experiment, file=sys.stderr)
        return 1
    prediction_set = int(entry['prediction_set_id'])
    con = sqlite3.connect('file:%s?mode=ro' % _database_file(), uri=True)
    try:
        for season in seasons:
            done = con.execute(
                "select count(distinct as_of_gameweek) from prediction_snapshots "
                "where prediction_set_id = ? and season_start_year = ? and status = 'complete'",
                (prediction_set, season)).fetchone()[0]
            if done < SNAPSHOTS:
                print('set %s (%s) has %s/%s complete snapshots for season %s'
                      % (prediction_set, experiment, done, SNAPSHOTS, season), file=sys.stderr)
                return 1
    finally:
        con.close()
    print(prediction_set)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == 'pick':
        return pick(argv[1])
    if len(argv) >= 2 and argv[0] == 'set-id':
        seasons = [23, 24, 25]
        if '--seasons' in argv:
            seasons = [int(value) for value in argv[argv.index('--seasons') + 1].split(',')]
        return set_id(argv[1], seasons)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main())
