"""Pick the best single LightGBM model scored so far, for the every-5-GW retraining run.

Run from api/:
  python -m simulation.model_eval.pick_lgbm --runs lgbm_v3 lgbm_v3_variants --out retrain_every5

Reads comparison.csv of each run (later runs win when a model is in several),
keeps single LightGBM experiments that have an `_every5` twin in EXPERIMENTS
(no mixes or blends), and picks the highest total. Writes the choice to
run_metrics/model_eval/<out>/lgbm_choice.json and the twin's name to
run_metrics/model_eval/<out>/lgbm_choice.txt (one line, for the .cmd).
"""

import argparse
import csv
import json

from simulation.model_eval.experiments import EXPERIMENTS, RESULTS_ROOT, load_manifest

SEASON = 25
SUFFIX = '_every5'


def candidates():
    return sorted(name for name in EXPERIMENTS
                  if name.startswith('lgbm_') and not name.endswith(SUFFIX) and name + SUFFIX in EXPERIMENTS)


def scores(runs):
    """{model: (total, run)} over the runs' comparison.csv files, later runs first to last."""
    found = {}
    for run in runs:
        path = RESULTS_ROOT / run / 'comparison.csv'
        if not path.is_file():
            print('No results for run %s, skipped' % run)
            continue
        with path.open(encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                found[row['model']] = (float(row['total']), run)
    return found


def pick(found, names, built=None):
    """Highest total; a tie (e.g. an id stage that earned 0 rounds, so identical
    predictions) goes to the base model, the shortest name (lgbm_v3 over lgbm_v3_id_stage)."""
    pool = [(round(found[name][0], 9), -len(name), name)
            for name in names if name in found and (built is None or name in built)]
    if not pool:
        raise SystemExit('No LightGBM candidate has been scored')
    total, _simpler, name = max(pool)
    return name, total, [(value, model) for value, _s, model in sorted(pool, reverse=True)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    manifest = load_manifest()
    built = {name for name, entry in manifest.items() if SEASON in entry.get('seasons', [])}
    name, total, ranked = pick(scores(args.runs), candidates(), built)
    out_dir = RESULTS_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    choice = {'winner': name, 'total': total, 'every5': name + SUFFIX, 'runs': args.runs,
              'ranking': [{'model': model, 'total': value} for value, model in ranked]}
    (out_dir / 'lgbm_choice.json').write_text(json.dumps(choice, indent=2), encoding='utf-8')
    (out_dir / 'lgbm_choice.txt').write_text(name, encoding='ascii')   # no newline: read by `set /p`
    for value, model in ranked:
        print('  %-28s %+.2f' % (model, value))
    print('Best LightGBM: %s (%+.2f); building %s' % (name, total, name + SUFFIX))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
