"""Pick the base model (B0) for the next wave from the candidates built so far.

Run from api/:
  python -m simulation.model_eval.pick_base --run-id wave_a_final \
      --candidates v2_log_minbands_teams v2_log_minbands_teams_score ...

Scores every candidate built for 2025/26, together with its 25% v2_log_h5 blend
(eval-<name>_blend25, from manager_sets --blend), in one evaluation, so the
totals are directly comparable. The candidate whose blend has the highest total
wins. Without any blends it falls back to the unblended totals. The winner's
model plan is copied to learning_v2/model_plans/<--plan-out>.json, and the
choice is written to run_metrics/model_eval/<run-id>/base_choice.json.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from simulation.model_eval.experiments import EXPERIMENTS, RESULTS_ROOT, load_manifest

API_DIR = Path(__file__).resolve().parents[2]
PLANS_DIR = API_DIR / 'premierlearning' / 'learning_v2' / 'model_plans'
SEASON = 25


def plan_of(name):
    args = EXPERIMENTS[name]['args']
    if '--model-plan' not in args:
        return 'simple_h1'
    return args[args.index('--model-plan') + 1]


def _existing_set_names(names):
    from flask import Flask

    from persistence import DB_Handler

    found = set()
    app = Flask(__name__)
    with app.app_context():
        db = DB_Handler()
        try:
            for name in names:
                db.cursor.execute(db._sql('select max(id) from prediction_sets where name = ?'), (name,))
                row = db.cursor.fetchone()
                if row and row[0] is not None:
                    found.add(name)
        finally:
            db.close_connection()
    return found


def pick(rows, candidates):
    """(name, total, blended) of the best candidate in comparison.csv rows."""
    totals = {row['model']: float(row['total']) for row in rows}
    blended = [(totals[name + '_blend25'], name) for name in candidates if name + '_blend25' in totals]
    if blended:
        total, name = max(blended)
        return name, total, True
    plain = [(totals[name], name) for name in candidates if name in totals]
    if not plain:
        raise SystemExit('None of the candidates were scored')
    total, name = max(plain)
    return name, total, False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--candidates', nargs='+', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--plan-out', default='simple_h1_b0')
    args = parser.parse_args(argv)

    manifest = load_manifest()
    built = [name for name in args.candidates
             if name in manifest and SEASON in manifest[name].get('seasons', [])]
    missing = sorted(set(args.candidates) - set(built))
    if missing:
        print('Not built for 2025/26, left out: %s' % ', '.join(missing))
    if not built:
        raise SystemExit('No candidate is built for 2025/26')
    blends = _existing_set_names(['eval-%s_blend25' % name for name in built])

    command = [
        sys.executable, '-m', 'simulation.model_eval', 'evaluate', '--seasons', str(SEASON), '--no-panel',
        '--run-id', args.run_id, '--experiments', 'v1', 'v1_h5', 'v2_log_minbands',
    ] + built
    if 'eval-v2_log_minbands_blend25' in _existing_set_names(['eval-v2_log_minbands_blend25']):
        command += ['--set', 'v2_log_minbands_blend25=eval-v2_log_minbands_blend25']
    for name in built:
        if 'eval-%s_blend25' % name in blends:
            command += ['--set', '%s_blend25=eval-%s_blend25' % (name, name)]
    print(' '.join(command), flush=True)
    if subprocess.call(command, cwd=str(API_DIR)) != 0:
        raise SystemExit('Evaluation failed')

    out_dir = RESULTS_ROOT / args.run_id
    with (out_dir / 'comparison.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    name, total, blended = pick(rows, built)
    source = PLANS_DIR / ('%s.json' % plan_of(name))
    target = PLANS_DIR / ('%s.json' % args.plan_out)
    shutil.copyfile(str(source), str(target))
    choice = {
        'winner': name, 'total': total, 'scored_as_blend25': blended,
        'plan': plan_of(name), 'copied_to': target.name, 'candidates': built,
    }
    with (out_dir / 'base_choice.json').open('w', encoding='utf-8') as handle:
        json.dump(choice, handle, indent=2)
    print('Base model: %s (total %.2f%s), plan %s copied to %s'
          % (name, total, ' as 25%% blend' if blended else '', plan_of(name), target.name))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
