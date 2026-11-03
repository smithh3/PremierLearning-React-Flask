"""Build the experiment prediction sets, then score them against v1.

Run from api/:
  python -m simulation.model_eval list
  python -m simulation.model_eval build v2_huber --seasons 25            (one experiment)
  python -m simulation.model_eval build-all --seasons 25                 (smoke test season)
  python -m simulation.model_eval evaluate --seasons 25
  python -m simulation.model_eval build-all                              (23, 24, 25)
  python -m simulation.model_eval evaluate
  python -m simulation.model_eval evaluate --set v1=12 --set v2_huber=14 --config my.json

Builds go through simulation.prediction_builder, so they land in the strict
point-in-time cache and resume where they stopped. Each experiment's prediction
set id is recorded in run_metrics/model_eval/experiments.json. Anything after
`--` on a build is passed to the builder (e.g. -- --quick).

Evaluate writes run_metrics/model_eval/<run_id>/: comparison.txt (read this
first), comparison.csv, scores.csv, raw_metrics.csv, buckets.csv,
calibration_bins.csv, ranking_single_gw.csv, ranking_five_gw.csv,
calibrator.csv, summary.json, config.json and panel.csv.gz.
"""

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from simulation.model_eval.experiments import (  # noqa: E402
    BASELINE,
    DERIVED,
    EXPERIMENTS,
    MANIFEST,
    RESULTS_ROOT,
    load_manifest,
    record_build,
)

DEFAULT_SEASONS = (23, 24, 25)


def _seasons(value):
    from simulation.prediction_builder import _parse_seasons
    return _parse_seasons(value)


def _builder_argv(name, seasons, passthrough):
    return (
        list(EXPERIMENTS[name]['args'])
        + ['--seasons', ','.join(str(s) for s in seasons), '--name', 'eval-%s' % name]
        + list(passthrough)
    )


def cmd_list(_args):
    manifest = load_manifest()
    for name, spec in list(EXPERIMENTS.items()) + list(DERIVED.items()):
        built = manifest.get(name)
        state = 'set %s, seasons %s' % (built['prediction_set_id'], built['seasons']) if built else (
            'derived from %s' % spec['base'] if 'base' in spec else 'not built')
        print('%-16s %-28s %s' % (name, state, spec['note']))
    return 0


def cmd_build(args):
    if args.experiment not in EXPERIMENTS:
        raise SystemExit('Unknown experiment %s. Choose from %s' % (args.experiment, ', '.join(EXPERIMENTS)))
    from simulation import prediction_builder

    argv = _builder_argv(args.experiment, args.seasons, args.passthrough)
    print('Building %s: %s' % (args.experiment, ' '.join(argv)))
    # Runner reads its flags from sys.argv, so it has to match the build.
    sys.argv = ['simulation.prediction_builder'] + argv
    code, builder = prediction_builder.run(argv)
    if builder is not None and builder.prediction_set_id is not None:
        record_build(args.experiment, builder.prediction_set_id, args.seasons, argv)
        print('Recorded %s -> prediction set %s in %s' % (args.experiment, builder.prediction_set_id, MANIFEST))
    return code


def cmd_build_all(args):
    """One child process per experiment, so TensorFlow state never carries over."""
    names = args.only or list(EXPERIMENTS)
    failed = []
    for name in names:
        command = [
            sys.executable, '-m', 'simulation.model_eval', 'build', name,
            '--seasons', ','.join(str(s) for s in args.seasons),
        ]
        if args.passthrough:
            command += ['--'] + list(args.passthrough)
        print('\n=== %s ===\n%s' % (name, ' '.join(command)), flush=True)
        # Training logs print characters like '≤'; a redirected Windows stdout is cp1252.
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        if subprocess.call(command, cwd=str(Path(__file__).resolve().parents[2]), env=env) != 0:
            failed.append(name)
    if failed:
        print('Failed or incomplete: %s' % ', '.join(failed))
        return 1
    return 0


def _resolve_sets(args):
    manifest = load_manifest()
    sets = {}
    for name in args.experiments or list(EXPERIMENTS):
        if name in manifest:
            sets[name] = int(manifest[name]['prediction_set_id'])
    for override in args.set or []:
        name, _, value = override.partition('=')
        if not value:
            raise SystemExit('--set needs name=prediction_set_id, got %s' % override)
        value = value.strip()
        # A set's name (eval-...) is looked up once the database is open.
        sets[name.strip()] = int(value) if value.isdigit() else value
    if BASELINE not in sets:
        raise SystemExit(
            'No %s prediction set. Run `python -m simulation.model_eval build %s` or pass --set %s=ID'
            % (BASELINE, BASELINE, BASELINE)
        )
    return sets


def _named_sets(db, sets):
    """Swap any set given by name (e.g. a blend from manager_sets) for its newest id."""
    resolved = {}
    for model, value in sets.items():
        if isinstance(value, int):
            resolved[model] = value
            continue
        db.cursor.execute(db._sql('select max(id) from prediction_sets where name = ?'), (value,))
        found = db.cursor.fetchone()
        if not found or found[0] is None:
            raise SystemExit('No prediction set named %s for %s' % (value, model))
        resolved[model] = int(found[0])
        print('  %s: set %s is %s' % (model, value, resolved[model]))
    return resolved


def cmd_evaluate(args):
    from flask import Flask

    from persistence import DB_Handler
    from simulation.model_eval.calibrate import rolling_linear
    from simulation.model_eval.config import load_config
    from simulation.model_eval.evaluate import evaluate, write_outputs
    from simulation.model_eval.panel import load_panel

    config = load_config(args.config)
    if args.resamples is not None:
        config['bootstrap']['resamples'] = int(args.resamples)
    sets = _resolve_sets(args)
    print('Evaluating seasons %s' % ', '.join(str(s) for s in args.seasons))
    app = Flask(__name__)
    with app.app_context():
        db = DB_Handler()
        try:
            sets = _named_sets(db, sets)
            units, scorers, dropped = load_panel(db, sets, args.seasons, config)
        finally:
            db.close_connection()
    models = list(sets)
    calibrators = {}
    for name, spec in DERIVED.items():
        if spec['base'] in sets:
            print('Deriving %s from %s (rolling linear calibration)' % (name, spec['base']))
            calibrators[name] = rolling_linear(units, spec['base'], name, config['calibrator']['min_pairs'])
            models.append(name)
    results = evaluate(units, models, BASELINE, config)
    run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = RESULTS_ROOT / run_id
    notes = {name: spec['note'] for name, spec in list(EXPERIMENTS.items()) + list(DERIVED.items())}
    text = write_outputs(
        results, units, out_dir, config,
        extra={'prediction_sets': sets, 'scorers': scorers, 'dropped_origins': dropped},
        write_panel=not args.no_panel,
        notes=notes,
    )
    if calibrators:
        with (out_dir / 'calibrator.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['model', 'season', 'origin', 'pairs', 'intercept', 'slope'])
            writer.writeheader()
            for name, records in calibrators.items():
                for record in records:
                    writer.writerow(dict(model=name, **record))
    from simulation.model_eval.slices import add_slice_report

    text += add_slice_report(units, results['models'], out_dir)
    print(text)
    print('Wrote %s' % out_dir)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog='python -m simulation.model_eval', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='Experiments and their recorded prediction sets')

    build = sub.add_parser('build', help='Build one experiment')
    build.add_argument('experiment')
    build.add_argument('--seasons', type=_seasons, default=DEFAULT_SEASONS)

    build_all = sub.add_parser('build-all', help='Build every experiment, one process each')
    build_all.add_argument('--seasons', type=_seasons, default=DEFAULT_SEASONS)
    build_all.add_argument('--only', nargs='+', choices=list(EXPERIMENTS))

    run = sub.add_parser('evaluate', help='Score the built experiments against v1')
    run.add_argument('--seasons', type=_seasons, default=DEFAULT_SEASONS)
    run.add_argument('--experiments', nargs='+', help='Subset of experiments (v1 is always needed)')
    run.add_argument('--set', action='append', help='name=prediction_set_id, overrides the manifest')
    run.add_argument('--config', help='JSON overriding simulation/model_eval/config.py defaults')
    run.add_argument('--resamples', type=int, help='Bootstrap resamples (default from config)')
    run.add_argument('--run-id')
    run.add_argument('--no-panel', action='store_true', help='Skip writing panel.csv.gz')
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough = []
    if '--' in argv:
        split = argv.index('--')
        argv, passthrough = argv[:split], argv[split + 1:]
    args = build_parser().parse_args(argv)
    args.passthrough = passthrough
    handlers = {
        'list': cmd_list,
        'build': cmd_build,
        'build-all': cmd_build_all,
        'evaluate': cmd_evaluate,
    }
    return handlers[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
