"""Build historical predictions, then run the shared roll / DGW / rotation sweep.

The trial grid lives in simulation.sweep. This entry runs that same plan on
every --seasons value (default 24 then 25). Change TRIALS once; both seasons
pick it up.

From api/:
  python -m simulation.rebuild_and_sweep
  python -m simulation.rebuild_and_sweep --seasons 25
  python -m simulation.rebuild_and_sweep --season 24
  python -m simulation.rebuild_and_sweep --dry-run
  python -m simulation.rebuild_and_sweep --skip-build
  python -m simulation.rebuild_and_sweep --from 10
  python -m simulation.rebuild_and_sweep --resume
  python -m simulation.rebuild_and_sweep --seasons 25 --workers 4
"""
import argparse
import sys

from simulation.prediction_builder import main as build_predictions
from simulation.sweep import main as run_sweep


DEFAULT_SEASONS = (24, 25)


def predict_name(season):
    season = int(season)
    return 'strict-historical-%s-%s-form3-7' % (season, season + 1)


def _parse_seasons(value):
    seasons = []
    for part in str(value).split(','):
        part = part.strip()
        if not part:
            continue
        year = int(part)
        if year >= 2000:
            year %= 100
        seasons.append(year)
    if not seasons:
        raise argparse.ArgumentTypeError('at least one season is required')
    return tuple(seasons)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Rebuild historical predictions then run the shared '
            'roll / DGW-wait / WC-rotation sim sweep'
        )
    )
    parser.add_argument(
        '--seasons',
        type=_parse_seasons,
        default=DEFAULT_SEASONS,
        help='Season start years to build and simulate, comma-separated. Default 24,25.',
    )
    parser.add_argument(
        '--season',
        type=int,
        help='Run a single season instead of --seasons',
    )
    parser.add_argument(
        '--skip-build',
        action='store_true',
        help='Skip prediction rebuild and sweep the latest cache for each season',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--from', dest='start', type=int, default=1)
    parser.add_argument('--only', type=int)
    parser.add_argument(
        '--resume',
        action='store_true',
        help=(
            'Skip trials already finished in hyperparam_sweep.jsonl for each '
            'season. Interrupted in-flight trials restart from GW1.'
        ),
    )
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Run the pre-sweep WC 30 / FH 11 config instead of the 40-trial grid',
    )
    parser.add_argument('--prediction-set-id', type=int)
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Parallel trial processes per season. 1 is sequential. 4 uses more CPU.',
    )
    parser.add_argument(
        '--name',
        default=None,
        help='Prediction-set name. Default strict-historical-<season>-<season+1>-form3-7',
    )
    return parser


def resolve_seasons(args):
    if args.season is not None:
        return (int(args.season),)
    return tuple(int(season) for season in args.seasons)


def _predict_argv(name, season):
    return [
        '--seasons', str(int(season)),
        '--from-gw', '1',
        '--to-gw', '38',
        '--train-once-per-season',
        '--horizon-reps', '5',
        '--minutes-reps', '5',
        '--name', name,
    ]


def _sweep_argv(args, season):
    argv = ['--season', str(int(season))]
    if args.dry_run:
        argv.append('--dry-run')
    if args.only is not None:
        argv.extend(['--only', str(args.only)])
    elif args.baseline:
        argv.append('--baseline')
    elif args.start != 1:
        argv.extend(['--from', str(args.start)])
    if args.resume:
        argv.append('--resume')
    if args.prediction_set_id is not None:
        argv.extend(['--prediction-set-id', str(args.prediction_set_id)])
    argv.extend(['--workers', str(max(1, int(args.workers)))])
    return argv


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    seasons = resolve_seasons(args)
    print(
        'Shared sweep plan: seasons %s'
        % ','.join(str(season) for season in seasons)
    )
    for season in seasons:
        name = args.name or predict_name(season)
        predict_argv = _predict_argv(name, season)
        if args.dry_run and not args.skip_build:
            print(
                'Would build %s/%s: python -m simulation.prediction_builder %s'
                % (season, season + 1, ' '.join(predict_argv))
            )
        elif not args.skip_build:
            saved_argv = list(sys.argv)
            print('=== historical predictions %s (%s/%s) ===' % (
                name, season, season + 1
            ))
            try:
                rc = build_predictions(predict_argv)
            finally:
                sys.argv = saved_argv
            if rc:
                return rc
            print('=== predictions complete; starting %s/%s sim sweep ===' % (
                season, season + 1
            ))
        rc = run_sweep(_sweep_argv(args, season))
        if rc:
            return rc
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
