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
  python -m simulation.rebuild_and_sweep --points-loss poisson --points-minutes-weight --no-horizon-model --trials 02_hz4,15_mh_max_hz4
"""
import argparse
import sys

from simulation.prediction_builder import main as build_predictions
from simulation.sweep import (
    H1_POISSON_OLD_WINNER_TRIALS,
    H1_POISSON_TRIALS,
    main as run_sweep,
)


DEFAULT_SEASONS = (24, 25)


def predict_name(season, args=None):
    season = int(season)
    if args is not None and getattr(args, 'no_horizon_model', False):
        return 'strict-historical-%s-%s-h1-poisson-mw' % (season, season + 1)
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
    parser.add_argument(
        '--trials',
        help='Comma-separated trial names to simulate (subset of the grid)',
    )
    parser.add_argument(
        '--tag',
        help='Write sim logs under run_metrics/simulations/<tag>/',
    )
    parser.add_argument(
        '--points-loss',
        choices=('huber', 'poisson'),
        default='huber',
        help='Loss for the h1 points net when building predictions',
    )
    parser.add_argument(
        '--points-minutes-weight',
        action='store_true',
        help='Weight h1 training rows by minutes/90',
    )
    parser.add_argument(
        '--no-horizon-model',
        action='store_true',
        help='Skip the horizon-N net; keep raw h1 fixture scores',
    )
    parser.add_argument(
        '--h1-poisson',
        action='store_true',
        help=(
            'Train 1GW Poisson + minutes weights, skip the horizon net, and '
            'sweep the last-run variants in H1_POISSON_TRIALS'
        ),
    )
    parser.add_argument(
        '--old-winners',
        action='store_true',
        help=(
            'Sweep the horizon-model cells that beat Poisson-h1 on 25/26 '
            '(or won 24/25) and were not in the first Poisson replay'
        ),
    )
    parser.add_argument(
        '--points-reps',
        type=int,
        default=None,
        help='Independent fits for the h1 points net. Default 1, or 5 with --no-horizon-model',
    )
    return parser


def resolve_seasons(args):
    if args.season is not None:
        return (int(args.season),)
    return tuple(int(season) for season in args.seasons)


def _predict_argv(name, season, args=None):
    argv = [
        '--seasons', str(int(season)),
        '--from-gw', '1',
        '--to-gw', '38',
        '--train-once-per-season',
        '--horizon-reps', '5',
        '--minutes-reps', '5',
        '--name', name,
    ]
    if args is None:
        return argv
    if getattr(args, 'no_horizon_model', False):
        argv.append('--no-horizon-model')
        argv[argv.index('--horizon-reps') + 1] = '1'
    loss = getattr(args, 'points_loss', None)
    if loss and loss != 'huber':
        argv.extend(['--points-loss', str(loss)])
    if getattr(args, 'points_minutes_weight', False):
        argv.append('--points-minutes-weight')
    points_reps = getattr(args, 'points_reps', None)
    if points_reps is None and getattr(args, 'no_horizon_model', False):
        points_reps = 5
    if points_reps is not None:
        argv.extend(['--points-reps', str(int(points_reps))])
    return argv


def _sweep_argv(args, season):
    argv = ['--season', str(int(season))]
    if args.dry_run:
        argv.append('--dry-run')
    if args.only is not None:
        argv.extend(['--only', str(args.only)])
    elif getattr(args, 'trials', None):
        argv.extend(['--trials', str(args.trials)])
    elif args.baseline:
        argv.append('--baseline')
    elif args.start != 1:
        argv.extend(['--from', str(args.start)])
    if args.resume:
        argv.append('--resume')
    if args.prediction_set_id is not None:
        argv.extend(['--prediction-set-id', str(args.prediction_set_id)])
    argv.extend(['--workers', str(max(1, int(args.workers)))])
    tag = getattr(args, 'tag', None)
    if isinstance(tag, str) and tag.strip():
        argv.extend(['--tag', tag.strip()])
    return argv


def apply_h1_poisson_defaults(args):
    """Fill in the recommended 1GW Poisson+mw pack if --h1-poisson was passed."""
    if getattr(args, 'old_winners', False):
        if not args.trials:
            args.trials = ','.join(H1_POISSON_OLD_WINNER_TRIALS)
        if not args.tag:
            args.tag = 'h1_poisson_old_winners'
    if not getattr(args, 'h1_poisson', False):
        return args
    args.points_loss = 'poisson'
    args.points_minutes_weight = True
    args.no_horizon_model = True
    if args.points_reps is None:
        args.points_reps = 5
    if not args.trials:
        args.trials = ','.join(H1_POISSON_TRIALS)
    if not args.tag:
        args.tag = 'h1_poisson_mw'
    return args


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_h1_poisson_defaults(args)
    seasons = resolve_seasons(args)
    print(
        'Shared sweep plan: seasons %s'
        % ','.join(str(season) for season in seasons)
    )
    if args.trials:
        print('  trials %s  tag=%s' % (args.trials, args.tag or '-'))
    for season in seasons:
        name = args.name or predict_name(season, args)
        predict_argv = _predict_argv(name, season, args)
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
