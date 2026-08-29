"""A/B feature flags on a horizon-only run, reusing horizon_sweep's trial runner.

This grid: last-season pp90 × mins/match at 25/26 GW20 (after H1). Training
stops at GW19 so the second half of 25/26 is not in the fit.

Run from api/:
  python -m premierlearning.feature_sweep
  python -m premierlearning.feature_sweep --reps 1 --top 50
  python -m premierlearning.feature_sweep --only 1
  python -m premierlearning.feature_sweep --dry-run

Each trial is --horizon-only (--no-persist): skip minutes, assume 90 played
so predicted pp90 is the points score. Report is player / team / price / pp90.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from premierlearning.horizon_sweep import (
    _BASE,
    _trial,
    run_trial,
    select_trials,
)

SWEEPS_ROOT = Path(__file__).resolve().parent.parent / 'run_metrics' / 'feature_sweeps'

# Mid-25/26 snapshot: train through GW19, predict GW20–24.
_MID_H1 = ['--as-of-season', '25', '--as-of-gw', '20']

TRIALS = [
    _trial(
        '01_mid_h1_control',
        '25/26 GW20, no last-season pp90×mins interaction.',
        extra_args=_MID_H1,
        PL_LS_PP90_X_MINS='0',
    ),
    _trial(
        '02_mid_h1_pp90_x_mins',
        '25/26 GW20, last-season pp90 × (mins per match / 90).',
        extra_args=_MID_H1,
        PL_LS_PP90_X_MINS='1',
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Horizon-5 feature-flag sweep')
    parser.add_argument('--reps', type=int, default=1, help='Independent fits per trial (default 1)')
    parser.add_argument('--top', type=int, default=50, help='Leaderboard rows to print (default 50)')
    parser.add_argument(
        '--only',
        help='Run a single trial by number or name (e.g. 2 or 02_mid_h1_pp90_x_mins)',
    )
    parser.add_argument(
        '--from',
        dest='from_name',
        help='Resume from this trial number or name, inclusive',
    )
    parser.add_argument('--dry-run', action='store_true', help='Print trials without running')
    parser.add_argument('--quick', action='store_true', help='Pass --quick through (debug only)')
    parser.add_argument(
        '--keep-going',
        action='store_true',
        help='Continue remaining trials after a failure (default: stop)',
    )
    return parser.parse_args()


def _load_email_top(trial_dir):
    path = trial_dir / 'email_top.json'
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding='utf-8'))


def write_comparison(sweep_dir, trials, top_n, print_stdout=True):
    rows_by_trial = {
        trial['name']: _load_email_top(sweep_dir / trial['name'])
        for trial in trials
    }
    names = [trial['name'] for trial in trials]
    lines = [
        'Feature sweep comparison',
        'Top %d by predicted pp90 over the next 5 GWs at 25/26 GW20 (player, team, price).' % top_n,
        '',
    ]
    for trial in trials:
        name = trial['name']
        text_path = sweep_dir / name / 'email_top.txt'
        lines.append('=== %s ===' % name)
        lines.append(trial['note'])
        if text_path.exists():
            lines.append(text_path.read_text(encoding='utf-8').rstrip())
        else:
            lines.append('(no email_top.txt — trial missing or failed)')
        lines.append('')

    if len(names) >= 2:
        control_name, treat_name = names[0], names[1]
        control = {row['player']: row for row in rows_by_trial.get(control_name) or []}
        treat = {row['player']: row for row in rows_by_trial.get(treat_name) or []}
        lines.append('=== Rank movement %s -> %s ===' % (control_name, treat_name))
        header = '%-28s %8s %8s %8s %8s %8s' % (
            'player', 'rk0', 'rk1', 'd_rk', 'pred0', 'pred1',
        )
        lines.append(header)
        lines.append('-' * len(header))
        players = []
        seen = set()
        for row in (rows_by_trial.get(treat_name) or []) + (rows_by_trial.get(control_name) or []):
            if row['player'] in seen:
                continue
            seen.add(row['player'])
            players.append(row['player'])
        for player in players:
            a = control.get(player)
            b = treat.get(player)
            rk0 = a['rank'] if a else ''
            rk1 = b['rank'] if b else ''
            pred0 = a['pred'] if a else ''
            pred1 = b['pred'] if b else ''
            delta = ''
            if a and b:
                delta = a['rank'] - b['rank']
            lines.append(
                '%-28s %8s %8s %8s %8s %8s'
                % (player, rk0, rk1, delta, pred0, pred1)
            )

    text_path = sweep_dir / 'comparison.txt'
    text_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if print_stdout:
        print('\nWrote %s' % text_path)
        print('\n'.join(lines))


def _with_email_top(trial, top_n):
    env = dict(_BASE)
    env.update(trial.get('env') or {})
    env['PL_EMAIL_TOP_N'] = str(int(top_n))
    out = dict(trial)
    out['env'] = env
    return out


def main():
    args = parse_args()
    trials = [_with_email_top(trial, args.top) for trial in select_trials(args, TRIALS)]
    if args.dry_run:
        for trial in trials:
            print('%s  %s  %s' % (trial['name'], trial['env'], trial['note']))
        return

    sweep_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_dir = SWEEPS_ROOT / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / 'sweep.json').write_text(
        json.dumps(
            {'id': sweep_id, 'reps': args.reps, 'top': args.top, 'trials': trials},
            indent=2,
        ),
        encoding='utf-8',
    )
    print('Sweep directory: %s' % sweep_dir)
    print('%s trials x %s reps (horizon-only, assume 90 mins, top %s)' % (
        len(trials), args.reps, args.top,
    ))

    failures = []
    for trial in trials:
        if not run_trial(sweep_dir, trial, args.reps, args.quick):
            failures.append(trial['name'])
            write_comparison(sweep_dir, trials, args.top, print_stdout=False)
            if not args.keep_going:
                print('Stopping after first failure (pass --keep-going to continue).')
                break
        else:
            write_comparison(sweep_dir, trials, args.top, print_stdout=False)

    write_comparison(sweep_dir, trials, args.top, print_stdout=True)

    if failures:
        print('Finished with failures: %s' % ', '.join(failures))
        sys.exit(1)
    print('Sweep finished: %s' % sweep_dir)


if __name__ == '__main__':
    main()
