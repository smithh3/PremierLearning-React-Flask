"""Sweep horizon-5 player-net hyperparameters.

Past grids and scores live in api/run_metrics/horizon_sweeps/HISTORY.md.
Update TRIALS here for the next run. HISTORY.md is updated automatically
when a sweep finishes (or after each trial, replacing that sweep's section).

Run from api/:
  python -m premierlearning.horizon_sweep
  python -m premierlearning.horizon_sweep --reps 10
  python -m premierlearning.horizon_sweep --from 3
  python -m premierlearning.horizon_sweep --only 2
  python -m premierlearning.horizon_sweep --dry-run

Each trial writes api/run_metrics/horizon_sweeps/<sweep_id>/<trial>/
    (summary.json, summary.txt, rep_*.txt, top100.txt, config.json, stdout.txt).
A comparison.csv / comparison.txt is written when the sweep finishes.

Pick on Kendall tau, top-20 Jaccard, pred std, and Haaland / Thiago / Fernandes.
Val MAE is a keep/reject band (~1.53 good; ~1.58+ reject), not the pick metric.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


API_DIR = Path(__file__).resolve().parent.parent
SWEEPS_ROOT = API_DIR / 'run_metrics' / 'horizon_sweeps'
HISTORY_PATH = SWEEPS_ROOT / 'HISTORY.md'

# Current bake: 20260819_085557 trial 06 (L2 2.25e-3 + ReduceLR 0.3).
# Pin the full recipe so learning.py defaults cannot drift remaining trials.
_BASE = {
    'PL_WIDTH_MULT': '1.5',
    'PL_N_HIDDEN': '3',
    'PL_DROPOUT_1': '0.1',
    'PL_DROPOUT_2': '0.05',
    'PL_L2': '0.00225',
    'PL_LEARNING_RATE': '0.0005',
    'PL_WEIGHT_DECAY': '0',
    'PL_OPTIMIZER': 'adamw',
    'PL_REDUCE_LR_FACTOR': '0.3',
}

CONTROL_NOTE = (
    'Control: 1.5x, 3 layers, Dropout 0.1/0.05, L2 2.25e-3, LR 5e-4, AdamW wd 0, '
    'ReduceLR 0.3 (20260819_085557 trial 06). 10 reps.'
)


def _trial(name, note, **env_delta):
    env = dict(_BASE)
    env.update(env_delta)
    return {'name': name, 'note': note, 'env': env}


TRIALS = [
    _trial(
        '01_control',
        'Bake / this-run 06: L2 2.25e-3 + ReduceLR 0.3. Best Jaccard and min J this run; tau 0.960. 10-rep check.',
    ),
    _trial(
        '02_rlr_0.5',
        'Previous bake (v5 02 / this-run 01): same body, ReduceLR 0.5. Tau held 0.973 then 0.964; min J did not (0.905 then 0.739).',
        PL_REDUCE_LR_FACTOR='0.5',
    ),
    _trial(
        '03_lr_6e-4',
        'This-run 02 / v5 07: LR 6e-4, ReduceLR 0.5. High tau both sweeps; Jaccard 0.943 then 0.889. Exact replay, not stacked on 0.3.',
        PL_LEARNING_RATE='0.0006',
        PL_REDUCE_LR_FACTOR='0.5',
    ),
    _trial(
        '04_l2_2e-3',
        'v4 winner 02_wd_0: L2 2e-3, ReduceLR 0.5. min J 0.905 that run; v5 control replay was fragile (tau 0.914, pstd 0.136).',
        PL_L2='0.002',
        PL_REDUCE_LR_FACTOR='0.5',
    ),
    _trial(
        '05_lr_6e-4_rlr_0.3',
        'Stack the two training-dynamics keeps: LR 6e-4 on the new ReduceLR 0.3 bake.',
        PL_LEARNING_RATE='0.0006',
    ),
    _trial(
        '06_huber_1.4',
        'Huber 1.4 on the bake (was 1.8). 1.0 only lost as a bundle with patience 40 on older bodies.',
        PL_HUBER_DELTA='1.4',
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Horizon-5 hyperparameter sweep')
    parser.add_argument('--reps', type=int, default=10, help='Independent fits per trial (default 10)')
    parser.add_argument(
        '--only',
        help='Run a single trial by number or name (e.g. 2 or 02_lr_6e-4)',
    )
    parser.add_argument(
        '--from',
        dest='from_name',
        help='Resume from this trial number or name, inclusive (e.g. 3 or 03_batch_16)',
    )
    parser.add_argument('--dry-run', action='store_true', help='Print trials without running')
    parser.add_argument('--quick', action='store_true', help='Pass --quick through (debug only)')
    parser.add_argument(
        '--keep-going',
        action='store_true',
        help='Continue remaining trials after a failure (default: stop)',
    )
    return parser.parse_args()


def _trial_index(token, trials):
    names = [t['name'] for t in trials]
    token = str(token).strip()
    if token in names:
        return names.index(token)
    if re.fullmatch(r'\d+', token):
        n = int(token)
        for i, name in enumerate(names):
            match = re.match(r'^0*(\d+)_', name)
            if match and int(match.group(1)) == n:
                return i
        raise SystemExit(
            'No trial numbered %s. Choose from: %s' % (token, ', '.join(names))
        )
    matches = [i for i, name in enumerate(names) if name.startswith(token)]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit('Unknown trial %s. Choose from: %s' % (token, ', '.join(names)))


def select_trials(args):
    trials = list(TRIALS)
    if args.only:
        i = _trial_index(args.only, trials)
        return [trials[i]]
    if args.from_name:
        start = _trial_index(args.from_name, trials)
        return trials[start:]
    return trials


def parse_top100_player(top100_text, needle):
    if not top100_text:
        return None
    needle = needle.lower()
    for line in top100_text.splitlines():
        if needle not in line.lower():
            continue
        match = re.match(r'^\s*(\d+)\s+(.+?)\s+(\d+)\s+(\S+)\s+([\d.]+)\s*$', line)
        if not match:
            continue
        return {
            'rank': int(match.group(1)),
            'name': match.group(2).strip(),
            'pts': float(match.group(5)),
        }
    return None


def trial_row(trial_dir):
    summary_path = trial_dir / 'summary.json'
    top100_path = trial_dir / 'top100.txt'
    config_path = trial_dir / 'config.json'
    row = {'trial': trial_dir.name, 'ok': summary_path.exists()}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding='utf-8'))
        row['note'] = config.get('note', '')
        row['env'] = json.dumps(config.get('env', {}), sort_keys=True)
        extra = config.get('extra_args') or []
        if extra:
            row['env'] = (row['env'] + ' extra=' + json.dumps(extra)).strip()
    if not summary_path.exists():
        return row
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    agreement = summary.get('rep_agreement') or {}
    session = summary.get('session') or {}
    stats = summary.get('rep_stats') or []
    model_config = summary.get('model_config') or {}
    row['optimizer'] = model_config.get('optimizer')
    row.update({
        'mean_kendall_tau': agreement.get('mean_kendall_tau'),
        'min_kendall_tau': agreement.get('min_kendall_tau'),
        'top_20_jaccard_mean': agreement.get('top_20_jaccard_mean'),
        'top_20_jaccard_min': agreement.get('top_20_jaccard_min'),
        'pred_std_mean': agreement.get('pred_std_mean'),
        'pred_std_max': agreement.get('pred_std_max'),
        'n_reps_averaged': session.get('n_reps_averaged'),
        'n_reps_requested': session.get('n_reps_requested'),
    })
    if stats:
        row['mean_epochs_ran'] = round(sum(s.get('epochs_ran', 0) for s in stats) / float(len(stats)), 1)
        row['mean_test_mae'] = round(sum(s.get('test_mae', 0) for s in stats) / float(len(stats)), 4)
        row['mean_pred_mean'] = round(sum(s.get('pred_mean', 0) for s in stats) / float(len(stats)), 3)
        row['mean_pred_max'] = round(sum(s.get('pred_max', 0) for s in stats) / float(len(stats)), 3)
    top100 = ''
    if top100_path.exists():
        top100 = top100_path.read_text(encoding='utf-8')
    else:
        stdout_path = trial_dir / 'stdout.txt'
        if stdout_path.exists():
            stdout = stdout_path.read_text(encoding='utf-8')
            marker = 'Top 100 players by predicted points over next'
            idx = stdout.find(marker)
            if idx >= 0:
                top100 = stdout[idx:]
    haaland = parse_top100_player(top100, 'Haaland')
    fernandes = parse_top100_player(top100, 'Fernandes')
    if haaland:
        row['haaland_rank'] = haaland['rank']
        row['haaland_pts'] = haaland['pts']
    if fernandes:
        row['fernandes_rank'] = fernandes['rank']
        row['fernandes_pts'] = fernandes['pts']
    return row


def write_comparison(sweep_dir, trials):
    rows = [trial_row(sweep_dir / t['name']) for t in trials]
    fieldnames = [
        'trial', 'ok', 'optimizer', 'mean_kendall_tau', 'min_kendall_tau',
        'top_20_jaccard_mean', 'top_20_jaccard_min', 'pred_std_mean', 'pred_std_max',
        'n_reps_averaged', 'n_reps_requested', 'mean_epochs_ran', 'mean_test_mae',
        'mean_pred_mean', 'mean_pred_max', 'haaland_rank', 'haaland_pts',
        'fernandes_rank', 'fernandes_pts', 'note', 'env',
    ]
    csv_path = sweep_dir / 'comparison.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        'Horizon-5 hyperparameter sweep comparison',
        CONTROL_NOTE,
        'Higher kendall/jaccard and lower pred_std = stabler ranks.',
        'haaland_pts is 5-GW total in --horizon-only mode (not h1 pp90).',
        '',
    ]
    header = (
        '%-28s %8s %8s %8s %8s %8s %8s %8s %8s'
        % ('trial', 'tau', 'j20', 'pstd', 'reps', 'epochs', 'H_rk', 'H_pts', 'testMAE')
    )
    lines.append(header)
    lines.append('-' * len(header))
    for row in rows:
        lines.append(
            '%-28s %8s %8s %8s %8s %8s %8s %8s %8s'
            % (
                row.get('trial', ''),
                row.get('mean_kendall_tau', ''),
                row.get('top_20_jaccard_mean', ''),
                row.get('pred_std_mean', ''),
                '%s/%s' % (row.get('n_reps_averaged', ''), row.get('n_reps_requested', '')),
                row.get('mean_epochs_ran', ''),
                row.get('haaland_rank', ''),
                row.get('haaland_pts', ''),
                row.get('mean_test_mae', ''),
            )
        )
        if row.get('note'):
            lines.append('    %s' % row['note'])
    text_path = sweep_dir / 'comparison.txt'
    text_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\nWrote %s\nWrote %s' % (csv_path, text_path))
    print('\n'.join(lines))
    append_history(sweep_dir)


def append_history(sweep_dir):
    """Write this sweep's comparison.txt into HISTORY.md under ### <sweep_id>."""
    comparison_path = sweep_dir / 'comparison.txt'
    if not comparison_path.exists():
        return
    sweep_id = sweep_dir.name
    comparison = comparison_path.read_text(encoding='utf-8').rstrip() + '\n'
    section = '### %s\n\n```\n%s```\n' % (sweep_id, comparison)

    if HISTORY_PATH.exists():
        text = HISTORY_PATH.read_text(encoding='utf-8')
    else:
        text = '# Horizon-5 sweep history\n\n## Runs\n\n'

    text = re.sub(r'\n### Next\b.*', '\n', text, flags=re.DOTALL)
    pattern = r'### %s\n.*?(?=\n### |\Z)' % re.escape(sweep_id)
    if re.search(pattern, text, flags=re.DOTALL):
        text = re.sub(pattern, section.rstrip() + '\n', text, count=1, flags=re.DOTALL)
    else:
        text = text.rstrip() + '\n\n' + section
    if not text.endswith('\n'):
        text += '\n'
    HISTORY_PATH.write_text(text, encoding='utf-8')
    print('Updated %s' % HISTORY_PATH)


def run_trial(sweep_dir, trial, reps, quick):
    extra_args = list(trial.get('extra_args') or [])
    trial_reps = int(trial.get('reps') or reps)
    if '--linear' in extra_args:
        trial_reps = 1
    trial_dir = sweep_dir / trial['name']
    trial_dir.mkdir(parents=True, exist_ok=True)
    config = {
        'name': trial['name'],
        'note': trial['note'],
        'env': trial['env'],
        'extra_args': extra_args,
        'reps': trial_reps,
    }
    (trial_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    env = os.environ.copy()
    env.update(trial['env'])
    env['PL_SWEEP_DIR'] = str(sweep_dir)
    env['PL_SWEEP_TRIAL'] = trial['name']
    env['PYTHONPATH'] = str(API_DIR) + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'

    cmd = [
        sys.executable, '-m', 'premierlearning',
        '--horizon-only', '--horizon-reps', str(trial_reps), '--no-persist',
    ]
    cmd.extend(extra_args)
    if quick:
        cmd.append('--quick')

    print('\n=== %s ===' % trial['name'])
    print(trial['note'])
    print('env:', trial['env'] or '(defaults)')
    print('cmd:', ' '.join(cmd))

    stdout_path = trial_dir / 'stdout.txt'
    with stdout_path.open('w', encoding='utf-8') as out:
        proc = subprocess.run(
            cmd,
            cwd=str(API_DIR),
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        print('FAILED %s (exit %s). See %s' % (trial['name'], proc.returncode, stdout_path))
        return False
    print('OK %s -> %s' % (trial['name'], trial_dir))
    return True


def main():
    args = parse_args()
    trials = select_trials(args)
    if args.dry_run:
        for trial in trials:
            extra = ' '.join(trial.get('extra_args') or [])
            print('%s  %s  %s  %s' % (trial['name'], trial['env'] or '{}', extra, trial['note']))
        return

    sweep_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_dir = SWEEPS_ROOT / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / 'sweep.json').write_text(
        json.dumps({'id': sweep_id, 'reps': args.reps, 'trials': trials}, indent=2),
        encoding='utf-8',
    )
    print('Sweep directory: %s' % sweep_dir)
    print('%s trials x %s reps (horizon-only, no persist)' % (len(trials), args.reps))

    failures = []
    for trial in trials:
        if not run_trial(sweep_dir, trial, args.reps, args.quick):
            failures.append(trial['name'])
            write_comparison(sweep_dir, trials)
            if not args.keep_going:
                print('Stopping after first failure (pass --keep-going to continue).')
                break
        else:
            write_comparison(sweep_dir, trials)

    if failures:
        print('Finished with failures: %s' % ', '.join(failures))
        sys.exit(1)
    print('Sweep finished: %s' % sweep_dir)


if __name__ == '__main__':
    main()
