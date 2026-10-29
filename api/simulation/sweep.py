"""20-trial sweep: map the horizon / chip-bar surface now the two are separable.

The previous 40-trial grid found that 19 of its trials were byte-identical to
01_mid in 25/26: every DGW-wait weight, the whole H2 multiplier family, the
early-WC delay, and light rotation changed nothing. What did move was the
transfer horizon (hz 6 was +75, hz 3 was -75) and the WC bar. Those two were
the same lever: chip gaps are summed over the horizon and compared to an
absolute bar, so hz 3 turned a 45 bar into 15 points a week while hz 6 turned
it into 7.5. That is why hz 3 pushed the first wildcard from GW8 to GW12, and
why rotation weight looked non-monotonic - at 6.0 it discounted the bar enough
to jump the chip to GW4, at 3.0 it only made the old week marginal.

Bars are now quoted over CHIP_BAR_REFERENCE_HORIZON weeks and rescaled to the
horizon in use, so this grid can vary them independently:
  - 10 trials on the horizon x bar surface, as a star: both axes through the
    baseline plus four corners, which is enough to tell an additive surface
    from one where the right bar depends on the horizon
  - 5 decompose last run's winner into horizon, patience and rotation
  - 2 ask whether rotation weight is anything more than a bar discount
  - 2 on the FH bar, 1 canary on the DGW-wait family

Planning cost grows fast with the horizon, so the horizon axis is 3 to 6 and
every trial that is not about the horizon sits at 4. 02_hz4 reproduces the
previous grid's 01_mid bars exactly (quoted values are the old ones times
5/4), which keeps the two runs comparable. Defaults to 25/26, 24/25, 23/24.

From api/:
  python -m simulation.sweep
  python -m simulation.sweep --season 24
  python -m simulation.sweep --from 2
  python -m simulation.sweep --only 2
  python -m simulation.sweep --resume
  python -m simulation.sweep --dry-run
  python -m simulation.sweep --workers 4
"""

import argparse
import atexit
import contextlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask

from persistence import DB_Handler
from simulation.sim_runner import SimRunner, SimulationConfig
from squadtools.chip_strategy import CHIP_BAR_REFERENCE_HORIZON
from squadtools.simulation_strategy import MIN_GAIN_REFERENCE_HORIZON


SEASON = 25


def _bar(points_per_week):
    """Quote a chip bar from the gain per week it should demand."""
    return round(float(points_per_week) * CHIP_BAR_REFERENCE_HORIZON, 4)


def _gain(points_per_week):
    """Quote min_horizon_gain from the gain per week it should demand."""
    return round(float(points_per_week) * MIN_GAIN_REFERENCE_HORIZON, 4)


# Knobs the last grid proved inert in 25/26 start at zero here, so a trial that
# turns one on is the only place it can be blamed. Bars are per-week values so
# the horizon axis leaves them alone.
SHARED = dict(
    horizon_gameweeks=5,
    min_current_gw_gain=1.0,
    min_horizon_gain=_gain(0.75),
    max_transfers_per_gameweek=5,
    relax_thresholds_at_max_ft=False,
    chip_strategy=True,
    fh_threshold=_bar(2.5),
    wc_threshold=_bar(11.25),
    fh_recovery_threshold=_bar(2.5),
    tc_above_average_threshold=1.6,
    tc_window=8,
    minutes_scale=None,
    bench_fill_weight=0.0,
    bench_weights=(),
    reserve_keeper_weight=None,
    earliest_chip_gameweek=3,
    roll_free_transfers=True,
    fh_dgw_weight=0.0,
    wc_dgw_weight=0.0,
    dgw_h2_mult=2.0,
    wc_rotation_weight=0.0,
    wc_early_weight=0.0,
)

TRIAL_FIELDS = {
    'wc': 'wc_threshold',
    'fh': 'fh_threshold',
    'mh': 'min_horizon_gain',
    'mc': 'min_current_gw_gain',
    'tc': 'tc_above_average_threshold',
    'hz': 'horizon_gameweeks',
    'hits': 'max_transfers_per_gameweek',
    'relax': 'relax_thresholds_at_max_ft',
    'win': 'tc_window',
    'fill': 'bench_fill_weight',
    'slots': 'bench_weights',
    'gk': 'reserve_keeper_weight',
    'rec': 'fh_recovery_threshold',
    'earliest': 'earliest_chip_gameweek',
    'ms': 'minutes_scale',
    'fh_gap': 'fh_after_wildcard_weeks',
    'tc_floor': 'tc_taper_floor',
    'tc_min': 'tc_min_captain_points',
    'tc_easy': 'tc_easy_until_week',
    'roll': 'roll_free_transfers',
    'fh_dgw': 'fh_dgw_weight',
    'wc_dgw': 'wc_dgw_weight',
    'dgw_h2': 'dgw_h2_mult',
    'wc_rot': 'wc_rotation_weight',
    'wc_early': 'wc_early_weight',
}

# Every trial states its horizon and WC bar: those two are the grid's axes, so
# the comparison table should show them even when they are the baseline value.
# The baseline horizon is 4, both because planning cost grows steeply with it
# (measured on GW1-2 of 25/26: 146s at hz 4, 266s at 5, 443s at 6, over 950s
# at 8) and because the previous grid ran at 4, so these cells stay comparable.
_MID = dict(hz=4, wc=_bar(11.25))
# Per-week WC bars. 11.25 is last grid's 01_mid; the last run showed 7.5 and 15
# are both clearly worse, so this brackets the optimum rather than spanning it.
_WC_LO = _bar(9.0)
_WC_MID = _bar(11.25)
_WC_HI = _bar(13.5)
# 1.5 a week per rotting player: four of them discount the WC bar by 6 a week,
# which is what pulled the wildcard to GW4 last run.
_ROT_HI = _bar(1.5)


def _t(name, **extra):
    trial = dict(_MID)
    trial.update(extra)
    trial['name'] = name
    return trial


# 20 trials so all three seasons fit in one run. The horizon x bar block is a
# star rather than a full grid: both axes through the baseline, then the four
# corners. If the corners match what the axes predict on their own, the two
# knobs are separable and the 15-cell grid was never needed.
TRIALS = (
    # Horizon at the baseline bar. 8 is left out: it more than doubles hz 6's
    # already 3x cost and the axis reads fine without it.
    _t('01_hz3', hz=3),
    _t('02_hz4'),
    _t('03_hz5', hz=5),
    _t('04_hz6', hz=6),
    # Per-week WC bar at the baseline horizon.
    _t('05_wc_lo', wc=_WC_LO),
    _t('06_wc_hi', wc=_WC_HI),
    # Corners across the widest horizon span we can afford. If these land
    # where the two axes predict on their own, the knobs are separable.
    _t('07_hz3_lo', hz=3, wc=_WC_LO),
    _t('08_hz3_hi', hz=3, wc=_WC_HI),
    _t('09_hz6_lo', hz=6, wc=_WC_LO),
    _t('10_hz6_hi', hz=6, wc=_WC_HI),
    # Last run's winner, decomposed. 04_hz6 is the third arm, so each factor
    # is measurable alone and the pair says whether they interact.
    _t('11_mh_off_hz6', hz=6, mh=_gain(0.0)),
    _t('12_mh_max_hz6', hz=6, mh=_gain(2.0)),
    _t('13_rot_hi_hz6', hz=6, wc_rot=_ROT_HI),
    _t('14_stuck_hz6', hz=6, mh=_gain(2.0), wc_rot=_ROT_HI),
    # Does patience need the long horizon, or does it stand on its own?
    _t('15_mh_max_hz4', mh=_gain(2.0)),
    # Rotation weight against the flat bar cut it is worth. Four rotting
    # players at 1.5 a week is the same 6 a week that 17 removes outright, so
    # matching weeks would mean rotation is a bar discount under another name.
    _t('16_rot_hi', wc_rot=_ROT_HI),
    _t('17_wc_equiv', wc=_bar(5.25)),
    # Free hit bar, which the last grid only moved at one horizon.
    _t('18_fh_lo', fh=_bar(1.0)),
    _t('19_fh_hi', fh=_bar(4.0)),
    # The DGW-wait family was inert in 25/26 but moved 24/25 by 18 points, so
    # one extreme trial keeps it under observation across all three seasons.
    _t('20_dgw_canary', fh_dgw=_bar(2.5), wc_dgw=_bar(5.0), dgw_h2=6.0),
)

TRIAL_BY_NAME = {trial['name']: trial for trial in TRIALS}

# Pre-sweep single-run knobs (WC 30 / FH 11 / horizon 5). Not part of the grid.
BASELINE = dict(
    name='00_baseline',
    roll=False,
    fh_dgw=0.0,
    wc_dgw=0.0,
    dgw_h2=2.0,
    wc_rot=0.0,
    wc_early=0.0,
    earliest=5,
    hz=5,
    mh=2.0,
    mc=0.0,
    wc=30.0,
    fh=11.0,
    rec=8.0,
    tc=1.3,
    win=10,
    fill=0.3,
    slots=(0.5, 0.3, 0.1),
    gk=0.2,
    hits=5,
    relax=False,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'run_metrics' / 'simulations'
JSONL_PATH = RESULTS_DIR / 'hyperparam_sweep.jsonl'
TABLE_PATH = RESULTS_DIR / 'hyperparam_sweep.txt'
_RESULTS_TAG = None

# Last 20-trial run's variants worth replaying under a new prediction model:
# baseline, best 25/26, long hz, high FH bar, short hz (2nd on 24/25).
H1_POISSON_TRIALS = (
    '02_hz4',
    '15_mh_max_hz4',
    '04_hz6',
    '19_fh_hi',
    '01_hz3',
)

# Horizon-model cells that beat or matched Poisson-h1 on 25/26, or won 24/25,
# and were not in the first Poisson replay. 15_mh_max_hz4 already ran (and
# collapsed to the baseline); 17_wc_equiv was identical to 16_rot_hi in 25/26.
H1_POISSON_OLD_WINNER_TRIALS = (
    '11_mh_off_hz6',
    '09_hz6_lo',
    '16_rot_hi',
    '13_rot_hi_hz6',
    '06_wc_hi',
)


def apply_results_tag(tag):
    """Write this sweep's logs under run_metrics/simulations/<tag>/."""
    global RESULTS_DIR, JSONL_PATH, TABLE_PATH, _RESULTS_TAG
    root = Path(__file__).resolve().parents[1] / 'run_metrics' / 'simulations'
    safe = None
    if tag:
        safe = ''.join(
            ch if ch.isalnum() or ch in '-_' else '-' for ch in str(tag)
        )
        if not safe:
            raise SystemExit('--tag must contain a letter or number')
        root = root / safe
    _RESULTS_TAG = safe
    RESULTS_DIR = root
    JSONL_PATH = RESULTS_DIR / 'hyperparam_sweep.jsonl'
    TABLE_PATH = RESULTS_DIR / 'hyperparam_sweep.txt'
    return RESULTS_DIR


def trial_overrides(trial):
    return {key: trial[key] for key in trial if key != 'name'}


def config_kwargs(trial, prediction_set_id, season):
    kwargs = dict(SHARED)
    kwargs.update(
        prediction_set_id=int(prediction_set_id),
        season_start_year=int(season),
        gameweek_from=1,
        gameweek_to=38,
    )
    for short, field in TRIAL_FIELDS.items():
        if short in trial:
            kwargs[field] = trial[short]
    return kwargs


def _options_label(options):
    if not options:
        return 'defaults'
    parts = []
    for key in sorted(options):
        parts.append('%s=%s' % (key, options[key]))
    return ' '.join(parts)


def chips_used(logs):
    used = {}
    for row in logs:
        chip = row.get('chip')
        if chip:
            used.setdefault(chip, []).append(int(row['gameweek']))
    return {chip: weeks for chip, weeks in used.items()}


def _format_chips(used):
    parts = []
    for chip in ('wildcard', 'bench_boost', 'free_hit', 'triple_captain'):
        weeks = used.get(chip) or []
        if weeks:
            short = {
                'wildcard': 'WC',
                'bench_boost': 'BB',
                'free_hit': 'FH',
                'triple_captain': 'TC',
            }[chip]
            parts.append('%s %s' % (short, ','.join(str(week) for week in weeks)))
    return ' '.join(parts) if parts else '-'


def _append_jsonl(row):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open('a', encoding='utf-8') as file:
        file.write(json.dumps(row, sort_keys=True) + '\n')


def _load_rows():
    if not JSONL_PATH.is_file():
        return []
    rows = []
    for line in JSONL_PATH.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _season_number(row):
    """Season start year. Older rows stored the label ('25/26') in this field."""
    value = row.get('season')
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    try:
        return int(str(value).split('/')[0].strip())
    except ValueError:
        return None


def _top_percent(row):
    info = row.get('overall_rank') or {}
    value = info.get('top_percent')
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def latest_by_trial_season(rows):
    """Newest row per (trial, season), so re-runs replace earlier attempts."""
    latest = {}
    for row in rows:
        season = _season_number(row)
        if season is None:
            continue
        latest[(row.get('name') or '-', season)] = row
    return latest


def consistency_lines(rows):
    """Per-trial top% in each season, to show which knobs travel between years.

    Points are not comparable across seasons, so rank against that season's
    real FPL field is the yardstick. Sorted by mean top%, best first.
    """
    latest = latest_by_trial_season(rows)
    if not latest:
        return []
    seasons = sorted({season for _name, season in latest})
    names = sorted({name for name, _season in latest})
    if len(seasons) < 2:
        return []
    header = '%-22s %2s' % ('trial', 'n')
    header += ''.join('  %6s' % ('s%s' % season) for season in seasons)
    header += '  %6s %6s %6s %7s' % ('mean', 'worst', 'spread', 'mean pts')
    body = []
    for name in names:
        cells = []
        tops = []
        points = []
        for season in seasons:
            row = latest.get((name, season))
            top = None if row is None else _top_percent(row)
            if row is not None:
                try:
                    points.append(float(row.get('total_points') or 0))
                except (TypeError, ValueError):
                    pass
            if top is None:
                cells.append('%6s' % '-')
            else:
                cells.append('%6.2f' % top)
                tops.append(top)
        line = '%-22s %2s' % (name[:22], len(tops))
        line += ''.join('  %s' % cell for cell in cells)
        if tops:
            mean = sum(tops) / len(tops)
            line += '  %6.2f %6.2f %6.2f' % (mean, max(tops), max(tops) - min(tops))
        else:
            mean = float('inf')
            line += '  %6s %6s %6s' % ('-', '-', '-')
        line += '  %7s' % (
            '%.0f' % (sum(points) / len(points)) if points else '-'
        )
        body.append((mean, name, line))
    body.sort(key=lambda item: (item[0], item[1]))
    return [
        '',
        'Per-trial consistency across seasons (top%% of the FPL field, lower is better)',
        'Newest run per trial and season only. Blank means that season has not run.',
        '',
        header,
    ] + [line for _mean, _name, line in body]


def write_comparison(rows=None):
    rows = list(_load_rows() if rows is None else rows)
    rows.sort(
        key=lambda row: (-float(row.get('total_points') or 0), row.get('name') or '')
    )
    lines = [
        'Simulation hyperparameter sweep',
        'Append-only log: %s' % JSONL_PATH.name,
        '',
        '%-4s %5s %3s %8s %6s %4s %5s %-22s %-32s %s'
        % ('#', 'pts', 'szn', 'rank', 'top%', 'trn', 'hold', 'label', 'knobs', 'chips'),
    ]
    for index, row in enumerate(rows, start=1):
        rank_info = row.get('overall_rank') or {}
        rank = rank_info.get('rank')
        top = rank_info.get('top_percent')
        season = _season_number(row)
        lines.append(
            '%-4s %5.0f %3s %8s %6s %4s %5s %-22s %-32s %s'
            % (
                index,
                float(row.get('total_points') or 0),
                '-' if season is None else season,
                '-' if rank is None else int(rank),
                '-' if top is None else '%.2f' % float(top),
                row.get('transfers_made', '-'),
                row.get('hold_gameweeks', '-'),
                (row.get('name') or '')[:22],
                _options_label(row.get('options') or {})[:32],
                row.get('chips_label') or _format_chips(row.get('chips') or {}),
            )
        )
    if not rows:
        lines.append('(no finished trials yet)')
    lines.extend(consistency_lines(rows))
    text = '\n'.join(lines) + '\n'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(text, encoding='utf-8')
    return text


def _describe_prediction_set(db_handler, prediction_set_id):
    meta = db_handler.get_prediction_set(prediction_set_id)
    if meta is None:
        print('Using prediction set %s (no metadata row)' % prediction_set_id)
        return
    config = meta.get('config') or {}
    print(
        'Using prediction set %s name=%s hash=%s'
        % (
            meta['id'],
            meta.get('name') or '-',
            (meta.get('model_config_hash') or '')[:12],
        )
    )
    print(
        '  training=%s minutes_reps=%s horizon_reps=%s points_loss=%s '
        'mw=%s horizon_model=%s team_expectancies=%s'
        % (
            config.get('training_schedule') or meta.get('training_policy'),
            config.get('minutes_reps'),
            config.get('horizon_reps'),
            config.get('points_loss') or 'huber',
            config.get('points_minutes_weight'),
            config.get('horizon_model'),
            config.get('team_expectancies'),
        )
    )
    minutes_reps = config.get('minutes_reps')
    horizon_reps = config.get('horizon_reps')
    if minutes_reps not in (None, 5):
        print(
            'WARNING: cache minutes_reps=%s (want 5). Rebuild with '
            'simulation.prediction_builder --minutes-reps 5'
            % minutes_reps
        )
    if config.get('horizon_model', True) and horizon_reps not in (None, 5):
        print(
            'WARNING: cache horizon_reps=%s (want 5). Rebuild with '
            '--horizon-reps 5'
            % horizon_reps
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description='20-trial sim sweep: horizon against the chip bar'
    )
    parser.add_argument('--from', dest='start', type=int, default=1)
    parser.add_argument('--only', type=int)
    parser.add_argument(
        '--trials',
        help='Comma-separated trial names or 1-based indices from the grid',
    )
    parser.add_argument(
        '--tag',
        help='Write jsonl/logs under run_metrics/simulations/<tag>/',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help=(
            'Skip trials already finished in hyperparam_sweep.jsonl for this '
            'season. Interrupted in-flight trials restart from GW1.'
        ),
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--prediction-set-id', type=int)
    parser.add_argument(
        '--season',
        type=int,
        default=SEASON,
        help='Season start year to simulate (25 or 24). Default %s.' % SEASON,
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Parallel trial processes. 1 is sequential. 4 uses more CPU.',
    )
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Run the pre-sweep WC 30 / FH 11 config instead of the grid',
    )
    parser.add_argument(
        '--internal-worker',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    return parser


def _parse_trial_tokens(value):
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split(',') if part.strip()]


def _trial_by_token(token):
    if token.isdigit():
        index = int(token)
        if not 1 <= index <= len(TRIALS):
            raise SystemExit('--trials index must be between 1 and %s' % len(TRIALS))
        return index, TRIALS[index - 1]
    trial = TRIAL_BY_NAME.get(token)
    if trial is None:
        raise SystemExit(
            'Unknown trial %s. Names: %s'
            % (token, ', '.join(t['name'] for t in TRIALS))
        )
    index = next(i for i, item in enumerate(TRIALS, start=1) if item['name'] == token)
    return index, trial


def selected_trials(args):
    if getattr(args, 'baseline', False) is True:
        return [(0, BASELINE)]
    if args.only is not None:
        index = int(args.only)
        if not 1 <= index <= len(TRIALS):
            raise SystemExit('--only must be between 1 and %s' % len(TRIALS))
        return [(index, TRIALS[index - 1])]
    tokens = _parse_trial_tokens(getattr(args, 'trials', None))
    if tokens:
        return [_trial_by_token(token) for token in tokens]
    start = int(args.start)
    if not 1 <= start <= len(TRIALS):
        raise SystemExit('--from must be between 1 and %s' % len(TRIALS))
    return [
        (index, trial)
        for index, trial in enumerate(TRIALS, start=1)
        if index >= start
    ]


def remaining_trials(chosen, season, rows=None):
    """Keep trials that have no finished JSONL row for this season.

    In-flight workers never reach the log, so stopping the sweep and
    resuming restarts those four (or however many were running).
    """
    done = {
        name
        for (name, row_season) in latest_by_trial_season(
            rows if rows is not None else _load_rows()
        )
        if row_season == int(season)
    }
    remaining = [
        (index, trial)
        for index, trial in chosen
        if trial['name'] not in done
    ]
    skipped = [
        (index, trial)
        for index, trial in chosen
        if trial['name'] in done
    ]
    return remaining, skipped


def _configure_workers(workers):
    workers = max(1, int(workers))
    # A `threads` option of any value deadlocks the bundled CBC once branch
    # and bound starts, so drop a stale value instead of pinning one.
    os.environ.pop('SIM_CBC_THREADS', None)
    return workers


def _venv_python():
    """Return the venv interpreter, not the system Python that created it."""
    prefix = Path(sys.prefix)
    if os.name == 'nt':
        candidates = (prefix / 'Scripts' / 'python.exe', prefix / 'python.exe')
    else:
        candidates = (prefix / 'bin' / 'python', prefix / 'bin' / 'python3')
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def _looks_like_debugger(executable):
    text = str(executable or '').replace('\\', '/').lower()
    return 'debugpy' in text or 'pydevd' in text


def _worker_python():
    venv_python = _venv_python()
    if venv_python:
        return venv_python
    executable = sys.executable
    if executable and not _looks_like_debugger(executable):
        return executable
    return getattr(sys, '_base_executable', None) or executable


def _use_real_python_executable():
    """Point sys.executable at venv Python, not debugpy or system Python."""
    python = _worker_python()
    if python:
        sys.executable = python


def _strip_debugger_env():
    for key in list(os.environ):
        upper = key.upper()
        if upper.startswith('PYDEVD') or upper.startswith('DEBUGPY'):
            os.environ.pop(key, None)


def _result_path(trial_name, season):
    return RESULTS_DIR / ('%s_s%s.result.json' % (trial_name, season))


def _worker_env():
    env = os.environ.copy()
    for key in list(env):
        upper = key.upper()
        if upper.startswith('PYDEVD') or upper.startswith('DEBUGPY'):
            env.pop(key, None)
    env.pop('PYTHONHOME', None)
    pythonpath = [
        part
        for part in env.get('PYTHONPATH', '').split(os.pathsep)
        if part
        and 'debugpy' not in part.lower()
        and 'pydevd' not in part.lower()
    ]
    if pythonpath:
        env['PYTHONPATH'] = os.pathsep.join(pythonpath)
    else:
        env.pop('PYTHONPATH', None)
    env.pop('SIM_CBC_THREADS', None)
    env['PYTHONUNBUFFERED'] = '1'
    env['VIRTUAL_ENV'] = sys.prefix
    return env


def _worker_cwd():
    return str(Path(__file__).resolve().parents[1])


def _worker_command(index, prediction_set_id, season, trial=None, tag=None):
    cmd = [
        _worker_python(),
        '-m',
        'simulation.sweep',
        '--internal-worker',
        '--season',
        str(season),
        '--prediction-set-id',
        str(prediction_set_id),
        '--workers',
        '1',
    ]
    if trial is not None and trial.get('name') == BASELINE['name']:
        cmd.append('--baseline')
    else:
        cmd.extend(['--only', str(index)])
    if tag is None:
        tag = _RESULTS_TAG
    if tag:
        cmd.extend(['--tag', str(tag)])
    return cmd


JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_HANDLE = None
_JOB_HAS_SELF = False
# Worker processes we spawned, so Ctrl+C does not leave solvers running.
_LIVE_WORKERS = set()


def _kernel32():
    """kernel32 with explicit prototypes.

    Without these, handles are passed as 32-bit ints on x64 and the
    GetCurrentProcess pseudo-handle arrives truncated, so joining the job fails
    with ERROR_INVALID_HANDLE.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
    ]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _kill_workers_with_this_process():
    """Put this process in a job whose members die when we do.

    A trial runs as a grandchild (launcher stub, then Python, then cbc.exe), so
    killing the sweep used to leave the whole tree solving in the background.
    Job membership is inherited by every descendant, and the job kills its
    remaining members as soon as our last handle to it closes, which happens
    however this process goes away, including taskkill /F.
    """
    global _JOB_HANDLE, _JOB_HAS_SELF
    if os.name != 'nt' or _JOB_HANDLE is not None:
        return _JOB_HANDLE
    import ctypes
    from ctypes import wintypes

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ('ReadOperationCount', ctypes.c_ulonglong),
            ('WriteOperationCount', ctypes.c_ulonglong),
            ('OtherOperationCount', ctypes.c_ulonglong),
            ('ReadTransferCount', ctypes.c_ulonglong),
            ('WriteTransferCount', ctypes.c_ulonglong),
            ('OtherTransferCount', ctypes.c_ulonglong),
        ]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ('PerProcessUserTimeLimit', wintypes.LARGE_INTEGER),
            ('PerJobUserTimeLimit', wintypes.LARGE_INTEGER),
            ('LimitFlags', wintypes.DWORD),
            ('MinimumWorkingSetSize', ctypes.c_size_t),
            ('MaximumWorkingSetSize', ctypes.c_size_t),
            ('ActiveProcessLimit', wintypes.DWORD),
            ('Affinity', ctypes.c_size_t),
            ('PriorityClass', wintypes.DWORD),
            ('SchedulingClass', wintypes.DWORD),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ('BasicLimitInformation', _BasicLimits),
            ('IoInfo', _IoCounters),
            ('ProcessMemoryLimit', ctypes.c_size_t),
            ('JobMemoryLimit', ctypes.c_size_t),
            ('PeakProcessMemoryUsed', ctypes.c_size_t),
            ('PeakJobMemoryUsed', ctypes.c_size_t),
        ]

    kernel32 = _kernel32()
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    limits = _ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    extended_limit_information = 9
    if not kernel32.SetInformationJobObject(
        job,
        extended_limit_information,
        ctypes.byref(limits),
        ctypes.sizeof(limits),
    ):
        kernel32.CloseHandle(job)
        return None
    _JOB_HANDLE = job
    # Joining ourselves means descendants inherit membership for free. If the
    # surrounding job forbids nesting we fall back to adding each worker.
    _JOB_HAS_SELF = bool(
        kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())
    )
    return job


def _add_worker_to_job(proc):
    if os.name != 'nt' or _JOB_HANDLE is None or _JOB_HAS_SELF:
        return
    kernel32 = _kernel32()
    process_set_quota_and_terminate = 0x0100 | 0x0001
    handle = kernel32.OpenProcess(
        process_set_quota_and_terminate, False, proc.pid
    )
    if not handle:
        return
    try:
        kernel32.AssignProcessToJobObject(_JOB_HANDLE, handle)
    finally:
        kernel32.CloseHandle(handle)


def kill_live_workers():
    """Stop tracked workers and their solvers on Ctrl+C or a clean exit.

    The job object handles hard kills of this process; this covers the paths
    where we still get to run code. proc.kill() would only reach the launcher
    stub, so on Windows the tree goes through taskkill.
    """
    for proc in list(_LIVE_WORKERS):
        _LIVE_WORKERS.discard(proc)
        try:
            if proc.poll() is not None:
                continue
            if os.name == 'nt':
                subprocess.run(
                    ['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                    capture_output=True,
                )
            else:
                proc.kill()
        except Exception:
            continue


def _popen_worker(cmd):
    from squadtools.simulation_strategy import (
        _restore_unpatched_createprocess,
        ensure_quiet_cbc,
    )

    ensure_quiet_cbc()
    _restore_unpatched_createprocess()
    kwargs = dict(env=_worker_env(), cwd=_worker_cwd())
    if os.name != 'nt':
        proc = subprocess.Popen(cmd, **kwargs)
        _LIVE_WORKERS.add(proc)
        return proc
    _kill_workers_with_this_process()
    # No CREATE_BREAKAWAY_FROM_JOB: leaving the job is what used to strand
    # workers when the sweep was killed or its terminal closed.
    new_group = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x00000200)
    proc = subprocess.Popen(cmd, creationflags=new_group, **kwargs)
    _add_worker_to_job(proc)
    _LIVE_WORKERS.add(proc)
    return proc


def _echo_worker_log(proc, log_path):
    """Copy the child's log to this process so F5 still shows GW output."""
    deadline = time.time() + 30
    while not log_path.is_file():
        if proc.poll() is not None:
            return proc.wait()
        if time.time() > deadline:
            break
        time.sleep(0.05)
    if not log_path.is_file():
        return proc.wait()
    with log_path.open('r', encoding='utf-8') as log:
        while True:
            line = log.readline()
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
                continue
            if proc.poll() is not None:
                rest = log.read()
                if rest:
                    sys.stdout.write(rest)
                    sys.stdout.flush()
                break
            time.sleep(0.05)
    return proc.wait()


def _await_trial_process(job, stream=False):
    index, trial, prediction_set_id, season = job
    result_path = _result_path(trial['name'], season)
    log_path = RESULTS_DIR / ('%s_s%s.log' % (trial['name'], season))
    if result_path.is_file():
        result_path.unlink()
    print('Starting %s' % trial['name'], flush=True)
    if stream:
        print('Live log %s' % log_path, flush=True)
    proc = _popen_worker(
        _worker_command(index, prediction_set_id, season, trial)
    )
    try:
        if stream:
            rc = _echo_worker_log(proc, log_path)
        else:
            rc = proc.wait()
    finally:
        _LIVE_WORKERS.discard(proc)
    if rc != 0 or not result_path.is_file():
        raise RuntimeError(
            'Trial %s exited %s. See %s' % (trial['name'], rc, log_path)
        )
    row = json.loads(result_path.read_text(encoding='utf-8'))
    return index, row, str(log_path)


def _run_internal_worker(args):
    from squadtools.simulation_strategy import ensure_quiet_cbc

    _strip_debugger_env()
    ensure_quiet_cbc()
    chosen = selected_trials(args)
    if len(chosen) != 1:
        raise SystemExit('--internal-worker requires --only')
    if args.prediction_set_id is None:
        raise SystemExit('--internal-worker requires --prediction-set-id')
    index, trial = chosen[0]
    season = int(args.season)
    _index, row, _log_path = _run_trial_job(
        (index, trial, int(args.prediction_set_id), season)
    )
    _result_path(trial['name'], season).write_text(
        json.dumps(row, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return 0


def _trial_row(trial, season, summary, used):
    return {
        'finished_at': datetime.now(timezone.utc).isoformat(),
        'name': trial['name'],
        'season': season,
        'season_label': summary['season'],
        'total_points': summary['total_points'],
        'transfers_made': summary['transfers_made'],
        'hold_gameweeks': summary.get('hold_gameweeks'),
        'simulation_run_id': summary['simulation_run_id'],
        'prediction_set_id': summary['prediction_set_id'],
        'overall_rank': summary.get('overall_rank'),
        'chips': used,
        'chips_label': _format_chips(used),
        'options': trial_overrides(trial),
    }


def _print_trial_ok(row, log_path=None):
    rank_info = row.get('overall_rank') or {}
    rank = rank_info.get('rank')
    top = rank_info.get('top_percent')
    print(
        'OK %s  %s pts  rank %s  top %s%%  %s'
        % (
            row['name'],
            row['total_points'],
            '-' if rank is None else int(rank),
            '-' if top is None else '%.2f' % float(top),
            row.get('chips_label') or _format_chips(row.get('chips') or {}),
        )
    )
    if log_path:
        print('  log %s' % log_path)


def _run_one_trial(trial, prediction_set_id, season, db_handler):
    runner = SimRunner(
        SimulationConfig(**config_kwargs(trial, prediction_set_id, season)),
        db_handler=db_handler,
    )
    summary = runner.run()
    used = chips_used(runner.gameweek_logs)
    return _trial_row(trial, season, summary, used)


def _run_trial_job(job):
    import traceback

    index, trial, prediction_set_id, season = job
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / ('%s_s%s.log' % (trial['name'], season))
    app = Flask(__name__)
    with app.app_context():
        db_handler = DB_Handler()
        try:
            with log_path.open('w', encoding='utf-8', buffering=1) as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    print(
                        '=== %s/%s trial %s %s ==='
                        % (season, season + 1, index, trial['name']),
                        flush=True,
                    )
                    try:
                        row = _run_one_trial(
                            trial, prediction_set_id, season, db_handler
                        )
                    except Exception:
                        traceback.print_exc()
                        raise
            return index, row, str(log_path)
        finally:
            db_handler.close_connection()


def _write_comparison_safely():
    """A broken summary row must not take down a multi-hour sweep."""
    try:
        return write_comparison()
    except Exception as exc:
        print('WARNING: could not write %s: %s' % (TABLE_PATH.name, exc))
        return ''


def _run_trials_parallel(chosen, prediction_set_id, season, workers):
    jobs = [
        (index, trial, prediction_set_id, season)
        for index, trial in chosen
    ]
    print(
        'Running %s trials with %s workers. Per-trial logs in %s'
        % (len(jobs), workers, RESULTS_DIR)
    )
    _use_real_python_executable()
    stream = len(jobs) == 1
    failures = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_await_trial_process, job, stream) for job in jobs
            ]
            for future in as_completed(futures):
                try:
                    _index, row, log_path = future.result()
                except Exception as exc:
                    print('FAIL %s' % exc)
                    failures.append(str(exc))
                    continue
                _append_jsonl(row)
                _write_comparison_safely()
                _print_trial_ok(row, log_path)
    except BaseException:
        # Workers run in their own process group, so Ctrl+C never reaches them.
        kill_live_workers()
        raise
    if failures:
        raise SystemExit(
            '%s trial(s) failed. Logs are in %s' % (len(failures), RESULTS_DIR)
        )


def main(argv=None):
    from squadtools.simulation_strategy import ensure_quiet_cbc

    args = build_parser().parse_args(argv)
    apply_results_tag(args.tag if isinstance(getattr(args, 'tag', None), str) else None)
    if args.internal_worker:
        return _run_internal_worker(args)
    season = int(args.season)
    workers = _configure_workers(args.workers)
    atexit.register(kill_live_workers)
    ensure_quiet_cbc()
    _use_real_python_executable()
    chosen = selected_trials(args)
    skipped = []
    if args.resume:
        chosen, skipped = remaining_trials(chosen, season)
    print(
        '%s/%s  %s trials (%s total in grid)  workers=%s'
        % (season, season + 1, len(chosen), len(TRIALS), workers)
    )
    if skipped:
        print(
            '  resume skip %s finished: %s'
            % (
                len(skipped),
                ', '.join(trial['name'] for _index, trial in skipped),
            )
        )
    for index, trial in chosen:
        extras = trial_overrides(trial)
        print(
            '  %02d %s  %s'
            % (index, trial['name'], _options_label(extras))
        )
    if args.dry_run or not chosen:
        if args.resume and not chosen:
            print('Resume: nothing left for %s/%s' % (season, season + 1))
            if not args.dry_run:
                print('\nSweep finished. Compare in %s' % TABLE_PATH)
                print(_write_comparison_safely().rstrip())
        return 0

    app = Flask(__name__)
    with app.app_context():
        db_handler = DB_Handler()
        try:
            prediction_set_id = args.prediction_set_id
            if prediction_set_id is None:
                prediction_set_id = db_handler.latest_prediction_set_id(season)
            if prediction_set_id is None:
                raise SystemExit(
                    'No complete prediction set exists for %s/%s'
                    % (season, season + 1)
                )
            _describe_prediction_set(db_handler, prediction_set_id)
            db_handler.close_connection()
            db_handler = None
            _run_trials_parallel(
                chosen, prediction_set_id, season, workers
            )
        except KeyboardInterrupt:
            _write_comparison_safely()
            print('Interrupted. Comparison written to %s' % TABLE_PATH)
            raise
        finally:
            if db_handler is not None:
                db_handler.close_connection()
    print('\nSweep finished. Compare in %s' % TABLE_PATH)
    print(_write_comparison_safely().rstrip())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
