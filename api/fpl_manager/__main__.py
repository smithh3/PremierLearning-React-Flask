"""Train the masked manager from the debugger or `python -m fpl_manager`."""
from __future__ import annotations

import argparse
import gc
import json
from functools import partial
from pathlib import Path

from flask import Flask

from persistence import DB_Handler

from fpl_manager.multi import MultiSeasonRun, SeasonByRun, note_completed_runs
from fpl_manager.observation import features_for, version_for_features
from fpl_manager.season_source import (
    CALENDARS,
    default_calendar,
    is_oracle_set,
    load_managed_season,
)
from fpl_manager.train import CheckpointMismatch, train_policy
from fpl_manager.workers import multi_process_env


# A launch with no step budget keeps training until the debugger is stopped.
# Resume adds this many steps on top of the saved counter.
_UNTIL_STOPPED = 10 ** 12

_ORACLE_NOTICE = (
    'ORACLE RUN - THIS CHEATS. These forecasts were fed the real minutes every player '
    'went on to play, so results here are an upper bound, not a usable manager.'
)


def _season_year(value: str) -> int:
    year = int(value)
    return year % 100 if year >= 2000 else year


def _season_list(value: str):
    years = [_season_year(part) for part in str(value).split(',') if part.strip()]
    if not years:
        raise argparse.ArgumentTypeError('name at least one season')
    return years


def _copy_list(value: str):
    numbers = []
    for part in str(value).split(','):
        part = part.strip()
        if '-' in part:
            low, high = part.split('-')
            numbers.extend(range(int(low), int(high) + 1))
        elif part:
            numbers.append(int(part))
    return tuple(numbers)


def build_parser():
    parser = argparse.ArgumentParser(description='Train the fantasy manager policy')
    parser.add_argument('--season', type=_season_year, default=25)
    parser.add_argument('--seasons', type=_season_list)
    parser.add_argument('--prediction-set-id', type=int)
    parser.add_argument('--from-gw', type=int, default=1)
    parser.add_argument('--to-gw', type=int, default=38)
    parser.add_argument('--timesteps', type=int)
    parser.add_argument(
        '--gamma', type=float,
        help='Discount. Default 0.99, or 0.995 for v3.',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint', default='fpl_manager_runs/weekly')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--observation', choices=('v1', 'v2', 'fast', 'v3'))
    parser.add_argument(
        '--solver', choices=('cbc', 'highs'),
        help='Squad solver. highs tries HiGHS first and hands hard problems to CBC. '
             'Default cbc, or highs for v3.',
    )
    parser.add_argument(
        '--ranged-cache', action='store_true', default=None,
        help='Store squad answers by money range (always on for v3; optional for v2).',
    )
    parser.add_argument(
        '--runs-per-update', type=int,
        help='Runs played between weight updates, across all workers. '
             'Default one per worker, or 8 for v3.',
    )
    parser.add_argument(
        '--gae-lambda', type=float,
        help='How far credit spreads back. Default 0.95, or 0.98 for v3.',
    )
    parser.add_argument(
        '--evaluate-every', type=int,
        help='Play one top-choice run (eval.csv) after every this many updates. '
             '0 turns it off. Default 0, or 5 for v3 (one per 40 training runs).',
    )
    parser.add_argument(
        '--calendar', choices=CALENDARS,
        help='known: each fixture in the gameweek FPL showed at that deadline, so blanks '
             'and doubles appear only once FPL announced them. final: the gameweek it was '
             'finally played in (knows reschedules early). Default known for v3, else final.',
    )
    parser.add_argument(
        '--copies', type=_copy_list, default=(),
        help='v3: synthetic copies to train on, e.g. 1-5. Each season of a run is then '
             'the real one or one of these, at random (synthetic_seasons/training.py).',
    )
    parser.add_argument(
        '--heldout-copies', type=_copy_list, default=(),
        help='With --copies: copies never trained on; evaluation runs cycle through them.',
    )
    parser.add_argument(
        '--allow-oracle', action='store_true',
        help='Accept a set fed real future minutes. The checkpoint name must say oracle.',
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    app = Flask(__name__)
    with app.app_context():
        db = DB_Handler()
        try:
            years = list(args.seasons or [args.season])
            version = _observation_version(args.checkpoint, args.observation)
            if version == 'fast' and years != [25]:
                raise ValueError('The fast run plays the 2025/26 season.')
            v3 = version == 'v3'
            gamma = args.gamma if args.gamma is not None else (0.995 if v3 else 0.99)
            gae_lambda = args.gae_lambda if args.gae_lambda is not None else (0.98 if v3 else 0.95)
            evaluate_every = (
                args.evaluate_every if args.evaluate_every is not None else (5 if v3 else 0)
            )
            solver = args.solver or ('highs' if v3 else 'cbc')
            ranged = True if v3 else bool(args.ranged_cache)
            calendar = args.calendar or default_calendar(version)
            if args.copies and not v3:
                raise ValueError('Training on synthetic copies is a v3 run.')
            if args.heldout_copies and not args.copies:
                raise ValueError('--heldout-copies needs --copies.')
            oracle = args.allow_oracle and is_oracle_set(db, args.prediction_set_id)
            if oracle and 'oracle' not in Path(args.checkpoint).name.lower():
                raise ValueError('An oracle run needs oracle in its checkpoint name.')
            seasons = [
                load_managed_season(
                    db, args.prediction_set_id, year, args.from_gw, args.to_gw,
                    observation_version=version, allow_oracle=args.allow_oracle,
                    solver=solver, ranged=ranged, calendar=calendar,
                )
                for year in years
            ]
            _write_features(args.checkpoint, version)
            if oracle:
                print(_ORACLE_NOTICE)
                (Path(args.checkpoint) / 'ORACLE_REAL_MINUTES.txt').write_text(_ORACLE_NOTICE + '\n')
            multi = len(seasons) > 1
            workers = max(1, int(args.workers))
            logged = multi or version == 'fast'
            if workers > 1 and not logged:
                raise ValueError('Several processes are for a logged run.')
            decisions = sum(season.last - season.first + 1 for season in seasons)
            manifest_path = Path(args.checkpoint) / 'manifest.json'
            resume = bool(args.resume or manifest_path.is_file())
            saved_steps = 0
            if resume and manifest_path.is_file():
                saved_steps = int(json.loads(manifest_path.read_text()).get('num_timesteps') or 0)
                print(
                    'Continuing the saved policy at %s from step %s.'
                    % (args.checkpoint, saved_steps)
                )
            if logged and decisions:
                note_completed_runs(args.checkpoint, saved_steps // decisions)
            described = ', '.join(
                '%s/%s set %s GW%s-%s' % (
                    season.season,
                    season.season + 1,
                    season.prediction_set_id,
                    season.first,
                    season.last,
                )
                for season in seasons
            )
            budget_text = (
                'until you stop it' if args.timesteps is None
                else '%s timesteps' % args.timesteps
            )
            print(
                'Fantasy manager %s, %s, gamma %s, checkpoint %s'
                % (described, budget_text, gamma, args.checkpoint)
            )
            print(
                'Fixture calendar: known. Each week\'s forecasts use the gameweeks FPL '
                'showed at that deadline, so blanks and doubles appear only once FPL '
                'announced them.'
                if calendar == 'known' else
                'Fixture calendar: final. Forecasts use the gameweek each fixture was '
                'finally played in, so the manager knows reschedules before they happen.'
            )
            if v3:
                print(
                    'v3. Each gameweek loads its saved snapshot, built only from earlier '
                    'weeks. The inputs compare this week with the best week for '
                    'transfers, wildcard, free hit, bench boost and triple captain, '
                    'using stored squads where they can. The action chosen is solved '
                    'exactly, %s first, planning 5 weeks ahead with no hits. Points '
                    'awarded are that season\'s historical totals with autosubs. Each '
                    'season is its own episode: a quarter of each week\'s points is paid '
                    'that week, and the season\'s points and finish percentile at its '
                    'end. GAE lambda %s. %s'
                    % (
                        'HiGHS' if solver == 'highs' else 'CBC',
                        gae_lambda,
                        'A top-choice run goes to eval.csv after every %s updates.'
                        % evaluate_every if evaluate_every else 'No evaluation runs.',
                    )
                )
            elif version == 'fast':
                print(
                    'Each gameweek loads its saved snapshot, built only from earlier '
                    'weeks. The squad optimiser '
                    'plans 5 weeks ahead and does not take hits: it rolls one week, '
                    'then spends the free transfers, and it prices a wildcard played '
                    'this week. A cached squad is reused when the team value is within '
                    '£0.5m. Points awarded each week are that season\'s historical totals.'
                )
            elif version == 'v2':
                print(
                    'Each gameweek loads its saved snapshot, built only from earlier '
                    'weeks. The squad optimiser plans 5 weeks ahead and does not '
                    'take hits. A 10-week column compares the squad held today with '
                    'a wildcard five weeks ahead. Points awarded each week are that '
                    'season\'s historical totals. Saved squads from the other run '
                    'stay in the same database and are not reused.'
                )
            else:
                print(
                    'Each gameweek loads its saved snapshot, built only from earlier '
                    'weeks. The squad optimiser plans 6 weeks ahead. Points awarded '
                    'each week are that season\'s historical totals.'
                )
            env_fns = None
            if logged:
                if v3:
                    print(
                        'Each run plays all %s seasons and is one row in runs.csv.'
                        % len(seasons)
                    )
                if args.copies:
                    from synthetic_seasons.training import describe

                    print(describe(args.copies, args.heldout_copies))
                elif multi:
                    print(
                        'Each run plays all %s seasons. The reward is their summed '
                        'points and finish percentiles, paid once at the end of the run.'
                        % len(seasons)
                    )
                else:
                    print(
                        'Each run is written to runs.csv. The reward is that '
                        'season\'s points and finish percentile, paid once at the end.'
                    )
                if workers > 1:
                    print(
                        '%s processes play at the same time. They share this policy '
                        'and the squad cache.' % workers
                    )
                    env_fns = [
                        partial(
                            multi_process_env,
                            tuple(int(year) for year in years),
                            int(args.from_gw),
                            int(args.to_gw),
                            args.prediction_set_id,
                            args.checkpoint,
                            version,
                            args.allow_oracle,
                            solver,
                            ranged,
                            calendar=calendar,
                            copies=tuple(args.copies),
                            heldout_copies=tuple(args.heldout_copies),
                        )
                        for _worker in range(workers)
                    ]
                    factory = None
                else:
                    log_path = Path(args.checkpoint) / 'runs.csv'
                    if args.copies:
                        from synthetic_seasons.training import mixed_factory

                        factory = mixed_factory(
                            db, seasons, args.copies, args.heldout_copies, log_path, seed=args.seed,
                        )
                    elif v3:
                        factory = SeasonByRun(seasons, log_path=log_path)
                    else:
                        factory = lambda: MultiSeasonRun(seasons, log_path=log_path)
                runs_per_update = args.runs_per_update or (8 if v3 else workers)
                runs_per_worker = max(1, int(runs_per_update) // workers)
                n_steps = decisions * runs_per_worker
                print(
                    'The weights update after every %s runs (%s per process).'
                    % (runs_per_worker * workers, runs_per_worker)
                )
            else:
                factory = seasons[0].episode
                n_steps = 38
            if env_fns is not None:
                # The worker processes load their own seasons. This process only
                # updates the policy, so it lets go of its copy.
                seasons = None
                gc.collect()
            train_policy(
                factory,
                args.checkpoint,
                total_timesteps=_UNTIL_STOPPED if args.timesteps is None else args.timesteps,
                resume=resume,
                gamma=gamma,
                seed=args.seed,
                n_steps=n_steps,
                batch_size=_batch_size(n_steps, decisions, v3) if logged else 38,
                env_fns=env_fns,
                features=features_for(version),
                gae_lambda=gae_lambda,
                evaluate_every=evaluate_every,
            )
        finally:
            db.close_connection()
    print('Fantasy manager checkpoint saved to %s' % args.checkpoint)
    return 0


def _batch_size(n_steps, decisions, v3) -> int:
    """v1/v2 learn from each worker's whole rollout at once; v3 in two-run pieces."""
    if v3:
        return int(min(int(n_steps), 2 * int(decisions)))
    return int(n_steps)


def _observation_version(checkpoint, requested):
    path = Path(checkpoint) / 'manifest.json'
    if not path.is_file():
        return requested or 'v1'
    saved = version_for_features(json.loads(path.read_text()).get('features'))
    if not saved:
        raise CheckpointMismatch(
            'Observation fields in %s do not match a known config' % checkpoint
        )
    if requested and requested != saved:
        raise CheckpointMismatch(
            'Checkpoint %s is observation %s. Start %s in a new directory.'
            % (checkpoint, saved, requested)
        )
    return saved


def _write_features(checkpoint, version):
    directory = Path(checkpoint)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        'observation': version,
        'features': list(features_for(version)),
    }
    (directory / 'features.json').write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
