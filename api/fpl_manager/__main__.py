"""Train the masked manager from the debugger or `python -m fpl_manager`."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from flask import Flask

from persistence import DB_Handler

from fpl_manager.multi import MultiSeasonRun, note_completed_runs
from fpl_manager.season_source import load_managed_season
from fpl_manager.train import train_policy
from fpl_manager.workers import multi_process_env


# A launch with no step budget keeps training until the debugger is stopped.
# Resume adds this many steps on top of the saved counter.
_UNTIL_STOPPED = 10 ** 12


def _season_year(value: str) -> int:
    year = int(value)
    return year % 100 if year >= 2000 else year


def _season_list(value: str):
    years = [_season_year(part) for part in str(value).split(',') if part.strip()]
    if not years:
        raise argparse.ArgumentTypeError('name at least one season')
    return years


def build_parser():
    parser = argparse.ArgumentParser(description='Train the fantasy manager policy')
    parser.add_argument('--season', type=_season_year, default=25)
    parser.add_argument('--seasons', type=_season_list)
    parser.add_argument('--prediction-set-id', type=int)
    parser.add_argument('--from-gw', type=int, default=1)
    parser.add_argument('--to-gw', type=int, default=38)
    parser.add_argument('--timesteps', type=int)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint', default='fpl_manager_runs/weekly')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    app = Flask(__name__)
    with app.app_context():
        db = DB_Handler()
        try:
            years = list(args.seasons or [args.season])
            seasons = [
                load_managed_season(
                    db, args.prediction_set_id, year, args.from_gw, args.to_gw,
                )
                for year in years
            ]
            multi = len(seasons) > 1
            workers = max(1, int(args.workers))
            if workers > 1 and not multi:
                raise ValueError('Several processes are for a multi-season run.')
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
            if multi and decisions:
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
                % (described, budget_text, args.gamma, args.checkpoint)
            )
            print(
                'Each gameweek loads its saved snapshot, built only from earlier '
                'weeks. The squad optimiser plans 6 weeks ahead. Points awarded '
                'each week are that season\'s historical totals.'
            )
            env_fns = None
            if multi:
                print(
                    'Each run plays all %s seasons. The reward is their summed '
                    'points and finish percentiles, paid once at the end of the run.'
                    % len(seasons)
                )
                if workers > 1:
                    print(
                        '%s processes play at the same time. They share this policy '
                        'and the squad cache. The weights update when all %s runs '
                        'have finished.'
                        % (workers, workers)
                    )
                    env_fns = [
                        partial(
                            multi_process_env,
                            tuple(int(year) for year in years),
                            int(args.from_gw),
                            int(args.to_gw),
                            args.prediction_set_id,
                            args.checkpoint,
                        )
                        for _worker in range(workers)
                    ]
                    factory = None
                else:
                    log_path = Path(args.checkpoint) / 'runs.csv'
                    factory = lambda: MultiSeasonRun(seasons, log_path=log_path)
                n_steps = decisions
            else:
                factory = seasons[0].episode
                n_steps = 38
            train_policy(
                factory,
                args.checkpoint,
                total_timesteps=_UNTIL_STOPPED if args.timesteps is None else args.timesteps,
                resume=resume,
                gamma=args.gamma,
                seed=args.seed,
                n_steps=n_steps,
                batch_size=n_steps if multi else 38,
                env_fns=env_fns,
            )
        finally:
            db.close_connection()
    print('Fantasy manager checkpoint saved to %s' % args.checkpoint)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
