import argparse
import json
from pathlib import Path

from flask import Flask

from persistence import DB_Handler
from simulation.sim_runner import SimRunner, SimulationConfig
from simulation.sweep import TRIAL_BY_NAME, config_kwargs, trial_overrides


# The grid's baseline cell: both sweep axes at their middle value.
TRIAL = TRIAL_BY_NAME['02_hz4']


def _seasons(value):
    result = []
    for part in str(value).split(','):
        year = int(part.strip())
        result.append(year % 100 if year >= 2000 else year)
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description='Replay historical FPL seasons from cached predictions'
    )
    parser.add_argument('--prediction-set-id', type=int)
    parser.add_argument(
        '--seasons',
        '--season',
        dest='seasons',
        type=_seasons,
        default=[23, 24, 25],
    )
    parser.add_argument('--output-dir')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    extras = trial_overrides(TRIAL)
    print(
        'In-process trial %s  %s'
        % (
            TRIAL['name'],
            ' '.join('%s=%s' % (key, extras[key]) for key in sorted(extras)),
        )
    )
    app = Flask(__name__)
    summaries = []
    with app.app_context():
        db_handler = DB_Handler()
        try:
            for season in args.seasons:
                prediction_set_id = args.prediction_set_id
                if prediction_set_id is None:
                    prediction_set_id = db_handler.latest_prediction_set_id(season)
                if prediction_set_id is None:
                    raise ValueError(
                        f'No complete prediction set exists for {season}/{season + 1}'
                    )
                kwargs = config_kwargs(TRIAL, prediction_set_id, season)
                if args.output_dir:
                    kwargs['output_dir'] = str(
                        Path(args.output_dir) / f'season_{season}_{season + 1}'
                    )
                runner = SimRunner(
                    SimulationConfig(**kwargs),
                    db_handler=db_handler,
                )
                summaries.append(runner.run())
        finally:
            db_handler.close_connection()
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
