import argparse
import json
from pathlib import Path

from flask import Flask

from persistence import DB_Handler
from .sim_runner import SimRunner, SimulationConfig


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
    parser.add_argument('--from-gw', type=int, default=1)
    parser.add_argument('--to-gw', type=int, default=38)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--min-current-gain', type=float, default=0.0)
    parser.add_argument('--min-horizon-gain', type=float, default=0.0)
    parser.add_argument('--max-transfers-per-gw', type=int, default=5)
    parser.add_argument('--output-dir')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not (1 <= args.from_gw <= args.to_gw <= 38):
        raise SystemExit('gameweek range must satisfy 1 <= from <= to <= 38')
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
                output_dir = None
                if args.output_dir:
                    output_dir = str(
                        Path(args.output_dir) / f'season_{season}_{season + 1}'
                    )
                runner = SimRunner(
                    SimulationConfig(
                        prediction_set_id=prediction_set_id,
                        season_start_year=season,
                        gameweek_from=args.from_gw,
                        gameweek_to=args.to_gw,
                        horizon_gameweeks=args.horizon,
                        min_current_gw_gain=args.min_current_gain,
                        min_horizon_gain=args.min_horizon_gain,
                        max_transfers_per_gameweek=args.max_transfers_per_gw,
                        output_dir=output_dir,
                    ),
                    db_handler=db_handler,
                )
                summaries.append(runner.run())
        finally:
            db_handler.close_connection()
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
