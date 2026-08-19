"""Build reusable point-in-time historical prediction snapshots."""

import argparse
import hashlib
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy
from flask import Flask

os.environ.setdefault('PL_SIMULATION_BUILD', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from persistence import DB_Handler
from premierlearning.runner import Runner
from simulation.historical_data import VaastavSeasonData
from simulation.season_model import FixedSeasonModels


DEFAULT_SEASONS = (23, 24, 25)
CODE_HASH_FILES = (
    'premierlearning/runner.py',
    'premierlearning/learning.py',
    'premierlearning/clean_match.py',
    'premierlearning/player.py',
    'premierlearning/team_expectancy.py',
    'simulation/historical_data.py',
    'simulation/season_model.py',
)


def _parse_seasons(value):
    result = []
    for part in str(value).split(','):
        part = part.strip()
        if not part:
            continue
        year = int(part)
        if year >= 2000:
            year %= 100
        result.append(year)
    if not result:
        raise argparse.ArgumentTypeError('at least one season is required')
    return tuple(result)


def _model_hash(config):
    digest = hashlib.sha256()
    digest.update(json.dumps(config, sort_keys=True).encode('utf-8'))
    api_dir = Path(__file__).resolve().parents[1]
    for relative in CODE_HASH_FILES:
        path = api_dir / relative
        digest.update(relative.encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _set_random_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


@contextmanager
def _simulation_environment():
    previous = os.environ.get('PL_SIMULATION_BUILD')
    os.environ['PL_SIMULATION_BUILD'] = '1'
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('PL_SIMULATION_BUILD', None)
        else:
            os.environ['PL_SIMULATION_BUILD'] = previous


def _prediction_rows(season, season_start_year, as_of_gameweek):
    rows = []
    for player in season.players:
        by_gameweek = {}
        for match in player.future_matches:
            if match.gameweek is None:
                continue
            by_gameweek.setdefault(int(match.gameweek), []).append(match)
        for target_gw, matches in by_gameweek.items():
            matches.sort(
                key=lambda match: (
                    str(getattr(match, 'fixture_date', None) or ''),
                    int(match.fixture_id),
                )
            )
            for fixture_order, match in enumerate(matches, start=1):
                rows.append({
                    'season_start_year': int(season_start_year),
                    'as_of_gameweek': int(as_of_gameweek),
                    'target_gameweek': int(target_gw),
                    'fixture_id': int(match.fixture_id),
                    'fixture_order': int(fixture_order),
                    'player_id': int(player.id),
                    'player_code': int(player.code),
                    'player_name': player.get_name(),
                    'position': int(player.position),
                    'team_id': int(player.team_id),
                    'opponent_team_id': int(match.opponent_team.id),
                    'opponent': match.opponent,
                    'is_home': bool(match.home),
                    'price': int(round(float(player.current_cost))),
                    'selectable': bool(
                        getattr(player, 'historical_selectable', True)
                    ),
                    'predicted_minutes': float(match.minutes or 0.0),
                    'predicted_points': float(match.points or 0.0),
                    'points_error': float(
                        getattr(match, 'points_error', 0.0) or 0.0
                    ),
                    'availability_mode': 'minutes_proxy',
                    'fixture_mode': 'final_calendar',
                })
    rows.sort(
        key=lambda row: (
            row['player_id'],
            row['target_gameweek'],
            row['fixture_order'],
            row['fixture_id'],
        )
    )
    return rows


class HistoricalPredictionBuilder:
    def __init__(
        self,
        db_handler,
        seasons=DEFAULT_SEASONS,
        gameweek_from=1,
        gameweek_to=38,
        horizon_gameweeks=5,
        seed=42,
        force=False,
        name='strict-historical-rest-v1',
        prediction_set_id=None,
        adopt_code_hash=False,
        train_once_per_season=False,
    ):
        self.db = db_handler
        self.seasons = tuple(int(season) for season in seasons)
        self.gameweek_from = int(gameweek_from)
        self.gameweek_to = int(gameweek_to)
        self.horizon_gameweeks = int(horizon_gameweeks)
        self.seed = int(seed)
        self.force = bool(force)
        self.name = name
        self.train_once_per_season = bool(train_once_per_season)
        self.sources = {}
        self.config = {
            'horizon_gameweeks': self.horizon_gameweeks,
            'seed': self.seed,
            'linear': '--linear' in sys.argv,
            'quick': '--quick' in sys.argv,
            'team_expectancies': '--no-team-expectancies' not in sys.argv,
            'training_policy': 'strict_point_in_time',
            'availability_mode': 'minutes_proxy',
            'fixture_mode': 'final_calendar',
            'prediction_scope': 'rest_of_season',
            'training_schedule': (
                'once_per_season'
                if self.train_once_per_season
                else 'once_per_snapshot'
            ),
        }
        metadata = {
            'name': self.name,
            'model_config_hash': _model_hash(self.config),
            'horizon_gameweeks': self.horizon_gameweeks,
            'training_policy': 'strict_point_in_time',
            'availability_mode': 'minutes_proxy',
            'fixture_mode': 'final_calendar',
            'random_seed': self.seed,
            'config': self.config,
        }
        if prediction_set_id is None:
            self.prediction_set_id = self.db.get_or_create_prediction_set(metadata)
        else:
            existing = self.db.get_prediction_set(prediction_set_id)
            if existing is None:
                raise ValueError(f'Unknown prediction set {prediction_set_id}')
            if (
                existing['horizon_gameweeks'] != self.horizon_gameweeks
                or existing['training_policy'] != metadata['training_policy']
                or existing['availability_mode'] != metadata['availability_mode']
                or existing['fixture_mode'] != metadata['fixture_mode']
            ):
                raise ValueError(
                    f'Prediction set {prediction_set_id} is incompatible with '
                    'the requested historical build'
                )
            if (
                existing['model_config_hash'] != metadata['model_config_hash']
                and not adopt_code_hash
            ):
                raise ValueError(
                    f'Prediction set {prediction_set_id} was built with a '
                    'different model/config hash; use --adopt-code-hash only '
                    'after verifying the change is non-semantic'
                )
            self.db.update_prediction_set_metadata(prediction_set_id, metadata)
            self.prediction_set_id = int(prediction_set_id)

    def source(self, season_start_year):
        if season_start_year not in self.sources:
            self.sources[season_start_year] = VaastavSeasonData(season_start_year)
        return self.sources[season_start_year]

    def _new_runner(self, season_start_year, as_of_gameweek):
        current_season = self.source(season_start_year).build_current_season(
            as_of_gameweek,
            self.horizon_gameweeks,
            predict_through_gameweek=38,
        )
        runner = Runner(
            current_season=current_season,
            target_season_year=season_start_year,
            as_of_gameweek=as_of_gameweek,
            horizon_gameweeks=self.horizon_gameweeks,
            update_data=False,
        )
        runner.populate_past_seasons()
        runner.validate_temporal_cutoff()
        return runner

    def build_one(
        self,
        season_start_year,
        as_of_gameweek,
        fixed_models=None,
        prepared_runner=None,
    ):
        snapshot_id, should_build = self.db.begin_prediction_snapshot(
            self.prediction_set_id,
            season_start_year,
            as_of_gameweek,
            force=self.force,
            expected_max_target_gameweek=38,
        )
        if not should_build:
            print(
                f'Skipping complete snapshot {season_start_year}/'
                f'{season_start_year + 1} GW{as_of_gameweek}'
            )
            return False

        print(
            f'\nBuilding prediction set {self.prediction_set_id}: '
            f'{season_start_year}/{season_start_year + 1} GW{as_of_gameweek}'
        )
        try:
            runner = prepared_runner or self._new_runner(
                season_start_year,
                as_of_gameweek,
            )
            with _simulation_environment():
                if fixed_models is None:
                    _set_random_seed(
                        self.seed + season_start_year * 100 + as_of_gameweek
                    )
                    runner.run(persist=False)
                else:
                    fixed_models.predict(
                        runner,
                        already_prepared=prepared_runner is not None,
                    )
            rows = _prediction_rows(
                runner.current_season,
                season_start_year,
                as_of_gameweek,
            )
            if not rows:
                raise ValueError('prediction snapshot produced no fixture rows')
            target_gws = {row['target_gameweek'] for row in rows}
            invalid = [
                gameweek
                for gameweek in target_gws
                if not (as_of_gameweek <= gameweek <= 38)
            ]
            if invalid:
                raise ValueError(
                    f'prediction rows outside remaining season: {invalid}'
                )
            if max(target_gws) != 38:
                raise ValueError(
                    f'prediction snapshot ends at GW{max(target_gws)}, not GW38'
                )
            self.db.persist_historical_predictions(snapshot_id, rows)
            warnings = [
                'Historical weekly injury percentages are unavailable; '
                'predicted minutes are used as the availability proxy.',
                'Fixture events use the final season calendar and may know later reschedules.',
                'Every remaining fixture through GW38 is predicted from this cutoff; '
                'later actual results are never used.',
            ]
            self.db.complete_prediction_snapshot(snapshot_id, len(rows), warnings)
            print(f'Persisted {len(rows)} player-fixture predictions')
            return True
        except Exception as exc:
            self.db.fail_prediction_snapshot(snapshot_id, exc)
            raise

    def build(self):
        built = 0
        for season_start_year in self.seasons:
            available = set(self.source(season_start_year).gameweeks)
            fixed_models = None
            training_runner = None
            if self.train_once_per_season:
                _set_random_seed(self.seed + season_start_year * 100)
                training_runner = self._new_runner(season_start_year, 1)
                with _simulation_environment():
                    fixed_models = FixedSeasonModels.fit(training_runner)
            for gameweek in range(self.gameweek_from, self.gameweek_to + 1):
                if gameweek not in available:
                    print(
                        f'Skipping season {season_start_year} GW{gameweek}: '
                        'no event in fixtures.csv'
                    )
                    continue
                prepared_runner = (
                    training_runner if gameweek == 1 else None
                )
                built += int(self.build_one(
                    season_start_year,
                    gameweek,
                    fixed_models=fixed_models,
                    prepared_runner=prepared_runner,
                ))
        return built

    def coverage(self):
        rows = self.db.prediction_coverage(self.prediction_set_id)
        completed = {
            (row['season_start_year'], row['as_of_gameweek'])
            for row in rows
            if (
                row['status'] == 'complete'
                and row['row_count'] > 0
                and row['stored_row_count'] == row['row_count']
                and row['min_target_gameweek'] == row['as_of_gameweek']
                and row['max_target_gameweek'] == 38
            )
        }
        expected = {
            (season, gameweek)
            for season in self.seasons
            for gameweek in range(self.gameweek_from, self.gameweek_to + 1)
            if gameweek in set(self.source(season).gameweeks)
        }
        missing = sorted(expected - completed)
        failed = [
            row for row in rows
            if (
                row['status'] != 'complete'
                or row['row_count'] <= 0
                or row['stored_row_count'] != row['row_count']
                or row['min_target_gameweek'] != row['as_of_gameweek']
                or row['max_target_gameweek'] != 38
            )
        ]
        report = {
            'prediction_set_id': self.prediction_set_id,
            'expected_snapshots': len(expected),
            'complete_snapshots': len(expected & completed),
            'missing': missing,
            'failed_or_empty': failed,
        }
        print(json.dumps(report, indent=2))
        return report


def build_parser():
    parser = argparse.ArgumentParser(
        description='Build strict point-in-time historical FPL predictions'
    )
    parser.add_argument('--seasons', type=_parse_seasons, default=DEFAULT_SEASONS)
    parser.add_argument('--from-gw', type=int, default=1)
    parser.add_argument('--to-gw', type=int, default=38)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', default='strict-historical-rest-v1')
    parser.add_argument(
        '--prediction-set-id',
        type=int,
        help='Resume an explicitly selected compatible cache',
    )
    parser.add_argument(
        '--adopt-code-hash',
        action='store_true',
        help='Allow an explicit cache to adopt a verified non-semantic code hash',
    )
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--coverage-only', action='store_true')
    parser.add_argument(
        '--linear',
        action='store_true',
        help='Use the existing Ridge prediction mode',
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use the existing short neural training mode',
    )
    parser.add_argument('--no-team-expectancies', action='store_true')
    parser.add_argument(
        '--train-once-per-season',
        action='store_true',
        help='Fit at the GW1 cutoff and reuse fixed weights for every GW snapshot',
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (1 <= args.from_gw <= args.to_gw <= 38):
        parser.error('gameweek range must satisfy 1 <= from <= to <= 38')
    if args.horizon < 1:
        parser.error('--horizon must be positive')

    app = Flask(__name__)
    with app.app_context():
        db_handler = DB_Handler()
        try:
            builder = HistoricalPredictionBuilder(
                db_handler=db_handler,
                seasons=args.seasons,
                gameweek_from=args.from_gw,
                gameweek_to=args.to_gw,
                horizon_gameweeks=args.horizon,
                seed=args.seed,
                force=args.force,
                name=args.name,
                prediction_set_id=args.prediction_set_id,
                adopt_code_hash=args.adopt_code_hash,
                train_once_per_season=args.train_once_per_season,
            )
            print(f'Prediction set ID: {builder.prediction_set_id}')
            if not args.coverage_only:
                builder.build()
            report = builder.coverage()
            if report['missing'] or report['failed_or_empty']:
                return 1
            return 0
        finally:
            db_handler.close_connection()


if __name__ == '__main__':
    raise SystemExit(main())
