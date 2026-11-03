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
from premierlearning.learning import (
    HORIZON_REPS_DEFAULT,
    MINUTES_REPS_DEFAULT,
    PlayerNetHParams,
    parse_horizon_reps,
    parse_minutes_reps,
    with_learning_rep_flags,
)
from premierlearning.learning_v2.score import (
    HUMP_HEIGHT,
    HUMP_PEAK,
    HUMP_POWER,
    HUMP_WIDTH,
    LOSS_TIER_MIX,
    SMOOTH_ALPHA,
    SMOOTH_POWER,
    SMOOTH_SCALE,
    WEIGHTING_KINDS,
    WEIGHTING_TIER,
    LossWeighting,
)
from premierlearning.learning_v2.train import PATIENCE_FLOOR, resolve_loss_name
from premierlearning.runner import Runner
from simulation.historical_data import VaastavSeasonData
from simulation.season_model import FixedSeasonModels
from simulation.season_model_v2 import FixedSeasonModelsV2


DEFAULT_SEASONS = (23, 24, 25)
LEARNING_V2_FLAG = '--learning-v2'
DEFAULT_V2_PLAN = 'simple_h1'
# Hashed only for learning_v2 builds, so v1 cache hashes do not move.
V2_CODE_HASH_FILES = (
    'premierlearning/learning_v2/dataset.py',
    'premierlearning/learning_v2/external_hook.py',
    'premierlearning/learning_v2/external_inputs.py',
    'premierlearning/learning_v2/kickoff.py',
    'premierlearning/learning_v2/learn.py',
    'premierlearning/learning_v2/model.py',
    'premierlearning/learning_v2/plan.py',
    'premierlearning/learning_v2/score.py',
    'premierlearning/learning_v2/staged.py',
    'premierlearning/learning_v2/teams.py',
    'premierlearning/learning_v2/train.py',
    'simulation/season_model_v2.py',
)
CODE_HASH_FILES = (
    'premierlearning/runner.py',
    'premierlearning/learning.py',
    'premierlearning/clean_match.py',
    'premierlearning/match.py',
    'premierlearning/player.py',
    'premierlearning/config_constants.py',
    'premierlearning/helper_functions.py',
    'premierlearning/team_expectancy.py',
    'squadtools/availability.py',
    'squadtools/chance_to_play.py',
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


def _model_hash(config, extra_files=()):
    digest = hashlib.sha256()
    digest.update(json.dumps(config, sort_keys=True).encode('utf-8'))
    api_dir = Path(__file__).resolve().parents[1]
    for relative in tuple(CODE_HASH_FILES) + tuple(extra_files):
        path = api_dir / relative
        digest.update(relative.encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _v2_plan_file(plan_name):
    return 'premierlearning/learning_v2/model_plans/%s.json' % plan_name


@contextmanager
def _learning_v2_argv(enabled):
    """Runner reads --learning-v2 from argv; make it match the build, then restore."""
    previous = list(sys.argv)
    if enabled and LEARNING_V2_FLAG not in sys.argv:
        sys.argv.append(LEARNING_V2_FLAG)
    if not enabled:
        sys.argv[:] = [arg for arg in sys.argv if arg != LEARNING_V2_FLAG]
    try:
        yield
    finally:
        sys.argv[:] = previous


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
                    'predicted_pp90': (
                        None
                        if getattr(match, 'predicted_pp90', None) is None
                        else float(match.predicted_pp90)
                    ),
                    'predicted_points': float(match.points or 0.0),
                    'chance_of_playing': (
                        None
                        if getattr(match, 'chance_of_playing', None) is None
                        else int(match.chance_of_playing)
                    ),
                    'points_error': float(
                        getattr(match, 'points_error', 0.0) or 0.0
                    ),
                    'availability_mode': 'deferred_chance',
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
        replace_cache=False,
        points_loss='huber',
        points_minutes_weight=False,
        use_horizon_model=True,
        points_reps=1,
        learning_v2=False,
        model_plan=DEFAULT_V2_PLAN,
        v2_weighting=None,
        v2_loss=None,
        points_tier_weight=False,
        v2_smear='off',
        v2_patience_floor=PATIENCE_FLOOR,
        v2_horizon=0,
        v2_minutes_head=0.0,
        v2_minutes_target='bands',
    ):
        self.db = db_handler
        self.learning_v2 = bool(learning_v2)
        self.model_plan = str(model_plan or DEFAULT_V2_PLAN)
        self.v2_weighting = v2_weighting or LossWeighting()
        self.v2_loss = resolve_loss_name(v2_loss)
        self.v2_smear = str(v2_smear or 'off')
        self.v2_patience_floor = int(v2_patience_floor)
        self.v2_horizon = int(v2_horizon or 0)
        self.v2_minutes_head = float(v2_minutes_head or 0.0)
        self.v2_minutes_target = str(v2_minutes_target or 'bands')
        self.points_tier_weight = bool(points_tier_weight)
        self.seasons = tuple(int(season) for season in seasons)
        self.gameweek_from = int(gameweek_from)
        self.gameweek_to = int(gameweek_to)
        self.horizon_gameweeks = int(horizon_gameweeks)
        self.seed = int(seed)
        self.force = bool(force)
        self.name = name
        self.train_once_per_season = bool(train_once_per_season)
        self.replace_on_build = bool(replace_cache)
        self.points_loss = str(points_loss or 'huber').lower()
        self.points_minutes_weight = bool(points_minutes_weight)
        self.use_horizon_model = bool(use_horizon_model)
        self.points_reps = max(1, int(points_reps))
        self.sources = {}
        self.config = {
            'horizon_gameweeks': self.horizon_gameweeks,
            'seed': self.seed,
            'linear': '--linear' in sys.argv,
            'quick': '--quick' in sys.argv,
            'team_expectancies': '--no-team-expectancies' not in sys.argv,
            'training_policy': 'strict_point_in_time',
            'availability_mode': 'deferred_chance',
            'fixture_mode': 'final_calendar',
            'prediction_scope': 'rest_of_season',
            'horizon_reps': (
                0 if not self.use_horizon_model else parse_horizon_reps()
            ),
            'horizon_model': self.use_horizon_model,
            'minutes_reps': parse_minutes_reps(),
            'points_loss': self.points_loss,
            'points_minutes_weight': self.points_minutes_weight,
            'points_reps': self.points_reps,
            'training_schedule': (
                'once_per_season'
                if self.train_once_per_season
                else 'once_per_snapshot'
            ),
        }
        if self.points_tier_weight:
            # Only when set, so unweighted v1 configs hash as before.
            self.config['points_tier_weight'] = LOSS_TIER_MIX
        extra_hash_files = ()
        if self.learning_v2:
            # v2 skips team expectancies, the minutes net and the horizon net, so
            # those keys stay as recorded but only these change its output.
            self.config.update({
                'model': 'learning_v2',
                'model_plan': self.model_plan,
                'v2_loss': self.v2_loss,
                'v2_weighting': self.v2_weighting.config(),
                'huber_delta': PlayerNetHParams.from_env().huber_delta,
                # Training rows carry the FPL flag known at their deadline
                # (fplcache, 2021/22 on); older v2 sets saw a constant 1.0.
                'v2_chance_input': 'deadline_flags',
            })
            if self.v2_smear != 'off':
                # Only when set, so existing v2 configs hash as before.
                self.config['v2_smear'] = self.v2_smear
            if self.v2_patience_floor != PATIENCE_FLOOR:
                self.config['v2_patience_floor'] = self.v2_patience_floor
            if self.v2_horizon:
                self.config['v2_horizon'] = self.v2_horizon
            if '_oracle_' in self.model_plan:
                # fpl_manager refuses these sets.
                self.config['uses_future_information'] = True
            if self.v2_minutes_head:
                self.config['v2_minutes_head'] = self.v2_minutes_head
                self.config['v2_minutes_target'] = self.v2_minutes_target
            extra_hash_files = V2_CODE_HASH_FILES + (_v2_plan_file(self.model_plan),)
        self.metadata = {
            'name': self.name,
            'model_config_hash': _model_hash(self.config, extra_hash_files),
            'horizon_gameweeks': self.horizon_gameweeks,
            'training_policy': 'strict_point_in_time',
            'availability_mode': 'deferred_chance',
            'fixture_mode': 'final_calendar',
            'random_seed': self.seed,
            'config': self.config,
        }
        if prediction_set_id is None:
            self.prediction_set_id = None
        else:
            existing = self.db.get_prediction_set(prediction_set_id)
            if existing is None:
                raise ValueError(f'Unknown prediction set {prediction_set_id}')
            if (
                existing['horizon_gameweeks'] != self.horizon_gameweeks
                or existing['training_policy'] != self.metadata['training_policy']
                or existing['availability_mode'] != self.metadata['availability_mode']
                or existing['fixture_mode'] != self.metadata['fixture_mode']
            ):
                raise ValueError(
                    f'Prediction set {prediction_set_id} is incompatible with '
                    'the requested historical build'
                )
            if (
                existing['model_config_hash'] != self.metadata['model_config_hash']
                and not adopt_code_hash
            ):
                raise ValueError(
                    f'Prediction set {prediction_set_id} was built with a '
                    'different model/config hash; use --adopt-code-hash only '
                    'after verifying the change is non-semantic'
                )
            self.db.update_prediction_set_metadata(prediction_set_id, self.metadata)
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
        if self.learning_v2:
            runner.v2_model_plan = self.model_plan
            runner.v2_weighting = self.v2_weighting
            runner.v2_points_reps = self.points_reps
        return runner

    def _fit_fixed_models(self, training_runner):
        if self.learning_v2:
            return FixedSeasonModelsV2.fit(
                training_runner,
                plan_name=self.model_plan,
                weighting=self.v2_weighting,
                points_reps=self.points_reps,
                loss_name=self.v2_loss,
                smear_mode=self.v2_smear,
                patience_floor=self.v2_patience_floor,
                horizon_gameweeks=self.v2_horizon,
                minutes_weight=self.v2_minutes_head,
                minutes_target=self.v2_minutes_target,
            )
        return FixedSeasonModels.fit(
            training_runner,
            points_loss=self.points_loss,
            points_minutes_weight=self.points_minutes_weight,
            use_horizon_model=self.use_horizon_model,
            points_reps=self.points_reps,
            points_tier_weight=self.points_tier_weight,
        )

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
            with _simulation_environment(), _learning_v2_argv(self.learning_v2):
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
                'Predicted pp90 and minutes are stored raw. Chance-to-play from '
                'Randdalf/fplcache (or live FPL) is applied in squadtools at '
                'optimiser time, so availability policy can change without a rebuild.',
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

    def _ensure_prediction_set(self):
        """Reuse the matching cache. Only --replace-cache drops other seasons."""
        if self.replace_on_build:
            self.prediction_set_id = self.db.replace_prediction_cache(self.metadata)
            print(
                'Replaced previous historical predictions; '
                f'prediction set ID: {self.prediction_set_id}'
            )
            return
        if self.prediction_set_id is None:
            self.prediction_set_id = self.db.get_or_create_prediction_set(
                self.metadata
            )
            print(
                'Using prediction set %s; existing season snapshots are kept'
                % self.prediction_set_id
            )

    def build(self):
        self._ensure_prediction_set()
        built = 0
        for season_start_year in self.seasons:
            available = set(self.source(season_start_year).gameweeks)
            fixed_models = None
            training_runner = None
            if self.train_once_per_season:
                _set_random_seed(self.seed + season_start_year * 100)
                training_runner = self._new_runner(season_start_year, 1)
                with _simulation_environment(), _learning_v2_argv(self.learning_v2):
                    fixed_models = self._fit_fixed_models(training_runner)
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
    parser.add_argument(
        '--replace-cache',
        action='store_true',
        help='Drop every prediction set and simulation run before building',
    )
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
        '--horizon-reps',
        type=int,
        default=HORIZON_REPS_DEFAULT,
        help='Independent fits for the horizon-N net (h1 points stay 1). Default %s.'
        % HORIZON_REPS_DEFAULT,
    )
    parser.add_argument(
        '--minutes-reps',
        type=int,
        default=MINUTES_REPS_DEFAULT,
        help='Independent fits for the minutes net, same keep/drop averaging as horizon. Default %s.'
        % MINUTES_REPS_DEFAULT,
    )
    parser.add_argument(
        '--train-once-per-season',
        action='store_true',
        help='Fit at the GW1 cutoff and reuse fixed weights for every GW snapshot',
    )
    parser.add_argument(
        '--points-loss',
        choices=('huber', 'poisson'),
        default='huber',
        help='Loss for the single-gameweek points net. Default huber (log1p).',
    )
    parser.add_argument(
        '--points-minutes-weight',
        action='store_true',
        help='Weight h1 training rows by minutes/90 (Poisson exposure).',
    )
    parser.add_argument(
        '--no-horizon-model',
        action='store_true',
        help='Skip the horizon-N net; leave h1 fixture scores unscaled.',
    )
    parser.add_argument(
        '--points-reps',
        type=int,
        default=1,
        help='Independent fits for the h1 points net (or the learning_v2 net). Default 1.',
    )
    parser.add_argument(
        LEARNING_V2_FLAG,
        dest='learning_v2',
        action='store_true',
        help='Predict with the learning_v2 points model instead of v1.',
    )
    parser.add_argument(
        '--model-plan',
        default=DEFAULT_V2_PLAN,
        help='learning_v2 model plan. Default %s.' % DEFAULT_V2_PLAN,
    )
    parser.add_argument(
        '--v2-weighting',
        choices=WEIGHTING_KINDS,
        default=WEIGHTING_TIER,
        help='learning_v2 loss weights by actual points. Default tier (the blended bands).',
    )
    parser.add_argument('--v2-tier-mix', type=float, default=LOSS_TIER_MIX)
    parser.add_argument('--v2-smooth-alpha', type=float, default=SMOOTH_ALPHA)
    parser.add_argument('--v2-smooth-scale', type=float, default=SMOOTH_SCALE)
    parser.add_argument('--v2-smooth-power', type=float, default=SMOOTH_POWER)
    parser.add_argument('--v2-smooth-cap', type=float, default=None,
                        help='Cap smooth weights before scaling to mean 1 (2.4 = the bands without steps).')
    parser.add_argument('--v2-hump-peak', type=float, default=HUMP_PEAK)
    parser.add_argument('--v2-hump-height', type=float, default=HUMP_HEIGHT)
    parser.add_argument('--v2-hump-power', type=float, default=HUMP_POWER)
    parser.add_argument('--v2-hump-width', type=float, default=HUMP_WIDTH)
    parser.add_argument('--v2-loss', choices=('huber_raw', 'mse', 'huber', 'poisson'), default=None,
                        help='learning_v2 loss. Default huber_raw; huber trains on log1p points.')
    parser.add_argument('--v2-smear', choices=('off', 'global', 'conditional'), default='off',
                        help='Smearing correction for log-target v2 losses.')
    parser.add_argument('--v2-horizon', type=int, default=0,
                        help='Also fit a learning_v2 window net over this many GWs; 0 is h1 only.')
    parser.add_argument('--v2-minutes-head', type=float, default=0.0,
                        help='Loss weight of a training-only minutes head on the h1 net; 0 is none.')
    parser.add_argument('--v2-minutes-target', choices=('bands', 'log'), default='bands',
                        help='What the minutes head predicts: bands (cross-entropy) or log minutes (Huber).')
    parser.add_argument('--v2-patience-floor', type=int, default=PATIENCE_FLOOR,
                        help='Minimum early-stop patience for learning_v2. Default %s; lower it only '
                             'for evaluation test runs.' % PATIENCE_FLOOR)
    parser.add_argument('--huber-delta', type=float, default=None,
                        help='Huber threshold; sets PL_HUBER_DELTA for this build.')
    parser.add_argument('--points-tier-weight', action='store_true',
                        help='v1: weight h1 points rows by the blended score bands of the fixture, '
                             'and horizon rows by the bands of the window points per 90.')
    return parser


def main(argv=None):
    code, _builder = run(argv)
    return code


def run(argv=None):
    """Build and report coverage. Returns (exit code, builder)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (1 <= args.from_gw <= args.to_gw <= 38):
        parser.error('gameweek range must satisfy 1 <= from <= to <= 38')
    if args.horizon < 1:
        parser.error('--horizon must be positive')
    if args.horizon_reps < 1:
        parser.error('--horizon-reps must be a positive integer')
    if args.minutes_reps < 1:
        parser.error('--minutes-reps must be a positive integer')
    if args.points_reps < 1:
        parser.error('--points-reps must be a positive integer')

    sys.argv = with_learning_rep_flags(
        sys.argv,
        horizon_reps=args.horizon_reps,
        minutes_reps=args.minutes_reps,
    )
    if args.huber_delta is not None:
        # PlayerNetHParams.from_env reads it, so the hash and every fit agree.
        os.environ['PL_HUBER_DELTA'] = str(args.huber_delta)

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
                replace_cache=args.replace_cache,
                points_loss=args.points_loss,
                points_minutes_weight=args.points_minutes_weight,
                use_horizon_model=not args.no_horizon_model,
                points_reps=args.points_reps,
                learning_v2=args.learning_v2,
                model_plan=args.model_plan,
                v2_weighting=LossWeighting(
                    kind=args.v2_weighting,
                    tier_mix=args.v2_tier_mix,
                    alpha=args.v2_smooth_alpha,
                    scale=args.v2_smooth_scale,
                    power=args.v2_smooth_power,
                    cap=args.v2_smooth_cap,
                    hump_peak=args.v2_hump_peak,
                    hump_height=args.v2_hump_height,
                    hump_power=args.v2_hump_power,
                    hump_width=args.v2_hump_width,
                ),
                v2_loss=args.v2_loss,
                points_tier_weight=args.points_tier_weight,
                v2_smear=args.v2_smear,
                v2_patience_floor=args.v2_patience_floor,
                v2_horizon=args.v2_horizon,
                v2_minutes_head=args.v2_minutes_head,
                v2_minutes_target=args.v2_minutes_target,
            )
            if not args.coverage_only:
                builder.build()
            elif builder.prediction_set_id is None:
                builder.prediction_set_id = db_handler.latest_prediction_set_id()
            print(f'Prediction set ID: {builder.prediction_set_id}')
            report = builder.coverage()
            if report['missing'] or report['failed_or_empty']:
                return 1, builder
            return 0, builder
        finally:
            db_handler.close_connection()


if __name__ == '__main__':
    raise SystemExit(main())
