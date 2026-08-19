from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json
import os
import numpy
import sys
import csv
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError as MAE
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_predict
from scipy.stats import kendalltau

from premierlearning.clean_match import (
    CleanMatch, get_inputs_from_data_key, get_outputs_from_data_key
)


def use_linear_model() -> bool:
    return '--linear' in sys.argv


def _ridge_pipeline():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("ridge", RidgeCV(alphas=numpy.logspace(-3, 3, 13), store_cv_values=False)),
    ])


@dataclass
class MLOptions:
    reps: int
    epochs: int
    batch_size: int
    input_keys: list
    output_key: str

EPOCH_MAP = {
    'total_points': 500,
    'minutes': 500,
}

EPOCHS = 8
BATCH_SIZE = 32
HORIZON_REPS_FLAG = '--horizon-reps'
RUN_METRICS_DIR = Path('run_metrics/horizon_reps')
LAST_HORIZON_METRICS_DIR = None

# Player net: 20260819_085557 trial 06_reduce_lr_0.3
# (L2 2.25e-3 wd-0 dropout body + ReduceLR factor 0.3)
LEARNING_RATE = 5e-4  # was 1e-3
EARLY_STOPPING_PATIENCE = 25  # was 10
REDUCE_LR_PATIENCE = 8  # was 5
REDUCE_LR_MIN = 1e-5
L2 = 2.25e-3  # was 2e-3 on the v4 wd-0 winner
WEIGHT_DECAY = 0.0  # sweep-4 02_wd_0; was 1e-4 on the v3 winner
WIDTH_MULT = 1.5  # sweep-1 trial 09 (was 1.0 option-6 baseline)
N_HIDDEN = 3  # sweep-1 trial 09 (was 2 option-6 baseline)
DROPOUT_1 = 0.1
DROPOUT_2 = 0.05
HUBER_DELTA = 1.8
REDUCE_LR_FACTOR = 0.3  # was 0.5; 20260819_085557 trial 06


@dataclass
class PlayerNetHParams:
    learning_rate: float = LEARNING_RATE
    l2: float = L2
    weight_decay: float = WEIGHT_DECAY
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    reduce_lr_patience: int = REDUCE_LR_PATIENCE
    reduce_lr_factor: float = REDUCE_LR_FACTOR
    reduce_lr_min: float = REDUCE_LR_MIN
    width_mult: float = WIDTH_MULT
    n_hidden: int = N_HIDDEN
    huber_delta: float = HUBER_DELTA
    batch_size: int = BATCH_SIZE
    average_all_reps: bool = False
    val_mae_keep_ratio: float = 1.08  # drop reps worse than 8% over best val MAE
    dropout_1: float = DROPOUT_1
    dropout_2: float = DROPOUT_2
    loss: str = 'huber'  # historic alt: 'mae'
    optimizer: str = 'adamw'  # 'adamw' or 'adam'

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_env(cls) -> 'PlayerNetHParams':
        def env(name, default, cast):
            raw = os.environ.get(name)
            if raw is None or raw == '':
                return default
            if cast is bool:
                return raw.strip().lower() in ('1', 'true', 'yes')
            return cast(raw)

        return cls(
            learning_rate=env('PL_LEARNING_RATE', LEARNING_RATE, float),
            l2=env('PL_L2', L2, float),
            weight_decay=env('PL_WEIGHT_DECAY', WEIGHT_DECAY, float),
            early_stopping_patience=env('PL_EARLY_STOPPING_PATIENCE', EARLY_STOPPING_PATIENCE, int),
            reduce_lr_patience=env('PL_REDUCE_LR_PATIENCE', REDUCE_LR_PATIENCE, int),
            reduce_lr_factor=env('PL_REDUCE_LR_FACTOR', REDUCE_LR_FACTOR, float),
            reduce_lr_min=env('PL_REDUCE_LR_MIN', REDUCE_LR_MIN, float),
            width_mult=env('PL_WIDTH_MULT', WIDTH_MULT, float),
            n_hidden=env('PL_N_HIDDEN', N_HIDDEN, int),
            huber_delta=env('PL_HUBER_DELTA', HUBER_DELTA, float),
            batch_size=env('PL_BATCH_SIZE', BATCH_SIZE, int),
            average_all_reps=env('PL_AVERAGE_ALL_REPS', False, bool),
            val_mae_keep_ratio=env('PL_VAL_MAE_KEEP_RATIO', 1.08, float),
            dropout_1=env('PL_DROPOUT_1', DROPOUT_1, float),
            dropout_2=env('PL_DROPOUT_2', DROPOUT_2, float),
            loss=env('PL_LOSS', 'huber', str),
            optimizer=env('PL_OPTIMIZER', 'adamw', str),
        )


def build_adamw(learning_rate, weight_decay):
    """AdamW on TF 2.11+ Keras, or tensorflow-addons on TF 2.4."""
    try:
        from tensorflow.keras.optimizers import AdamW
        return AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    except ImportError:
        pass
    try:
        from tensorflow_addons.optimizers import AdamW
        return AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    except ImportError as exc:
        raise ImportError(
            'AdamW needs TensorFlow >= 2.11 or tensorflow-addons (for TF 2.4). '
            'Install tensorflow-addons or upgrade TensorFlow.'
        ) from exc


def _build_optimizer(hp: PlayerNetHParams):
    """Adam vs AdamW with the same learning_rate; weight_decay only applies to AdamW."""
    name = (hp.optimizer or 'adamw').strip().lower()
    if name == 'adam':
        return Adam(learning_rate=hp.learning_rate), 'Adam'
    if name in ('adamw', 'adam_w'):
        return build_adamw(hp.learning_rate, hp.weight_decay), 'AdamW'
    raise ValueError("Unknown optimizer %s (use 'adam' or 'adamw')" % hp.optimizer)


if '--quick' in sys.argv:
    EPOCHS = 2


def parse_horizon_reps() -> int:
    """Independent fits for the horizon model only; default 1 unless --horizon-reps N."""
    for i, arg in enumerate(sys.argv):
        if arg == HORIZON_REPS_FLAG and i + 1 < len(sys.argv):
            try:
                n = int(sys.argv[i + 1])
                if n < 1:
                    raise ValueError
                return n
            except ValueError:
                raise SystemExit(f'{HORIZON_REPS_FLAG} requires a positive integer')
        if arg.startswith(HORIZON_REPS_FLAG + '='):
            try:
                n = int(arg.split('=', 1)[1])
                if n < 1:
                    raise ValueError
                return n
            except ValueError:
                raise SystemExit(f'{HORIZON_REPS_FLAG} requires a positive integer')
    return 1


def _flatten_predictions(preds) -> numpy.ndarray:
    return numpy.asarray(preds, dtype=float).reshape(-1)


def compute_rep_agreement(all_rep_preds, top_k: int = 20) -> Optional[dict]:
    """Run-to-run agreement across independent fits (ranking + prediction spread)."""
    reps = [_flatten_predictions(p) for p in all_rep_preds]
    n_reps = len(reps)
    if n_reps < 2:
        return None

    stacked = numpy.stack(reps, axis=0)
    per_player_std = numpy.std(stacked, axis=0, ddof=0)

    taus = []
    top_k_jaccards = []
    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            tau, _ = kendalltau(reps[i], reps[j])
            taus.append(float(tau) if tau == tau else 1.0)
            top_i = set(numpy.argsort(-reps[i])[:top_k])
            top_j = set(numpy.argsort(-reps[j])[:top_k])
            top_k_jaccards.append(len(top_i & top_j) / len(top_i | top_j))

    return {
        'n_reps': n_reps,
        'mean_kendall_tau': round(float(numpy.mean(taus)), 4),
        'min_kendall_tau': round(float(numpy.min(taus)), 4),
        f'top_{top_k}_jaccard_mean': round(float(numpy.mean(top_k_jaccards)), 4),
        f'top_{top_k}_jaccard_min': round(float(numpy.min(top_k_jaccards)), 4),
        'pred_std_mean': round(float(per_player_std.mean()), 4),
        'pred_std_p90': round(float(numpy.percentile(per_player_std, 90)), 4),
        'pred_std_max': round(float(per_player_std.max()), 4),
    }


def print_rep_agreement(all_rep_preds, top_k: int = 20):
    agreement = compute_rep_agreement(all_rep_preds, top_k=top_k)
    if agreement:
        print({f'rep agreement ({agreement["n_reps"]} fits)': agreement})


def _round_stats(stats: dict) -> dict:
    rounded = {}
    for key, value in stats.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return rounded


def _write_text_report(path: Path, title: str, sections: Dict[str, dict]):
    lines = [title, '=' * len(title), '']
    for section_name, payload in sections.items():
        lines.append(section_name)
        lines.append('-' * len(section_name))
        if isinstance(payload, dict):
            lines.append(json.dumps(payload, indent=2, sort_keys=True))
        elif isinstance(payload, list):
            for item in payload:
                lines.append(str(item))
        else:
            lines.append(str(payload))
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def _horizon_metrics_dir() -> Path:
    sweep_dir = os.environ.get('PL_SWEEP_DIR')
    trial = os.environ.get('PL_SWEEP_TRIAL')
    if sweep_dir and trial:
        run_dir = Path(sweep_dir) / trial
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RUN_METRICS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_last_horizon_metrics_dir():
    """Resolve the metrics folder at write time (import * would freeze a None)."""
    if LAST_HORIZON_METRICS_DIR is not None:
        return LAST_HORIZON_METRICS_DIR
    sweep_dir = os.environ.get('PL_SWEEP_DIR')
    trial = os.environ.get('PL_SWEEP_TRIAL')
    if sweep_dir and trial:
        return Path(sweep_dir) / trial
    return None


def save_horizon_rep_run_metrics(
    ml_data: 'MLData',
    rep_stats: List[dict],
    baselines: dict,
    agreement: Optional[dict],
    reps_used_for_avg: int,
    selection_threshold: float,
):
    """Persist one directory per multi-rep horizon session for later comparison."""
    if os.environ.get('PL_SIMULATION_BUILD'):
        return
    global LAST_HORIZON_METRICS_DIR
    run_dir = _horizon_metrics_dir()
    LAST_HORIZON_METRICS_DIR = run_dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    session_meta = {
        'timestamp': timestamp,
        'argv': sys.argv,
        'sweep_trial': os.environ.get('PL_SWEEP_TRIAL'),
        'output_key': ml_data.output_key,
        'log_targets': ml_data.ln_inputs,
        'n_reps_requested': len(rep_stats),
        'n_reps_averaged': reps_used_for_avg,
        'selection_threshold_val_mae': round(selection_threshold, 6) if selection_threshold else None,
        'n_train_rows': int(ml_data.input_learn.shape[0]),
        'n_test_rows': int(ml_data.input_test.shape[0]),
        'n_predict_rows': int(ml_data.input_to_pred.shape[0]),
    }
    payload = {
        'session': session_meta,
        'model_config': ml_data.model_config,
        'baselines': _round_stats(baselines),
        'rep_agreement': agreement or {},
        'rep_stats': [_round_stats(s) for s in rep_stats],
        'feature_columns': ml_data.player_feature_columns,
    }
    (run_dir / 'summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    _write_text_report(
        run_dir / 'summary.txt',
        f'Horizon multi-rep run {timestamp}',
        {
            'session': session_meta,
            'model_config': ml_data.model_config,
            'baselines': _round_stats(baselines),
            'rep_agreement': agreement or {},
            'feature_columns': ml_data.player_feature_columns,
        },
    )

    for i, stats in enumerate(rep_stats, start=1):
        _write_text_report(
            run_dir / f'rep_{i:02d}.txt',
            f'Horizon rep {i}/{len(rep_stats)} — {timestamp}',
            {
                'session': session_meta,
                'model_config': ml_data.model_config,
                'rep_stats': _round_stats(stats),
                'feature_columns': ml_data.player_feature_columns,
            },
        )

    print(f'Saved horizon rep metrics to {run_dir.resolve()}')


def get_linear_predictions(ml_data):
    """Fit Ridge (same pipeline as baselines) and predict; skip the neural net."""
    print('(linear-only run — RidgeCV predictions)')
    X_train = ml_data.input_learn
    y_train = ml_data.output_learn.copy()
    X_test = ml_data.input_test
    y_test = ml_data.output_test.copy()
    X_pred = ml_data.input_to_pred

    if ml_data.ln_inputs:
        y_train = numpy.log1p(numpy.maximum(y_train, 0))
        y_test_orig = y_test.copy()
        y_test_fit = numpy.log1p(numpy.maximum(y_test, 0))
    else:
        y_test_orig = y_test
        y_test_fit = y_test

    lin_pipe = _ridge_pipeline()
    lin_pipe.fit(X_train, y_train.ravel())
    ml_data.linear_pipe = lin_pipe

    y_pred_test = lin_pipe.predict(X_test)
    if ml_data.ln_inputs:
        y_pred_test = numpy.expm1(y_pred_test)
        mae = mean_absolute_error(y_test_orig, y_pred_test)
        rmse = mean_squared_error(y_test_orig, y_pred_test, squared=False)
    else:
        mae = mean_absolute_error(y_test_fit, y_pred_test)
        rmse = mean_squared_error(y_test_fit, y_pred_test, squared=False)

    print({
        'linear test absolute error': float(mae),
        'linear test rms error': float(rmse),
    })

    preds = lin_pipe.predict(X_pred)
    if ml_data.ln_inputs:
        preds = numpy.expm1(preds)
    pred_list = [numpy.asarray([float(p)], dtype=float) for p in preds]
    pred_flat = _flatten_predictions(pred_list)
    ridge = lin_pipe.named_steps['ridge']
    pca = lin_pipe.named_steps['pca']
    ml_data.model_config = {
        'optimizer': 'RidgeCV',
        'model': 'ridge',
        'pca_var_explained': 0.95,
        'ridge_alpha': float(getattr(ridge, 'alpha_', 0.0) or 0.0),
        'pca_components': int(getattr(pca, 'n_components_', 0) or 0),
        'n_features': int(ml_data.input_learn.shape[1]),
        'loss': 'mse',
        'log_targets': bool(ml_data.ln_inputs),
    }
    fit_stats = {
        'epochs_ran': 0,
        'test_mae': float(mae),
        'test_rmse': float(rmse),
        'best_val_mae': float(mae),
        'pred_min': float(pred_flat.min()) if pred_flat.size else 0.0,
        'pred_max': float(pred_flat.max()) if pred_flat.size else 0.0,
        'pred_mean': float(pred_flat.mean()) if pred_flat.size else 0.0,
        'kept_for_avg': True,
        'rep_index': 1,
    }
    return pred_list, fit_stats


class ReusablePredictor:
    """A fitted player model that can score new point-in-time feature rows."""

    def __init__(
        self,
        input_keys,
        include_standard_stats,
        log_targets,
        feature_columns,
        scaler=None,
        models=None,
        linear_pipe=None,
    ):
        self.input_keys = list(input_keys)
        self.include_standard_stats = bool(include_standard_stats)
        self.log_targets = bool(log_targets)
        self.feature_columns = list(feature_columns)
        self.scaler = scaler
        self.models = list(models or [])
        self.linear_pipe = linear_pipe

    def predict(self, clean_matches):
        if not clean_matches:
            return []
        rows = get_inputs_from_data_key(
            clean_matches,
            self.input_keys,
            self.include_standard_stats,
            include_opponent_features=True,
        )
        columns = list(rows[0].keys()) if rows else []
        if columns != self.feature_columns:
            raise ValueError(
                f'Prediction feature mismatch: expected {self.feature_columns}, got {columns}'
            )
        inputs = numpy.array([list(row.values()) for row in rows], dtype=float)
        if self.linear_pipe is not None:
            predictions = self.linear_pipe.predict(inputs).reshape(-1, 1)
        else:
            scaled = self.scaler.transform(inputs)
            model_predictions = [
                model.predict(scaled, verbose=0) for model in self.models
            ]
            predictions = numpy.mean(model_predictions, axis=0)
        if self.log_targets:
            predictions = numpy.expm1(predictions)
        return [
            numpy.asarray([float(value)], dtype=float)
            for value in predictions.reshape(-1)
        ]


def fit_reusable_predictor(
    clean_matches,
    input_keys,
    output_key,
    include_standard_stats,
    output_learn=None,
    log_targets=False,
    reps=1,
):
    """Fit once and return a predictor reusable across historical GW snapshots."""
    if not clean_matches:
        raise ValueError(f'No training rows for reusable {output_key} model')
    ml_data = MLData(
        clean_matches,
        clean_matches[:1],
        input_keys,
        output_key,
        include_standard_stats,
        output_learn=output_learn,
        log_targets=log_targets,
    )
    print(calc_baselines(ml_data))
    print(f'Fitting reusable {output_key} model once for the season')
    if use_linear_model():
        get_linear_predictions(ml_data)
        return ReusablePredictor(
            input_keys=input_keys,
            include_standard_stats=include_standard_stats,
            log_targets=log_targets,
            feature_columns=ml_data.player_feature_columns,
            linear_pipe=ml_data.linear_pipe,
        )

    fitted_models = []
    for rep in range(max(1, int(reps))):
        if reps > 1:
            print(f'  reusable fit {rep + 1}/{reps}')
        ml_data.setup_model()
        ml_data.get_predictions()
        fitted_models.append(ml_data.model)
    return ReusablePredictor(
        input_keys=input_keys,
        include_standard_stats=include_standard_stats,
        log_targets=log_targets,
        feature_columns=ml_data.player_feature_columns,
        scaler=ml_data.scaler,
        models=fitted_models,
    )


class OrigRMSE(tf.keras.metrics.Metric):
    """RMSE on original scale when training targets are log1p-transformed."""

    def __init__(self, name='orig_rmse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sse = self.add_weight(name='sse', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.expm1(tf.cast(y_true, tf.float32))
        y_pred = tf.math.expm1(tf.cast(y_pred, tf.float32))
        self.sse.assign_add(tf.reduce_sum(tf.square(y_true - y_pred)))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return tf.sqrt(tf.math.divide_no_nan(self.sse, self.count))

    def reset_state(self):
        self.sse.assign(0.0)
        self.count.assign(0.0)

    def reset_states(self):
        self.reset_state()


class OrigMAE(tf.keras.metrics.Metric):
    """MAE on original scale when training targets are log1p-transformed."""

    def __init__(self, name='orig_mae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.abs_err = self.add_weight(name='abs_err', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.expm1(tf.cast(y_true, tf.float32))
        y_pred = tf.math.expm1(tf.cast(y_pred, tf.float32))
        self.abs_err.assign_add(tf.reduce_sum(tf.abs(y_true - y_pred)))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.abs_err, self.count)

    def reset_state(self):
        self.abs_err.assign(0.0)
        self.count.assign(0.0)

    def reset_states(self):
        self.reset_state()


class MLData:
    """Flat Sequential model. Same architecture for 1-GW or N-GW targets;
    caller packs future-window features (e.g. summed team xg over N) into the row.
    Past form always comes from the usual last-5 / season windows on the clean match.
    """

    def __init__(self, clean_matches: List[CleanMatch], input_to_pred, input_keys, output_key, include_standard_stats,
                 output_learn: Optional[numpy.ndarray] = None, log_targets: bool = False):
        self.output_key = output_key
        self.ln_inputs = False
        self.mae_name = 'mean_absolute_error'
        self.rmse_name = 'root_mean_squared_error'

        inputs_with_headings = get_inputs_from_data_key(
            clean_matches, input_keys, include_standard_stats, include_opponent_features=True
        )
        self.player_feature_columns = list(inputs_with_headings[0].keys()) if inputs_with_headings else []
        self.input_learn = numpy.array([list(x.values()) for x in inputs_with_headings], dtype=float)

        if output_learn is not None:
            self.output_learn = numpy.asarray(output_learn, dtype=float).reshape(-1, 1)
            outputs_with_headings = [{output_key: float(v[0])} for v in self.output_learn]
        else:
            outputs_with_headings = get_outputs_from_data_key(clean_matches, output_key)
            self.output_learn = numpy.array([list(x.values()) for x in outputs_with_headings], dtype=float)

        inputs_pred_with_headings = get_inputs_from_data_key(
            input_to_pred, input_keys, include_standard_stats, include_opponent_features=True
        )
        self.input_to_pred = numpy.array([list(x.values()) for x in inputs_pred_with_headings], dtype=float)

        if (
            output_key == 'total_points'
            and not os.environ.get('PL_SWEEP_TRIAL')
            and not os.environ.get('PL_SIMULATION_BUILD')
        ):
            self.write_clean_data_csvs(
                inputs_with_headings, outputs_with_headings, inputs_pred_with_headings,
                clean_matches, input_to_pred
            )

        self.input_learn, self.input_test, self.output_learn, self.output_test = train_test_split(
            self.input_learn, self.output_learn, test_size=0.2, random_state=42, shuffle=True
        )

        if log_targets:
            self.set_options(ln_inputs=True)
        self._inputs_prepared = False
        self.hparams = PlayerNetHParams.from_env()
        self.model_config = None

    def prepare_inputs(self):
        if self._inputs_prepared:
            return
        if self.ln_inputs:
            self.output_learn[self.output_learn < 0] = 0
            self.output_learn = numpy.log1p(self.output_learn)
            self.output_test[self.output_test < 0] = 0
            self.output_test = numpy.log1p(self.output_test)
        self.scaler = StandardScaler().fit(self.input_learn)
        self.input_learn = self.scaler.transform(self.input_learn)
        self.input_to_pred = self.scaler.transform(self.input_to_pred)
        self.input_test = self.scaler.transform(self.input_test)
        self.validate_inputs()
        self._inputs_prepared = True

    def _hidden_sizes(self, n_in: int) -> list:
        hp = self.hparams
        net_size = max(8, int(n_in * hp.width_mult))
        sizes = [net_size]
        if hp.n_hidden >= 2:
            sizes.append(max(4, net_size // 2))
        if hp.n_hidden >= 3:
            sizes.append(max(2, net_size // 4))
        if hp.n_hidden >= 4:
            sizes.append(max(2, net_size // 8))
        return sizes

    def setup_model(self):
        self.prepare_inputs()
        if getattr(self, 'model', None) is not None:
            del self.model
        hp = self.hparams
        # 1.5× width, 3 layers + L2 1e-3 (sweep-1 trial 09; stabler horizon ranks)
        reg = tf.keras.regularizers.l2(hp.l2)
        n_in = self.input_learn.shape[1]
        hidden_layers = self._hidden_sizes(n_in)
        dropouts = [hp.dropout_1, hp.dropout_2]
        self.model = Sequential()
        self.model.add(Dense(
            hidden_layers[0], kernel_regularizer=reg, input_shape=(n_in,), activation='relu'
        ))
        if dropouts[0] > 0:
            self.model.add(Dropout(dropouts[0]))
        for i, units in enumerate(hidden_layers[1:], start=1):
            self.model.add(Dense(units, kernel_regularizer=reg, activation='relu'))
            if i < len(dropouts) and dropouts[i] > 0:
                self.model.add(Dropout(dropouts[i]))
        self.model.add(Dense(1))
        loss_name = (hp.loss or 'huber').lower()
        opt, opt_label = _build_optimizer(hp)
        self.model_config = {
            **hp.as_dict(),
            'hidden_layers': hidden_layers,
            'n_features': int(n_in),
            'max_epochs': EPOCH_MAP.get(self.output_key, EPOCHS),
            'loss': f'MAE' if loss_name == 'mae' else f'Huber(delta={hp.huber_delta})',
            'optimizer': opt_label,
        }
        print(
            f'Model capacity: hidden={hidden_layers}, optimizer={opt_label}, '
            f'l2={hp.l2}, weight_decay={hp.weight_decay}, lr={hp.learning_rate} '
            f'(n_features={n_in}, loss={self.model_config["loss"]})'
        )

        if loss_name == 'mae':
            loss_fn = MeanAbsoluteError()
        else:
            loss_fn = Huber(delta=hp.huber_delta)
        if self.ln_inputs:
            metrics = [OrigRMSE(), OrigMAE()]
        else:
            metrics = [RootMeanSquaredError(), MAE()]
        self.model.compile(loss=loss_fn, optimizer=opt, metrics=metrics)

    def set_options(self, ln_inputs=None):
        self.ln_inputs = ln_inputs if ln_inputs else self.ln_inputs
        if self.ln_inputs:
            self.mae_name = 'orig_mae'
            self.rmse_name = 'orig_rmse'

    def write_clean_data_csvs(self, inputs_with_headings, outputs_with_headings, inputs_pred_with_headings, clean_matches, input_to_pred):
        non_learning_cols = ['id', 'player_id', 'player_name']
        in_cols = non_learning_cols + [heading for heading in inputs_with_headings[0].keys() if heading not in non_learning_cols]
        out_cols = non_learning_cols + [heading for heading in outputs_with_headings[0].keys() if heading not in non_learning_cols]
        self.write_dicts_to_csv(self.add_names_to_dicts(clean_matches, inputs_with_headings), "X Train.csv", in_cols)
        self.write_dicts_to_csv(self.add_names_to_dicts(clean_matches, outputs_with_headings), "y Train.csv", out_cols)
        self.write_dicts_to_csv(self.add_names_to_dicts(input_to_pred, inputs_pred_with_headings), "X Predict.csv", in_cols)

    def add_names_to_dicts(self, clean_matches, data):
        data_copy = data.copy()
        for row, clean_match in zip(data_copy, clean_matches):
            row['player_name'] = clean_match['player_name']
            row['player_id'] = clean_match['player_id']
            row['id'] = clean_match['id']
        return data_copy

    def write_dicts_to_csv(self, data, filename, columns=None):
        if not data:
            raise ValueError("The data list is empty.")

        columns = data[0].keys() if not columns else columns

        with open(filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)

    def validate_inputs(self):
        assert self.input_learn.shape[0] == self.output_learn.shape[0], \
            f"Train inputs ({self.input_learn.shape}) and outputs ({self.output_learn.shape}) mismatch"
        assert self.input_test.shape[0] == self.output_test.shape[0], \
            f"Test inputs ({self.input_test.shape}) and outputs ({self.output_test.shape}) mismatch"
        inputs = [self.input_learn, self.input_test, self.output_learn, self.output_test]
        names = ["input_learn", "input_test", "output_learn", "output_test"]
        for name, arr in zip(names, inputs):
            assert isinstance(arr, numpy.ndarray), f"{name} is not a NumPy array"
            assert numpy.issubdtype(arr.dtype, numpy.number), f"{name} has non-numeric dtype {arr.dtype}"
            assert numpy.all(numpy.isfinite(arr)), f"{name} contains NaN or Inf values"

        print("All inputs/outputs have matching shapes and valid numeric values")

    def get_predictions(self):
        if self.output_key in EPOCH_MAP and '--quick' not in sys.argv:
            epochs = EPOCH_MAP[self.output_key]
        else:
            epochs = EPOCHS

        hp = self.hparams
        # callbacks = [
        #     EarlyStopping(monitor=f"val_{self.mae_name}", patience=10, restore_best_weights=True),
        #     ReduceLROnPlateau(monitor=f"val_{self.rmse_name}", factor=0.5, patience=5),
        # ]
        callbacks = [
            EarlyStopping(
                monitor=f"val_{self.mae_name}",
                patience=hp.early_stopping_patience,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor=f"val_{self.rmse_name}",
                factor=hp.reduce_lr_factor,
                patience=hp.reduce_lr_patience,
                min_lr=hp.reduce_lr_min,
            ),
        ]
        history = self.model.fit(
            self.input_learn, self.output_learn,
            epochs=epochs, batch_size=hp.batch_size, validation_split=0.2,
            verbose=2, callbacks=callbacks, shuffle=True
        )
        epochs_ran = len(history.history['loss'])
        val_mae_key = f'val_{self.mae_name}'
        val_rmse_key = f'val_{self.rmse_name}'
        fit_stats = {
            'epochs_ran': epochs_ran,
            'max_epochs': epochs,
            'train_mae': float(history.history[self.mae_name][-1]),
            'train_rmse': float(history.history[self.rmse_name][-1]),
            'best_val_mae': float(min(history.history[val_mae_key])),
            'final_val_mae': float(history.history[val_mae_key][-1]),
            'best_val_rmse': float(min(history.history[val_rmse_key])),
            'final_val_rmse': float(history.history[val_rmse_key][-1]),
            'selection_rmse': float(history.history[self.rmse_name][-1]),
        }
        if 'lr' in history.history:
            fit_stats['final_lr'] = float(history.history['lr'][-1])
        print({
            'epochs_ran': fit_stats['epochs_ran'],
            'model absolute error': fit_stats['train_mae'],
            'model rms error': fit_stats['train_rmse'],
            'best val mae': fit_stats['best_val_mae'],
        })

        test_loss, test_mae, test_rmse = self.model.evaluate(self.input_test, self.output_test, verbose=0)
        fit_stats['test_loss'] = float(test_loss)
        fit_stats['test_mae'] = float(test_mae)
        fit_stats['test_rmse'] = float(test_rmse)
        print({
            "test loss": test_loss,
            "test_orig_rmse": test_mae,
            "test_orig_mae": test_rmse
        })

        results = self.model.predict(self.input_to_pred, verbose=0)

        if self.ln_inputs:
            results = numpy.expm1(results)

        pred_flat = _flatten_predictions(results)
        fit_stats['pred_min'] = float(pred_flat.min())
        fit_stats['pred_max'] = float(pred_flat.max())
        fit_stats['pred_mean'] = float(pred_flat.mean())

        return results, fit_stats


def calc_baselines(ml_data: MLData):
    output_mean_baseline = numpy.full_like(ml_data.output_learn, numpy.mean(ml_data.output_learn))
    mae_mean_baseline = mean_absolute_error(ml_data.output_learn, output_mean_baseline)
    rmse_mean_baseline = mean_squared_error(ml_data.output_learn, output_mean_baseline, squared=False)

    output_median_baseline = numpy.full_like(ml_data.output_learn, numpy.median(ml_data.output_learn))
    mae_median_baseline = mean_absolute_error(ml_data.output_learn, output_median_baseline)
    rmse_median_baseline = mean_squared_error(ml_data.output_learn, output_median_baseline, squared=False)

    lin_pipe = _ridge_pipeline()
    lin_pipe.fit(ml_data.input_learn, ml_data.output_learn.ravel())
    y_lin = lin_pipe.predict(ml_data.input_learn)
    mae_lin = mean_absolute_error(ml_data.output_learn, y_lin)
    rmse_lin = mean_squared_error(ml_data.output_learn, y_lin, squared=False)

    return {
        'mean baseline absolute error': mae_mean_baseline,
        'mean baseline rms error': rmse_mean_baseline,
        'median baseline absolute error': mae_median_baseline,
        'median baseline rms error': rmse_median_baseline,
        "linear regression absolute error": mae_lin,
        "linear regression rms error": rmse_lin,
    }


class ReusablePointsErrorPredictor:
    def __init__(self, pipe, input_keys, include_standard_stats, global_rmse):
        self.pipe = pipe
        self.input_keys = list(input_keys)
        self.include_standard_stats = bool(include_standard_stats)
        self.global_rmse = float(global_rmse)

    def predict(self, clean_matches):
        if not clean_matches:
            return []
        rows = get_inputs_from_data_key(
            clean_matches,
            self.input_keys,
            self.include_standard_stats,
            include_opponent_features=True,
        )
        inputs = numpy.array([list(row.values()) for row in rows], dtype=float)
        return numpy.maximum(self.pipe.predict(inputs), 0.0).astype(float).tolist()


def fit_points_error_predictor(
    clean_matches, input_keys, include_standard_stats=True
):
    """Fit the per-fixture uncertainty model once for a simulated season."""
    print('Fitting horizon-1 linear models for per-match points error...')
    inputs_with_headings = get_inputs_from_data_key(
        clean_matches, input_keys, include_standard_stats, include_opponent_features=True
    )
    outputs_with_headings = get_outputs_from_data_key(clean_matches, 'total_points')
    X = numpy.array([list(x.values()) for x in inputs_with_headings], dtype=float)
    y = numpy.array([list(x.values()) for x in outputs_with_headings], dtype=float).ravel()

    if len(X) < 20:
        raise ValueError('Not enough rows for reusable points error model')

    points_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", RidgeCV(alphas=numpy.logspace(-3, 3, 13), store_cv_values=False)),
    ])
    n_cv = 5 if len(X) >= 50 else 3
    oof_pred = cross_val_predict(points_pipe, X, y, cv=n_cv)
    abs_resid = numpy.abs(y - oof_pred)
    global_rmse = float(mean_squared_error(y, oof_pred, squared=False))

    err_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", RidgeCV(alphas=numpy.logspace(-3, 3, 13), store_cv_values=False)),
    ])
    err_pipe.fit(X, abs_resid)
    return ReusablePointsErrorPredictor(
        err_pipe,
        input_keys,
        include_standard_stats,
        global_rmse,
    )


def fit_and_predict_points_error_margins(
    clean_matches, future_clean_matches, input_keys, include_standard_stats=True
):
    """Per-match points_error via a second Ridge that predicts |residual| from the same features."""
    if not future_clean_matches:
        return [], 0.0
    predictor = fit_points_error_predictor(
        clean_matches,
        input_keys,
        include_standard_stats,
    )
    errs = predictor.predict(future_clean_matches)

    print(
        f'Linear horizon-1 OOF RMSE: {predictor.global_rmse:.3f}; '
        f'per-match |residual| model — future err mean={float(numpy.mean(errs)):.3f} '
        f'range=({float(numpy.min(errs)):.3f}, {float(numpy.max(errs)):.3f})'
    )
    return errs, predictor.global_rmse


def fit_linear_points_error_margin(clean_matches, input_keys, include_standard_stats=True) -> float:
    """Global OOF RMSE from the points Ridge (logging / legacy). Prefer per-match margins."""
    inputs_with_headings = get_inputs_from_data_key(
        clean_matches, input_keys, include_standard_stats, include_opponent_features=True
    )
    outputs_with_headings = get_outputs_from_data_key(clean_matches, 'total_points')
    X = numpy.array([list(x.values()) for x in inputs_with_headings], dtype=float)
    y = numpy.array([list(x.values()) for x in outputs_with_headings], dtype=float).ravel()
    if len(X) < 20:
        return 0.0
    points_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", RidgeCV(alphas=numpy.logspace(-3, 3, 13), store_cv_values=False)),
    ])
    n_cv = 5 if len(X) >= 50 else 3
    oof_pred = cross_val_predict(points_pipe, X, y, cv=n_cv)
    return float(mean_squared_error(y, oof_pred, squared=False))


def get_avg_predictions(clean_matches, input_pred, input_keys, output_key, include_standard_stats,
                        output_learn=None, log_targets: bool = False, reps: int = 1):
    print('Getting predictions for %s...' % output_key)
    if reps > 1:
        print(f'  ({reps} independent fits; rep agreement reported after training)')
    is_quick = '--quick' in sys.argv
    if is_quick:
        print("(quick run)")
    ml_data = MLData(
        clean_matches, input_pred, input_keys, output_key, include_standard_stats,
        output_learn=output_learn, log_targets=log_targets
    )
    baselines = calc_baselines(ml_data)
    print(baselines)
    print('Model input columns:')
    for col in ml_data.player_feature_columns:
        print(f'  {col}')

    if use_linear_model():
        preds, stats = get_linear_predictions(ml_data)
        save_horizon_rep_run_metrics(
            ml_data,
            rep_stats=[stats],
            baselines=baselines,
            agreement=None,
            reps_used_for_avg=1,
            selection_threshold=0.0,
        )
        return preds

    preds_and_stats = []
    for rep in range(reps):
        if reps > 1:
            print(f'  fit {rep + 1}/{reps}...')
        ml_data.setup_model()
        preds_and_stats.append(ml_data.get_predictions())

    if reps > 1:
        agreement = compute_rep_agreement([m[0] for m in preds_and_stats])
        print_rep_agreement([m[0] for m in preds_and_stats])
    else:
        agreement = None

    hp = ml_data.hparams
    if hp.average_all_reps or len(preds_and_stats) == 1:
        kept_idx = list(range(len(preds_and_stats)))
        selection_threshold = None
        rule = 'all reps'
    else:
        maes = [float(m[1]['best_val_mae']) for m in preds_and_stats]
        best_mae = min(maes)
        selection_threshold = best_mae * hp.val_mae_keep_ratio
        kept_idx = [i for i, mae in enumerate(maes) if mae <= selection_threshold]
        if not kept_idx:
            kept_idx = [int(numpy.argmin(maes))]
        rule = (
            f'best_val_mae ≤ {selection_threshold:.4f} '
            f'(best {best_mae:.4f} × {hp.val_mae_keep_ratio:g})'
        )

    for i, (_pred, stats) in enumerate(preds_and_stats):
        stats['kept_for_avg'] = i in kept_idx
        stats['rep_index'] = i + 1

    model_predictions = [preds_and_stats[i][0] for i in kept_idx]
    if reps > 1:
        dropped = [i + 1 for i in range(len(preds_and_stats)) if i not in kept_idx]
        print(f'  averaging {len(model_predictions)}/{reps} fits ({rule})')
        if dropped:
            print(f'  dropped reps {dropped} (best_val_mae too high)')
        save_horizon_rep_run_metrics(
            ml_data,
            rep_stats=[m[1] for m in preds_and_stats],
            baselines=baselines,
            agreement=agreement,
            reps_used_for_avg=len(model_predictions),
            selection_threshold=selection_threshold or 0.0,
        )

    averaged_predictions = []
    for i in range(len(model_predictions[0])):
        single_match_predictions = []
        for prediction_array in model_predictions:
            single_match_predictions.append(prediction_array[i])
        averaged_predictions.append(sum(single_match_predictions) / len(single_match_predictions))

    return averaged_predictions
