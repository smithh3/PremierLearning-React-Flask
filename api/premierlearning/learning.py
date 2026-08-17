from dataclasses import dataclass
from typing import List, Optional
import numpy
import sys
import csv
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError as MAE
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_predict

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
REPS = 1
if '--quick' in sys.argv:
    EPOCHS = 2
    # REPS = 2


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
    return [numpy.asarray([float(p)], dtype=float) for p in preds]


def orig_rmse(y_true_log, y_pred_log):
    """RMSE on original scale when training targets are log1p-transformed."""
    y_true = tf.math.expm1(y_true_log)
    y_pred = tf.math.expm1(y_pred_log)
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def orig_mae(y_true_log, y_pred_log):
    """MAE on original scale when training targets are log1p-transformed."""
    y_true = tf.math.expm1(y_true_log)
    y_pred = tf.math.expm1(y_pred_log)
    return tf.reduce_mean(tf.abs(y_true - y_pred))


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

        if output_key == 'total_points':
            self.write_clean_data_csvs(
                inputs_with_headings, outputs_with_headings, inputs_pred_with_headings,
                clean_matches, input_to_pred
            )

        self.input_learn, self.input_test, self.output_learn, self.output_test = train_test_split(
            self.input_learn, self.output_learn, test_size=0.2, random_state=42, shuffle=True
        )

        if log_targets:
            self.set_options(ln_inputs=True)

    def setup_model(self):
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
        # Smaller net + stronger L2 / weight decay → stabler rankings across runs
        reg = tf.keras.regularizers.l2(1e-3)
        n_in = self.input_learn.shape[1]
        net_size = max(8, n_in)  # was 1.5× features, 3 hidden layers
        self.model = Sequential()
        self.model.add(Dense(
            net_size, kernel_regularizer=reg, input_shape=(n_in,), activation='relu'
        ))
        self.model.add(Dense(max(4, net_size // 2), kernel_regularizer=reg, activation='relu'))
        self.model.add(Dense(1))
        print(
            f'Model capacity: hidden=[{net_size}, {max(4, net_size // 2)}], '
            f'l2=1e-3, weight_decay=1e-4 (n_features={n_in})'
        )

        opt = AdamW(learning_rate=1e-3, weight_decay=1e-4)
        huber_loss = Huber(delta=1.8)
        if self.ln_inputs:
            metrics = [orig_rmse, orig_mae]
        else:
            metrics = [RootMeanSquaredError(), MAE()]
        self.model.compile(loss=huber_loss, optimizer=opt, metrics=metrics)

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

        callbacks = [
            EarlyStopping(monitor=f"val_{self.mae_name}", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor=f"val_{self.rmse_name}", factor=0.5, patience=5),
        ]
        history = self.model.fit(
            self.input_learn, self.output_learn,
            epochs=epochs, batch_size=BATCH_SIZE, validation_split=0.2,
            verbose=2, callbacks=callbacks, shuffle=True
        )
        print({
            'model absolute error': history.history[self.mae_name][-1],
            'model rms error': history.history[self.rmse_name][-1]
        })

        test_loss, test_mae, test_rmse = self.model.evaluate(self.input_test, self.output_test, verbose=0)
        print({
            "test loss": test_loss,
            "test_orig_rmse": test_mae,
            "test_orig_mae": test_rmse
        })

        results = self.model.predict(self.input_to_pred, verbose=0)

        if self.ln_inputs:
            results = numpy.expm1(results)

        return results, history.history[self.rmse_name][-1]


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


def fit_and_predict_points_error_margins(
    clean_matches, future_clean_matches, input_keys, include_standard_stats=True
):
    """Per-match points_error via a second Ridge that predicts |residual| from the same features."""
    print('Fitting horizon-1 linear models for per-match points error...')
    inputs_with_headings = get_inputs_from_data_key(
        clean_matches, input_keys, include_standard_stats, include_opponent_features=True
    )
    outputs_with_headings = get_outputs_from_data_key(clean_matches, 'total_points')
    X = numpy.array([list(x.values()) for x in inputs_with_headings], dtype=float)
    y = numpy.array([list(x.values()) for x in outputs_with_headings], dtype=float).ravel()

    if len(X) < 20 or not future_clean_matches:
        print('Not enough rows for linear error margin; defaulting to 0')
        return [0.0] * len(future_clean_matches), 0.0

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

    future_inputs = get_inputs_from_data_key(
        future_clean_matches, input_keys, include_standard_stats, include_opponent_features=True
    )
    X_future = numpy.array([list(x.values()) for x in future_inputs], dtype=float)
    errs = err_pipe.predict(X_future)
    errs = numpy.maximum(errs, 0.0)

    print(
        f'Linear horizon-1 OOF RMSE: {global_rmse:.3f}; '
        f'per-match |residual| model — future err mean={float(errs.mean()):.3f} '
        f'range=({float(errs.min()):.3f}, {float(errs.max()):.3f})'
    )
    return errs.astype(float).tolist(), global_rmse


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
                        output_learn=None, log_targets: bool = False):
        print('Getting predictions for %s...' % output_key)
        is_quick = '--quick' in sys.argv
        if is_quick:
            print("(quick run)")
        ml_data = MLData(
            clean_matches, input_pred, input_keys, output_key, include_standard_stats,
            output_learn=output_learn, log_targets=log_targets
        )
        print(calc_baselines(ml_data))
        print('Model input columns:')
        for col in ml_data.player_feature_columns:
            print(f'  {col}')

        if use_linear_model():
            return get_linear_predictions(ml_data)

        ml_data.setup_model()

        preds_and_loss = []
        for _ in range(REPS):
            preds_and_loss.append(ml_data.get_predictions())

        min_loss = min(m[1] for m in preds_and_loss)

        model_predictions = [m[0] for m in preds_and_loss if m[1] <= min_loss * 1.2]

        averaged_predictions = []
        for i in range(len(model_predictions[0])):
            single_match_predictions = []
            for prediction_array in model_predictions:
                single_match_predictions.append(prediction_array[i])
            averaged_predictions.append(sum(single_match_predictions) / len(single_match_predictions))

        return averaged_predictions
