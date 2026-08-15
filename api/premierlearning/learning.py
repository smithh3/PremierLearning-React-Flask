from dataclasses import dataclass
from typing import List
import numpy
import sys
import csv
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError as MAE
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from premierlearning.clean_match import CleanMatch, get_inputs_from_data_key, get_outputs_from_data_key


@dataclass
class MLOptions:
    reps: int
    epochs: int
    batch_size: int
    input_keys: list
    output_key: str

EPOCH_MAP = {
    'total_points': 500
}

EPOCHS = 8
BATCH_SIZE = 32
REPS = 1
if '--quick' in sys.argv:
    EPOCHS = 2
    # REPS = 2


class MLData:

    def __init__(self, clean_matches: List[CleanMatch], input_to_pred, input_keys, output_key, include_standard_stats):
        self.output_key = output_key
        self.ln_inputs = False
        self.mae_name = 'mean_absolute_error'
        self.rmse_name = 'root_mean_squared_error'

        inputs_with_headings = get_inputs_from_data_key(clean_matches, input_keys, include_standard_stats)
        self.input_learn = numpy.array([list(x.values()) for x in inputs_with_headings], dtype=float)
        outputs_with_headings = get_outputs_from_data_key(clean_matches, output_key)
        self.output_learn = numpy.array([list(x.values()) for x in outputs_with_headings], dtype=float)
        inputs_pred_with_headings = get_inputs_from_data_key(input_to_pred, input_keys, include_standard_stats)
        self.input_to_pred = numpy.array([list(x.values()) for x in inputs_pred_with_headings], dtype=float)

        if output_key == 'total_points':
            self.write_clean_data_csvs(inputs_with_headings, outputs_with_headings, inputs_pred_with_headings, clean_matches, input_to_pred)
        
        self.input_learn, self.input_test, self.output_learn, self.output_test = train_test_split(
            self.input_learn, self.output_learn, test_size=0.2, random_state=42, shuffle=True
        )
    
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
        reg = tf.keras.regularizers.l2(3e-5)
        net_size = self.input_learn.shape[1] * 3 / 2
        self.model = Sequential()
        self.model.add(Dense(int(net_size), kernel_regularizer=reg,input_shape=(self.input_learn.shape[1],), activation='relu'))
        # self.model.add(Dropout(0.3))
        self.model.add(Dense(int(net_size / 2), kernel_regularizer=reg, activation='relu'))
        # self.model.add(Dropout(0.1))
        self.model.add(Dense(int(net_size / 4), kernel_regularizer=reg, activation='relu'))
        self.model.add(Dense(1))

        opt = AdamW(learning_rate=1e-3, weight_decay=1e-5)
        # mae_loss = MeanAbsoluteError() # looked to be working quite well if huber doesn't work out
        huber_loss = Huber(delta=1.8) # delta is the loss value where evaluation switches from quadratic to linear
        self.model.compile(loss=huber_loss, optimizer=opt, metrics=[self.get_rmse_orig_func(), self.get_mae_orig_func()])
    
    def get_rmse_orig_func(self):
        if self.ln_inputs:
            def orig_rmse(y_true_log, y_pred_log):
                y_true = tf.math.expm1(y_true_log)
                y_pred = tf.math.expm1(y_pred_log)
                return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
            return orig_rmse
        else:
            return RootMeanSquaredError()

    def get_mae_orig_func(self):
        if self.ln_inputs:
            def orig_mae(y_true_log, y_pred_log):
                y_true = tf.math.expm1(y_true_log)
                y_pred = tf.math.expm1(y_pred_log)
                return tf.reduce_mean(tf.abs(y_true - y_pred))
            return orig_mae
        else:
            return MAE()
    
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

        # Extract headers from the keys of the first dictionary
        columns = data[0].keys() if not columns else columns

        with open(filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)

    def validate_inputs(self):
        # 1. Check shapes are consistent for X/y pairs
        assert self.input_learn.shape[0] == self.output_learn.shape[0], \
            f"Train inputs ({self.input_learn.shape}) and outputs ({self.output_learn.shape}) mismatch"
        assert self.input_test.shape[0] == self.output_test.shape[0], \
            f"Test inputs ({self.input_test.shape}) and outputs ({self.output_test.shape}) mismatch"
        inputs = [self.input_learn, self.input_test, self.output_learn, self.output_test]

        # 2. Check all arrays are numeric and finite
        for name, arr in zip(["input_learn", "input_test", "output_learn", "output_test"], inputs):
            assert isinstance(arr, numpy.ndarray), f"{name} is not a NumPy array"
            assert numpy.issubdtype(arr.dtype, numpy.number), f"{name} has non-numeric dtype {arr.dtype}"
            assert numpy.all(numpy.isfinite(arr)), f"{name} contains NaN or Inf values"

        print("All inputs/outputs have matching shapes and valid numeric values")

    def get_predictions(self):
        if self.output_key in EPOCH_MAP and '--quick' not in sys.argv:
            epochs = EPOCH_MAP[self.output_key]
        else:
            epochs = EPOCHS
        
        # patience is number of epochs before seeing improvement
        callbacks = [
            EarlyStopping(monitor=f"val_{self.mae_name}", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor=f"val_{self.rmse_name}", factor=0.5, patience=5),
            # ModelCheckpoint(filepath="best.keras", monitor="val_mean_absolute_error", mode="min", save_best_only=True, verbose=1),
        ]
        history = self.model.fit(self.input_learn, self.output_learn,
            epochs=epochs, batch_size=BATCH_SIZE, validation_split=0.2,
            verbose=1, callbacks=callbacks, shuffle=True)
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
        # test_results = self.model.predict(self.output_test)

        results = self.model.predict(self.input_to_pred)

        if self.ln_inputs:
            results = numpy.expm1(results)

        return results, history.history[self.rmse_name][-1]

# def loss_func(y_true, y_pred):
#         y_true = float(y_true)
#         y_pred = float(y_pred)

#         return abs(y_true - y_pred) ** 2
def calc_baselines(ml_data: MLData):
    output_mean_baseline = numpy.full_like(ml_data.output_learn, numpy.mean(ml_data.output_learn))
    mae_mean_baseline = mean_absolute_error(ml_data.output_learn, output_mean_baseline)
    rmse_mean_baseline = mean_squared_error(ml_data.output_learn, output_mean_baseline, squared=False)

    output_median_baseline = numpy.full_like(ml_data.output_learn, numpy.median(ml_data.output_learn))
    mae_median_baseline = mean_absolute_error(ml_data.output_learn, output_median_baseline)
    rmse_median_baseline = mean_squared_error(ml_data.output_learn, output_median_baseline, squared=False)

    lin_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca",    PCA(n_components=0.95, random_state=42)),  # keep 95% variance
        ("ridge",  RidgeCV(alphas=numpy.logspace(-3, 3, 13), store_cv_values=False))
    ])
    lin_pipe.fit(ml_data.input_learn, ml_data.output_learn)
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


def get_avg_predictions(clean_matches, input_pred, input_keys, output_key, include_standard_stats):
        print('Getting predictions for %s...' % output_key)
        is_quick = '--quick' in sys.argv
        if is_quick:
            print("(quick run)")
        ml_data = MLData(clean_matches, input_pred, input_keys, output_key, include_standard_stats)
        if output_key == 'total_points':
            ml_data.set_options(ln_inputs=True)
        print(calc_baselines(ml_data))
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
