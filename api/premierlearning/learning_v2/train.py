"""Train the v2 points head. Default loss is Huber on raw points."""
from __future__ import annotations

import sys

import numpy
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanAbsoluteError as MAE
from tensorflow.keras.metrics import RootMeanSquaredError

from premierlearning.learning import (
    EPOCHS,
    EPOCH_MAP,
    POINTS_LOSS_FLAG,
    OrigMAE,
    OrigRMSE,
    PlayerNetHParams,
    _build_optimizer,
    _flag_present,
    _loss_fn,
    parse_h1_poisson_recipe,
    parse_points_loss,
    parse_points_reps,
    select_reps_for_average,
    uses_log_targets,
)
from premierlearning.learning_v2.dataset import mask_array
from premierlearning.learning_v2.model import build_model
from premierlearning.learning_v2.plan import input_specs
from premierlearning.learning_v2.score import (
    blended_tier_weights,
    format_form_report,
    format_tier_report,
    recent_gameweek_rates,
    score_by_recent_form,
    score_by_tier,
)


def resolve_loss_name(loss_name=None) -> str:
    """Huber on raw points, unless the process asked for another loss."""
    if loss_name:
        return str(loss_name).strip().lower()
    if _flag_present(POINTS_LOSS_FLAG) or parse_h1_poisson_recipe():
        return parse_points_loss()
    return 'huber_raw'


def _copy_arrays(arrays):
    return {key: numpy.array(value, copy=True) for key, value in arrays.items()}


def _take(arrays, index):
    index = numpy.asarray(index)
    return {key: value[index] for key, value in arrays.items()}


def _fit_scalers(arrays, specs):
    stats = {}
    for spec in specs:
        if spec.kind != 'float':
            continue
        mask = mask_array(spec.path, arrays)
        values = arrays[spec.key]
        flat = values.reshape(-1) if mask is None else values[mask > 0.5]
        if flat.size == 0:
            stats[spec.key] = (0.0, 1.0)
            continue
        std = float(numpy.std(flat))
        if std < 1e-6:
            std = 1.0
        stats[spec.key] = (float(numpy.mean(flat)), std)
    return stats


def _apply_scalers(arrays, specs, stats):
    for spec in specs:
        if spec.kind != 'float':
            continue
        mean, std = stats[spec.key]
        values = arrays[spec.key]
        values[...] = (values - mean) / std
        mask = mask_array(spec.path, arrays)
        if mask is not None:
            values *= (mask > 0.5).astype(values.dtype)


def stop_after_plateau(current_epoch, best_epoch, fraction=0.2, cap=40, floor=5) -> bool:
    """True once the wait since the best epoch reaches 20% of epochs run, between 5 and 40."""
    waited = int(current_epoch) - int(best_epoch)
    if waited <= 0:
        return False
    allowed = min(float(cap), max(float(floor), float(fraction) * int(current_epoch)))
    return waited >= allowed


class PlateauStopping(Callback):
    """Early stop with patience = min(40, max(5, 20% of epochs so far)). Restores the best weights."""

    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        self.best = None
        self.best_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = int(epoch) + 1
        value = logs.get(self.monitor)
        if value is None:
            return
        value = float(value)
        if self.best is None or value < self.best:
            self.best = value
            self.best_epoch = current
            self.best_weights = self.model.get_weights()
            return
        if not stop_after_plateau(current, self.best_epoch):
            return
        print(
            'learning_v2 early stop at epoch %s: best validation at epoch %s '
            '(patience is 20%% of epochs run, at least 5 and at most 40)'
            % (current, self.best_epoch)
        )
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
        self.model.stop_training = True


def _metric_names(log_targets: bool):
    if log_targets:
        return 'orig_mae', 'orig_rmse'
    return 'mean_absolute_error', 'root_mean_squared_error'


def fit_and_predict(plan, vocab, train_batch, predict_batch, hparams=None, loss_name=None, reps=None, report_batch=None):
    """Fit on train_batch and return one non-negative points prediction per predict row."""
    hparams = hparams or PlayerNetHParams.from_env()
    loss_name = resolve_loss_name(loss_name)
    reps = parse_points_reps() if reps is None else int(reps)
    specs = input_specs(plan)
    log_targets = uses_log_targets(loss_name)
    labels = numpy.asarray(train_batch.labels, dtype=numpy.float32).reshape(-1, 1)
    count = int(labels.shape[0])
    if count < 1:
        raise ValueError('learning_v2 has no training rows')

    index = numpy.arange(count)
    if count >= 20:
        train_index, test_index = train_test_split(
            index, test_size=0.2, random_state=42, shuffle=True,
        )
    else:
        train_index, test_index = index, index[:0]

    train_x = _take(train_batch.arrays, train_index)
    test_x = _take(train_batch.arrays, test_index)
    pred_x = _copy_arrays(predict_batch.arrays)
    stats = _fit_scalers(train_x, specs)
    _apply_scalers(train_x, specs, stats)
    if len(test_index):
        _apply_scalers(test_x, specs, stats)
    if predict_batch.matches:
        _apply_scalers(pred_x, specs, stats)
    report_x = None
    if report_batch is not None and report_batch.matches:
        report_x = _copy_arrays(report_batch.arrays)
        _apply_scalers(report_x, specs, stats)

    y_train = labels[train_index]
    y_test = labels[test_index]
    raw_test = numpy.asarray(y_test, dtype=numpy.float32).reshape(-1)
    if log_targets:
        y_train = numpy.log1p(numpy.maximum(y_train, 0.0))
        y_test = numpy.log1p(numpy.maximum(y_test, 0.0))

    mae_name, rmse_name = _metric_names(log_targets)
    epochs = EPOCHS if '--quick' in sys.argv else EPOCH_MAP.get('total_points', EPOCHS)
    val_split = 0.2 if len(train_index) >= 10 else 0.0
    pred_count = len(predict_batch.matches)
    collected = []

    for rep in range(max(1, reps)):
        if reps > 1:
            print('  learning_v2 fit %s/%s' % (rep + 1, reps))
        model = build_model(plan, vocab, hparams, loss_name=loss_name)
        optimizer, opt_label = _build_optimizer(hparams)
        metrics = [OrigRMSE(), OrigMAE()] if log_targets else [RootMeanSquaredError(), MAE()]
        # Weighted metrics so early stopping follows the same weights as the loss.
        model.compile(
            loss=_loss_fn(loss_name, hparams),
            optimizer=optimizer,
            weighted_metrics=metrics,
        )
        sample_weight = blended_tier_weights(labels[train_index].reshape(-1))
        print(
            'learning_v2 %s: loss=%s, fixture weights 0.2 of the way from equal to '
            'the score bands (blank x1, 3-5 x1.2, 6-9 x1.6, 10+ x2.4, average weight 1), '
            'optimizer=%s, epochs=%s, train=%s, early stop patience=20%% of epochs run (at least 5, at most 40)'
            % (plan.name, loss_name, opt_label, epochs, len(train_index))
        )
        callbacks = []
        if val_split > 0:
            callbacks = [
                PlateauStopping(monitor='val_%s' % mae_name),
                ReduceLROnPlateau(
                    monitor='val_%s' % rmse_name,
                    factor=hparams.reduce_lr_factor,
                    patience=hparams.reduce_lr_patience,
                    min_lr=hparams.reduce_lr_min,
                ),
            ]
        history = model.fit(
            train_x, y_train,
            epochs=epochs,
            batch_size=hparams.batch_size,
            validation_split=val_split,
            verbose=2,
            callbacks=callbacks,
            shuffle=True,
            sample_weight=sample_weight,
        )
        val_key = 'val_%s' % mae_name
        if val_key in history.history:
            best_val = float(min(history.history[val_key]))
        else:
            best_val = float(history.history[mae_name][-1])
        test_pred = None
        if len(test_index):
            evaluated = model.evaluate(test_x, y_test, verbose=0)
            print({'test_loss': float(evaluated[0]), 'test_metrics': [float(v) for v in evaluated[1:]]})
            test_pred = numpy.asarray(model.predict(test_x, verbose=0), dtype=numpy.float32)
        if pred_count:
            raw = model.predict(pred_x, verbose=0)
        else:
            raw = numpy.zeros((0, 1), dtype=numpy.float32)
        report_raw = None
        if report_x is not None:
            report_raw = numpy.asarray(model.predict(report_x, verbose=0), dtype=numpy.float32)
        collected.append((
            numpy.asarray(raw, dtype=numpy.float32),
            {'best_val_mae': best_val},
            test_pred,
            report_raw,
        ))
        del model

    kept, _threshold, rule = select_reps_for_average([item[1] for item in collected], hparams)
    if reps > 1:
        print('  averaging %s/%s learning_v2 fits (%s)' % (len(kept), reps, rule))
    averaged = numpy.mean([collected[i][0] for i in kept], axis=0)
    if raw_test.size and collected[kept[0]][2] is not None:
        averaged_test = numpy.mean([collected[i][2] for i in kept], axis=0)
        if log_targets:
            averaged_test = numpy.expm1(averaged_test)
        averaged_test = numpy.maximum(0.0, numpy.asarray(averaged_test, dtype=float)).reshape(-1)
        print(format_tier_report(score_by_tier(raw_test, averaged_test)))
        test_matches = [train_batch.matches[int(i)] for i in test_index]
        print(format_form_report(score_by_recent_form(
            raw_test, averaged_test, recent_gameweek_rates(test_matches),
        )))
    if log_targets:
        averaged = numpy.expm1(averaged)
    averaged = numpy.maximum(0.0, averaged.reshape(-1))
    report_averaged = None
    if report_x is not None and collected[kept[0]][3] is not None:
        report_averaged = numpy.mean([collected[i][3] for i in kept], axis=0)
        if log_targets:
            report_averaged = numpy.expm1(report_averaged)
        report_averaged = numpy.maximum(0.0, numpy.asarray(report_averaged, dtype=float)).reshape(-1)
    return averaged, report_averaged
