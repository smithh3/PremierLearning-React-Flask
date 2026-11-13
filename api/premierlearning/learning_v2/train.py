"""Train the v2 points head. Default loss is Huber on raw points."""
from __future__ import annotations

import gc
import math
import sys

import numpy
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanAbsoluteError as MAE
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.utils import Sequence

from premierlearning.learning import (
    EPOCHS,
    EPOCH_MAP,
    POINTS_LOSS_FLAG,
    OrigMAE,
    OrigRMSE,
    PlayerNetHParams,
    SMEAR_OFF,
    _build_optimizer,
    _flag_present,
    _loss_fn,
    _parse_string_flag,
    apply_smearing,
    parse_h1_poisson_recipe,
    parse_points_loss,
    parse_points_reps,
    select_reps_for_average,
    smearing_factor,
    uses_log_targets,
)
from premierlearning.learning_v2.dataset import MINUTES_TARGET_BANDS, mask_array, minutes_targets, score_targets
from premierlearning.learning_v2.model import build_model, masked_poisson, minutes_only, points_only
from premierlearning.learning_v2.plan import POINTS_STAGED
from premierlearning.learning_v2.staged import goal_band_loss
from premierlearning.learning_v2.plan import input_specs
from premierlearning.learning_v2.score import (
    LOSS_TIER_MIX,
    SMOOTH_ALPHA,
    SMOOTH_POWER,
    SMOOTH_SCALE,
    WEIGHTING_TIER,
    LossWeighting,
    format_form_report,
    format_tier_report,
    recent_gameweek_rates,
    score_by_recent_form,
    score_by_tier,
)

WEIGHTING_FLAG = '--v2-weighting'
TIER_MIX_FLAG = '--v2-tier-mix'
SMOOTH_ALPHA_FLAG = '--v2-smooth-alpha'
SMOOTH_SCALE_FLAG = '--v2-smooth-scale'
SMOOTH_POWER_FLAG = '--v2-smooth-power'
SMOOTH_CAP_FLAG = '--v2-smooth-cap'


def resolve_loss_name(loss_name=None) -> str:
    """Huber on raw points, unless the process asked for another loss."""
    if loss_name:
        return str(loss_name).strip().lower()
    if _flag_present(POINTS_LOSS_FLAG) or parse_h1_poisson_recipe():
        return parse_points_loss()
    return 'huber_raw'


def _float_flag(flag, default) -> float:
    raw = _parse_string_flag(flag, None)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        raise SystemExit('%s requires a number' % flag)


def resolve_weighting(weighting=None) -> LossWeighting:
    """The blended score bands, unless the process asked for none or smooth."""
    if weighting is not None:
        return weighting
    return LossWeighting(
        kind=_parse_string_flag(WEIGHTING_FLAG, WEIGHTING_TIER),
        tier_mix=_float_flag(TIER_MIX_FLAG, LOSS_TIER_MIX),
        alpha=_float_flag(SMOOTH_ALPHA_FLAG, SMOOTH_ALPHA),
        scale=_float_flag(SMOOTH_SCALE_FLAG, SMOOTH_SCALE),
        power=_float_flag(SMOOTH_POWER_FLAG, SMOOTH_POWER),
        cap=_float_flag(SMOOTH_CAP_FLAG, None) if _parse_string_flag(SMOOTH_CAP_FLAG, None) else None,
    )


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


# Team histories are stored once per (club, cut, kickoff) in batch.tables and
# looked up per batch of rows, instead of copied into every player row.
PREDICT_BATCH_SIZE = 256


def _table_inputs(batch) -> dict:
    return getattr(batch, 'table_inputs', None) or {}


def _row_specs(specs, batch) -> list:
    """The specs whose arrays sit on each row (everything not in a shared table)."""
    shared = _table_inputs(batch)
    return [spec for spec in specs if spec.key not in shared]


def _fit_table_scalers(batch, specs) -> dict:
    """Mean and spread of each shared-table float input, over its real (unpadded) slots."""
    stats = {}
    shared = _table_inputs(batch)
    for spec in specs:
        if spec.key not in shared or spec.kind != 'float':
            continue
        name, field, _ref = shared[spec.key]
        table = batch.tables[name]
        flat = table[field][table['mask'] > 0.5]
        if flat.size == 0:
            stats[spec.key] = (0.0, 1.0)
            continue
        std = float(numpy.std(flat))
        stats[spec.key] = (float(numpy.mean(flat)), std if std >= 1e-6 else 1.0)
    return stats


def _feed_tables(batch, specs, stats) -> dict:
    """{input key: (scaled copy of its table column, row array of indexes)}."""
    feed = {}
    shared = _table_inputs(batch)
    for spec in specs:
        if spec.key not in shared:
            continue
        name, field, ref = shared[spec.key]
        table = batch.tables[name]
        values = numpy.array(table[field], copy=True)
        if spec.kind == 'float':
            mean, std = stats[spec.key]
            values = ((values - mean) / std).astype(numpy.float32)
            values *= (table['mask'] > 0.5).astype(numpy.float32)
        feed[spec.key] = (values, ref)
    return feed


class TableFeed(Sequence):
    """Keras batches built on demand: row inputs sliced, shared-table inputs looked up."""

    def __init__(self, arrays, input_keys, feed, y=None, weights=None, batch_size=32,
                 shuffle=False, seed=42):
        self.arrays = arrays
        self.input_keys = [key for key in input_keys if key in arrays]
        self.feed = feed
        self.y = y
        self.weights = weights
        self.batch_size = max(1, int(batch_size))
        self.count = len(next(iter(arrays.values()))) if arrays else 0
        self.order = numpy.arange(self.count)
        self.shuffle = bool(shuffle)
        self.rng = numpy.random.RandomState(int(seed))
        if self.shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return int(math.ceil(self.count / float(self.batch_size)))

    def __getitem__(self, index):
        rows = self.order[index * self.batch_size:(index + 1) * self.batch_size]
        x = {key: self.arrays[key][rows] for key in self.input_keys}
        for key, (values, ref) in self.feed.items():
            x[key] = values[self.arrays[ref][rows]]
        if self.y is None:
            return x
        if isinstance(self.y, dict):
            y = {name: value[rows] for name, value in self.y.items()}
        else:
            y = self.y[rows]
        if self.weights is None:
            return x, y
        return x, y, self.weights[rows]

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def _rows(arrays, start, stop):
    return {key: value[start:stop] for key, value in arrays.items()}


def _slice_y(y, start, stop):
    if y is None:
        return None
    if isinstance(y, dict):
        return {name: value[start:stop] for name, value in y.items()}
    return y[start:stop]


def _fit_with_tables(model, train_x, fit_y, sample_weight, validation, val_split, feed, input_keys,
                     batch_size, epochs, callbacks, seed):
    """model.fit for a plan with shared tables. validation is (x, y) or None.

    With no validation set, the last val_split of the training rows validate,
    as Keras's own validation_split does.
    """
    if validation is None and val_split > 0:
        count = len(next(iter(train_x.values())))
        split_at = int(math.floor(count * (1.0 - val_split)))
        validation = (_rows(train_x, split_at, count), _slice_y(fit_y, split_at, count))
        val_weights = None if sample_weight is None else sample_weight[split_at:]
        train_x = _rows(train_x, 0, split_at)
        fit_y = _slice_y(fit_y, 0, split_at)
        sample_weight = None if sample_weight is None else sample_weight[:split_at]
    else:
        val_weights = None
    validation_feed = None
    if validation is not None:
        validation_feed = TableFeed(
            validation[0], input_keys, feed, y=validation[1], weights=val_weights, batch_size=batch_size,
        )
    training = TableFeed(
        train_x, input_keys, feed, y=fit_y, weights=sample_weight, batch_size=batch_size,
        shuffle=True, seed=seed,
    )
    return model.fit(
        training, epochs=epochs, validation_data=validation_feed, verbose=2,
        callbacks=callbacks, shuffle=False,
    )


# Above ReduceLROnPlateau's patience (8), so the learning rate drops at least once
# before a fit can stop. At 5, v2 fits stopped by epoch ~10 at the starting rate;
# at 15 they ran ~40 epochs and scored better on held-out rows. Live and any
# predictions we act on use this; evaluation test runs can pass a lower floor.
PATIENCE_FLOOR = 15


def stop_after_plateau(current_epoch, best_epoch, fraction=0.2, cap=40, floor=5) -> bool:
    """True once the wait since the best epoch reaches 20% of epochs run, between 5 and 40."""
    waited = int(current_epoch) - int(best_epoch)
    if waited <= 0:
        return False
    allowed = min(float(cap), max(float(floor), float(fraction) * int(current_epoch)))
    return waited >= allowed


class PlateauStopping(Callback):
    """Early stop with patience = min(40, max(floor, 20% of epochs so far)). Restores the best weights."""

    def __init__(self, monitor, floor=PATIENCE_FLOOR):
        super().__init__()
        self.monitor = monitor
        self.floor = floor
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
        if not stop_after_plateau(current, self.best_epoch, floor=self.floor):
            return
        print(
            'learning_v2 early stop at epoch %s: best validation at epoch %s '
            '(patience is 20%% of epochs run, at least %s and at most 40)'
            % (current, self.best_epoch, self.floor)
        )
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
        self.model.stop_training = True


def _metric_names(log_targets: bool):
    if log_targets:
        return 'orig_mae', 'orig_rmse'
    return 'mean_absolute_error', 'root_mean_squared_error'


class FittedPoints:
    """The kept fits and the scalers they were trained with.

    Reusable on later snapshots, provided their batches were built with the same
    vocab: embedding indexes come from it.
    """

    def __init__(self, models, specs, stats, log_targets, smearing=None, smear_mode=SMEAR_OFF,
                 minutes_models=None):
        self.models = list(models)
        self.specs = specs
        self.stats = stats
        self.log_targets = bool(log_targets)
        self.smearing = smearing
        self.smear_mode = smear_mode
        # The kept fits' minutes-band heads, when they were trained with one.
        self.minutes_models = [model for model in (minutes_models or []) if model is not None]

    def _outputs(self, models, batch):
        """Each model's output on the batch, inputs scaled as in training."""
        arrays = _copy_arrays(batch.arrays)
        row_specs = _row_specs(self.specs, batch)
        _apply_scalers(arrays, row_specs, self.stats)
        if not _table_inputs(batch):
            return [numpy.asarray(model.predict(arrays, verbose=0), dtype=numpy.float32) for model in models]
        feed = _feed_tables(batch, self.specs, self.stats)
        rows = TableFeed(arrays, [spec.key for spec in row_specs], feed, batch_size=PREDICT_BATCH_SIZE)
        return [numpy.asarray(model.predict(rows, verbose=0), dtype=numpy.float32) for model in models]

    def predict_minutes_bands(self, batch):
        """Chance of 0, 1-59 and 60+ minutes per row, or None without a bands head.

        The head is trained alongside points only to shape the shared layers;
        this reads what it learned. The kept fits are averaged.
        """
        if not self.minutes_models:
            return None
        if not batch.matches:
            return numpy.zeros((0, 3), dtype=numpy.float32)
        return numpy.mean(self._outputs(self.minutes_models, batch), axis=0)

    def raw_predict(self, batch) -> numpy.ndarray:
        """Mean model output before the log transform is undone, one row per match."""
        if not batch.matches:
            return numpy.zeros((0, 1), dtype=numpy.float32)
        return numpy.mean(self._outputs(self.models, batch), axis=0)

    def predict(self, batch) -> numpy.ndarray:
        """Non-negative raw points, one per match in the batch."""
        averaged = self.raw_predict(batch)
        if self.log_targets:
            averaged = apply_smearing(numpy.expm1(averaged), self.smearing, self.smear_mode)
        return numpy.maximum(0.0, numpy.asarray(averaged, dtype=float).reshape(-1))


def fit_and_predict(plan, vocab, train_batch, predict_batch, hparams=None, loss_name=None,
                    reps=None, report_batch=None, weighting=None):
    """Fit on train_batch and return one non-negative points prediction per predict row."""
    fitted = fit_points(
        plan, vocab, train_batch, hparams=hparams, loss_name=loss_name, reps=reps,
        weighting=weighting,
    )
    report = None
    if report_batch is not None and report_batch.matches:
        report = fitted.predict(report_batch)
    return fitted.predict(predict_batch), report


def _group_split(positions, groups, test_size):
    """Split positions so no group lands on both sides."""
    kept, held = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42).split(
        positions, groups=groups[positions],
    ))
    return positions[kept], positions[held]


def fit_points(plan, vocab, train_batch, hparams=None, loss_name=None, reps=None, weighting=None,
               smear_mode=SMEAR_OFF, patience_floor=PATIENCE_FLOOR, groups=None,
               minutes_weight=0.0, minutes_target=MINUTES_TARGET_BANDS) -> FittedPoints:
    """Fit `reps` models on train_batch, keep the ones select_reps_for_average keeps.

    smear_mode applies to log-target losses, whose expm1 is a geometric-mean-like
    value below the true mean. The factor is measured on the held-out test rows.
    groups keeps related rows (overlapping windows) out of both sides of the test
    and validation splits. minutes_weight above 0 trains a minutes_target head at
    that loss weight: cross-entropy on bands, or Huber on scaled log minutes. Early
    stopping stays on points and the head is dropped before anything predicts.
    """
    hparams = hparams or PlayerNetHParams.from_env()
    loss_name = resolve_loss_name(loss_name)
    weighting = resolve_weighting(weighting)
    reps = parse_points_reps() if reps is None else int(reps)
    specs = input_specs(plan)
    log_targets = uses_log_targets(loss_name)
    labels = numpy.asarray(train_batch.labels, dtype=numpy.float32).reshape(-1, 1)
    count = int(labels.shape[0])
    if count < 1:
        raise ValueError('learning_v2 has no training rows')

    index = numpy.arange(count)
    val_index = index[:0]
    if count >= 20 and groups is not None:
        train_index, test_index = _group_split(index, groups, 0.2)
        train_index, val_index = _group_split(train_index, groups, 0.2)
    elif count >= 20:
        train_index, test_index = train_test_split(
            index, test_size=0.2, random_state=42, shuffle=True,
        )
    else:
        train_index, test_index = index, index[:0]

    train_x = _take(train_batch.arrays, train_index)
    test_x = _take(train_batch.arrays, test_index)
    val_x = _take(train_batch.arrays, val_index)
    row_specs = _row_specs(specs, train_batch)
    stats = _fit_scalers(train_x, row_specs)
    _apply_scalers(train_x, row_specs, stats)
    if len(test_index):
        _apply_scalers(test_x, row_specs, stats)
    if len(val_index):
        _apply_scalers(val_x, row_specs, stats)
    feed = None
    if _table_inputs(train_batch):
        stats.update(_fit_table_scalers(train_batch, specs))
        feed = _feed_tables(train_batch, specs, stats)
    input_keys = [spec.key for spec in row_specs]

    def _test_rows(y=None):
        if feed is None:
            return test_x
        return TableFeed(test_x, input_keys, feed, y=y, batch_size=PREDICT_BATCH_SIZE)

    y_train = labels[train_index]
    y_test = labels[test_index]
    y_val = labels[val_index]
    raw_test = numpy.asarray(y_test, dtype=numpy.float32).reshape(-1)
    if log_targets:
        y_train = numpy.log1p(numpy.maximum(y_train, 0.0))
        y_test = numpy.log1p(numpy.maximum(y_test, 0.0))
        y_val = numpy.log1p(numpy.maximum(y_val, 0.0))
    elif loss_name == 'poisson':
        # Poisson needs y >= 0; the rare negative scores floor at 0, as log targets do.
        y_train = numpy.maximum(y_train, 0.0)
        y_test = numpy.maximum(y_test, 0.0)

    mae_name, rmse_name = _metric_names(log_targets)
    minutes_weight = float(minutes_weight or 0.0)
    score_weight = float(getattr(plan.config, 'score_head_weight', 0.0) or 0.0)
    fit_y, eval_y, val_y = y_train, y_test, y_val
    extra_targets = {}
    if minutes_weight:
        extra_targets['minutes'] = minutes_targets(train_batch.matches, minutes_target)
    staged = getattr(plan.config, 'points_structure', 'direct') == POINTS_STAGED
    xg_weight = 0.0
    if score_weight:
        # Staged: goal bands, each team fixture counted once; otherwise Poisson per row.
        xg_weight = float(getattr(plan.config, 'score_xg_weight', 0.0) or 0.0) if staged else 0.0
        extra_targets['score'] = score_targets(
            train_batch.matches, weighted=staged,
            team_xg=getattr(train_batch, 'team_xg', None) if xg_weight > 0 else None,
        )
        if xg_weight > 0 and extra_targets['score'].shape[1] < 6:
            raise ValueError('score_xg_weight is set but the training batch has no team xG')
    if extra_targets:
        fit_y = {'points': y_train}
        eval_y = {'points': y_test}
        val_y = {'points': y_val}
        for name, target in extra_targets.items():
            fit_y[name] = target[train_index]
            eval_y[name] = target[test_index]
            val_y[name] = target[val_index]
        # Keras names a multi-output model's metrics after the output.
        mae_name, rmse_name = 'points_%s' % mae_name, 'points_%s' % rmse_name
    epochs = EPOCHS if '--quick' in sys.argv else EPOCH_MAP.get('total_points', EPOCHS)
    val_split = 0.2 if len(train_index) >= 10 and not len(val_index) else 0.0
    validation_data = (val_x, val_y) if len(val_index) else None
    collected = []

    for rep in range(max(1, reps)):
        if reps > 1:
            print('  learning_v2 fit %s/%s' % (rep + 1, reps))
        model = build_model(
            plan, vocab, hparams, loss_name=loss_name, minutes_head=minutes_target if minutes_weight else None,
        )
        optimizer, opt_label = _build_optimizer(hparams)
        metrics = [OrigRMSE(), OrigMAE()] if log_targets else [RootMeanSquaredError(), MAE()]
        # Unweighted metrics, so early stopping (val RMSE) and rep selection
        # (val MAE) use the same yardstick whichever loss weighting is under test.
        if extra_targets:
            losses = {'points': _loss_fn(loss_name, hparams)}
            loss_weights = {'points': 1.0}
            if minutes_weight:
                losses['minutes'] = (
                    'sparse_categorical_crossentropy' if minutes_target == MINUTES_TARGET_BANDS
                    else _loss_fn('huber', hparams)
                )
                loss_weights['minutes'] = minutes_weight
            if score_weight:
                if staged:
                    losses['score'] = goal_band_loss(float(getattr(plan.config, 'score_xg_weight', 0.0) or 0.0))
                else:
                    losses['score'] = masked_poisson
                loss_weights['score'] = score_weight
            model.compile(
                loss=losses,
                loss_weights=loss_weights,
                optimizer=optimizer,
                metrics={'points': metrics},
            )
        else:
            model.compile(
                loss=_loss_fn(loss_name, hparams),
                optimizer=optimizer,
                metrics=metrics,
            )
        sample_weight = weighting.weights(labels[train_index].reshape(-1))
        print(
            'learning_v2 %s: loss=%s, %s, optimizer=%s, epochs=%s, train=%s, '
            'early stop on unweighted val RMSE, patience=20%% of epochs run (at least %s, at most 40)'
            % (plan.name, loss_name, weighting.describe(), opt_label, epochs, len(train_index), patience_floor)
        )
        if minutes_weight:
            print('learning_v2 minutes head: %s, training only, loss weight %s' % (minutes_target, minutes_weight))
        if score_weight:
            print('learning_v2 score head: team goals for and against, %s, loss weight %s' % (
                'goal bands 0-4+, one weight per team fixture' if staged else 'Poisson', score_weight))
        if getattr(plan.config, 'points_structure', 'direct') != 'direct':
            print('learning_v2 points structure: %s' % plan.config.points_structure)
        if staged and getattr(plan.config, 'scoreline_squads', False):
            print('learning_v2 scoreline squads: each side\'s squad summary feeds the scoreline')
        if staged and xg_weight > 0:
            known = float(extra_targets['score'][:, 5].mean()) if 'score' in extra_targets else 0.0
            print('learning_v2 scoreline xG: band means fitted to match xG at weight %s (%.0f%% of rows have xG)'
                  % (xg_weight, 100.0 * known))
        callbacks = []
        if val_split > 0 or validation_data is not None:
            # RMSE, not MAE: MAE is best at the median and stopped fits within a
            # few epochs, before they learned the hauls the mean needs.
            callbacks = [
                PlateauStopping(monitor='val_%s' % rmse_name, floor=patience_floor),
                ReduceLROnPlateau(
                    monitor='val_%s' % rmse_name,
                    factor=hparams.reduce_lr_factor,
                    patience=hparams.reduce_lr_patience,
                    min_lr=hparams.reduce_lr_min,
                ),
            ]
        if feed is None:
            history = model.fit(
                train_x, fit_y,
                epochs=epochs,
                batch_size=hparams.batch_size,
                validation_split=val_split,
                validation_data=validation_data,
                verbose=2,
                callbacks=callbacks,
                shuffle=True,
                sample_weight=sample_weight,
            )
        else:
            history = _fit_with_tables(
                model, train_x, fit_y, sample_weight, validation_data, val_split, feed, input_keys,
                hparams.batch_size, epochs, callbacks, seed=42 + rep,
            )
        val_key = 'val_%s' % mae_name
        if val_key in history.history:
            best_val = float(min(history.history[val_key]))
        else:
            best_val = float(history.history[mae_name][-1])
        test_pred = None
        if len(test_index):
            if feed is None:
                evaluated = model.evaluate(test_x, eval_y, verbose=0)
            else:
                evaluated = model.evaluate(_test_rows(eval_y), verbose=0)
            print({'test_loss': float(evaluated[0]), 'test_metrics': [float(v) for v in evaluated[1:]]})
        minutes_model = None
        if extra_targets:
            if minutes_weight and minutes_target == MINUTES_TARGET_BANDS:
                minutes_model = minutes_only(model)
            model = points_only(model)
        if len(test_index):
            test_pred = numpy.asarray(model.predict(_test_rows(), verbose=0), dtype=numpy.float32)
        collected.append((model, {'best_val_mae': best_val}, test_pred, minutes_model))
        del history
        # Each fit's input copies run to gigabytes; let them go before the next.
        gc.collect()

    kept, _threshold, rule = select_reps_for_average([item[1] for item in collected], hparams)
    if reps > 1:
        print('  averaging %s/%s learning_v2 fits (%s)' % (len(kept), reps, rule))
    smearing = None
    if raw_test.size and collected[kept[0]][2] is not None:
        averaged_test = numpy.mean([collected[i][2] for i in kept], axis=0)
        if log_targets:
            if smear_mode != SMEAR_OFF:
                smearing = smearing_factor(y_test, averaged_test)
                print(
                    'learning_v2 smearing (%s, %s held-out rows): S=%.3f; by level %s'
                    % (smear_mode, smearing['n_rows'], smearing['S'], ', '.join(
                        '%.1f->%.2f' % (b['pred_mean'], b['S']) for b in smearing['by_pred_level']
                    ))
                )
            averaged_test = apply_smearing(numpy.expm1(averaged_test), smearing, smear_mode)
        averaged_test = numpy.maximum(0.0, numpy.asarray(averaged_test, dtype=float)).reshape(-1)
        print(format_tier_report(score_by_tier(raw_test, averaged_test)))
        test_matches = [train_batch.matches[int(i)] for i in test_index]
        print(format_form_report(score_by_recent_form(
            raw_test, averaged_test, recent_gameweek_rates(test_matches),
        )))
    return FittedPoints(
        [collected[i][0] for i in kept], specs, stats, log_targets,
        smearing=smearing, smear_mode=smear_mode,
        minutes_models=[collected[i][3] for i in kept],
    )
