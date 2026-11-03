"""Raw metrics for one model over a list of units (repeats allowed, for the bootstrap).

Single-GW rows: eligible players with a fixture in the origin gameweek, the
prediction made at that deadline. 5-GW rows: every eligible player at an origin
whose whole window is inside the season, totals over the next five absolute
gameweeks.

Lower is better for every error; higher is better for the Spearman correlations.
"""

import math

import numpy
from scipy.stats import spearmanr


def bucket_rows(values, player_ids, buckets):
    """Row indices per bucket, ranked by value descending, ties by player id.

    Fraction buckets cover rank positions [ceil(from * n), ceil(to * n)); a top_n
    bucket is the first n positions and overlaps the fraction buckets.
    """
    values = numpy.asarray(values, dtype=float)
    count = int(values.size)
    order = numpy.lexsort((numpy.asarray(player_ids), -values))
    chosen = {}
    for bucket in buckets:
        if 'top_n' in bucket:
            chosen[bucket['name']] = order[:min(int(bucket['top_n']), count)]
            continue
        start = int(math.ceil(float(bucket['from_fraction']) * count))
        end = int(math.ceil(float(bucket['to_fraction']) * count))
        chosen[bucket['name']] = order[start:end]
    return chosen


def _spearman(predicted, actual, minimum):
    if predicted.size < int(minimum):
        return None
    if numpy.all(predicted == predicted[0]) or numpy.all(actual == actual[0]):
        return None
    value = spearmanr(predicted, actual).correlation
    if value is None or not numpy.isfinite(value):
        return None
    return float(value)


def fixture_swing_rows(unit, min_minutes):
    """Players with one fixture in every gameweek of a full window, each played min_minutes+.

    Their points still swing from fixture to fixture, but not because they were
    injured, rotated or had a blank or double, so the swing is down to the
    fixtures. None when a unit has no such players or no realised minutes.
    """
    if not unit.full_window or getattr(unit, 'minutes', None) is None:
        return None
    if getattr(unit, 'fixture_count', None) is None:
        return None
    keep = (unit.fixture_count == 1).all(axis=1) & (unit.minutes >= float(min_minutes)).all(axis=1)
    return keep if keep.any() else None


def swing_sums(predicted, actual):
    """(cells, sum pd*ad, sum pd^2, sum ad^2) for deviations from each player's own window mean."""
    pd = predicted - predicted.mean(axis=1, keepdims=True)
    ad = actual - actual.mean(axis=1, keepdims=True)
    return int(pd.size), float((pd * ad).sum()), float((pd * pd).sum()), float((ad * ad).sum())


def swing_metrics(cells, cross, pred_sq, actual_sq):
    """Fixture-swing metrics from pooled sums.

    corr: how well the predicted rises and falls across a player's next five
    fixtures line up with his real ones (higher is better; a model that predicts
    no swing scores 0). slope: real swing per point of predicted swing (1 is
    right, above 1 means the model's swings are too small). mse: error on the
    swings themselves.
    """
    if not cells or actual_sq <= 1e-12:
        return {'fixture_swing_n': int(cells), 'fixture_swing_corr': None,
                'fixture_swing_slope': None, 'fixture_swing_mse': None}
    return {
        'fixture_swing_n': int(cells),
        'fixture_swing_corr': cross / math.sqrt(pred_sq * actual_sq) if pred_sq > 1e-12 else 0.0,
        'fixture_swing_slope': cross / pred_sq if pred_sq > 1e-12 else None,
        'fixture_swing_mse': (pred_sq - 2.0 * cross + actual_sq) / cells,
    }


class RankCache:
    """Per-unit correlations do not depend on which units a resample draws."""

    def __init__(self):
        self._values = {}

    def swing(self, model, unit, min_minutes):
        key = (model, id(unit), 'swing', float(min_minutes))
        if key not in self._values:
            keep = fixture_swing_rows(unit, min_minutes)
            if keep is None:
                self._values[key] = (0, 0.0, 0.0, 0.0)
            else:
                self._values[key] = swing_sums(unit.preds[model][keep], unit.actual[keep])
        return self._values[key]

    def get(self, model, unit, kind, minimum):
        key = (model, id(unit), kind)
        if key not in self._values:
            if kind == 'single_gw':
                mask = unit.has_fixture[:, 0]
                value = _spearman(unit.preds[model][mask, 0], unit.actual[mask, 0], minimum)
            else:
                value = _spearman(
                    unit.preds[model].sum(axis=1), unit.actual.sum(axis=1), minimum,
                )
            self._values[key] = value
        return self._values[key]


class _BucketSums:
    __slots__ = ('count', 'abs_error', 'signed', 'predicted', 'actual')

    def __init__(self):
        self.count = 0
        self.abs_error = 0.0
        self.signed = 0.0
        self.predicted = 0.0
        self.actual = 0.0

    def add(self, predicted, actual):
        error = predicted - actual
        self.count += int(predicted.size)
        self.abs_error += float(numpy.abs(error).sum())
        self.signed += float(error.sum())
        self.predicted += float(predicted.sum())
        self.actual += float(actual.sum())

    def row(self):
        if not self.count:
            return {'count': 0, 'mae': None, 'mean_predicted': None, 'mean_actual': None, 'signed_error': None}
        return {
            'count': self.count,
            'mae': self.abs_error / self.count,
            'mean_predicted': self.predicted / self.count,
            'mean_actual': self.actual / self.count,
            'signed_error': self.signed / self.count,
        }


def _weighted_bucket_mae(sums, buckets):
    """Weighted mean of bucket MAEs. Empty buckets drop out and the rest are rescaled."""
    total = 0.0
    weight = 0.0
    for bucket in buckets:
        cell = sums[bucket['name']]
        if cell.count:
            total += float(bucket['weight']) * cell.abs_error / cell.count
            weight += float(bucket['weight'])
    return total / weight if weight else None


def calibration_bins(predicted, actual, bins):
    """Equal-count bins by prediction; ties keep input order."""
    order = numpy.argsort(predicted, kind='mergesort')
    rows = []
    for index, chunk in enumerate(numpy.array_split(order, int(bins))):
        if chunk.size == 0:
            rows.append({'bin': index, 'n': 0, 'lo': None, 'hi': None,
                         'mean_predicted': None, 'mean_actual': None, 'gap': None})
            continue
        mean_predicted = float(predicted[chunk].mean())
        mean_actual = float(actual[chunk].mean())
        rows.append({
            'bin': index,
            'n': int(chunk.size),
            'lo': float(predicted[chunk].min()),
            'hi': float(predicted[chunk].max()),
            'mean_predicted': mean_predicted,
            'mean_actual': mean_actual,
            'gap': abs(mean_predicted - mean_actual),
        })
    return rows


def compute(units, model, config, rank_cache=None, details=False):
    """Every raw metric for `model`. details=True adds the bucket and bin tables."""
    rank_cache = rank_cache or RankCache()
    buckets = config['buckets']
    minimum = int(config['min_rank_players'])
    single_pred = []
    single_actual = []
    sums = {
        (horizon, side): {bucket['name']: _BucketSums() for bucket in buckets}
        for horizon in ('2A', '2B') for side in ('actual', 'predicted')
    }
    single_ranks = []
    five_ranks = []
    skipped_single = 0
    skipped_five = 0
    min_minutes = float((config.get('fixture_swing') or {}).get('min_minutes', 60))
    swing = [0, 0.0, 0.0, 0.0]
    for unit in units:
        for index, value in enumerate(rank_cache.swing(model, unit, min_minutes)):
            swing[index] += value
        predicted = unit.preds[model]
        mask = unit.has_fixture[:, 0]
        single_pred.append(predicted[mask, 0])
        single_actual.append(unit.actual[mask, 0])
        value = rank_cache.get(model, unit, 'single_gw', minimum)
        if value is None:
            skipped_single += 1
        else:
            single_ranks.append(value)
        if not unit.full_window or unit.player_ids.size == 0:
            continue
        value = rank_cache.get(model, unit, 'five_gw', minimum)
        if value is None:
            skipped_five += 1
        else:
            five_ranks.append(value)
        total_pred = predicted.sum(axis=1)
        total_actual = unit.actual.sum(axis=1)
        for side, ranked_by in (('actual', total_actual), ('predicted', total_pred)):
            for name, rows in bucket_rows(ranked_by, unit.player_ids, buckets).items():
                sums[('2A', side)][name].add(total_pred[rows], total_actual[rows])
                sums[('2B', side)][name].add(predicted[rows, 0], unit.actual[rows, 0])

    single_pred = numpy.concatenate(single_pred) if single_pred else numpy.zeros(0)
    single_actual = numpy.concatenate(single_actual) if single_actual else numpy.zeros(0)
    error = single_pred - single_actual
    result = {
        'overall_n': int(error.size),
        'overall_mse': float(numpy.mean(error ** 2)) if error.size else None,
        'overall_mae': float(numpy.mean(numpy.abs(error))) if error.size else None,
        'overall_bias': float(numpy.mean(error)) if error.size else None,
        'spearman_single_gw': float(numpy.mean(single_ranks)) if single_ranks else None,
        'spearman_five_gw': float(numpy.mean(five_ranks)) if five_ranks else None,
        'rank_scored_single_gw': len(single_ranks),
        'rank_skipped_single_gw': skipped_single,
        'rank_scored_five_gw': len(five_ranks),
        'rank_skipped_five_gw': skipped_five,
    }
    result['overall_rmse'] = None if result['overall_mse'] is None else math.sqrt(result['overall_mse'])
    result.update(swing_metrics(*swing))
    for (horizon, side), cells in sums.items():
        result['%s_%s' % (horizon, side)] = _weighted_bucket_mae(cells, buckets)
    bins = calibration_bins(single_pred, single_actual, config['calibration_bins']) if error.size else []
    gaps = [row['gap'] for row in bins if row['gap'] is not None]
    result['calibration_gap'] = float(numpy.mean(gaps)) if gaps else None
    if error.size > 1 and numpy.var(single_pred) > 1e-12:
        slope, intercept = numpy.polyfit(single_pred, single_actual, 1)
        result['calibration_slope'] = float(slope)
        result['calibration_intercept'] = float(intercept)
    else:
        result['calibration_slope'] = result['calibration_intercept'] = None
    for side in ('actual', 'predicted'):
        for name in ('top_5pct', 'top_5'):
            cell = sums[('2A', side)].get(name)
            if cell is not None:
                result['2A_%s_%s_signed' % (side, name)] = cell.row()['signed_error']
    if details:
        result['buckets'] = [
            dict(horizon=horizon, side=side, bucket=name, **cell.row())
            for (horizon, side), cells in sums.items() for name, cell in cells.items()
        ]
        result['calibration_bins'] = bins
    return result


def rank_tables(units, model, config, rank_cache):
    """Per-gameweek and per-origin correlations for the report."""
    minimum = int(config['min_rank_players'])
    single = []
    five = []
    for unit in units:
        single.append({
            'season': unit.season, 'gameweek': unit.origin,
            'n': int(unit.has_fixture[:, 0].sum()),
            'spearman': rank_cache.get(model, unit, 'single_gw', minimum),
        })
        if unit.full_window:
            five.append({
                'season': unit.season, 'origin': unit.origin,
                'n': int(unit.player_ids.size),
                'spearman': rank_cache.get(model, unit, 'five_gw', minimum),
            })
    return single, five
