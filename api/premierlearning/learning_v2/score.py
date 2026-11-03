"""Held-out points score that cares more about high actual returns.

Weighting an error by the predicted score does the opposite of what we want:
a model that calls a 12-point game a 1 hides that miss. The weight comes from
the actual points instead, in four bands.

A second report groups rows by the player's own recent level: mean points per
gameweek over the previous five gameweeks. That separates a forward who has
been scoring every week from a one-off haul.
"""
from __future__ import annotations

import numpy

from premierlearning.learning_v2.dataset import game_stat, is_before, match_sort_key

# (name, inclusive lower bound, exclusive upper bound or None, weight)
TIERS = (
    ('low', 0.0, 3.0, 1.0),
    ('medium', 3.0, 6.0, 2.0),
    ('high', 6.0, 10.0, 4.0),
    ('very_high', 10.0, None, 8.0),
)


def _tier_index(points: float) -> int:
    value = float(points)
    if value < TIERS[0][1]:
        return 0
    for index, (_name, lower, upper, _weight) in enumerate(TIERS):
        if value >= lower and (upper is None or value < upper):
            return index
    return len(TIERS) - 1


def tier_weights(points) -> numpy.ndarray:
    """Per-row loss weights from actual points. Same bands as the tier report."""
    values = numpy.asarray(points, dtype=float).reshape(-1)
    weights = numpy.empty(values.shape, dtype=numpy.float32)
    for index, value in enumerate(values):
        weights[index] = TIERS[_tier_index(value)][3]
    return weights


def normalized_tier_weights(points) -> numpy.ndarray:
    """Band weights with mean 1, so a haul still counts 8x a blank without raising the learning rate."""
    weights = tier_weights(points)
    return _mean_one(weights)


# 0 = every fixture equal, 1 = the report bands. 0.2 keeps a 10-point game at 2.4x a blank.
LOSS_TIER_MIX = 0.2


def blended_tier_weights(points, mix=LOSS_TIER_MIX, plateau=False) -> numpy.ndarray:
    """Blend of equal weights and the score bands, then scaled to mean 1.

    A blank stays at 1. With mix 0.2 a 3-5 point game is 1.2, a 6-9 is 1.6
    and a 10-point game is 2.4, before the mean is put back to 1. plateau keeps
    10+ at the 6-9 band: a haul's loss already grows with the size of the miss,
    so it needs no extra weight on top.
    """
    bands = tier_weights(points)
    if plateau:
        bands = numpy.minimum(bands, TIERS[2][3])
    mix = float(mix)
    weights = (1.0 - mix) + mix * bands
    return _mean_one(weights)


def _mean_one(weights) -> numpy.ndarray:
    weights = numpy.asarray(weights, dtype=numpy.float32)
    if weights.size == 0:
        return weights
    mean = float(weights.mean())
    if mean <= 0:
        return numpy.ones_like(weights)
    return weights / mean


# 3 points is 1.23 (bands 1.2), 6 is 1.65 (1.6), 10 is 2.4 (2.4); 20 keeps going to 4.96.
SMOOTH_ALPHA = 1.4
SMOOTH_SCALE = 10.0
SMOOTH_POWER = 1.5


def smooth_weights(points, alpha=SMOOTH_ALPHA, scale=SMOOTH_SCALE, power=SMOOTH_POWER,
                   cap=None) -> numpy.ndarray:
    """1 + alpha * (max(points, 0) / scale) ** power, capped, then scaled to mean 1.

    The defaults sit on the blended bands where those are defined. With no cap
    the curve keeps rising past 10 points; cap 2.4 makes it the bands without steps.
    """
    if float(scale) <= 0:
        raise ValueError('smooth weight scale must be positive')
    values = numpy.maximum(numpy.asarray(points, dtype=float).reshape(-1), 0.0)
    weights = 1.0 + float(alpha) * (values / float(scale)) ** float(power)
    if cap is not None:
        weights = numpy.minimum(weights, float(cap))
    return _mean_one(weights)


# Rises like the bands to a peak at 8 points, then eases off: 3 is ~1.22, 6 ~1.50,
# 8 is 1.70, 10 ~1.67, 15 ~1.43, 20 ~1.17. Hauls already carry a big miss.
HUMP_PEAK = 8.0
HUMP_HEIGHT = 0.7
HUMP_POWER = 1.2
HUMP_WIDTH = 10.0


def hump_weights(points, peak=HUMP_PEAK, height=HUMP_HEIGHT, power=HUMP_POWER,
                 width=HUMP_WIDTH) -> numpy.ndarray:
    """1 + height * (points / peak) ** power up to the peak, a Gaussian fall after; mean 1."""
    values = numpy.maximum(numpy.asarray(points, dtype=float).reshape(-1), 0.0)
    rising = (numpy.minimum(values, peak) / float(peak)) ** float(power)
    falling = numpy.exp(-((numpy.maximum(values - peak, 0.0) / float(width)) ** 2))
    weights = 1.0 + float(height) * numpy.where(values <= peak, rising, falling)
    return _mean_one(weights)


WEIGHTING_NONE = 'none'
WEIGHTING_TIER = 'tier'
WEIGHTING_PLATEAU = 'plateau'
WEIGHTING_SMOOTH = 'smooth'
WEIGHTING_HUMP = 'hump'
WEIGHTING_KINDS = (WEIGHTING_NONE, WEIGHTING_TIER, WEIGHTING_PLATEAU, WEIGHTING_SMOOTH, WEIGHTING_HUMP)


class LossWeighting:
    """Per-fixture training weights from actual points. Never from predictions."""

    def __init__(self, kind=WEIGHTING_TIER, tier_mix=LOSS_TIER_MIX,
                 alpha=SMOOTH_ALPHA, scale=SMOOTH_SCALE, power=SMOOTH_POWER, cap=None,
                 hump_peak=HUMP_PEAK, hump_height=HUMP_HEIGHT, hump_power=HUMP_POWER,
                 hump_width=HUMP_WIDTH):
        kind = str(kind or WEIGHTING_TIER).strip().lower()
        if kind not in WEIGHTING_KINDS:
            raise ValueError('loss weighting must be one of %s' % ', '.join(WEIGHTING_KINDS))
        if float(scale) <= 0:
            raise ValueError('smooth weight scale must be positive')
        self.kind = kind
        self.tier_mix = float(tier_mix)
        self.alpha = float(alpha)
        self.scale = float(scale)
        self.power = float(power)
        self.cap = None if cap is None else float(cap)
        self.hump_peak = float(hump_peak)
        self.hump_height = float(hump_height)
        self.hump_power = float(hump_power)
        self.hump_width = float(hump_width)

    def weights(self, points) -> numpy.ndarray:
        if self.kind == WEIGHTING_NONE:
            return numpy.ones(numpy.asarray(points).reshape(-1).shape, dtype=numpy.float32)
        if self.kind == WEIGHTING_TIER:
            return blended_tier_weights(points, mix=self.tier_mix)
        if self.kind == WEIGHTING_PLATEAU:
            return blended_tier_weights(points, mix=self.tier_mix, plateau=True)
        if self.kind == WEIGHTING_HUMP:
            return hump_weights(points, self.hump_peak, self.hump_height, self.hump_power, self.hump_width)
        return smooth_weights(points, self.alpha, self.scale, self.power, self.cap)

    def config(self) -> dict:
        """Only the settings that change the weights, for cache hashes."""
        if self.kind in (WEIGHTING_TIER, WEIGHTING_PLATEAU):
            return {'kind': self.kind, 'tier_mix': self.tier_mix}
        if self.kind == WEIGHTING_SMOOTH:
            config = {'kind': self.kind, 'alpha': self.alpha, 'scale': self.scale, 'power': self.power}
            if self.cap is not None:
                config['cap'] = self.cap
            return config
        if self.kind == WEIGHTING_HUMP:
            return {'kind': self.kind, 'peak': self.hump_peak, 'height': self.hump_height,
                    'power': self.hump_power, 'width': self.hump_width}
        return {'kind': self.kind}

    def describe(self) -> str:
        if self.kind == WEIGHTING_NONE:
            return 'every fixture weighted equally'
        if self.kind in (WEIGHTING_TIER, WEIGHTING_PLATEAU):
            top = 3.0 if self.kind == WEIGHTING_PLATEAU else 7.0
            return (
                'fixture weights %s of the way from equal to the score bands '
                '(blank x1, 3-5 x%.2f, 6-9 x%.2f, 10+ x%.2f, average weight 1)'
                % (
                    self.tier_mix,
                    1.0 + self.tier_mix, 1.0 + 3.0 * self.tier_mix, 1.0 + top * self.tier_mix,
                )
            )
        if self.kind == WEIGHTING_HUMP:
            return ('hump fixture weights peaking at x%.2f on %g points, easing off for hauls '
                    '(average weight 1)' % (1.0 + self.hump_height, self.hump_peak))
        capped = '' if self.cap is None else ', capped at %g' % self.cap
        return (
            'smooth fixture weights 1 + %s * (points / %s)^%s%s (average weight 1)'
            % (self.alpha, self.scale, self.power, capped)
        )


def score_by_tier(actual, predicted) -> dict:
    """Absolute error by actual-points band, plus one weighted MAE.

    low is 0–2 (weight 1), medium 3–5 (2), high 6–9 (4), very high 10+ (8).
    Weighted MAE is sum(|error| * weight) / sum(weight). Lower is better.
    """
    actual = numpy.asarray(actual, dtype=float).reshape(-1)
    predicted = numpy.asarray(predicted, dtype=float).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError('actual and predicted must have the same length')
    error = numpy.abs(predicted - actual)
    weights = tier_weights(actual)
    weight_sum = float(weights.sum())
    weighted_mae = float((error * weights).sum() / weight_sum) if weight_sum else 0.0
    tiers = []
    for index, (name, lower, upper, weight) in enumerate(TIERS):
        chosen = numpy.array([_tier_index(value) == index for value in actual])
        count = int(chosen.sum())
        if count:
            mean_actual = float(actual[chosen].mean())
            mean_predicted = float(predicted[chosen].mean())
            mae = float(error[chosen].mean())
        else:
            mean_actual = mean_predicted = mae = None
        tiers.append({
            'name': name,
            'lower': lower,
            'upper': upper,
            'weight': weight,
            'count': count,
            'mae': mae,
            'mean_actual': mean_actual,
            'mean_predicted': mean_predicted,
            'bias': None if mae is None else mean_predicted - mean_actual,
        })
    return {
        'count': int(actual.size),
        'mae': float(error.mean()) if actual.size else 0.0,
        'weighted_mae': weighted_mae,
        'tiers': tiers,
    }


def format_tier_report(report: dict) -> str:
    """Text table for the end of a training run."""
    lines = [
        'Held-out points by actual score '
        '(weights: low 0-2 x1, medium 3-5 x2, high 6-9 x4, very high 10+ x8)',
        '  weighted MAE %.3f    unweighted MAE %.3f    (lower is better, bias = predicted - actual)'
        % (report['weighted_mae'], report['mae']),
        '  %-10s %7s %7s %8s %10s %8s' % ('tier', 'n', 'MAE', 'actual', 'predicted', 'bias'),
    ]
    for tier in report['tiers']:
        if not tier['count']:
            lines.append('  %-10s %7d %7s %8s %10s %8s' % (tier['name'], 0, '-', '-', '-', '-'))
            continue
        lines.append(
            '  %-10s %7d %7.3f %8.2f %10.2f %+8.2f'
            % (
                tier['name'], tier['count'], tier['mae'],
                tier['mean_actual'], tier['mean_predicted'], tier['bias'],
            )
        )
    return '\n'.join(lines)


def _gameweek_key(match):
    carry = 1 if getattr(match, 'is_last_season_carry', False) else 0
    gameweek = getattr(match, 'gameweek', None)
    if gameweek is None:
        return (carry, 'match', id(match))
    return (carry, int(gameweek))


def recent_gameweek_rates(matches, n=5, minimum=3) -> numpy.ndarray:
    """Mean actual points per gameweek over up to `n` prior gameweeks.

    Rows with fewer than `minimum` prior gameweeks are NaN and drop out of the
    player-level report.
    """
    ordered = {}
    rates = numpy.full(len(matches), numpy.nan, dtype=float)
    for index, match in enumerate(matches):
        player = getattr(match, 'player', None)
        if player is None:
            continue
        key = id(player)
        if key not in ordered:
            ordered[key] = sorted(getattr(player, 'matches', []) or [], key=match_sort_key)
        totals = {}
        for earlier in reversed(ordered[key]):
            if earlier is match or not is_before(earlier, match):
                continue
            gameweek_key = _gameweek_key(earlier)
            if gameweek_key not in totals:
                if len(totals) >= n:
                    break
                totals[gameweek_key] = 0.0
            totals[gameweek_key] += game_stat(earlier, 'total_points')
        if len(totals) < minimum:
            continue
        rates[index] = float(sum(totals.values()) / len(totals))
    return rates


def score_by_recent_form(actual, predicted, rates) -> dict:
    """Tier each fixture by the player's recent points per gameweek, not by this score."""
    actual = numpy.asarray(actual, dtype=float).reshape(-1)
    predicted = numpy.asarray(predicted, dtype=float).reshape(-1)
    rates = numpy.asarray(rates, dtype=float).reshape(-1)
    if not (actual.shape == predicted.shape == rates.shape):
        raise ValueError('actual, predicted, and rates must have the same length')
    kept = numpy.isfinite(rates)
    error = numpy.abs(predicted - actual)
    tiers = []
    for index, (name, lower, upper, weight) in enumerate(TIERS):
        chosen = kept & numpy.array([
            numpy.isfinite(rate) and _tier_index(rate) == index for rate in rates
        ])
        count = int(chosen.sum())
        if count:
            mean_actual = float(actual[chosen].mean())
            mean_predicted = float(predicted[chosen].mean())
            mae = float(error[chosen].mean())
            mean_form = float(rates[chosen].mean())
            bias = mean_predicted - mean_actual
        else:
            mean_actual = mean_predicted = mae = mean_form = bias = None
        tiers.append({
            'name': name,
            'lower': lower,
            'upper': upper,
            'weight': weight,
            'count': count,
            'mae': mae,
            'mean_actual': mean_actual,
            'mean_predicted': mean_predicted,
            'bias': bias,
            'mean_form': mean_form,
        })
    kept_error = error[kept]
    return {
        'count': int(kept.sum()),
        'kept': int(kept.sum()),
        'skipped': int((~kept).sum()),
        'mae': float(kept_error.mean()) if kept_error.size else 0.0,
        'weighted_mae': 0.0,
        'tiers': tiers,
    }


def format_form_report(report: dict) -> str:
    """Players grouped by how many points they had been scoring lately."""
    lines = [
        'Held-out points by recent player level '
        '(mean points per gameweek over the previous 5, at least 3 gameweeks)',
        '  same bands: low under 3, medium 3-5, high 6-9, very high 10+',
        '  rows %s, skipped %s with too little history    unweighted MAE %.3f'
        % (report.get('kept', report['count']), report.get('skipped', 0), report['mae']),
        '  %-10s %7s %7s %8s %10s %8s %8s'
        % ('tier', 'n', 'MAE', 'actual', 'predicted', 'bias', 'form'),
    ]
    for tier in report['tiers']:
        if not tier['count']:
            lines.append(
                '  %-10s %7d %7s %8s %10s %8s %8s' % (tier['name'], 0, '-', '-', '-', '-', '-')
            )
            continue
        lines.append(
            '  %-10s %7d %7.3f %8.2f %10.2f %+8.2f %8.2f'
            % (
                tier['name'], tier['count'], tier['mae'],
                tier['mean_actual'], tier['mean_predicted'], tier['bias'],
                tier['mean_form'],
            )
        )
    return '\n'.join(lines)
