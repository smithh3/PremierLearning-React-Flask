"""Baseline-relative scores in noise units, the composite, and guardrail flags.

Why not 100 * (1 - E / E_v1). That ratio explodes for metrics near zero: a
calibration gap going 0.12 -> 0.06 scores +50 while MSE improving 3% scores +3,
so the 15% category would outweigh the 25% one many times over. Instead every
component is

    S = (M_v1 - M_model) / sigma        (lower is better)
    S = (M_model - M_v1) / sigma        (higher is better)

where sigma is that metric's spread for v1 across block-bootstrap resamples of
the evaluation gameweeks. One unit is "one standard deviation of evaluation
noise" in every component, so the category weights mean what they say. v1
scores exactly 0. The plan's relative-improvement numbers are still reported
alongside as relative_pct, as a diagnostic only.

A sigma that is missing or zero cannot scale anything: that component scores 0
and the model is flagged, rather than returning inf or NaN.
"""

import math

import numpy

from simulation.model_eval.metrics import compute

LOWER = 'lower'
HIGHER = 'higher'


def components(config):
    """(component, metric key, direction, weight in the total)."""
    category = config['category_weights']
    future = config['future_weights']
    sides = config['future_sides']
    ranking = config['ranking_weights']
    rows = [('overall', 'overall_mse', LOWER, category['overall'])]
    for horizon in ('2A', '2B'):
        for side in ('actual', 'predicted'):
            rows.append((
                '%s_%s' % (horizon, side), '%s_%s' % (horizon, side), LOWER,
                category['future'] * future[horizon] * sides[side],
            ))
    rows.append(('calibration', 'calibration_gap', LOWER, category['calibration']))
    rows.append(('single_gw_rank', 'spearman_single_gw', HIGHER, category['ranking'] * ranking['single_gw']))
    rows.append(('five_gw_rank', 'spearman_five_gw', HIGHER, category['ranking'] * ranking['five_gw']))
    if category.get('fixtures'):
        rows.append(('fixture_swing', 'fixture_swing_corr', HIGHER, category['fixtures']))
    return rows


CATEGORY_OF = {
    'overall': 'overall',
    '2A_actual': 'future', '2A_predicted': 'future',
    '2B_actual': 'future', '2B_predicted': 'future',
    'calibration': 'calibration',
    'single_gw_rank': 'ranking', 'five_gw_rank': 'ranking',
    'fixture_swing': 'fixtures',
}


def _valid(value):
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def noise_score(model_value, baseline_value, sigma, direction):
    """Score in noise units, or None when it cannot be computed."""
    if not (_valid(model_value) and _valid(baseline_value) and _valid(sigma)) or sigma <= 1e-12:
        return None
    delta = model_value - baseline_value
    return (delta if direction == HIGHER else -delta) / sigma


def relative_pct(model_value, baseline_value, direction):
    """The plan's formulas: 100(1 - E/E_v1), and 100(R - R_v1)/(1 - R_v1)."""
    if not (_valid(model_value) and _valid(baseline_value)):
        return None
    if direction == LOWER:
        if baseline_value <= 0:
            return None
        return 100.0 * (1.0 - model_value / baseline_value)
    if baseline_value >= 1.0:
        return None
    return 100.0 * (model_value - baseline_value) / (1.0 - baseline_value)


def capped(score, cap):
    """Clip a noise score to +-cap; past a few sd an improvement is already clearly real."""
    if score is None or cap is None:
        return score
    return max(-float(cap), min(float(cap), score))


def score_model(model_metrics, baseline_metrics, sigmas, config):
    """Component, category and total scores for one model against v1.

    Each component is capped at +-score_cap noise units before weighting. A
    metric that barely moves for v1 (the calibration gap) has a tiny sigma, so a
    fix aimed squarely at it scored 17 sd and outweighed every other category
    despite its 15% weight. total_uncapped keeps the old sum for comparison.
    """
    cap = config.get('score_cap')
    rows = []
    invalid = []
    categories = {name: 0.0 for name in config['category_weights']}
    total = 0.0
    total_uncapped = 0.0
    for component, key, direction, weight in components(config):
        raw = noise_score(model_metrics.get(key), baseline_metrics.get(key), sigmas.get(key), direction)
        score = capped(raw, cap)
        if score is None:
            invalid.append(key)
        used = 0.0 if score is None else score
        categories[CATEGORY_OF[component]] += used * weight / config['category_weights'][CATEGORY_OF[component]]
        total += used * weight
        total_uncapped += (0.0 if raw is None else raw) * weight
        rows.append({
            'component': component,
            'metric': key,
            'direction': direction,
            'weight': weight,
            'model_value': model_metrics.get(key),
            'v1_value': baseline_metrics.get(key),
            'sigma': sigmas.get(key),
            'score': score,
            'uncapped_score': raw,
            'weighted': used * weight,
            'relative_pct': relative_pct(model_metrics.get(key), baseline_metrics.get(key), direction),
        })
    return {
        'components': rows, 'categories': categories, 'total': total,
        'total_uncapped': total_uncapped, 'invalid': invalid,
    }


def bootstrap_draws(units, block, resamples, seed):
    """Index lists for a moving-block bootstrap of origins, blocks kept inside a season."""
    by_season = {}
    for index, unit in enumerate(units):
        by_season.setdefault(unit.season, []).append(index)
    blocks = []
    for indexes in by_season.values():
        indexes = sorted(indexes, key=lambda i: units[i].origin)
        length = min(int(block), len(indexes))
        for start in range(0, len(indexes) - length + 1):
            blocks.append(indexes[start:start + length])
    rng = numpy.random.RandomState(int(seed))
    draws = []
    for _ in range(int(resamples)):
        drawn = []
        while len(drawn) < len(units):
            drawn.extend(blocks[rng.randint(len(blocks))])
        draws.append(drawn[:len(units)])
    return draws


def bootstrap(units, models, baseline, config, rank_cache):
    """Sigma per metric from v1, and each model's total score per resample."""
    settings = config['bootstrap']
    draws = bootstrap_draws(units, settings['block_origins'], settings['resamples'], settings['seed'])
    keys = [key for _component, key, _direction, _weight in components(config)]
    per_draw = []
    for drawn in draws:
        sample = [units[i] for i in drawn]
        per_draw.append({
            model: compute(sample, model, config, rank_cache) for model in models
        })
    sigmas = {}
    for key in keys:
        values = [draw[baseline][key] for draw in per_draw if _valid(draw[baseline][key])]
        sigmas[key] = float(numpy.std(values, ddof=1)) if len(values) > 1 else None
    totals = {model: [] for model in models}
    for draw in per_draw:
        for model in models:
            totals[model].append(score_model(draw[model], draw[baseline], sigmas, config)['total'])
    return sigmas, totals


def interval(values):
    values = numpy.asarray(values, dtype=float)
    if values.size == 0:
        return {'lo': None, 'hi': None, 'p_above_zero': None}
    return {
        'lo': float(numpy.percentile(values, 2.5)),
        'hi': float(numpy.percentile(values, 97.5)),
        'p_above_zero': float((values > 0).mean()),
    }


def _worse_pct(model_value, baseline_value):
    if not (_valid(model_value) and _valid(baseline_value)) or baseline_value <= 0:
        return None
    return 100.0 * (model_value / baseline_value - 1.0)


def guardrails(model_metrics, baseline_metrics, total, season_totals, config):
    """Flags only; nothing is rejected automatically."""
    limits = config['guardrails']
    flags = []
    for key, limit in (('overall_mse', 'overall_mse_worse_pct'), ('overall_mae', 'overall_mae_worse_pct')):
        worse = _worse_pct(model_metrics.get(key), baseline_metrics.get(key))
        if worse is not None and worse > limits[limit]:
            flags.append('%s is %.1f%% worse than v1' % (key, worse))
    gap = model_metrics.get('calibration_gap')
    base_gap = baseline_metrics.get('calibration_gap')
    worse = _worse_pct(gap, base_gap)
    if (
        worse is not None and worse > limits['calibration_gap_worse_pct']
        and gap - base_gap > limits['calibration_gap_worse_points']
    ):
        flags.append('calibration gap %.3f vs v1 %.3f' % (gap, base_gap))
    for bucket in ('top_5pct', 'top_5'):
        signed = model_metrics.get('2A_predicted_%s_signed' % bucket)
        if _valid(signed) and signed > limits['own_top_overprediction_points']:
            flags.append(
                'overpredicts its own %s picks by %.2f points over 5 GWs' % (bucket, signed)
            )
    if total > 0:
        for season, season_total in sorted(season_totals.items()):
            if season_total < limits['season_total_below']:
                flags.append(
                    'season %s scores %.2f despite a positive pooled score' % (season, season_total)
                )
    return flags
