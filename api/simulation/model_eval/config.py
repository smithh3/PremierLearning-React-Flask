"""Every weight, bucket, horizon, eligibility rule and guardrail the evaluation uses.

Fixed across model variations. Override with --config path.json (same keys,
nested dicts merge) rather than editing the scoring code.
"""

import copy
import json
from pathlib import Path


DEFAULT_CONFIG = {
    'horizon': 5,
    # Single-GW metrics use every origin; 5-GW metrics only origins whose
    # window fits before GW38 (1-34). Incomplete windows are never scored.
    'origin_from': 1,
    'origin_to': 38,
    'eligibility': {
        # Known at the deadline and the same for every model: played in any of
        # the previous `recent_gws` gameweeks of this season. Until that many
        # gameweeks exist, last season's minutes (matched by player code) also
        # qualify. The target fixture's own minutes are never used.
        'recent_gws': 5,
        'last_season_min_minutes': 900.0,
        # Players the builder does not predict at this deadline (no recent
        # minutes and little usage over last season plus this one so far).
        # Leave them out rather than hand every model free zeros.
        'exclude_builder_skipped': True,
    },
    # fixtures: how well a player's predicted rises and falls across his next five
    # fixtures match the real ones (metrics.swing_metrics). Added 2026-11-03; the
    # other four were 0.25 / 0.25 / 0.15 / 0.35 before it.
    'category_weights': {
        'overall': 0.22,
        'future': 0.22,
        'calibration': 0.13,
        'ranking': 0.28,
        'fixtures': 0.15,
    },
    # Swing rows: one fixture in each of the five gameweeks and at least this many
    # minutes in each, so injuries, rotation, blanks and doubles do not count as
    # fixture swings. Uses realised minutes, the same rows for every model.
    'fixture_swing': {'min_minutes': 60},
    'future_weights': {'2A': 0.6, '2B': 0.4},
    # Buckets picked by realised 5-GW points reward inflating likely hauliers;
    # the same buckets picked by the model's own 5-GW prediction punish it.
    'future_sides': {'actual': 0.5, 'predicted': 0.5},
    'ranking_weights': {'single_gw': 0.5, 'five_gw': 0.5},
    # Rank-position buckets within each origin's eligible players. Ties are
    # broken by player id, so a tie on a boundary is split, never shared.
    'buckets': [
        {'name': 'bottom_50', 'from_fraction': 0.50, 'to_fraction': 1.00, 'weight': 0.05},
        {'name': 'p50_80', 'from_fraction': 0.20, 'to_fraction': 0.50, 'weight': 0.10},
        {'name': 'p80_95', 'from_fraction': 0.05, 'to_fraction': 0.20, 'weight': 0.15},
        {'name': 'top_5pct', 'from_fraction': 0.00, 'to_fraction': 0.05, 'weight': 0.30},
        {'name': 'top_5', 'top_n': 5, 'weight': 0.40},
    ],
    # Each component's noise score is clipped to +-score_cap before weighting,
    # so no single metric with a tiny sigma can swamp the category weights.
    # null turns the cap off.
    'score_cap': 3.0,
    'calibration_bins': 5,
    # Fewer eligible players than this and that gameweek's correlation is
    # skipped and counted, not scored as zero.
    'min_rank_players': 30,
    'bootstrap': {
        'resamples': 200,
        # Consecutive origins drawn together, so overlapping 5-GW windows
        # stay correlated inside a resample.
        'block_origins': 5,
        'seed': 7,
    },
    'calibrator': {
        # Rolling y = a + b * prediction, fitted on fixtures already played
        # before each deadline. Identity until this many pairs exist.
        'min_pairs': 500,
    },
    'guardrails': {
        'overall_mse_worse_pct': 5.0,
        'overall_mae_worse_pct': 5.0,
        'calibration_gap_worse_pct': 50.0,
        'calibration_gap_worse_points': 0.1,
        # Mean (predicted - actual) 5-GW points for the model's own top picks.
        'own_top_overprediction_points': 3.0,
        'season_total_below': -1.0,
    },
    'noise_pair': ['v2_huber', 'v2_huber_seed2'],
}


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with Path(path).open(encoding='utf-8') as handle:
            _merge(config, json.load(handle))
    validate_config(config)
    return config


def validate_config(config):
    """Weights inside each group must sum to 1, or the percentages lie."""
    groups = {
        'category_weights': config['category_weights'],
        'future_weights': config['future_weights'],
        'future_sides': config['future_sides'],
        'ranking_weights': config['ranking_weights'],
        'buckets': {bucket['name']: bucket['weight'] for bucket in config['buckets']},
    }
    for name, weights in groups.items():
        total = sum(float(value) for value in weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError('%s must sum to 1, got %s' % (name, total))
    if int(config['horizon']) < 1:
        raise ValueError('horizon must be positive')
    if int(config['calibration_bins']) < 2:
        raise ValueError('calibration_bins must be at least 2')
