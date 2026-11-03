"""Combine model outputs into the points the linear optimiser should see.

Predicted pp90 and minutes stay raw. Chance-to-play (live FPL or historic
fplcache) is applied here so availability policy can change without rebuilding
the prediction cache.

Live/Fargate scoring uses step_75: ≥75 → 90, 65–75 → 65–90, below 65 identity.
Historical simulations default to raw minutes and can still pass step_80 /
step_60 / step_70.
"""

from premierlearning.helper_functions import effective_minutes_for_points, implied_pp90


MINUTES_SCALE_FULL_IF_LIKELY = 'full_if_likely'
MINUTES_SCALE_RAW = 'raw'
# Same step idea as live, different cutoffs. full_at → 90; optional mid band
# maps [mid_at, full_at) onto [mid_lo, mid_hi]. Below mid_at (or full_at if
# there is no mid band) is identity.
MINUTES_SCALE_STEP_80 = 'step_80'
MINUTES_SCALE_STEP_75 = 'step_75'
MINUTES_SCALE_STEP_60 = 'step_60'
MINUTES_SCALE_STEP_70 = 'step_70'
_STEP_PRESETS = {
    # ≥80 → 90; 55–80 → 70–85. Harder to count as a starter than live.
    MINUTES_SCALE_STEP_80: (80.0, 55.0, 70.0, 85.0),
    # Live/Fargate: ≥75 → 90; 65–75 → 65–90.
    MINUTES_SCALE_STEP_75: (75.0, 65.0, 65.0, 90.0),
    # ≥60 → 90; 45–60 → 60–80. Rotation-risk minutes still look like starters.
    MINUTES_SCALE_STEP_60: (60.0, 45.0, 60.0, 80.0),
    # ≥70 → 90; no 60–75 interpolation. Cliff only.
    MINUTES_SCALE_STEP_70: (70.0, None, None, None),
}
LIVE_MINUTES_SCALE = MINUTES_SCALE_STEP_75

# Starter is always 1.0. Live/Fargate: first two outfield bench slots are
# 0.25 each; third outfield and the reserve keeper are 0. Historical replay
# still uses fill 0.3 and (0.5, 0.3, 0.1) plus keeper 0.2.
LIVE_OUTFIELD_BENCH_WEIGHTS = (0.25, 0.25, 0.0)
LIVE_RESERVE_KEEPER_WEIGHT = 0.0
LIVE_BENCH_WEIGHTS = LIVE_OUTFIELD_BENCH_WEIGHTS + (LIVE_RESERVE_KEEPER_WEIGHT,)
# Historical replay: pass 1 uses a uniform leftover weight; pass 2 applies
# these slot weights when refilling the frozen XI's bench.
SIM_BENCH_FILL_WEIGHT = 0.3
SIM_OUTFIELD_BENCH_WEIGHTS = (0.5, 0.3, 0.1)
SIM_RESERVE_KEEPER_WEIGHT = 0.2
SIM_BENCH_WEIGHTS = SIM_OUTFIELD_BENCH_WEIGHTS


def chance_multiplier(chance) -> float:
    if chance is None or chance == '':
        return 1.0
    return max(0.0, min(100.0, float(chance))) / 100.0


def stepped_minutes(mins, full_at, mid_at=None, mid_lo=None, mid_hi=None):
    """Map predicted minutes with a starter cliff and optional rotation band."""
    m = max(0.0, min(90.0, float(mins or 0)))
    if m >= full_at:
        return 90.0
    if mid_at is not None and m >= mid_at:
        span = float(full_at) - float(mid_at)
        if span <= 0:
            return 90.0
        return float(mid_lo) + (m - mid_at) / span * (float(mid_hi) - float(mid_lo))
    return m


def minutes_for_points(mins, minutes_scale=None) -> float:
    """Minutes used when turning pp90 into a points total.

    ``raw`` (simulation default): predicted minutes with no mapping.
    ``step_75`` (live/Fargate): ≥75 → 90; 65–75 → 65–90; below 65 identity.
    ``full_if_likely``: identity via effective_minutes_for_points.
    ``step_80`` / ``step_60`` / ``step_70``: same step shape, different ranges.
    """
    scale = minutes_scale or MINUTES_SCALE_RAW
    if scale == MINUTES_SCALE_RAW:
        return max(0.0, float(mins or 0.0))
    if scale == MINUTES_SCALE_FULL_IF_LIKELY:
        return effective_minutes_for_points(mins)
    preset = _STEP_PRESETS.get(scale)
    if preset is None:
        raise ValueError('Unknown minutes_scale %r' % (minutes_scale,))
    return stepped_minutes(mins, *preset)


def slot_bench_weights(outfield=(), reserve_keeper=None):
    """Four MIP slots: three outfield bench weights, then the reserve keeper.

    Empty outfield and no keeper weight means bench is unused (starters only).
    A 4-tuple ``outfield`` with no keeper arg is treated as already-combined.
    """
    values = tuple(float(weight) for weight in (outfield or ()))
    if len(values) >= 4 and reserve_keeper is None:
        return (values + (0.0, 0.0, 0.0, 0.0))[:4]
    if not values and reserve_keeper is None:
        return ()
    outfield3 = (values + (0.0, 0.0, 0.0))[:3]
    keeper = 0.0 if reserve_keeper is None else float(reserve_keeper)
    return outfield3 + (keeper,)


def normalize_bench_weights(weights, reserve_keeper=None):
    return slot_bench_weights(weights, reserve_keeper)


def unadjusted_points(pp90, minutes, minutes_scale=None) -> float:
    return (
        max(0.0, float(pp90 or 0.0))
        * minutes_for_points(minutes, minutes_scale)
        / 90.0
    )


def optimiser_points(pp90, minutes, chance=None, minutes_scale=None) -> float:
    return unadjusted_points(pp90, minutes, minutes_scale) * chance_multiplier(chance)


POINTS_BASIS_RAW = 'raw'


def match_optimiser_points(match, minutes_scale=None) -> float:
    """Score one fixture the way the linear optimiser should see it.

    Live/Fargate defaults to ``step_75``. Pass ``raw`` to skip the mapping.
    A fixture whose points_basis is 'raw' (learning_v2) already holds expected
    points with minutes and chance-to-play risk inside them, so it is used as
    it is, as the historical manager uses stored v2 points.
    """
    if getattr(match, 'points_basis', None) == POINTS_BASIS_RAW:
        return max(0.0, float(getattr(match, 'points', 0.0) or 0.0))
    stats = getattr(match, 'game_stats', None) or {}
    try:
        stats_mins = float(stats.get('minutes', 0) or 0)
    except (TypeError, ValueError):
        stats_mins = 0.0
    minutes = stats_mins
    if minutes <= 0:
        minutes = getattr(match, 'minutes', None)
    pp90 = getattr(match, 'predicted_pp90', None)
    if pp90 is None:
        pp90 = implied_pp90(getattr(match, 'points', 0.0), minutes)
    chance = getattr(match, 'chance_of_playing', None)
    scale = LIVE_MINUTES_SCALE if minutes_scale is None else minutes_scale
    return optimiser_points(pp90, minutes, chance, minutes_scale=scale)


def row_optimiser_points(row, minutes_scale=None) -> float:
    """Score a stored prediction row. Old caches without pp90 keep baked points."""
    pp90 = row.get('predicted_pp90')
    if pp90 is None or pp90 == '':
        return float(row.get('predicted_points') or 0.0)
    return optimiser_points(
        pp90,
        row.get('predicted_minutes'),
        row.get('chance_of_playing'),
        minutes_scale=minutes_scale,
    )
