"""Season points dominate.

v1 and v2: the weekly term is a small rank signal in the same universe, and
a multi-season run is paid once at its end. v3: see WEEKLY_POINTS_WEIGHT.
"""
from __future__ import annotations

TERMINAL_DIVISOR = 1000.0
WEEKLY_COEFFICIENT = 0.01
PERCENTILE_DIVISOR = 100.0


def terminal_reward(final_points: float) -> float:
    return float(final_points) / TERMINAL_DIVISOR


def multi_season_reward(points, percentiles) -> float:
    """One reward for a run that played several seasons.

    Points are summed and divided by 1000, the same scale as one season.
    Each finish percentile is 0-100 and is divided by 100, so a season's
    rank adds about as much as a thousand points.
    """
    total_points = sum(float(value) for value in points)
    total_percentiles = sum(float(value) for value in percentiles)
    return terminal_reward(total_points) + total_percentiles / PERCENTILE_DIVISOR


# v3 pays some points every week so each decision gets credit near the time
# it is made. The season-end payment is still the larger part:
#   each week:    0.25 x points / 1000      (about 0.5 over a season)
#   season end:   points / 1000 + finish percentile / 100   (about 2.8)
# Summed over a three-season run, the season-end parts equal the old
# end-of-run reward exactly.
WEEKLY_POINTS_WEIGHT = 0.25


def weekly_points_reward(points: float) -> float:
    return WEEKLY_POINTS_WEIGHT * float(points) / TERMINAL_DIVISOR


def season_end_reward(points: float, percentile: float) -> float:
    return terminal_reward(points) + float(percentile) / PERCENTILE_DIVISOR


def percentile_rank(score: float, reference) -> float:
    """Fraction of the reference population this score beats or ties. Mid-rank on ties."""
    others = [float(value) for value in reference]
    if not others:
        return 0.5
    below = sum(1 for value in others if value < score)
    equal = sum(1 for value in others if value == score)
    return (below + 0.5 * equal) / float(len(others))


def weekly_reward(score: float, reference) -> float:
    return WEEKLY_COEFFICIENT * (percentile_rank(score, reference) - 0.5)
