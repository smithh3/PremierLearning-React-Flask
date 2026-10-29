"""Season points dominate. The weekly term is a small rank signal in the same universe."""
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
