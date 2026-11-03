"""Rolling linear calibration: y = a + b * prediction, refitted at every deadline.

At deadline g the fit uses (prediction, actual) pairs from earlier snapshots
whose target gameweek had already been played: any earlier season, or this
season with target < g. Nothing from g onward is used, so the calibrated model
is scored on every season like the others instead of losing the first to a
fitting period. Until `min_pairs` exist the prediction passes through unchanged.

Three choices keep the correction from misleading the optimiser:
- It works on points before chance-to-play and multiplies by the flag after,
  fitting only on players with no doubt. Applied after the flag, the +1 or so
  intercept handed every 0%-flagged player about a point.
- It is fitted on gameweeks with exactly one fixture and applied to each fixture
  before a double is summed. Applied to the summed double, the slope (~0.6)
  shrank it like one big prediction and undervalued doubles.
- v2 sets whose points already carry the flag calibrate unflagged players only;
  flagged players keep the model's own number.

Blanks stay at 0 and calibrated fixtures are floored at 0. A positive slope
never reorders unflagged single-fixture players within a gameweek.
"""

from collections import defaultdict

import numpy


class _Sums:
    def __init__(self):
        self.n = 0.0
        self.x = 0.0
        self.y = 0.0
        self.xx = 0.0
        self.xy = 0.0

    def add(self, x, y):
        x = numpy.asarray(x, dtype=float)
        y = numpy.asarray(y, dtype=float)
        self.n += float(x.size)
        self.x += float(x.sum())
        self.y += float(y.sum())
        self.xx += float((x * x).sum())
        self.xy += float((x * y).sum())

    def merge(self, other):
        self.n += other.n
        self.x += other.x
        self.y += other.y
        self.xx += other.xx
        self.xy += other.xy

    def fit(self, min_pairs):
        """(a, b), or None while there are too few pairs or no spread in x."""
        if self.n < max(2.0, float(min_pairs)):
            return None
        variance = self.xx - self.x * self.x / self.n
        if variance <= 1e-12:
            return None
        slope = (self.xy - self.x * self.y / self.n) / variance
        intercept = (self.y - slope * self.x) / self.n
        return intercept, slope


def apply_fit(parts, has_fixture, intercept, slope):
    """Calibrated player x lead grid from per-fixture parts."""
    slots = numpy.arange(parts.base.shape[-1])
    present = slots[None, None, :] < parts.count[..., None]
    fixtures = numpy.where(present, numpy.maximum(0.0, intercept + slope * parts.base), 0.0)
    calibrated = fixtures.sum(axis=-1) * parts.multiplier
    if parts.keep_flagged:
        raw = numpy.where(present, parts.base, 0.0).sum(axis=-1)
        calibrated = numpy.where(parts.flagged, raw, calibrated)
    return numpy.where(has_fixture, calibrated, 0.0)


def rolling_linear(units, base, name, min_pairs):
    """Add `name` to every unit: `base` corrected by the fit known at that deadline.

    Returns one record per unit with the coefficients used, for the report.
    """
    resolved = _Sums()
    pending = defaultdict(_Sums)
    records = []
    for unit in sorted(units, key=lambda item: item.key):
        for key in [key for key in pending if key[0] < unit.season or (key[0] == unit.season and key[1] < unit.origin)]:
            resolved.merge(pending.pop(key))
        coefficients = resolved.fit(min_pairs)
        parts = unit.parts[base]
        if coefficients is None:
            calibrated = unit.preds[base].copy()
        else:
            calibrated = apply_fit(parts, unit.has_fixture, *coefficients)
        unit.preds[name] = calibrated
        unit.missing[name] = unit.missing.get(base, 0)
        records.append({
            'season': unit.season,
            'origin': unit.origin,
            'pairs': int(resolved.n),
            'intercept': None if coefficients is None else round(coefficients[0], 6),
            'slope': None if coefficients is None else round(coefficients[1], 6),
        })
        for lead in range(unit.valid_leads):
            mask = unit.has_fixture[:, lead] & (parts.count[:, lead] == 1) & ~parts.flagged[:, lead]
            pending[(unit.season, unit.target(lead))].add(parts.base[mask, lead, 0], unit.actual[mask, lead])
    return records
