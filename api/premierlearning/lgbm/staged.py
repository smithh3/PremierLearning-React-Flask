"""LightGBM v5: expected points built in stages, the way FPL scores them.

1. Minutes: the chance of 0, 1-59 and 60+ minutes (multiclass), on every row.
2. Clean sheet: the chance his team keeps one, on rows where a goalkeeper,
   defender or midfielder played 60+ (binary). Its columns include the team
   ratings (feature set 3), so it is those ratings plus learned corrections
   (rest, squads, form).
3. The rest of his points when he plays: goals, assists, bonus, saves, defcon,
   cards and goals conceded (regression on rows where he played, with a 60+
   flag column, so the same trees give "rest if 1-59" and "rest if 60+").

Expected points = P(1-59) x (1 + rest if 1-59)
                + P(60+) x (2 + rest if 60+ + clean-sheet points x P(clean sheet)),
with clean-sheet points 4 / 4 / 1 / 0 for GK / DEF / MID / FWD. Each stage
picks its rounds on the held-out newest 38 gameweeks, then refits on every row
and averages `seeds` refits, as lgbm/model.py does for the single model.
"""
from dataclasses import dataclass, field

import numpy

from premierlearning.lgbm.model import (
    BASE_PARAMS, EARLY_STOP_ROUNDS, HOLDOUT_GAMEWEEKS, MAX_ROUNDS, POSITION, REFIT_EXTRA, _errors, importance_lines,
)

SIXTY = 'stage.sixty'
BANDS = 3
APPEARANCE = (0.0, 1.0, 2.0)
CS_POINTS = {1: 4.0, 2: 4.0, 3: 1.0, 4: 0.0}
CS_POSITIONS = (1, 2, 3)
STAGES = ('minutes', 'clean_sheet', 'rest')
STAGE_PARAMS = {
    'minutes': {'objective': 'multiclass', 'num_class': BANDS, 'metric': 'multi_logloss'},
    'clean_sheet': {'objective': 'binary', 'metric': 'binary_logloss'},
    'rest': {'objective': 'regression', 'metric': 'l2'},
}


def minutes_band(minutes):
    """0: did not play, 1: 1-59 minutes, 2: 60+."""
    minutes = numpy.asarray(minutes, dtype=numpy.float64).reshape(-1)
    return numpy.where(minutes >= 60.0, 2, numpy.where(minutes >= 1.0, 1, 0)).astype(numpy.int32)


def holdout_mask(years, groups):
    """Rows of the newest HOLDOUT_GAMEWEEKS gameweeks (else the latest season)."""
    years = numpy.asarray(years).reshape(-1)
    valid = years == years.max()
    if groups is not None:
        keys = numpy.unique(groups)
        if len(keys) > HOLDOUT_GAMEWEEKS:
            valid = numpy.asarray(groups).reshape(-1) >= keys[-HOLDOUT_GAMEWEEKS]
    return valid


@dataclass
class _Stage:
    boosters: list
    rounds: int
    report: dict

    def predict(self, values):
        return sum(numpy.asarray(b.predict(values, num_iteration=self.rounds), dtype=numpy.float64)
                   for b in self.boosters) / len(self.boosters)


def _train_stage(lightgbm, name, values, names, categorical, target, rows, valid, seed, seeds):
    """Early-stop on the held-out part of `rows`, then refit on all of `rows`."""
    params = dict(BASE_PARAMS, seed=int(seed), **STAGE_PARAMS[name])
    report = {'rows': int(len(rows))}

    def dataset(subset, reference=None):
        return lightgbm.Dataset(values[subset], label=target[subset], feature_name=list(names),
                                categorical_feature=list(categorical), reference=reference, free_raw_data=True)

    train_rows, valid_rows = rows[~valid[rows]], rows[valid[rows]]
    held = None
    if len(train_rows) and len(valid_rows):
        train_set = dataset(train_rows)
        booster = lightgbm.train(
            params, train_set, num_boost_round=MAX_ROUNDS, valid_sets=[dataset(valid_rows, train_set)],
            valid_names=['latest'], callbacks=[lightgbm.early_stopping(EARLY_STOP_ROUNDS, verbose=False)])
        best = int(booster.best_iteration or booster.current_iteration())
        held = _Stage([booster], best, {})
        rounds = max(50, int(round(best * REFIT_EXTRA)))
        report.update(best_round=best, holdout_rows=int(len(valid_rows)))
    else:
        rounds = 500
        report['note'] = 'no hold-out, %s rounds' % rounds
    boosters = [lightgbm.train(dict(params, seed=int(seed) + extra), dataset(rows), num_boost_round=rounds)
                for extra in range(max(1, int(seeds)))]
    report['rounds'] = rounds
    return _Stage(boosters, rounds, report), held


@dataclass
class FittedStaged:
    names: list
    categorical: list
    minutes: _Stage
    clean_sheet: _Stage
    rest: _Stage
    report: dict = field(default_factory=dict)

    def _check(self, table):
        if list(table.names) != list(self.names):
            raise ValueError('feature columns differ from the fitted model')

    def predict_bands(self, table):
        """Chance of 0, 1-59 and 60+ minutes per row."""
        self._check(table)
        return _parts(table, self.minutes, self.clean_sheet, self.rest)[0]

    def predict(self, table, groups=None):
        """Expected points, floored at 0 (groups is accepted for the single model's signature)."""
        self._check(table)
        return _parts(table, self.minutes, self.clean_sheet, self.rest)[1]


def _with_sixty(values, flag):
    return numpy.hstack([values, numpy.full((values.shape[0], 1), flag, dtype=numpy.float32)])


def _parts(table, minutes, clean_sheet, rest):
    """(bands, expected points) for every row of a table."""
    values = table.values
    bands = minutes.predict(values).reshape(-1, BANDS)
    clean = clean_sheet.predict(values) if clean_sheet is not None else numpy.zeros(len(values))
    rest_short = rest.predict(_with_sixty(values, 0.0))
    rest_long = rest.predict(_with_sixty(values, 1.0))
    positions = numpy.nan_to_num(table.column(POSITION)).astype(numpy.int64)
    cs_points = numpy.array([CS_POINTS.get(int(p), 0.0) for p in positions])
    points = (bands[:, 1] * (APPEARANCE[1] + rest_short)
              + bands[:, 2] * (APPEARANCE[2] + rest_long + cs_points * clean))
    return bands, numpy.maximum(points, 0.0)


def fit_staged(table, labels, years, minutes, clean_sheets, seed=42, seeds=1, groups=None, verbose=True):
    """labels: points; minutes and clean_sheets: the fixture's real minutes and FPL clean-sheet flag."""
    import lightgbm

    labels = numpy.asarray(labels, dtype=numpy.float64).reshape(-1)
    band = minutes_band(minutes)
    clean_sheets = numpy.asarray(clean_sheets, dtype=numpy.float64).reshape(-1)
    if POSITION not in table.names:
        raise ValueError('the staged model needs the position column')
    positions = numpy.nan_to_num(table.column(POSITION)).astype(numpy.int64)
    valid = holdout_mask(years, groups)
    everyone = numpy.arange(len(labels))
    names, categorical = list(table.names), list(table.categorical)

    minutes_stage, minutes_held = _train_stage(
        lightgbm, 'minutes', table.values, names, categorical, band.astype(numpy.float64), everyone, valid, seed, seeds)

    cs_rows = numpy.flatnonzero((band == 2) & numpy.isin(positions, CS_POSITIONS))
    cs_stage = cs_held = None
    if len(cs_rows):
        cs_stage, cs_held = _train_stage(
            lightgbm, 'clean_sheet', table.values, names, categorical, (clean_sheets > 0).astype(numpy.float64),
            cs_rows, valid, seed, seeds)

    played = numpy.flatnonzero(band > 0)
    cs_points = numpy.array([CS_POINTS.get(int(p), 0.0) for p in positions])
    rest_target = labels - numpy.asarray(APPEARANCE)[band] - cs_points * (clean_sheets > 0)
    rest_values = numpy.hstack([table.values[played],
                                (band[played] == 2).astype(numpy.float32).reshape(-1, 1)])
    rest_stage, rest_held = _train_stage(
        lightgbm, 'rest', rest_values, names + [SIXTY], categorical, rest_target[played],
        numpy.arange(len(played)), valid[played], seed, seeds)
    del rest_values

    report = {'objective': 'staged', 'rows': int(len(labels)), 'columns': len(names),
              'stages': {'minutes': minutes_stage.report,
                         'clean_sheet': cs_stage.report if cs_stage else {'rows': 0},
                         'rest': rest_stage.report}}
    if seeds > 1:
        report['seeds'] = int(seeds)
    if minutes_held is not None and rest_held is not None and valid.any():
        rows = numpy.flatnonzero(valid)
        from premierlearning.lgbm.features import FeatureTable
        held_table = FeatureTable(names, table.values[rows], categorical)
        held_bands, held_points = _parts(held_table, minutes_held, cs_held, rest_held)
        report['holdout'] = _errors(held_points, labels[rows])
        report['holdout_rows'] = int(len(rows))
        report['holdout_mean_actual'] = float(labels[rows].mean())
        report['holdout_bands'] = [float(value) for value in held_bands.mean(axis=0)]
        report['actual_bands'] = [float((band[rows] == k).mean()) for k in range(BANDS)]
    fitted = FittedStaged(names, categorical, minutes_stage, cs_stage, rest_stage, report)
    if verbose:
        print(describe_staged(fitted))
    return fitted


def describe_staged(fitted):
    report = fitted.report
    lines = ['lightgbm staged (minutes, clean sheet, rest of points): %s rows x %s columns'
             % (report['rows'], report['columns'])]
    if 'seeds' in report:
        lines.append('  average of %s seeds per stage' % report['seeds'])
    for name in STAGES:
        stage = report['stages'][name]
        lines.append('  %-12s %7s rows, %s rounds%s' % (
            name, stage.get('rows', 0), stage.get('rounds', 0),
            ' (best held-out round %s)' % stage['best_round'] if 'best_round' in stage else ''))
    if 'holdout' in report:
        held = report['holdout']
        lines.append('  held-out (%s rows, actual mean %.3f): RMSE %.3f, MAE %.3f, bias %+.3f'
                     % (report['holdout_rows'], report['holdout_mean_actual'], held['rmse'], held['mae'], held['bias']))
        lines.append('  minutes bands 0 / 1-59 / 60+: predicted %s, actual %s' % (
            ' / '.join('%.3f' % v for v in report['holdout_bands']),
            ' / '.join('%.3f' % v for v in report['actual_bands'])))
    lines.append('  minutes stage:')
    lines.extend(importance_lines(fitted.minutes.boosters[0], fitted.minutes.boosters[0].feature_name(), 15))
    lines.append('  rest-of-points stage:')
    lines.extend(importance_lines(fitted.rest.boosters[0], fitted.rest.boosters[0].feature_name(), 15))
    return '\n'.join(lines)
