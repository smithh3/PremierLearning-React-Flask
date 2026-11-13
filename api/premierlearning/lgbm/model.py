"""Fit and apply the LightGBM points model.

Two passes: the first holds out the latest training season to find how many
boosting rounds help (early stopping), the second refits on every season with
that many rounds plus 10% for the extra season of data. Predictions are the
expected points of the fixture, minutes risk included, floored at 0.
"""
import re
from dataclasses import dataclass, field

import numpy

OBJECTIVES = ('regression', 'tweedie', 'huber', 'poisson', 'lambdarank')
DEFAULT_OBJECTIVE = 'regression'
RANK_OBJECTIVE = 'lambdarank'
NON_NEGATIVE_OBJECTIVES = ('tweedie', 'poisson')
MAX_ROUNDS = 4000
EARLY_STOP_ROUNDS = 150
REFIT_EXTRA = 1.1
IMPORTANCE_ROWS = 30

# Ranking: rows grouped by gameweek, relevance = points clipped to 0..30 with a
# linear gain (a 12-pointer counts 6x a 2-pointer, not 2^10x), scored on the
# top 300 of each gameweek (about 650 rows): nearly everyone who plays, so cheap
# defenders are ranked too, not just the top 50 (which saturates in a few rounds).
# The scores are not points, so they are mapped back: each row's percentile
# within its gameweek -> mean points at that percentile, learned on the
# held-out season (isotonic, so a higher rank never means fewer points).
RANK_MAX_LABEL = 30
RANK_PARAMS = {
    'metric': 'ndcg',
    'eval_at': [300],
    'lambdarank_truncation_level': 300,
    'label_gain': [float(gain) for gain in range(RANK_MAX_LABEL + 1)],
}
LINEAR_PARAMS = {'linear_tree': True, 'linear_lambda': 1.0}

# Team-id correction stage: the main model never sees the ids. A second, small
# model then fits only what the ids add on top of it (team / opponent id and
# position), heavily smoothed, and its rounds are chosen on the held-out season.
# If knowing "this is Arsenal" does not help on a season the trees never saw,
# early stopping leaves the stage at 0 rounds and the ids change nothing.
ID_STAGE_PARAMS = {
    'objective': 'regression',
    'learning_rate': 0.03,
    'num_leaves': 4,
    'min_data_in_leaf': 2000,
    'lambda_l2': 50.0,
    'cat_l2': 50.0,
    'cat_smooth': 100.0,
    'max_cat_threshold': 8,
    'min_data_per_group': 1000,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'num_threads': 4,
    'verbose': -1,
}
ID_STAGE_MAX_ROUNDS = 1500
ID_STAGE_EARLY_STOP = 100
ID_STAGE_MIN_GAIN = 1e-4   # held-out RMSE must drop by at least this, or the stage is dropped
POSITION = 'position'
# With gameweek groups, the hold-out is the newest 38 gameweeks of training
# rows: the latest season for a GW1 fit, and a rolling year for mid-season refits.
HOLDOUT_GAMEWEEKS = 38

BASE_PARAMS = {
    'learning_rate': 0.03,
    'num_leaves': 63,
    'min_data_in_leaf': 200,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l2': 5.0,
    'cat_smooth': 20.0,
    'min_data_per_group': 200,
    'max_bin': 255,
    'num_threads': 4,
    'verbose': -1,
}


# Rules we know hold for every player, all else equal: +1 means a higher value
# never lowers predicted points, -1 never raises them. Matched against the
# column names from features.py; anything unmatched is free. Left free on
# purpose: squads, rest / congestion and recent minutes load (a regular starter
# vs fatigue), and anything whose direction depends on position.
SAFE_RULES = (
    # The player's own output, recent or last season.
    (r'^hist\.(points_scored|minutes_played|xG|xA|goals|assists|defcon)\.(mean\d+|faded[\d.]+|per90_\d+)$', 1),
    (r'^hist\.xGC\.(mean\d+|faded[\d.]+|per90_\d+)$', -1),
    (r'^hist\.(played_share|sixty_share)\d+$', 1),
    (r'^hist\.since_(played|sixty)$', -1),
    (r'^seasons\.(points_per_90|xgi_per_90)\.(last|before)$', 1),
    (r'^context\.chance_of_playing$', 1),
    # His team's attack and defence, and the opponent's.
    (r'^current_team\.team_features\.(goals_last_n|goals_for_decay|xg_for_decay)$', 1),
    (r'^current_team\.team_features\.(goals_conceded_last_n|goals_against_decay|xg_against_decay)$', -1),
    (r'^current_opposition\.team_features\.(goals_conceded_last_n|goals_against_decay|xg_against_decay)$', 1),
    (r'^current_opposition\.team_features\.(goals_last_n|goals_for_decay|xg_for_decay)$', -1),
    (r'^team_hist\.(goals_for|xg_for)\.', 1),
    (r'^team_hist\.(goals_against|xg_against)\.', -1),
    (r'^opp_hist\.(goals_against|xg_against)\.', 1),
    (r'^opp_hist\.(goals_for|xg_for)\.', -1),
)
RULE_SETS = {'safe': SAFE_RULES}
MONOTONE_METHOD = 'intermediate'


def monotone_directions(names, rules=SAFE_RULES):
    """-1 / 0 / +1 per column; the first matching rule wins."""
    compiled = [(re.compile(pattern), direction) for pattern, direction in rules]
    directions = []
    for name in names:
        directions.append(next((direction for pattern, direction in compiled if pattern.search(name)), 0))
    return directions


def params_for(objective=DEFAULT_OBJECTIVE, seed=42, linear=False):
    if objective not in OBJECTIVES:
        raise ValueError('objective must be one of %s' % ', '.join(OBJECTIVES))
    params = dict(BASE_PARAMS, objective=objective, seed=int(seed))
    if objective == 'tweedie':
        params['tweedie_variance_power'] = 1.3
    if objective == 'huber':
        params['alpha'] = 3.0
    if objective == RANK_OBJECTIVE:
        params.update(RANK_PARAMS)
    if linear:
        params.update(LINEAR_PARAMS)
    return params


def gameweek_groups(batch):
    """Per row a (season, gameweek) key: ranking groups and the rolling hold-out."""
    years = list(batch.years) or [0] * len(batch.matches)
    return numpy.array([int(year or 0) * 100 + int(getattr(match, 'gameweek', 0) or 0)
                        for year, match in zip(years, batch.matches)], dtype=numpy.int64)


def recency_weights(years, half_life):
    """Row weights halving every half_life seasons back from the newest; None when off."""
    if not half_life:
        return None
    years = numpy.asarray(years, dtype=numpy.float64).reshape(-1)
    return 0.5 ** ((years.max() - years) / float(half_life))


def percentiles(scores, groups=None):
    """Each score's place within its group, 0 (lowest) to 1 (highest)."""
    scores = numpy.asarray(scores, dtype=numpy.float64).reshape(-1)
    out = numpy.empty(len(scores))
    keys = numpy.zeros(len(scores)) if groups is None else numpy.asarray(groups).reshape(-1)
    for key in numpy.unique(keys):
        rows = numpy.flatnonzero(keys == key)
        if len(rows) == 1:
            out[rows] = 0.5
            continue
        order = numpy.argsort(numpy.argsort(scores[rows], kind='stable'), kind='stable')
        out[rows] = order / float(len(rows) - 1)
    return out


def _fit_calibration(scores, points, groups):
    from sklearn.isotonic import IsotonicRegression

    curve = IsotonicRegression(out_of_bounds='clip').fit(percentiles(scores, groups), points)
    return numpy.asarray(curve.X_thresholds_, dtype=numpy.float64), numpy.asarray(curve.y_thresholds_, dtype=numpy.float64)


@dataclass
class _View:
    """Some of a table's columns, as LightGBM sees them."""
    values: numpy.ndarray
    names: list
    categorical: list


def _view(table, columns=None):
    if columns is None:
        return _View(table.values, list(table.names), list(table.categorical))
    names = [table.names[index] for index in columns]
    return _View(numpy.ascontiguousarray(table.values[:, columns]), names,
                 [name for name in names if name in table.categorical])


def _id_columns(table):
    return [index for index, name in enumerate(table.names) if name in table.categorical and name != POSITION]


@dataclass
class FittedLGBM:
    booster: object
    names: list
    categorical: list
    rounds: int
    report: dict = field(default_factory=dict)
    others: list = field(default_factory=list)   # extra seeds, averaged with booster
    main_columns: list = None    # columns the main boosters read (None: all)
    id_columns: list = None      # columns the id stage reads
    id_booster: object = None
    calibration: tuple = None    # ranking: (percentile, points) curve

    def raw_scores(self, table):
        values = table.values if self.main_columns is None else table.values[:, self.main_columns]
        boosters = [self.booster] + list(self.others)
        return sum(numpy.asarray(b.predict(values, num_iteration=self.rounds), dtype=numpy.float64)
                   for b in boosters) / len(boosters)

    def predict(self, table, groups=None):
        """Expected points, floored at 0. groups: each row's gameweek (ranking models only)."""
        if list(table.names) != list(self.names):
            raise ValueError('feature columns differ from the fitted model')
        raw = self.raw_scores(table)
        if self.calibration is not None:
            raw = numpy.interp(percentiles(raw, groups), *self.calibration)
        if self.id_booster is not None:
            raw = raw + numpy.asarray(self.id_booster.predict(table.values[:, self.id_columns]), dtype=numpy.float64)
        return numpy.maximum(raw, 0.0)


def _pick(array, rows):
    if array is None:
        return None
    return array if rows is None else array[rows]


def _dataset(lightgbm, view, target, rows=None, reference=None, weights=None, groups=None, init_score=None,
             linear=False):
    """groups: per-row group keys; rows must then be sorted by group."""
    group = None
    if groups is not None:
        _keys, group = numpy.unique(_pick(groups, rows), return_counts=True)
    return lightgbm.Dataset(
        _pick(view.values, rows), label=_pick(target, rows), weight=_pick(weights, rows), group=group,
        init_score=init_score, feature_name=list(view.names), categorical_feature=list(view.categorical),
        reference=reference, free_raw_data=True, params=dict(LINEAR_PARAMS) if linear else None,
    )


def _by_group(rows, groups):
    if groups is None:
        return rows
    return rows[numpy.argsort(groups[rows], kind='stable')]


def _errors(predicted, actual):
    predicted = numpy.maximum(predicted, 0.0)
    return {
        'rmse': float(numpy.sqrt(numpy.mean((predicted - actual) ** 2))),
        'mae': float(numpy.mean(numpy.abs(predicted - actual))),
        'bias': float(numpy.mean(predicted - actual)),
    }


def importance_lines(booster, names, limit=IMPORTANCE_ROWS):
    gains = numpy.asarray(booster.feature_importance(importance_type='gain'), dtype=numpy.float64)
    total = gains.sum() or 1.0
    order = numpy.argsort(-gains)[:limit]
    lines = ['  top %s columns by gain (share of all splits\' gain):' % min(limit, len(order))]
    for rank, index in enumerate(order, start=1):
        lines.append('  %3d  %-55s %5.1f%%' % (rank, names[index], 100.0 * gains[index] / total))
    return lines


def _id_stage_holdout(lightgbm, ids, labels, train_rows, valid_rows, base_train, base_valid, weights, seed):
    """Rounds the id stage earns on the held-out season (0: the ids do not help there)."""
    train_set = _dataset(lightgbm, ids, labels, train_rows, weights=weights, init_score=base_train)
    valid_set = _dataset(lightgbm, ids, labels, valid_rows, reference=train_set, init_score=base_valid)
    stage = lightgbm.train(
        dict(ID_STAGE_PARAMS, seed=int(seed)), train_set, num_boost_round=ID_STAGE_MAX_ROUNDS,
        valid_sets=[valid_set], valid_names=['latest'],
        callbacks=[lightgbm.early_stopping(ID_STAGE_EARLY_STOP, verbose=False)],
    )
    best = int(stage.best_iteration or 0)
    before = _errors(base_valid, labels[valid_rows])['rmse']
    after = _errors(base_valid + stage.predict(ids.values[valid_rows], num_iteration=max(best, 1)),
                    labels[valid_rows])['rmse']
    if before - after < ID_STAGE_MIN_GAIN:
        best = 0
    return best, {'rmse_without_ids': before, 'rmse_with_ids': after, 'rounds_earned': best}


def fit(table, labels, years, objective=DEFAULT_OBJECTIVE, seed=42, max_rounds=MAX_ROUNDS,
        early_stop=EARLY_STOP_ROUNDS, verbose=True, rules=None, seeds=1, linear=False, id_stage=False,
        recency=0.0, groups=None):
    """Train on every row; the latest season only steers the number of rounds.

    rules: a RULE_SETS name (e.g. 'safe') for monotone constraints, or None.
    seeds: the final model is the average of this many refits with seeds
    seed, seed + 1, ... (different row and column samples), which steadies
    the predictions; the number of rounds is found once, with the first seed.
    linear: linear leaves (each leaf fits a small line, not a constant).
    id_stage: keep the team / opponent ids out of the main model and add them
    as a separate correction that must earn its rounds on the held-out season.
    recency: half-life in seasons for row weights (0 = all seasons equal).
    groups: each row's gameweek key; needed for the ranking objective.
    """
    seeds = max(1, int(seeds))
    import lightgbm

    labels = numpy.asarray(labels, dtype=numpy.float64).reshape(-1)
    years = numpy.asarray(years).reshape(-1)
    if len(labels) != table.values.shape[0] or len(years) != len(labels):
        raise ValueError('table, labels and years must have the same rows')
    rank = objective == RANK_OBJECTIVE
    if rank and groups is None:
        raise ValueError('the ranking objective needs each row\'s gameweek (groups)')
    if rank and id_stage:
        raise ValueError('the id stage works on points, not ranking scores')
    groups = None if groups is None else numpy.asarray(groups).reshape(-1)
    rank_groups = groups if rank else None
    if rank:
        target = numpy.clip(numpy.round(labels), 0, RANK_MAX_LABEL).astype(numpy.int32)
    elif objective in NON_NEGATIVE_OBJECTIVES:
        # Tweedie and Poisson need labels >= 0: the rare negative score (own
        # goal, red card) trains as 0. Held-out errors still use the real points.
        target = numpy.maximum(labels, 0.0)
    else:
        target = labels
    weights = recency_weights(years, recency)
    id_columns = _id_columns(table) if id_stage else None
    if id_stage and not id_columns:
        raise ValueError('the id stage needs the team id columns (flatten with ids=True)')
    main_columns = None
    if id_columns:
        main_columns = [index for index in range(len(table.names)) if index not in set(id_columns)]
    main = _view(table, main_columns)
    params = params_for(objective, seed, linear=linear)
    report = {'objective': objective, 'rows': int(len(labels)), 'columns': len(main.names)}
    if linear:
        report['linear'] = True
    if recency:
        report['recency'] = float(recency)
    if rules:
        if rules not in RULE_SETS:
            raise ValueError('rules must be one of %s' % ', '.join(RULE_SETS))
        directions = monotone_directions(main.names, RULE_SETS[rules])
        params['monotone_constraints'] = directions
        params['monotone_constraints_method'] = MONOTONE_METHOD
        report['rules'] = '%s: %s columns rising, %s falling' % (
            rules, directions.count(1), directions.count(-1))
    latest = years.max()
    valid = years == latest
    holdout_from = None
    if groups is not None:
        keys = numpy.unique(groups)
        if len(keys) > HOLDOUT_GAMEWEEKS:
            holdout_from = int(keys[-HOLDOUT_GAMEWEEKS])
            valid = groups >= holdout_from
    calibration = None
    id_rounds = 0
    ids = None
    if id_columns:
        position = [index for index, name in enumerate(table.names) if name == POSITION]
        ids = _view(table, id_columns + position)
    if valid.all() or not valid.any():
        if rank:
            raise ValueError('the ranking objective needs a held-out season to map scores to points')
        if id_columns:
            raise ValueError('the id stage needs a held-out season to earn its rounds')
        rounds = int(max_rounds if early_stop is None else min(max_rounds, 500))
        report['note'] = 'one season only: no hold-out, %s rounds' % rounds
    else:
        train_rows = _by_group(numpy.flatnonzero(~valid), rank_groups)
        valid_rows = _by_group(numpy.flatnonzero(valid), rank_groups)
        train_set = _dataset(lightgbm, main, target, train_rows, weights=weights, groups=rank_groups, linear=linear)
        valid_set = _dataset(lightgbm, main, target, valid_rows, reference=train_set, groups=rank_groups,
                             linear=linear)
        booster = lightgbm.train(
            params, train_set, num_boost_round=max_rounds, valid_sets=[valid_set], valid_names=['latest'],
            callbacks=[lightgbm.early_stopping(early_stop, first_metric_only=True, verbose=False)],
        )
        best = int(booster.best_iteration or booster.current_iteration())
        held = numpy.asarray(booster.predict(main.values[valid_rows], num_iteration=best), dtype=numpy.float64)
        if rank:
            calibration = _fit_calibration(held, labels[valid_rows], groups[valid_rows])
            held = numpy.interp(percentiles(held, groups[valid_rows]), *calibration)
        report.update({
            'holdout_season': int(latest),
            'holdout_from': holdout_from,
            'holdout_rows': int(len(valid_rows)),
            'best_round': best,
            'holdout': _errors(held, labels[valid_rows]),
            'holdout_mean_actual': float(labels[valid_rows].mean()),
        })
        if id_columns:
            base_train = numpy.asarray(booster.predict(main.values[train_rows], num_iteration=best), dtype=numpy.float64)
            id_rounds, report['id_stage'] = _id_stage_holdout(
                lightgbm, ids, labels, train_rows, valid_rows, base_train, held, weights, seed)
            del base_train
        rounds = max(50, int(round(best * REFIT_EXTRA)))
        del booster, train_set, valid_set
    all_rows = _by_group(numpy.arange(len(labels)), rank_groups) if rank else None
    full_set = _dataset(lightgbm, main, target, all_rows, weights=weights, groups=rank_groups, linear=linear)
    booster = lightgbm.train(params, full_set, num_boost_round=rounds)
    others = []
    del full_set
    for extra in range(1, seeds):
        # A fresh Dataset each time: LightGBM refuses changed seeds on a built one.
        others.append(lightgbm.train(
            dict(params, seed=int(seed) + extra),
            _dataset(lightgbm, main, target, all_rows, weights=weights, groups=rank_groups, linear=linear),
            num_boost_round=rounds))
    report['rounds'] = rounds
    if seeds > 1:
        report['seeds'] = seeds
    fitted = FittedLGBM(booster, list(table.names), list(table.categorical), rounds, report, others,
                        main_columns=main_columns, id_columns=None, calibration=calibration)
    if id_columns and id_rounds:
        # Refit the stage on every season, on top of the final main model.
        fitted.id_columns = id_columns + position
        id_set = _dataset(lightgbm, ids, labels, weights=weights, init_score=fitted.raw_scores(table))
        fitted.id_booster = lightgbm.train(dict(ID_STAGE_PARAMS, seed=int(seed)), id_set,
                                           num_boost_round=max(1, int(round(id_rounds * REFIT_EXTRA))))
        report['id_stage']['rounds'] = fitted.id_booster.current_iteration()
    elif id_columns:
        report['id_stage']['rounds'] = 0
    if verbose:
        print(describe(fitted))
    return fitted


def describe(fitted):
    report = fitted.report
    lines = ['lightgbm %s: %s rows x %s columns, %s rounds' % (
        report.get('objective'), report.get('rows'), report.get('columns'), fitted.rounds)]
    if 'rules' in report:
        lines.append('  monotone rules %s' % report['rules'])
    if 'seeds' in report:
        lines.append('  average of %s seeds' % report['seeds'])
    if report.get('linear'):
        lines.append('  linear leaves')
    if report.get('recency'):
        lines.append('  recency weights: half-life %g seasons' % report['recency'])
    if fitted.calibration is not None:
        lines.append('  ranking scores mapped to points by gameweek percentile (%s steps)' % len(fitted.calibration[0]))
    if 'id_stage' in report:
        stage = report['id_stage']
        lines.append('  team-id stage: %s rounds; held-out RMSE %.4f without ids, %.4f with'
                     % (stage.get('rounds', 0), stage['rmse_without_ids'], stage['rmse_with_ids']))
    if 'holdout' in report:
        held = report['holdout']
        start = report.get('holdout_from')
        if start is None or start % 100 == 1:
            label = 'season %s' % report['holdout_season']
        else:
            label = 'last %s GWs (from %s GW%s)' % (HOLDOUT_GAMEWEEKS, start // 100, start % 100)
        lines.append(
            '  held-out %s (%s rows, actual mean %.3f): best round %s, RMSE %.3f, MAE %.3f, bias %+.3f'
            % (label, report['holdout_rows'], report['holdout_mean_actual'],
               report['best_round'], held['rmse'], held['mae'], held['bias'])
        )
    elif 'note' in report:
        lines.append('  ' + report['note'])
    lines.extend(importance_lines(fitted.booster, fitted.booster.feature_name()))
    return '\n'.join(lines)
