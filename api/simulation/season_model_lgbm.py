"""Fit the LightGBM points model once per season, then reuse it for every GW snapshot.

Rows and inputs come from learning_v2's build_dataset with the chosen plan, so
LightGBM sees exactly what the net with that plan sees as of each deadline;
premierlearning/lgbm/features.py flattens the lists into columns. As with
FixedSeasonModelsV2, each snapshot rebuilds its features from the matches
before its own deadline and only the trees stay at the GW1 fit. The vocab is
frozen at fit time so team ids keep their meaning.

Feature set 3 adds season columns (team ratings, what the opponent concedes to
each position; premierlearning/lgbm/extras.py), rebuilt from the runner's
seasons for every batch, and shrunk per-90s whose position rates are fixed at
the fit. staged=True fits lgbm/staged.py (minutes, clean sheet, rest of points)
instead of one model, and also stores each fixture's minutes bands.

Feature set 4 adds FPL's player news (lgbm/news.py): played fixtures read the
snapshot from their own gameweek's deadline, upcoming ones the as-of deadline.
"""

import numpy

from premierlearning.learning_v2.dataset import build_dataset, game_stat
from premierlearning.learning_v2.learn import store_minutes_bands, store_points
from premierlearning.learning_v2.plan import load_plan
from premierlearning.lgbm.extras import season_columns
from premierlearning.lgbm.features import DEFAULT_FEATURES, FEATURE_SETS, POSITION_COLUMN, flatten
from premierlearning.lgbm.model import DEFAULT_OBJECTIVE, fit as fit_lgbm
from premierlearning.lgbm.news import news_columns
from premierlearning.lgbm.staged import fit_staged

DEFAULT_LGBM_PLAN = 'simple_h1'


def _seasons(runner):
    return list(getattr(runner, 'past_seasons', []) or []) + [runner.current_season]


def gameweek_groups(batch):
    """Per row a (season, gameweek) key: the groups the ranking objective orders within."""
    years = list(batch.years) or [0] * len(batch.matches)
    return numpy.array([int(year or 0) * 100 + int(getattr(match, 'gameweek', 0) or 0)
                        for year, match in zip(years, batch.matches)], dtype=numpy.int64)


def flatten_with_seasons(batch, plan, seasons, features=DEFAULT_FEATURES, ids=True, priors=None,
                         as_of_gameweek=None):
    """flatten(), plus the season and news columns when the feature set has them.

    as_of_gameweek: the deadline an upcoming batch is predicted from (its news
    snapshot); None for played fixtures, which read their own gameweek's.
    """
    table = flatten(batch, plan, features=features, ids=ids, priors=priors)
    settings = FEATURE_SETS[features]
    if settings.season_columns:
        if POSITION_COLUMN not in table.names:
            raise ValueError('feature set %s needs the position column' % features)
        table = table.with_columns(season_columns(batch, seasons, table.column(POSITION_COLUMN)))
    if settings.news:
        table = table.with_columns(news_columns(batch, as_of_gameweek=as_of_gameweek))
    return table


class FixedSeasonModelsLGBM:
    def __init__(self, plan, vocab, fitted, features=DEFAULT_FEATURES, ids=True, priors=None, staged=False):
        self.plan = plan
        self.vocab = vocab
        self.fitted = fitted
        self.features = features
        self.ids = ids
        self.priors = priors
        self.staged = staged

    def _flatten(self, batch, runner):
        return flatten_with_seasons(batch, self.plan, _seasons(runner), features=self.features, ids=self.ids,
                                    priors=self.priors, as_of_gameweek=runner.as_of_gameweek)

    @classmethod
    def fit(cls, runner, plan_name=DEFAULT_LGBM_PLAN, objective=DEFAULT_OBJECTIVE, seed=42, rules=None,
            features=DEFAULT_FEATURES, ids=True, seeds=1, linear=False, id_stage=False, recency=0.0,
            staged=False):
        """Train on everything before the runner's gameweek (GW1: pre-season data only).

        Mid-season fits (--retrain-every) include this season's played
        gameweeks; the runner's training cutoff keeps later labels out.
        """
        plan = load_plan(plan_name)
        print('Training one fixed LightGBM model on plan %s as of GW%s...' % (plan.name, runner.as_of_gameweek))
        runner.process_player_matches()
        train_batch, _predict, vocab = build_dataset(
            _seasons(runner), plan,
            predict_season=runner.current_season,
            cutoff_gameweek=runner._trained_through_gameweek(),
        )
        if not train_batch.matches:
            raise ValueError('LightGBM found no training fixtures before GW%s' % runner.as_of_gameweek)
        vocab.freeze()
        table = flatten_with_seasons(train_batch, plan, _seasons(runner), features=features, ids=ids)
        labels = train_batch.labels.reshape(-1)
        years = list(train_batch.years)
        groups = gameweek_groups(train_batch)
        print('  %s train rows, %s columns (%s categorical), feature set %s, %s teams (vocab frozen)' % (
            table.values.shape[0], len(table.names), len(table.categorical), features, vocab.team_size - 1))
        minutes = clean_sheets = None
        if staged:
            minutes = numpy.array([game_stat(match, 'minutes') for match in train_batch.matches])
            clean_sheets = numpy.array([game_stat(match, 'clean_sheets') for match in train_batch.matches])
        del train_batch, _predict
        if staged:
            if rules or linear or id_stage or recency or objective != DEFAULT_OBJECTIVE:
                raise ValueError('the staged LightGBM model takes none of rules, linear, id stage, recency '
                                 'or another objective')
            fitted = fit_staged(table, labels, years, minutes, clean_sheets, seed=seed, seeds=seeds, groups=groups)
        else:
            fitted = fit_lgbm(table, labels, years, objective=objective, seed=seed, rules=rules, seeds=seeds,
                              linear=linear, id_stage=id_stage, recency=recency, groups=groups)
        return cls(plan, vocab, fitted, features=features, ids=ids, priors=table.priors, staged=staged)

    def prepare_runner(self, runner):
        """Rebuild causal player features, as Runner.run does before learning."""
        runner.process_player_matches()

    def predict(self, runner, already_prepared=False):
        if not already_prepared:
            self.prepare_runner(runner)
        _train, predict_batch, _vocab = build_dataset(
            _seasons(runner), self.plan,
            predict_season=runner.current_season,
            vocab=self.vocab,
            include_train=False,
        )
        if predict_batch.matches:
            table = self._flatten(predict_batch, runner)
            store_points(predict_batch.matches, self.fitted.predict(table, groups=gameweek_groups(predict_batch)))
            if self.staged:
                store_minutes_bands(predict_batch.matches, self.fitted.predict_bands(table))
        runner.current_season.calculate_player_points_over_next_5_gameweeks()
        return runner.current_season
