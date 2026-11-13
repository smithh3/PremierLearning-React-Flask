"""LightGBM for the live pipeline: `--learning-v2 --lightgbm` (Runner.learn).

v6 (the default since 2026-11-09, Hal asked) is the evaluated lgbm_v6
(simulation/model_eval/experiments.py): plan simple_h1_b1 flattened with
feature set 4 (set 2's windows and trends, set 3's team ratings, what the
opponent concedes to each position and shrunk per-90s, plus FPL's player news),
no team ids, 3 seeds averaged, squared error on raw points. It scored +2.32
over 2023/24-2025/26 against v3's +1.87 on 2025/26 alone, and led the
fantasy manager runs. `--lgbm-version v3` keeps the previous live model
(feature set 2).

Live, it trains on every played gameweek up to now (the evaluation trained
once at GW1); the hold-out that picks the number of rounds is the newest 38
gameweeks. Played fixtures read the news saved at their own deadline;
upcoming ones the news for the next deadline, which the runner saves from the
live API before learning (premierlearning/lgbm/news.py). Predictions are raw
expected points with minutes risk in them, stored like learning_v2's, so
squadtools uses them as they are. There is no minutes-bands head; the report
leaves that column out.

The plan reads the external all-competition features (api/data/external/
features), which must cover the upcoming fixtures: refresh them before
building the image.
"""
import sys

from premierlearning.learning_v2.dataset import build_dataset
from premierlearning.learning_v2.learn import (
    PREDICTION_TABLE_SIZE,
    format_opening_predictions,
    store_points,
    upcoming_gameweeks,
)
from premierlearning.learning_v2.plan import load_plan
from premierlearning.lgbm.model import fit, gameweek_groups

VERSIONS = {
    'v3': {'plan': 'simple_h1_b1', 'features': 2, 'ids': False, 'seeds': 3,
           'objective': 'regression', 'seed': 42},
    'v6': {'plan': 'simple_h1_b1', 'features': 4, 'ids': False, 'seeds': 3,
           'objective': 'regression', 'seed': 42},
}
DEFAULT_VERSION = 'v6'
VERSION_FLAG = '--lgbm-version'
# Kept for callers of the old name.
V3 = VERSIONS['v3']


def live_version(argv=None):
    """The LightGBM version the args ask for (`--lgbm-version v3`), else the default."""
    argv = list(sys.argv if argv is None else argv)
    if VERSION_FLAG in argv[:-1]:
        version = argv[argv.index(VERSION_FLAG) + 1]
        if version not in VERSIONS:
            raise SystemExit('%s must be one of %s' % (VERSION_FLAG, ', '.join(sorted(VERSIONS))))
        return version
    return DEFAULT_VERSION


def learn_lightgbm(runner, settings=None, version=None):
    from simulation.season_model_lgbm import flatten_with_seasons

    version = version or live_version()
    settings = dict(VERSIONS[version], **(settings or {}))
    plan = load_plan(settings['plan'])
    seasons = list(getattr(runner, 'past_seasons', []) or [])
    current = runner.current_season
    seasons.append(current)
    cutoff = runner._trained_through_gameweek() if hasattr(runner, '_trained_through_gameweek') else None
    print('LightGBM %s live: plan %s, feature set %s, team ids %s, %s seeds' % (
        version, plan.name, settings['features'], 'kept' if settings['ids'] else 'left out', settings['seeds']))
    train_batch, predict_batch, _vocab = build_dataset(
        seasons, plan, predict_season=current, cutoff_gameweek=cutoff,
    )
    print('  %s train rows, %s predict rows' % (len(train_batch.matches), len(predict_batch.matches)))
    if not train_batch.matches:
        raise SystemExit('LightGBM found no training fixtures')
    table = flatten_with_seasons(train_batch, plan, seasons, features=settings['features'], ids=settings['ids'])
    priors = table.priors
    labels = train_batch.labels.reshape(-1)
    years = list(train_batch.years)
    groups = gameweek_groups(train_batch)
    del train_batch
    fitted = fit(table, labels, years, objective=settings['objective'], seed=settings['seed'],
                 seeds=settings['seeds'], groups=groups)
    del table
    upcoming = upcoming_gameweeks(current)
    predictions = []
    if predict_batch.matches:
        predictions = fitted.predict(
            flatten_with_seasons(predict_batch, plan, seasons, features=settings['features'], ids=settings['ids'],
                                 priors=priors, as_of_gameweek=upcoming[0]),
            groups=gameweek_groups(predict_batch),
        )
        store_points(predict_batch.matches, predictions)
    print(format_opening_predictions(
        predict_batch.matches,
        gameweeks=upcoming,
        limit=PREDICTION_TABLE_SIZE,
        title='Top %s predicted points over the next %s gameweeks (GW%s-%s), LightGBM %s'
        % (PREDICTION_TABLE_SIZE, len(upcoming), upcoming[0], upcoming[-1], version),
    ))
    return predictions
