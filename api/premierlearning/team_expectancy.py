"""Team goals expectancy model: predict goals scored from pre-match form."""
import json
import sys
from collections import defaultdict

import numpy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import MeanAbsoluteError as MAE, RootMeanSquaredError
from tensorflow.keras.models import Sequential
from premierlearning.learning import build_adamw, use_linear_model

TEAM_FEATURE_KEYS = [
    'team_scored_last_5',
    'team_scored_avg',
    'team_conceded_last_5',
    'team_conceded_avg',
    'opp_scored_last_5',
    'opp_scored_avg',
    'opp_conceded_last_5',
    'opp_conceded_avg',
    'is_home',
    'fixture_difficulty',
    'opp_fixture_difficulty',
]

TEAM_EPOCHS = 2 if '--quick' in sys.argv else 500
TEAM_BATCH_SIZE = 32


def feature_row(team_match, opp_match, team=None, opp_team=None):
    return {
        'team_scored_last_5': float(team_match.scored_last_5_avg),
        'team_scored_avg': float(team_match.scored_season_avg),
        'team_conceded_last_5': float(team_match.conceded_last_5_avg),
        'team_conceded_avg': float(team_match.conceded_season_avg),
        'opp_scored_last_5': float(opp_match.scored_last_5_avg),
        'opp_scored_avg': float(opp_match.scored_season_avg),
        'opp_conceded_last_5': float(opp_match.conceded_last_5_avg),
        'opp_conceded_avg': float(opp_match.conceded_season_avg),
        'is_home': float(1 if team_match.home else 0),
        'fixture_difficulty': float(team_match.fixture_difficulty),
        'opp_fixture_difficulty': float(team_match.opp_fixture_difficulty),
    }


def _row_vector(row_dict):
    return [row_dict[k] for k in TEAM_FEATURE_KEYS]


def _season_key(season):
    return season.years


def index_fixture_sides(seasons):
    """(season_years, fixture_id) -> {'home': (team, match), 'away': (team, match)}

    Fixture IDs restart each season, so the key must include the season.
    """
    by_fixture = defaultdict(dict)
    for season in seasons:
        sk = _season_key(season)
        for team in season.teams.values():
            for match in list(team.team_match_dict.values()) + list(team.future_team_match_dict.values()):
                side = 'home' if match.home else 'away'
                by_fixture[(sk, int(match.id))][side] = (team, match)
    return by_fixture


def _apply_form_to_future_matches(team, scored, conceded, last5_s, last5_c):
    team.average_scored_so_far = scored
    team.average_conceded_so_far = conceded
    team._final_scored_last_5 = last5_s
    team._final_conceded_last_5 = last5_c
    for match in team.future_team_match_dict.values():
        match.scored_season_avg = scored
        match.conceded_season_avg = conceded
        match.scored_last_5_avg = last5_s
        match.conceded_last_5_avg = last5_c
        match.average_scored_in_previous_matches = scored
        match.average_conceded_in_previous_matches = conceded


def _bottom_team_form_priors(last_season, n: int = 3):
    """Use last season's weakest n sides as a proxy for newly promoted teams."""
    teams = [t for t in last_season.teams.values() if t.team_match_dict]
    if not teams:
        return 0.9, 1.85, 0.9, 1.85
    ranked = sorted(teams, key=lambda t: (t.average_points_so_far, -t.average_conceded_so_far))
    bottom = ranked[: min(n, len(ranked))]
    scored = sum(t.average_scored_so_far for t in bottom) / len(bottom)
    conceded = sum(t.average_conceded_so_far for t in bottom) / len(bottom)
    last5_s = sum(getattr(t, '_final_scored_last_5', t.average_scored_so_far) for t in bottom) / len(bottom)
    last5_c = sum(getattr(t, '_final_conceded_last_5', t.average_conceded_so_far) for t in bottom) / len(bottom)
    return float(scored), float(conceded), float(last5_s), float(last5_c)


def seed_form_from_last_season(current_season, past_seasons):
    """When a team has no finished matches yet, copy last season's scoring rates.

    Newly promoted sides (no PL history last year) get weak priors from last
    season's bottom-3 teams — not zeros (zeros look like 'unknown midtable').
    """
    if not past_seasons:
        return
    last_season = None
    target_year = current_season.years[0] - 1
    for season in past_seasons:
        if season.years[0] == target_year:
            last_season = season
            break
    if last_season is None:
        last_season = past_seasons[0]

    last_by_name = {t.name: t for t in last_season.teams.values()}
    promo_scored, promo_conceded, promo_l5_s, promo_l5_c = _bottom_team_form_priors(last_season)
    seeded = 0
    promoted = []
    for team in current_season.teams.values():
        if team.team_match_dict:
            continue
        prev = last_by_name.get(team.name)
        if prev is not None:
            scored = float(prev.average_scored_so_far or 0.0)
            conceded = float(prev.average_conceded_so_far or 0.0)
            last5_s = float(getattr(prev, '_final_scored_last_5', scored) or scored)
            last5_c = float(getattr(prev, '_final_conceded_last_5', conceded) or conceded)
            _apply_form_to_future_matches(team, scored, conceded, last5_s, last5_c)
            seeded += 1
        else:
            _apply_form_to_future_matches(
                team, promo_scored, promo_conceded, promo_l5_s, promo_l5_c
            )
            promoted.append(team.name)
            seeded += 1
    if seeded:
        print(f'Seeded early-season form for {seeded} teams from season {last_season.years}')
    if promoted:
        print(
            f'Promoted/new team prior (bottom-3 PL rates '
            f'scored={promo_scored:.2f} conceded={promo_conceded:.2f}): {", ".join(promoted)}'
        )


def build_training_rows(seasons):
    by_fixture = index_fixture_sides(seasons)
    X, y = [], []
    for sides in by_fixture.values():
        if 'home' not in sides or 'away' not in sides:
            continue
        home_team, home_match = sides['home']
        away_team, away_match = sides['away']
        # Only finished fixtures (have real scores) for labels
        if int(home_match.id) not in home_team.team_match_dict:
            continue
        X.append(_row_vector(feature_row(home_match, away_match, home_team, away_team)))
        y.append(float(home_match.scored))
        X.append(_row_vector(feature_row(away_match, home_match, away_team, home_team)))
        y.append(float(away_match.scored))
    return numpy.array(X, dtype=float), numpy.array(y, dtype=float).reshape(-1, 1)


def build_predict_targets(seasons):
    """List of (team, team_match, opp_team, opp_match) for every fixture side."""
    by_fixture = index_fixture_sides(seasons)
    targets = []
    fixture_pairs = []
    for sides in by_fixture.values():
        if 'home' not in sides or 'away' not in sides:
            continue
        home_team, home_match = sides['home']
        away_team, away_match = sides['away']
        fixture_pairs.append((home_team, home_match, away_team, away_match))
        targets.append((home_team, home_match, away_team, away_match))
        targets.append((away_team, away_match, home_team, home_match))
    return targets, fixture_pairs


def train_team_goals_model_linear(X, y):
    print('Training team goals expectancy model (linear Ridge)...')
    print(f'Training rows: {len(X)}')
    print('Team model input columns:')
    for col in TEAM_FEATURE_KEYS:
        print(f'  {col}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    # No PCA — only a handful of team features; PCA(0.95) can collapse too aggressively.
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", RidgeCV(alphas=numpy.logspace(-3, 3, 13), store_cv_values=False)),
    ])
    pipe.fit(X_train, y_train.ravel())
    y_pred = pipe.predict(X_test)
    print({
        'team_model_test_mae': float(mean_absolute_error(y_test, y_pred)),
        'team_model_test_rmse': float(mean_squared_error(y_test, y_pred, squared=False)),
        'pred_range': (float(y_pred.min()), float(y_pred.max())),
    })
    return {'pipe': pipe, 'linear': True}


def train_team_goals_model(X, y):
    if use_linear_model():
        return train_team_goals_model_linear(X, y)

    print('Training team goals expectancy model (neural net)...')
    print(f'Training rows: {len(X)}; max epochs: {TEAM_EPOCHS}')
    print('Team model input columns:')
    for col in TEAM_FEATURE_KEYS:
        print(f'  {col}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    net_size = max(8, int(X_train_s.shape[1] * 3 / 2))
    reg = tf.keras.regularizers.l2(3e-5)
    model = Sequential([
        Dense(net_size, kernel_regularizer=reg, input_shape=(X_train_s.shape[1],), activation='relu'),
        Dense(max(4, net_size // 2), kernel_regularizer=reg, activation='relu'),
        Dense(max(2, net_size // 4), kernel_regularizer=reg, activation='relu'),
        Dense(1),
    ])
    model.compile(
        loss=Huber(delta=1.0),
        optimizer=build_adamw(1e-3, 1e-5),
        metrics=[RootMeanSquaredError(), MAE()],
    )
    callbacks = [
        EarlyStopping(monitor='val_mean_absolute_error', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_root_mean_squared_error', factor=0.5, patience=5),
    ]
    history = model.fit(
        X_train_s, y_train,
        epochs=TEAM_EPOCHS,
        batch_size=TEAM_BATCH_SIZE,
        validation_split=0.2,
        verbose=2,
        callbacks=callbacks,
        shuffle=True,
    )
    y_pred = model.predict(X_test_s, verbose=0).ravel()
    print({
        'team_model_train_mae': float(history.history['mean_absolute_error'][-1]),
        'team_model_test_mae': float(mean_absolute_error(y_test, y_pred)),
        'team_model_test_rmse': float(mean_squared_error(y_test, y_pred, squared=False)),
        'epochs_ran': len(history.history['loss']),
        'pred_range': (float(y_pred.min()), float(y_pred.max())),
    })
    return {'model': model, 'scaler': scaler, 'linear': False}


def apply_predictions(trained, seasons):
    targets, fixture_pairs = build_predict_targets(seasons)
    if not targets:
        print('No fixtures to predict for team expectancies')
        return

    X = numpy.array([
        _row_vector(feature_row(tm, opp, team, opp_team))
        for team, tm, opp_team, opp in targets
    ], dtype=float)
    if trained.get('linear'):
        preds = trained['pipe'].predict(X).ravel()
    else:
        preds = trained['model'].predict(trained['scaler'].transform(X), verbose=0).ravel()
    preds = numpy.maximum(preds, 0.05)

    for i, (home_team, home_match, away_team, away_match) in enumerate(fixture_pairs):
        home_xg = float(preds[i * 2])
        away_xg = float(preds[i * 2 + 1])
        home_match.predicted_xg = home_xg
        home_match.predicted_xc = away_xg
        away_match.predicted_xg = away_xg
        away_match.predicted_xc = home_xg


def fit_team_expectancy_once(past_seasons, current_season):
    seasons = list(past_seasons) + [current_season]
    seed_form_from_last_season(current_season, past_seasons)
    X, y = build_training_rows(seasons)
    if len(X) < 50:
        print(f'Skipping team expectancies (only {len(X)} training rows)')
        return None
    return train_team_goals_model(X, y)


def apply_team_expectancy_model(model, past_seasons, current_season):
    if model is None:
        return False
    seasons = list(past_seasons) + [current_season]
    seed_form_from_last_season(current_season, past_seasons)
    apply_predictions(model, seasons)
    print('Team expectancies applied to all fixtures')
    return True


def run_team_expectancies(past_seasons, current_season):
    model = fit_team_expectancy_once(past_seasons, current_season)
    return apply_team_expectancy_model(model, past_seasons, current_season)


SCORELINES_PATH = 'predicted_scorelines.json'


def collect_future_scorelines(current_season):
    """Unfinished fixtures with predicted home/away xG, sorted by gameweek."""
    by_fixture = index_fixture_sides([current_season])
    rows = []
    for sides in by_fixture.values():
        if 'home' not in sides or 'away' not in sides:
            continue
        home_team, home_match = sides['home']
        away_team, away_match = sides['away']
        if int(home_match.id) in home_team.team_match_dict:
            continue
        rows.append({
            'gameweek': int(home_match.event or 0),
            'fixture_id': int(home_match.id),
            'home': home_team.name,
            'away': away_team.name,
            'home_xg': float(getattr(home_match, 'predicted_xg', 0) or 0),
            'away_xg': float(getattr(away_match, 'predicted_xg', 0) or 0),
        })
    rows.sort(key=lambda row: (row['gameweek'], row['fixture_id']))
    return rows


def write_future_scorelines(current_season, path=SCORELINES_PATH):
    rows = collect_future_scorelines(current_season)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(rows, file)
    print('Wrote %d predicted scorelines to %s' % (len(rows), path))
    return rows


def print_future_scorelines(current_season):
    """Print predicted scorelines for unfinished fixtures, grouped by gameweek."""
    rows = collect_future_scorelines(current_season)
    print('\nPredicted team scorelines:')
    gameweek = None
    for row in rows:
        if row['gameweek'] != gameweek:
            gameweek = row['gameweek']
            print('\nGameweek %s' % gameweek)
        print(
            '%s %.2f - %.2f %s'
            % (row['home'], row['home_xg'], row['away_xg'], row['away'])
        )
