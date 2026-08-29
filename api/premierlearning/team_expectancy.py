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
from premierlearning.squad_cost import (
    nudge_rate,
    squad_cost_early_weight,
    squad_cost_z_by_code,
    use_squad_cost,
)

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


def _remember_observed_form(match):
    if getattr(match, '_observed_form_saved', False):
        return
    match._observed_scored_season_avg = float(match.scored_season_avg or 0.0)
    match._observed_conceded_season_avg = float(match.conceded_season_avg or 0.0)
    match._observed_scored_last_5 = float(match.scored_last_5_avg or 0.0)
    match._observed_conceded_last_5 = float(match.conceded_last_5_avg or 0.0)
    match._observed_form_saved = True


def _prior_team_for(team, prev_season):
    prev_by_code = {
        t.code: t for t in prev_season.teams.values() if getattr(t, 'code', None) is not None
    }
    prev_by_name = {t.name: t for t in prev_season.teams.values()}
    code = getattr(team, 'code', None)
    if code is not None and code in prev_by_code:
        return prev_by_code[code]
    return prev_by_name.get(team.name)


def _apply_squad_cost_form(team, prior_rates, z, finished_events):
    """Idempotent: blend/nudge match form from saved observed windows.

    No prior results as of this fixture → last-season rates, with the cost
    nudge faded by GW. After results exist, shrink those rates toward observed
    form; both the last-season mix and the cost nudge are gone by GW5.
    """
    prior_s, prior_c, prior_l5s, prior_l5c = prior_rates
    matches = list(team.team_match_dict.values()) + list(team.future_team_match_dict.values())
    for match in matches:
        _remember_observed_form(match)
        gw = match.event
        weight = squad_cost_early_weight(gw)
        nudged_s = nudge_rate(prior_s, z, weight, sign=1.0)
        nudged_c = nudge_rate(prior_c, z, weight, sign=-1.0)
        nudged_l5s = nudge_rate(prior_l5s, z, weight, sign=1.0)
        nudged_l5c = nudge_rate(prior_l5c, z, weight, sign=-1.0)
        has_obs = any(event < int(gw or 0) for event in finished_events)
        if has_obs:
            match.scored_season_avg = (
                (1.0 - weight) * match._observed_scored_season_avg + weight * nudged_s
            )
            match.conceded_season_avg = (
                (1.0 - weight) * match._observed_conceded_season_avg + weight * nudged_c
            )
            match.scored_last_5_avg = (
                (1.0 - weight) * match._observed_scored_last_5 + weight * nudged_l5s
            )
            match.conceded_last_5_avg = (
                (1.0 - weight) * match._observed_conceded_last_5 + weight * nudged_l5c
            )
        else:
            match.scored_season_avg = nudged_s
            match.conceded_season_avg = nudged_c
            match.scored_last_5_avg = nudged_l5s
            match.conceded_last_5_avg = nudged_l5c
        match.average_scored_in_previous_matches = match.scored_season_avg
        match.average_conceded_in_previous_matches = match.conceded_season_avg

    if not team.team_match_dict:
        next_gw = min(
            (m.event for m in team.future_team_match_dict.values() if m.event is not None),
            default=1,
        )
        weight = squad_cost_early_weight(next_gw)
        team.average_scored_so_far = nudge_rate(prior_s, z, weight, sign=1.0)
        team.average_conceded_so_far = nudge_rate(prior_c, z, weight, sign=-1.0)
        team._final_scored_last_5 = nudge_rate(prior_l5s, z, weight, sign=1.0)
        team._final_conceded_last_5 = nudge_rate(prior_l5c, z, weight, sign=-1.0)


def _seed_form_without_squad_cost(current_season, past_seasons):
    """Pre-squad-cost seed: last-season rates on current-season future matches only."""
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
        prev = _prior_team_for(team, last_season) or last_by_name.get(team.name)
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
    print('Squad cost YoY prior off (--no-squad-cost / PL_SQUAD_COST=0)')
    if seeded:
        print(
            'Seeded early-season form for %d teams from season %s'
            % (seeded, last_season.years)
        )
    if promoted:
        print(
            'Promoted/new team prior (bottom-3 PL rates '
            'scored=%.2f conceded=%.2f): %s'
            % (promo_scored, promo_conceded, ', '.join(promoted))
        )


def seed_form_from_last_season(current_season, past_seasons):
    """Last-season scoring rates as an early-GW prior, nudged by squad-cost YoY.

    Newly promoted sides (no PL history last year) get weak priors from last
    season's bottom-3 teams — not zeros (zeros look like 'unknown midtable').
    Top-15 FPL cost change vs last season's end is a centered z; it scales the
    prior up/down and fades to zero by GW5. Applied to every season that has a
    predecessor so train rows match serve. PL_SQUAD_COST=0 restores the old
    current-season-only seed with no cost nudge.
    """
    if not use_squad_cost():
        _seed_form_without_squad_cost(current_season, past_seasons)
        return
    seasons = [s for s in list(past_seasons or []) + [current_season] if s is not None]
    if len(seasons) < 2:
        return
    by_year = {int(season.years[0]): season for season in seasons}
    for season in sorted(seasons, key=lambda s: s.years[0]):
        prev = by_year.get(int(season.years[0]) - 1)
        if prev is None:
            continue
        z_by_code = squad_cost_z_by_code(
            int(prev.years[0]),
            int(season.years[0]),
            prev_season=prev,
            curr_season=season,
        )
        promo = _bottom_team_form_priors(prev)
        promoted = []
        seeded = 0
        for team in season.teams.values():
            prev_team = _prior_team_for(team, prev)
            if prev_team is not None:
                scored = float(prev_team.average_scored_so_far or 0.0)
                conceded = float(prev_team.average_conceded_so_far or 0.0)
                last5_s = float(getattr(prev_team, '_final_scored_last_5', scored) or scored)
                last5_c = float(getattr(prev_team, '_final_conceded_last_5', conceded) or conceded)
            else:
                scored, conceded, last5_s, last5_c = promo
                promoted.append(team.name)
            code = getattr(team, 'code', None)
            z = float(z_by_code.get(code, 0.0) or 0.0) if code is not None else 0.0
            team.squad_cost_yoy_z = z
            finished_events = [
                int(match.event)
                for match in team.team_match_dict.values()
                if match.event is not None
            ]
            _apply_squad_cost_form(
                team,
                (scored, conceded, last5_s, last5_c),
                z,
                finished_events,
            )
            seeded += 1
        if season is current_season:
            if seeded:
                print(
                    'Seeded early-season form for %d teams from season %s'
                    % (seeded, prev.years)
                )
            if promoted:
                print(
                    'Promoted/new team prior (bottom-3 PL rates '
                    'scored=%.2f conceded=%.2f): %s'
                    % (promo[0], promo[1], ', '.join(promoted))
                )
            if z_by_code:
                labeled = []
                for team in season.teams.values():
                    z = float(getattr(team, 'squad_cost_yoy_z', 0.0) or 0.0)
                    if abs(z) < 0.05:
                        continue
                    labeled.append((z, team.name))
                labeled.sort(reverse=True)
                if labeled:
                    print(
                        'Squad cost YoY (top 15, centered z, fade-out GW5): '
                        + ', '.join('%s %+0.2f' % (name, z) for z, name in labeled)
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
