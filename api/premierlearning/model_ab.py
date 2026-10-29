"""Compare the horizon model against the single-gameweek model on a forward holdout.

Why not val MAE. Each arm below trains on a different target or a different row
population: pooled window pp90 against single-fixture pp90, appearance-stripped
labels, poisson means against log-space medians, overlapping windows against
non-overlapping ones. Their val MAEs are not on one scale, and worse, the
de-leaked arm's val MAE becomes honest while the leaky arm's stays inflated, so
ranking arms by val MAE actively rewards the leak. Every arm here is instead
scored on the same held-out fixtures, by the same measures.

The split is by time, not at random. Training uses seasons before the holdout
season plus that season's gameweeks up to --split-gw; scoring uses windows
anchored after it. A horizon training window must complete by the split, so no
training label overlaps a scored window. Holdout features legitimately use form
up to the fixture being scored, which is what happens live.

Default scoring (--pack full) uses realised minutes on both sides, so the
minutes model never enters. --pack live fits minutes once, then every arm's
pp90 is turned into expected points the way squadtools does (pp90 x minutes
/ 90). Ranking and calibration are then on expected vs actual points.

With --workers the fits are split over child processes, one group of fits per
child, and each child scores its own arms so only small result files come back.
Reps of a fit stay on one worker, so its training rows are built once and reused.
The minutes net is fit first in its own child and shared. Children die with the
parent, and their thread pools are sized to share the machine rather than each
claiming every core.

Run from api/:
  python -m premierlearning.model_ab --reps 5
  python -m premierlearning.model_ab --reps 5 --workers 4
  python -m premierlearning.model_ab --pack live --holdout-seasons 25,24 --reps 5 --workers 4
  python -m premierlearning.model_ab --reps 1 --only 01_h1_base     (smoke test)
  python -m premierlearning.model_ab --dry-run --workers 4
  python -m premierlearning.model_ab --split-gw 20 --holdout-season 25

Writes api/run_metrics/model_ab/<run_id>/ (comparison.csv, comparison.txt,
arms.json, per_fit_stats.json, plus worker_*.log when run with workers).
"""
import argparse
import atexit
import copy
import csv
import json
import os
import statistics
import sys
import time
from collections import defaultdict, namedtuple
from datetime import datetime
from pathlib import Path

import numpy

# Keep MLData from dumping X Train.csv on every one of the 50 fits.
os.environ.setdefault('PL_SWEEP_TRIAL', 'model_ab')

from premierlearning.clean_match import (  # noqa: E402
    TEAM_XC_NEXT_KEY,
    TEAM_XG_NEXT_KEY,
    apply_feature_interactions,
    get_input_keys,
)
from premierlearning.config_constants import POINTS_HORIZON  # noqa: E402
from premierlearning.learning import (  # noqa: E402
    MLData,
    SMEAR_GLOBAL,
    SMEAR_OFF,
    SMEAR_RELATIVE,
    apply_smearing,
    uses_log_targets,
)
from premierlearning import worker_jobs  # noqa: E402


API_DIR = Path(__file__).resolve().parent.parent
RESULTS_ROOT = API_DIR / 'run_metrics' / 'model_ab'

DEFAULT_SPLIT_GW = 20
DEFAULT_REPS = 5
# Below this the pooled label is too thin to mean much; matches the live builder.
MIN_PLAYED_IN_WINDOW = 3
TOP_N = 20
# Ranking is only meaningful among players who actually played a full-ish game;
# below this a single goal produces a pp90 no starter can reach.
STARTER_MINUTES = 60.0
SLICE_N = 10
SLICE_MIN_ROWS = 20
# Live pack: the arms that answer "is Poisson special, and does it still win
# once expected points go through the minutes model?"
LIVE_ARM_NAMES = (
    '01_h1_base',
    '04_h1_mw',
    '07_h1_poisson_mw',
    '10_hz_base',
    '16_hz_poisson_mw',
    '20_h1_huber_raw_mw',
    '21_h1_mse_raw_mw',
    '22_h1_tweedie_mw',
    '23_h1_hurdle',
)


FitSpec = namedtuple(
    'FitSpec', 'key model stride loss minutes_weight strip_appearance subsample note'
)
ArmSpec = namedtuple('ArmSpec', 'name fit smear note')


def _fit(key, model, note, stride=None, loss='huber', minutes_weight=False,
         strip_appearance=False, subsample=None):
    return FitSpec(
        key, model, stride, loss, minutes_weight, strip_appearance, subsample, note
    )


# Fifteen fits. Smearing is a transform on a fitted model's output, not a different
# fit, so the smeared arms below ride along for free.
FITS = (
    _fit('h1_base', 'h1', 'Single gameweek, exactly as it runs now.'),
    _fit('h1_mw', 'h1', 'Single gameweek, rows weighted by minutes played.',
         minutes_weight=True),
    _fit('h1_poisson', 'h1', 'Single gameweek, poisson loss on the raw rate.',
         loss='poisson'),
    _fit('h1_poisson_mw', 'h1',
         'Single gameweek, poisson with exposure weights (the textbook form).',
         loss='poisson', minutes_weight=True),
    _fit('h1_appearance', 'h1',
         'Single gameweek trained without appearance points, re-added to score.',
         strip_appearance=True),
    _fit('h1_huber_raw_mw', 'h1',
         'Single gameweek, Huber on raw pp90 with exposure weights (log off).',
         loss='huber_raw', minutes_weight=True),
    _fit('h1_mse_raw_mw', 'h1',
         'Single gameweek, MSE on raw pp90 with exposure weights (log off).',
         loss='mse', minutes_weight=True),
    _fit('h1_tweedie_mw', 'h1',
         'Single gameweek, Tweedie p=1.5 with exposure weights.',
         loss='tweedie', minutes_weight=True),
    _fit('h1_hurdle', 'h1',
         'Hurdle: P(played) times E[pp90|played] (Poisson+mw on the positive part).',
         loss='hurdle', minutes_weight=True),
    _fit('hz_base', 'horizon', 'Horizon as it runs now: every anchor, so windows overlap.',
         stride=1),
    _fit('hz_mw', 'horizon', 'Horizon, windows weighted by minutes played.',
         stride=1, minutes_weight=True),
    _fit('hz_poisson', 'horizon', 'Horizon, poisson loss on the raw rate.',
         stride=1, loss='poisson'),
    _fit('hz_poisson_mw', 'horizon', 'Horizon, poisson with exposure weights.',
         stride=1, loss='poisson', minutes_weight=True),
    _fit('hz_deleak', 'horizon',
         'Horizon with non-overlapping windows (1-5, 6-10, ...): no shared labels.',
         stride=POINTS_HORIZON),
    # Striding to de-leak also cuts the rows by a factor of the stride, so on its
    # own hz_deleak cannot tell leakage apart from having less data. This arm keeps
    # the overlap but matches the row count, making that difference measurable.
    _fit('hz_subsample', 'horizon',
         'Horizon, overlapping windows subsampled to the de-leaked row count.',
         stride=1, subsample=POINTS_HORIZON),
)

FIT_BY_KEY = {fit.key: fit for fit in FITS}


ARMS = (
    ArmSpec('01_h1_base', 'h1_base', SMEAR_OFF, 'Baseline single-gameweek model.'),
    ArmSpec('02_h1_smear', 'h1_base', SMEAR_GLOBAL,
            'Baseline plus a global smearing correction: level moves, ranks do not.'),
    ArmSpec('03_h1_relsmear', 'h1_base', SMEAR_RELATIVE,
            'Baseline plus relative smearing only: ranks move, level holds.'),
    ArmSpec('04_h1_mw', 'h1_mw', SMEAR_OFF, 'Minutes weighting alone.'),
    ArmSpec('05_h1_mw_smear', 'h1_mw', SMEAR_GLOBAL, 'Minutes weighting plus smearing.'),
    ArmSpec('06_h1_poisson', 'h1_poisson', SMEAR_OFF, 'Poisson, no weighting.'),
    ArmSpec('07_h1_poisson_mw', 'h1_poisson_mw', SMEAR_OFF, 'Poisson with exposure weights.'),
    ArmSpec('08_h1_appearance', 'h1_appearance', SMEAR_OFF,
            'Appearance points stripped for training, re-added for scoring.'),
    ArmSpec('09_h1_appearance_smear', 'h1_appearance', SMEAR_GLOBAL,
            'Appearance-stripped plus smearing.'),
    ArmSpec('10_hz_base', 'hz_base', SMEAR_OFF, 'Baseline horizon model (with the leak).'),
    ArmSpec('11_hz_smear', 'hz_base', SMEAR_GLOBAL, 'Horizon plus global smearing.'),
    ArmSpec('12_hz_relsmear', 'hz_base', SMEAR_RELATIVE, 'Horizon plus relative smearing.'),
    ArmSpec('13_hz_mw', 'hz_mw', SMEAR_OFF, 'Horizon with minutes weighting.'),
    ArmSpec('14_hz_mw_smear', 'hz_mw', SMEAR_GLOBAL, 'Horizon, minutes weighting plus smearing.'),
    ArmSpec('15_hz_poisson', 'hz_poisson', SMEAR_OFF, 'Horizon, poisson.'),
    ArmSpec('16_hz_poisson_mw', 'hz_poisson_mw', SMEAR_OFF, 'Horizon, poisson with exposure.'),
    ArmSpec('17_hz_deleak', 'hz_deleak', SMEAR_OFF, 'Horizon, non-overlapping windows.'),
    ArmSpec('18_hz_deleak_smear', 'hz_deleak', SMEAR_GLOBAL,
            'Horizon, non-overlapping windows plus smearing.'),
    ArmSpec('19_hz_subsample', 'hz_subsample', SMEAR_OFF,
            'Row-count control for 17: same rows, overlap kept. 17 vs 19 is the leak.'),
    ArmSpec('20_h1_huber_raw_mw', 'h1_huber_raw_mw', SMEAR_OFF,
            'Huber on raw pp90 plus minutes weights: was log1p the bug, or is Poisson special?'),
    ArmSpec('21_h1_mse_raw_mw', 'h1_mse_raw_mw', SMEAR_OFF,
            'MSE on raw pp90 plus minutes weights.'),
    ArmSpec('22_h1_tweedie_mw', 'h1_tweedie_mw', SMEAR_OFF,
            'Tweedie p=1.5 with exposure weights (zero-inflated hauls).'),
    ArmSpec('23_h1_hurdle', 'h1_hurdle', SMEAR_OFF,
            'P(played) x E[pp90|played], then the shared minutes model like every other arm.'),
)

ORIGINAL_ARM_NAMES = tuple(arm.name for arm in ARMS if int(arm.name.split('_', 1)[0]) < 20)


def match_minutes(match) -> float:
    stats = getattr(match, 'game_stats', None)
    if stats:
        return float(stats.get('minutes', 0) or 0)
    return float(getattr(match, 'minutes', 0) or 0)


def match_points(match) -> float:
    stats = getattr(match, 'game_stats', None)
    if stats and stats.get('total_points') is not None:
        return float(stats.get('total_points') or 0)
    return float(getattr(match, 'points', 0) or 0)


def appearance_points(minutes) -> float:
    """FPL's participation points, which every player collects for turning up."""
    if minutes <= 0:
        return 0.0
    return 2.0 if minutes >= 60 else 1.0


def fixture_pp90(match, strip_appearance=False) -> float:
    minutes = match_minutes(match)
    if minutes <= 0:
        return 0.0
    points = match_points(match)
    if strip_appearance:
        points -= appearance_points(minutes)
    return points * 90.0 / minutes


def pooled_pp90(window, strip_appearance=False) -> float:
    points = 0.0
    minutes = 0.0
    for match in window:
        match_mins = match_minutes(match)
        if match_mins <= 0:
            continue
        value = match_points(match)
        if strip_appearance:
            value -= appearance_points(match_mins)
        points += value
        minutes += match_mins
    if minutes <= 0:
        return 0.0
    return points * 90.0 / minutes


def _season_year(season) -> int:
    years = getattr(season, 'years', None)
    if isinstance(years, (tuple, list)) and years:
        return int(years[0])
    return int(years or 0)


def _sum_team_xg_xc(window):
    xg = sum(float(getattr(m.team_match, 'predicted_xg', 0.0) or 0.0) for m in window)
    xc = sum(float(getattr(m.team_match, 'predicted_xc', 0.0) or 0.0) for m in window)
    return xg, xc


def _horizon_row(match, window):
    """A window row: the anchor's features with team xG/xC summed over the window."""
    clean = match.to_clean_match()
    xg_sum, xc_sum = _sum_team_xg_xc(window)
    clean[TEAM_XG_NEXT_KEY] = xg_sum
    clean[TEAM_XC_NEXT_KEY] = xc_sum
    apply_feature_interactions(clean)
    return clean


def _season_matches(player):
    return [
        match for match in player.matches
        if not getattr(match, 'is_last_season_carry', False)
        and getattr(match, 'gameweek', None) is not None
    ]


def load_seasons(update_data=False):
    """Load players, teams and team expectancies without running the learn path."""
    from premierlearning.runner import Runner
    from premierlearning.season import CurrentSeason
    from premierlearning.team_expectancy import run_team_expectancies

    runner = Runner(update_data=update_data)
    if runner.current_season is None:
        runner.current_season = CurrentSeason(runner.target_season_year)
    runner.past_seasons = []
    runner.populate_past_seasons()
    runner.validate_temporal_cutoff()
    print('Running team expectancies (needed for the team xG features)...')
    run_team_expectancies(runner.past_seasons, runner.current_season)
    runner.process_player_matches()
    return runner


def pick_holdout_season(runner, requested=None):
    """The most recent season that actually finished, so windows have outcomes."""
    complete = [
        season for season in runner.past_seasons
        if int(getattr(season, 'training_complete_through_gw', 0) or 0) >= 30
    ]
    if not complete:
        raise SystemExit('No complete past season to hold out.')
    if requested is not None:
        for season in complete:
            if _season_year(season) == int(requested):
                return season
        raise SystemExit(
            'Season %s is not a complete past season. Available: %s'
            % (requested, ', '.join(str(_season_year(s)) for s in complete))
        )
    return max(complete, key=_season_year)


def build_h1_train(runner, holdout_season, split_gw, strip_appearance=False):
    """One row per completed fixture, from before the holdout point in time."""
    holdout_year = _season_year(holdout_season)
    rows = []
    labels = []
    weights = []
    seasons = list(runner.past_seasons) + [runner.current_season]
    for season in seasons:
        year = _season_year(season)
        if year > holdout_year:
            continue
        limit = split_gw if year == holdout_year else 38
        for player in season.players:
            for match in _season_matches(player):
                if not match.valid_for_learning:
                    continue
                if int(match.gameweek) > limit:
                    continue
                minutes = match_minutes(match)
                rows.append(match.to_clean_match())
                labels.append(fixture_pp90(match, strip_appearance))
                weights.append(minutes / 90.0)
    for i, row in enumerate(rows):
        row['id'] = i
    return rows, labels, weights


def build_horizon_train(runner, holdout_season, split_gw, horizon, stride=1):
    """One row per player-window. stride>1 stops windows sharing fixtures."""
    holdout_year = _season_year(holdout_season)
    rows = []
    labels = []
    weights = []
    skipped = 0
    seasons = list(runner.past_seasons) + [runner.current_season]
    for season in seasons:
        year = _season_year(season)
        if year > holdout_year:
            continue
        season_through = int(getattr(season, 'training_complete_through_gw', 38) or 0)
        # The window must finish before the holdout begins, or a training label
        # would contain a fixture we are about to score on.
        complete_through = min(season_through, split_gw if year == holdout_year else 38)
        for player in season.players:
            matches = _season_matches(player)
            seen = set()
            for match in matches:
                if not match.valid_for_learning:
                    continue
                anchor = int(match.gameweek)
                if anchor in seen:
                    continue
                seen.add(anchor)
                if (anchor - 1) % stride != 0:
                    continue
                if anchor + horizon - 1 > complete_through:
                    continue
                window = [
                    candidate for candidate in matches
                    if anchor <= int(candidate.gameweek) < anchor + horizon
                ]
                played = [m for m in window if match_minutes(m) > 0]
                if len(played) < MIN_PLAYED_IN_WINDOW:
                    skipped += 1
                    continue
                rows.append(_horizon_row(match, window))
                labels.append(pooled_pp90(window))
                weights.append(
                    sum(match_minutes(m) for m in played) / (90.0 * horizon)
                )
    for i, row in enumerate(rows):
        row['id'] = i
    print(
        'Horizon train rows (stride %s): %s (skipped %s thin windows)'
        % (stride, len(rows), skipped)
    )
    return rows, labels, weights


def build_holdout(holdout_season, split_gw, horizon, include_dnps=False):
    """Windows anchored after the split, each carrying its realised outcomes.

    Played-only is the original yardstick (pp90 among those who turned up).
    include_dnps adds the blanks so a minutes model can be wrong about sitting
    players the same way live scoring is.
    """
    windows = []
    last_gw = int(getattr(holdout_season, 'training_complete_through_gw', 38) or 38)
    for player in holdout_season.players:
        matches = _season_matches(player)
        by_gw = {}
        for match in matches:
            by_gw.setdefault(int(match.gameweek), []).append(match)
        player_id = getattr(player, 'id', None)
        for anchor in range(split_gw + 1, last_gw - horizon + 2):
            if anchor not in by_gw:
                continue
            window = [
                match for match in matches
                if anchor <= int(match.gameweek) < anchor + horizon
            ]
            played = [m for m in window if match_minutes(m) > 0]
            if len(played) < MIN_PLAYED_IN_WINDOW:
                continue
            anchor_match = by_gw[anchor][0]
            fixtures = []
            for match in window:
                minutes = match_minutes(match)
                if minutes <= 0 and not include_dnps:
                    continue
                fixtures.append({
                    'row': match.to_clean_match(),
                    'gameweek': int(match.gameweek),
                    'player_id': player_id,
                    'minutes': minutes,
                    'actual_pp90': fixture_pp90(match),
                    'actual_points': match_points(match),
                })
            if not fixtures:
                continue
            windows.append({
                'player_id': player_id,
                'anchor': anchor,
                'window_row': _horizon_row(anchor_match, window),
                'fixtures': fixtures,
                'actual_window_pp90': pooled_pp90(window),
            })
    fixture_rows = []
    for window in windows:
        for fixture in window['fixtures']:
            fixture['index'] = len(fixture_rows)
            fixture_rows.append(fixture['row'])
    window_rows = [w['window_row'] for w in windows]
    for i, row in enumerate(fixture_rows):
        row['id'] = i
    for i, row in enumerate(window_rows):
        row['id'] = i
    print(
        'Holdout: %s windows, %s scored fixtures, GW%s-%s of %s/%s'
        % (
            len(windows), len(fixture_rows), split_gw + 1, last_gw,
            _season_year(holdout_season), _season_year(holdout_season) + 1,
        )
    )
    return windows, fixture_rows, window_rows


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    if len(a) < 3:
        return float('nan')
    value = spearmanr(a, b).correlation
    return float('nan') if value is None else float(value)


def _skill(mae, baseline_mae):
    if not baseline_mae:
        return float('nan')
    return 1.0 - (mae / baseline_mae)


def combine_hurdle(p_play, pp90_if_play):
    """E[pp90] = P(played) x E[pp90 | played]."""
    p = numpy.clip(numpy.asarray(p_play, dtype=float).reshape(-1), 0.0, 1.0)
    mu = numpy.maximum(numpy.asarray(pp90_if_play, dtype=float).reshape(-1), 0.0)
    if p.shape != mu.shape:
        raise ValueError('hurdle parts must be the same length')
    return p * mu


def expected_points(pp90, minutes):
    """Same mapping squadtools uses: pp90 x clip(minutes, 0, 90) / 90."""
    rate = numpy.asarray(pp90, dtype=float).reshape(-1)
    mins = numpy.clip(numpy.asarray(minutes, dtype=float).reshape(-1), 0.0, 90.0)
    return rate * mins / 90.0


def slice_calibration(pred_points, actual_points, n_slices=SLICE_N, min_rows=SLICE_MIN_ROWS):
    """Calibration by predicted-points decile, not by who actually scored.

    Slicing on actuals would always make the top bin look underpredicted.
    """
    pred = numpy.asarray(pred_points, dtype=float).reshape(-1)
    actual = numpy.asarray(actual_points, dtype=float).reshape(-1)
    empty = {'n': 0, 'cal': None, 'lo': None, 'hi': None, 'pred_mean': None, 'actual_mean': None}
    if pred.size == 0:
        return [{'slice': i, **empty} for i in range(n_slices)]
    edges = numpy.quantile(pred, numpy.linspace(0.0, 1.0, n_slices + 1))
    rows = []
    for i in range(n_slices):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_slices - 1:
            mask = (pred >= lo) & (pred <= hi)
        else:
            mask = (pred >= lo) & (pred < hi)
        n = int(mask.sum())
        pred_sum = float(pred[mask].sum()) if n else 0.0
        actual_sum = float(actual[mask].sum()) if n else 0.0
        if n < min_rows or actual_sum == 0.0:
            cal = None
        else:
            cal = round(pred_sum / actual_sum, 4)
        rows.append({
            'slice': i,
            'lo': round(lo, 4),
            'hi': round(hi, 4),
            'n': n,
            'cal': cal,
            'pred_mean': None if n == 0 else round(float(pred[mask].mean()), 4),
            'actual_mean': None if n == 0 else round(float(actual[mask].mean()), 4),
        })
    return rows


def _rank_metrics(pred, actual, gameweeks, player_ids=None, mask=None):
    """Per-gameweek Spearman and top-20 overlap. Optionally sum a player's DGW."""
    pred = numpy.asarray(pred, dtype=float).reshape(-1)
    actual = numpy.asarray(actual, dtype=float).reshape(-1)
    gameweeks = numpy.asarray(gameweeks, dtype=int).reshape(-1)
    if mask is None:
        mask = numpy.ones(pred.shape[0], dtype=bool)
    taus = []
    overlaps = []
    for gameweek in numpy.unique(gameweeks):
        pick = mask & (gameweeks == gameweek)
        if player_ids is not None:
            pred_by = defaultdict(float)
            actual_by = defaultdict(float)
            for i in numpy.where(pick)[0]:
                key = player_ids[i]
                pred_by[key] += float(pred[i])
                actual_by[key] += float(actual[i])
            if len(pred_by) < TOP_N:
                continue
            keys = list(pred_by)
            gw_hat = numpy.array([pred_by[k] for k in keys], dtype=float)
            gw_actual = numpy.array([actual_by[k] for k in keys], dtype=float)
        else:
            if int(pick.sum()) < TOP_N:
                continue
            gw_hat = pred[pick]
            gw_actual = actual[pick]
        taus.append(_spearman(gw_hat, gw_actual))
        top_hat = set(numpy.argsort(gw_hat)[::-1][:TOP_N].tolist())
        top_actual = set(numpy.argsort(gw_actual)[::-1][:TOP_N].tolist())
        overlaps.append(len(top_hat & top_actual) / float(TOP_N))
    taus = [t for t in taus if t == t]
    return (
        float(numpy.mean(taus)) if taus else float('nan'),
        float(numpy.mean(overlaps)) if overlaps else float('nan'),
    )


def score(windows, fixture_pred, window_pred, pred_minutes=None):
    """Common yardstick: every arm is judged on these same realised outcomes.

    Unweighted pp90 error is reported but should not be led with: a five-minute
    goalscorer posts a pp90 over 100, so the mean realised pp90 sits far above
    anything a starter produces and the cameos dominate both the error and the
    ranking. Weighting by minutes played is the fix, and the minutes-weighted
    calibration is exactly total predicted points over total actual points.
    Ranking is measured on starters only, for the same reason.

    When pred_minutes is set, expected points use the minutes model
    (pp90 x predicted minutes / 90) and ranking is on those points vs realised
    points, all players that gameweek, which is what the optimiser sees.
    """
    actual = []
    hat = []
    minutes = []
    gameweeks = []
    player_ids = []
    realised_points = []
    indexes = []
    for window in windows:
        for fixture in window['fixtures']:
            actual.append(fixture['actual_pp90'])
            hat.append(float(fixture_pred[fixture['index']]))
            minutes.append(fixture['minutes'])
            gameweeks.append(fixture['gameweek'])
            player_ids.append(fixture.get('player_id', window.get('player_id')))
            realised_points.append(float(fixture.get(
                'actual_points', fixture['actual_pp90'] * fixture['minutes'] / 90.0
            )))
            indexes.append(fixture['index'])

    actual = numpy.array(actual, dtype=float)
    hat = numpy.array(hat, dtype=float)
    minutes = numpy.array(minutes, dtype=float)
    gameweeks = numpy.array(gameweeks, dtype=int)
    realised_points = numpy.array(realised_points, dtype=float)
    window_actual = numpy.array([w['actual_window_pp90'] for w in windows], dtype=float)
    window_hat = numpy.array([float(v) for v in window_pred], dtype=float)

    err = numpy.abs(hat - actual)
    window_err = numpy.abs(window_hat - window_actual)
    fixture_baseline = float(numpy.mean(numpy.abs(actual - numpy.median(actual))))
    window_baseline = float(
        numpy.mean(numpy.abs(window_actual - numpy.median(window_actual)))
    )
    played = minutes > 0
    if played.any():
        weighted_mae = float(numpy.average(err[played], weights=minutes[played]))
        weighted_baseline = float(
            numpy.average(
                numpy.abs(actual[played] - numpy.median(actual[played])),
                weights=minutes[played],
            )
        )
    else:
        weighted_mae = float('nan')
        weighted_baseline = float('nan')

    if pred_minutes is not None:
        minutes_for_points = numpy.array(
            [float(pred_minutes[i]) for i in indexes], dtype=float
        )
        pred_points = expected_points(hat, minutes_for_points)
        rho, top20 = _rank_metrics(
            pred_points, realised_points, gameweeks, player_ids=player_ids
        )
        pred_by_index = numpy.zeros(int(max(indexes)) + 1, dtype=float)
        actual_by_index = numpy.zeros(int(max(indexes)) + 1, dtype=float)
        for i, idx in enumerate(indexes):
            pred_by_index[idx] = pred_points[i]
            actual_by_index[idx] = realised_points[i]
        window_pred_pts = numpy.array([
            sum(pred_by_index[f['index']] for f in window['fixtures'])
            for window in windows
        ], dtype=float)
        window_actual_pts = numpy.array([
            sum(actual_by_index[f['index']] for f in window['fixtures'])
            for window in windows
        ], dtype=float)
        window_err = numpy.abs(window_pred_pts - window_actual_pts)
        window_baseline = float(
            numpy.mean(numpy.abs(window_actual_pts - numpy.median(window_actual_pts)))
        )
    else:
        pred_points = expected_points(hat, minutes)
        starters = minutes >= STARTER_MINUTES
        rho, top20 = _rank_metrics(
            hat, actual, gameweeks, mask=starters
        )

    points_mae = float(numpy.mean(numpy.abs(pred_points - realised_points)))
    predicted_points_sum = float(pred_points.sum())
    actual_points_sum = float(realised_points.sum())
    slices = slice_calibration(pred_points, realised_points)
    starters = minutes >= STARTER_MINUTES

    fixture_mae = float(err.mean())
    window_mae = float(window_err.mean())
    return {
        'fixture_mae': round(fixture_mae, 4),
        'fixture_skill': round(_skill(fixture_mae, fixture_baseline), 4),
        'weighted_mae': round(weighted_mae, 4),
        'weighted_skill': round(_skill(weighted_mae, weighted_baseline), 4),
        'window_mae': round(window_mae, 4),
        'window_skill': round(_skill(window_mae, window_baseline), 4),
        'points_mae': round(points_mae, 4),
        'points_calibration': round(
            predicted_points_sum / actual_points_sum if actual_points_sum else float('nan'), 4
        ),
        'spearman': round(rho, 4),
        'top20': round(top20, 4),
        'pred_mean': round(float(hat.mean()), 4),
        'actual_mean': round(float(actual.mean()) if actual.size else float('nan'), 4),
        'starter_pp90_mean': round(
            float(actual[starters].mean()) if starters.any() else float('nan'), 4
        ),
        'slice_cal': slices,
        'slice_cal_lo': slices[0]['cal'] if slices else None,
        'slice_cal_hi': slices[-1]['cal'] if slices else None,
    }


def to_fixture_and_window(fit, windows, raw_predictions):
    """Put an arm's output onto both scales so all arms share the yardstick.

    A single-gameweek arm predicts each fixture, so its window number is the
    minutes-weighted average of its fixtures. A horizon arm predicts one rate for
    the window, so that rate is also its prediction for every fixture in it.
    Realised minutes do the weighting, keeping the minutes model out of it.
    """
    values = numpy.asarray(raw_predictions, dtype=float).reshape(-1)
    if fit.model == 'horizon':
        window_pred = values
        fixture_pred = numpy.zeros(
            sum(len(w['fixtures']) for w in windows), dtype=float
        )
        for i, window in enumerate(windows):
            for fixture in window['fixtures']:
                fixture_pred[fixture['index']] = values[i]
        return fixture_pred, window_pred

    fixture_pred = values.copy()
    if fit.strip_appearance:
        # Trained without participation points, so add them back before scoring
        # against outcomes that include them.
        for window in windows:
            for fixture in window['fixtures']:
                minutes = fixture['minutes']
                if minutes > 0:
                    idx = fixture['index']
                    fixture_pred[idx] += (
                        appearance_points(minutes) * 90.0 / minutes
                    )
    window_pred = numpy.zeros(len(windows), dtype=float)
    for i, window in enumerate(windows):
        total_minutes = sum(f['minutes'] for f in window['fixtures'])
        if total_minutes <= 0:
            continue
        window_pred[i] = sum(
            fixture_pred[f['index']] * f['minutes'] for f in window['fixtures']
        ) / total_minutes
    return fixture_pred, window_pred


def _subsample(rows, labels, weights, factor, seed=42):
    """Thin rows by `factor` so a strided arm has a same-size overlapping control."""
    keep = len(rows) // int(factor)
    rng = numpy.random.default_rng(seed)
    picked = sorted(rng.choice(len(rows), size=keep, replace=False).tolist())
    rows = [rows[i] for i in picked]
    for i, row in enumerate(rows):
        row['id'] = i
    print('Subsampled to %s rows (1/%s)' % (len(rows), factor))
    return rows, [labels[i] for i in picked], [weights[i] for i in picked]


def _train_match_iter(runner, holdout_season, split_gw):
    holdout_year = _season_year(holdout_season)
    seasons = list(runner.past_seasons) + [runner.current_season]
    for season in seasons:
        year = _season_year(season)
        if year > holdout_year:
            continue
        limit = split_gw if year == holdout_year else 38
        for player in season.players:
            for match in _season_matches(player):
                if int(match.gameweek) > limit:
                    continue
                yield match


def build_h1_play_train(runner, holdout_season, split_gw):
    """Binary P(played) on minutes-valid rows, including DNPs."""
    rows = []
    labels = []
    for match in _train_match_iter(runner, holdout_season, split_gw):
        if not getattr(match, 'valid_for_minutes', False):
            continue
        rows.append(match.to_clean_match())
        labels.append(1.0 if match_minutes(match) > 0 else 0.0)
    for i, row in enumerate(rows):
        row['id'] = i
    return rows, labels


def build_minutes_train(runner, holdout_season, split_gw):
    rows = []
    labels = []
    for match in _train_match_iter(runner, holdout_season, split_gw):
        if not getattr(match, 'valid_for_minutes', False):
            continue
        rows.append(match.to_clean_match())
        labels.append(match_minutes(match))
    for i, row in enumerate(rows):
        row['id'] = i
    return rows, labels


def _points_mldata(rows, labels, predict_rows, loss, sample_weight):
    return MLData(
        rows,
        predict_rows,
        get_input_keys('total_points'),
        'total_points',
        True,
        output_learn=numpy.array(labels, dtype=float).reshape(-1, 1),
        log_targets=uses_log_targets(loss),
        sample_weight=sample_weight,
        loss=loss,
    )


def _rep_record(predictions, stats, clip_nonneg=False):
    values = numpy.asarray(predictions, dtype=float).reshape(-1)
    if clip_nonneg:
        values = numpy.maximum(values, 0.0)
    return {
        'predictions': values,
        'smearing': stats.get('smearing'),
        'stats': {
            key: stats.get(key)
            for key in ('epochs_ran', 'test_mae', 'best_val_mae', 'pred_mean')
        },
    }


def fit_minutes_predictions(runner, holdout_season, split_gw, fixture_rows, realised_minutes):
    """One minutes net for the holdout; every points arm reuses this vector."""
    rows, labels = build_minutes_train(runner, holdout_season, split_gw)
    if not rows:
        raise SystemExit('No minutes training rows')
    data = MLData(
        rows,
        fixture_rows,
        get_input_keys('minutes'),
        'minutes',
        True,
        output_learn=numpy.array(labels, dtype=float).reshape(-1, 1),
        log_targets=False,
        loss='huber',
    )
    print('\n--- fit minutes: %s rows (shared across all points arms) ---' % len(rows))
    data.setup_model()
    predictions, stats = data.get_predictions()
    pred = numpy.clip(numpy.asarray(predictions, dtype=float).reshape(-1), 0.0, 90.0)
    realised = numpy.asarray(realised_minutes, dtype=float).reshape(-1)
    if realised.size and pred.size == realised.size:
        mae = float(numpy.mean(numpy.abs(pred - realised)))
    else:
        mae = float('nan')
    stats = dict(stats)
    stats['holdout_minutes_mae'] = None if mae != mae else round(mae, 4)
    stats['n_train'] = len(rows)
    print('Minutes holdout MAE vs realised: %s' % stats['holdout_minutes_mae'])
    return pred, stats


def run_hurdle_fit(fit, runner, holdout_season, split_gw, fixture_rows, reps):
    """Two nets: P(played) on minutes-valid rows, Poisson+mw pp90 on players who played."""
    play_rows, play_labels = build_h1_play_train(runner, holdout_season, split_gw)
    pos_rows, pos_labels, pos_weights = build_h1_train(runner, holdout_season, split_gw)
    if not play_rows or not pos_rows:
        raise SystemExit('No hurdle training rows')
    play_data = _points_mldata(play_rows, play_labels, fixture_rows, 'bce', None)
    pos_weight = numpy.array(pos_weights, dtype=float) if fit.minutes_weight else None
    pos_data = _points_mldata(pos_rows, pos_labels, fixture_rows, 'poisson', pos_weight)
    print(
        '\n--- fit %s: %s play rows, %s positive rows ---'
        % (fit.key, len(play_rows), len(pos_rows))
    )
    per_rep = []
    for rep in range(reps):
        print('  %s play rep %s/%s' % (fit.key, rep + 1, reps))
        play_data.setup_model()
        p_hat, p_stats = play_data.get_predictions()
        print('  %s positive rep %s/%s' % (fit.key, rep + 1, reps))
        pos_data.setup_model()
        mu_hat, mu_stats = pos_data.get_predictions()
        combined = combine_hurdle(p_hat, mu_hat)
        record = _rep_record(combined, mu_stats)
        record['stats'] = {
            'play': {
                key: p_stats.get(key)
                for key in ('epochs_ran', 'test_mae', 'best_val_mae', 'pred_mean')
            },
            'positive': record['stats'],
        }
        per_rep.append(record)
    return per_rep


def run_fit(fit, runner, holdout_season, split_gw, horizon, windows,
            fixture_rows, window_rows, reps, quick=False):
    """Fit one training configuration `reps` times; return per-rep predictions."""
    if fit.loss == 'hurdle':
        return run_hurdle_fit(
            fit, runner, holdout_season, split_gw, fixture_rows, reps
        )
    if fit.model == 'h1':
        rows, labels, weights = build_h1_train(
            runner, holdout_season, split_gw, fit.strip_appearance
        )
        predict_rows = fixture_rows
    else:
        rows, labels, weights = build_horizon_train(
            runner, holdout_season, split_gw, horizon, fit.stride
        )
        predict_rows = window_rows
    if not rows:
        raise SystemExit('No training rows for fit %s' % fit.key)

    if fit.subsample:
        rows, labels, weights = _subsample(rows, labels, weights, fit.subsample)

    sample_weight = numpy.array(weights, dtype=float) if fit.minutes_weight else None
    data = _points_mldata(rows, labels, predict_rows, fit.loss, sample_weight)
    print(
        '\n--- fit %s: %s rows, loss=%s, minutes_weight=%s, stride=%s, log=%s ---'
        % (
            fit.key, len(rows), fit.loss, fit.minutes_weight, fit.stride,
            uses_log_targets(fit.loss),
        )
    )
    clip_nonneg = fit.loss in ('huber_raw', 'mse')
    per_rep = []
    for rep in range(reps):
        print('  %s rep %s/%s' % (fit.key, rep + 1, reps))
        data.setup_model()
        predictions, stats = data.get_predictions()
        per_rep.append(_rep_record(predictions, stats, clip_nonneg=clip_nonneg))
    return per_rep


def _mean_pstdev(values, ndigits=4):
    vals = [float(v) for v in values if v is not None and v == v]
    if not vals:
        return None, None
    mean = round(sum(vals) / len(vals), ndigits)
    if len(vals) < 2:
        return mean, 0.0
    return mean, round(statistics.pstdev(vals), ndigits)


METRIC_KEYS = (
    'weighted_mae', 'weighted_skill', 'points_mae', 'points_calibration',
    'spearman', 'top20', 'window_mae', 'window_skill', 'fixture_mae',
    'fixture_skill', 'pred_mean', 'slice_cal_lo', 'slice_cal_hi',
)


def aggregate(arm, per_rep_metrics):
    row = {'arm': arm.name, 'fit': arm.fit, 'smear': arm.smear, 'note': arm.note,
           'reps': len(per_rep_metrics)}
    for key in METRIC_KEYS:
        mean, std = _mean_pstdev([m[key] for m in per_rep_metrics])
        row[key] = mean
        row['%s_std' % key] = std
    if per_rep_metrics:
        row['actual_mean'] = per_rep_metrics[0]['actual_mean']
        row['starter_pp90_mean'] = per_rep_metrics[0]['starter_pp90_mean']
        slice_means = []
        n_slices = len(per_rep_metrics[0].get('slice_cal') or [])
        for i in range(n_slices):
            mean, _std = _mean_pstdev([
                (m.get('slice_cal') or [{}])[i].get('cal')
                if i < len(m.get('slice_cal') or []) else None
                for m in per_rep_metrics
            ])
            slice_means.append(mean)
        row['slice_cal'] = slice_means
        if slice_means:
            row['slice_cal_lo'] = slice_means[0]
            row['slice_cal_hi'] = slice_means[-1]
    return row


def write_results(run_dir, rows, fit_stats, context):
    run_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ['arm', 'fit', 'smear', 'reps', 'actual_mean', 'starter_pp90_mean']
    for key in METRIC_KEYS:
        fieldnames.extend([key, '%s_std' % key])
    fieldnames.append('note')
    csv_path = run_dir / 'comparison.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / 'arms.json').write_text(
        json.dumps({'context': context, 'rows': rows}, indent=2), encoding='utf-8'
    )
    (run_dir / 'per_fit_stats.json').write_text(
        json.dumps(fit_stats, indent=2), encoding='utf-8'
    )

    lines = [
        'Horizon vs single-gameweek, scored on one forward holdout',
        'Holdout %s/%s GW%s-%s; trained on earlier seasons + GW1-%s.'
        % (
            context['holdout_season'], context['holdout_season'] + 1,
            context['split_gw'] + 1, context['last_gw'], context['split_gw'],
        ),
        '%s windows, %s fixtures. %s'
        % (
            context['n_windows'], context['n_fixtures'],
            (
                'Expected points use a shared minutes model (pp90 x predicted minutes / 90). '
                'Minutes holdout MAE %s.'
                % context.get('minutes_mae')
                if context.get('score_minutes') == 'predicted'
                else 'Realised minutes on both sides, so the minutes model is not involved.'
            ),
        ),
        '',
        'wMAE   minutes-weighted pp90 error, and sd across reps (lower better).',
        '       Unweighted fMAE is in the csv but a 5-minute goalscorer posts a pp90',
        '       over 100, so cameos dominate it. Mean realised pp90 is %s overall but'
        % context['actual_mean'],
        '       %s among starters.' % context['starter_pp90_mean'],
        'skill  1 - MAE/constant-median MAE. 0 means no better than a constant.',
        'cal    total predicted points / total actual points. 1.0 is calibrated.',
        'rho    within-gameweek rank correlation'
        + (
            ', all players, on expected vs actual points.'
            if context.get('score_minutes') == 'predicted'
            else ', starters only (60+ minutes).'
        ),
        'top20  share of the actual top 20 that the arm also ranked top 20.',
        'cal_lo / cal_hi  calibration in the lowest and highest predicted-points decile.',
        'wMAE and cal disagree on purpose: MAE on a heavy tail is minimised near the',
        'median, so an arm can win wMAE while being badly mis-levelled on cal.',
        '',
    ]
    header = (
        '%-24s %8s %6s %7s %6s %6s %7s %7s %7s'
        % ('arm', 'wMAE', 'sd', 'wSkill', 'cal', 'rho', 'top20', 'cal_lo', 'cal_hi')
    )
    lines.append(header)
    lines.append('-' * len(header))
    for row in rows:
        lines.append(
            '%-24s %8s %6s %7s %6s %6s %7s %7s %7s'
            % (
                row['arm'],
                row.get('weighted_mae', ''), row.get('weighted_mae_std', ''),
                row.get('weighted_skill', ''), row.get('points_calibration', ''),
                row.get('spearman', ''), row.get('top20', ''),
                row.get('slice_cal_lo', ''), row.get('slice_cal_hi', ''),
            )
        )
    slice_values = [row.get('slice_cal') for row in rows if row.get('slice_cal')]
    if slice_values:
        n_slices = max(len(v) for v in slice_values)
        lines.extend(['', 'Predicted-points calibration by decile (d1 = we called lowest):'])
        slice_header = '%-24s' % 'arm' + ''.join(' %7s' % ('d%s' % (i + 1)) for i in range(n_slices))
        lines.append(slice_header)
        lines.append('-' * len(slice_header))
        for row in rows:
            vals = row.get('slice_cal') or []
            lines.append(
                '%-24s' % row['arm']
                + ''.join(' %7s' % (vals[i] if i < len(vals) else '') for i in range(n_slices))
            )
    text = '\n'.join(lines) + '\n'
    (run_dir / 'comparison.txt').write_text(text, encoding='utf-8')
    slice_path = run_dir / 'slice_calibration.csv'
    with slice_path.open('w', newline='', encoding='utf-8') as f:
        n_slices = max((len(row.get('slice_cal') or []) for row in rows), default=0)
        writer = csv.writer(f)
        writer.writerow(['arm'] + ['d%s' % (i + 1) for i in range(n_slices)])
        for row in rows:
            vals = row.get('slice_cal') or []
            writer.writerow([row['arm']] + vals)
    print('\n' + text)
    print('Wrote %s' % run_dir)


def _fit_cost(fit):
    """Rough relative cost, for balancing fits across workers.

    Driven by training rows: h1 has one row per fixture, an overlapping horizon
    has one per anchor, and a strided or subsampled one has a fraction of that.
    """
    if fit.loss == 'hurdle':
        return 2.0
    if fit.subsample:
        return 1.0 / float(fit.subsample)
    if fit.model == 'h1':
        return 1.0
    if fit.stride and fit.stride > 1:
        return 1.0 / float(fit.stride)
    return 0.8


def partition_fits(fits, workers):
    """Longest-processing-time-first, so no worker is left holding all the big fits."""
    workers = max(1, int(workers))
    groups = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for fit in sorted(fits, key=_fit_cost, reverse=True):
        target = loads.index(min(loads))
        groups[target].append(fit)
        loads[target] += _fit_cost(fit)
    return [group for group in groups if group]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare horizon and single-gameweek models on a forward holdout'
    )
    parser.add_argument('--reps', type=int, default=DEFAULT_REPS,
                        help='Independent fits per configuration (default %s)' % DEFAULT_REPS)
    parser.add_argument('--workers', type=int, default=1,
                        help='Child processes to spread the fits over (default 1)')
    parser.add_argument('--internal-worker', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--fits', help=argparse.SUPPRESS)
    parser.add_argument('--arms', help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', help=argparse.SUPPRESS)
    parser.add_argument('--worker-index', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--split-gw', type=int, default=DEFAULT_SPLIT_GW,
                        help='Last holdout-season gameweek used for training (default %s)'
                             % DEFAULT_SPLIT_GW)
    parser.add_argument('--holdout-season', type=int, default=None,
                        help='Season start year to hold out (default: latest complete)')
    parser.add_argument(
        '--holdout-seasons',
        help='Comma-separated season start years to run in sequence (e.g. 25,24)',
    )
    parser.add_argument('--horizon', type=int, default=POINTS_HORIZON,
                        help='Window length (default %s)' % POINTS_HORIZON)
    parser.add_argument('--pack', choices=('full', 'live'), default='full',
                        help='full = original 19-arm grid; live = minutes-in-score pack')
    parser.add_argument(
        '--score-minutes', choices=('realised', 'predicted'), default=None,
        help='predicted = pp90 x minutes-model / 90 (default for --pack live)',
    )
    parser.add_argument('--only',
                        help='Comma-separated fits or arms to run (smoke test)')
    parser.add_argument('--dry-run', action='store_true',
                        help='List fits and arms without loading data')
    parser.add_argument('--quick', action='store_true', help='Short epochs (debug only)')
    parser.add_argument('--internal-minutes', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--minutes-pred', help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_args(args):
    if not args.score_minutes:
        args.score_minutes = 'predicted' if args.pack == 'live' else 'realised'
    return args


def _holdout_years(args):
    if args.holdout_seasons:
        return [int(part.strip()) for part in args.holdout_seasons.split(',') if part.strip()]
    return [args.holdout_season]


def _selected(args):
    pack = getattr(args, 'pack', 'full')
    if pack == 'live':
        allowed = set(LIVE_ARM_NAMES)
    else:
        allowed = set(ORIGINAL_ARM_NAMES)
    arms = [arm for arm in ARMS if arm.name in allowed]
    if args.only:
        tokens = [t.strip() for t in args.only.split(',') if t.strip()]
        arms = [arm for arm in arms if arm.name in tokens or arm.fit in tokens]
        if not arms:
            raise SystemExit(
                'Unknown arm/fit %s. Arms: %s'
                % (args.only, ', '.join(a.name for a in ARMS))
            )
    keys = {arm.fit for arm in arms}
    return [fit for fit in FITS if fit.key in keys], arms


def _realised_minutes_vector(windows, n_fixtures):
    out = numpy.zeros(int(n_fixtures), dtype=float)
    for window in windows:
        for fixture in window['fixtures']:
            out[fixture['index']] = fixture['minutes']
    return out


def _persist_minutes(run_dir, pred, stats, n_fixtures, holdout_year):
    run_dir = Path(run_dir)
    numpy.save(run_dir / 'minutes_pred.npy', pred)
    payload = {
        'stats': stats,
        'n_fixtures': n_fixtures,
        'holdout_season': holdout_year,
    }
    (run_dir / 'minutes_stats.json').write_text(
        json.dumps(payload, indent=2, default=str), encoding='utf-8'
    )


def compute(fits, arms, args, run_dir=None):
    """Fit and score one set of arms. Runs in the parent, or in each worker."""
    runner = load_seasons()
    holdout_season = pick_holdout_season(runner, args.holdout_season)
    horizon = int(args.horizon)
    split_gw = int(args.split_gw)
    include_dnps = args.score_minutes == 'predicted'
    windows, fixture_rows, window_rows = build_holdout(
        holdout_season, split_gw, horizon, include_dnps=include_dnps
    )
    if not windows:
        raise SystemExit('Holdout is empty; try a lower --split-gw.')

    predictions_by_fit = {}
    fit_stats = {}
    pred_minutes = None
    minutes_mae = None
    if args.score_minutes == 'predicted':
        minutes_path = getattr(args, 'minutes_pred', None)
        if minutes_path:
            pred_minutes = numpy.load(minutes_path)
            if len(pred_minutes) != len(fixture_rows):
                raise SystemExit(
                    'minutes_pred length %s does not match %s fixture rows'
                    % (len(pred_minutes), len(fixture_rows))
                )
            stats_file = Path(minutes_path).with_name('minutes_stats.json')
            if stats_file.is_file():
                payload = json.loads(stats_file.read_text(encoding='utf-8'))
                minutes_mae = (payload.get('stats') or payload).get('holdout_minutes_mae')
        else:
            realised = _realised_minutes_vector(windows, len(fixture_rows))
            pred_minutes, minutes_stats = fit_minutes_predictions(
                runner, holdout_season, split_gw, fixture_rows, realised
            )
            fit_stats['minutes'] = minutes_stats
            minutes_mae = minutes_stats.get('holdout_minutes_mae')
            if run_dir is not None:
                _persist_minutes(
                    run_dir, pred_minutes, minutes_stats,
                    len(fixture_rows), _season_year(holdout_season),
                )

    for fit in fits:
        per_rep = run_fit(
            fit, runner, holdout_season, split_gw, horizon, windows,
            fixture_rows, window_rows, args.reps, args.quick
        )
        predictions_by_fit[fit.key] = per_rep
        fit_stats[fit.key] = {
            'note': fit.note,
            'reps': [rep['stats'] for rep in per_rep],
            'smearing': [rep['smearing'] for rep in per_rep],
        }

    rows = []
    for arm in arms:
        fit = FIT_BY_KEY[arm.fit]
        per_rep_metrics = []
        for rep in predictions_by_fit[arm.fit]:
            corrected = apply_smearing(rep['predictions'], rep['smearing'], arm.smear)
            fixture_pred, window_pred = to_fixture_and_window(fit, windows, corrected)
            per_rep_metrics.append(
                score(windows, fixture_pred, window_pred, pred_minutes=pred_minutes)
            )
        rows.append(aggregate(arm, per_rep_metrics))

    last_gw = int(getattr(holdout_season, 'training_complete_through_gw', 38) or 38)
    context = {
        'holdout_season': _season_year(holdout_season),
        'split_gw': split_gw,
        'last_gw': last_gw,
        'horizon': horizon,
        'reps': args.reps,
        'pack': getattr(args, 'pack', 'full'),
        'score_minutes': args.score_minutes,
        'n_windows': len(windows),
        'n_fixtures': len(fixture_rows),
        'actual_mean': rows[0].get('actual_mean') if rows else None,
        'starter_pp90_mean': rows[0].get('starter_pp90_mean') if rows else None,
        'minutes_mae': minutes_mae,
    }
    return rows, fit_stats, context


def run_internal_minutes(args):
    if not args.run_dir:
        raise SystemExit('--internal-minutes needs --run-dir')
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = load_seasons()
    holdout_season = pick_holdout_season(runner, args.holdout_season)
    split_gw = int(args.split_gw)
    horizon = int(args.horizon)
    windows, fixture_rows, _window_rows = build_holdout(
        holdout_season, split_gw, horizon, include_dnps=True
    )
    realised = _realised_minutes_vector(windows, len(fixture_rows))
    pred, stats = fit_minutes_predictions(
        runner, holdout_season, split_gw, fixture_rows, realised
    )
    _persist_minutes(
        run_dir, pred, stats, len(fixture_rows), _season_year(holdout_season)
    )
    print('Wrote minutes predictions for %s fixtures' % len(pred))


def run_minutes_job(args, run_dir):
    """Fit minutes once in a child so every points worker shares the same vector."""
    log_path = run_dir / 'minutes.log'
    cmd = [
        worker_jobs.worker_python(), '-m', 'premierlearning.model_ab',
        '--internal-minutes',
        '--run-dir', str(run_dir),
        '--split-gw', str(args.split_gw),
        '--horizon', str(args.horizon),
    ]
    if args.holdout_season is not None:
        cmd.extend(['--holdout-season', str(args.holdout_season)])
    if args.quick:
        cmd.append('--quick')
    env = worker_jobs.worker_env(worker_jobs.thread_limit_env(1))
    print('Fitting shared minutes model -> %s' % log_path.name)
    handle = log_path.open('w', encoding='utf-8')
    started = time.time()
    proc = worker_jobs.popen_worker(cmd, env=env, cwd=str(API_DIR), stdout=handle)
    next_heartbeat = started + 120
    try:
        while proc.poll() is None:
            if time.time() >= next_heartbeat:
                next_heartbeat = time.time() + 120
                print(
                    '  [%.0f min] minutes: %s'
                    % ((time.time() - started) / 60.0, _log_tail(log_path))
                )
                sys.stdout.flush()
            time.sleep(5)
        rc = proc.returncode
    finally:
        worker_jobs.LIVE_WORKERS.discard(proc)
        handle.close()
    if rc != 0 or not (run_dir / 'minutes_pred.npy').is_file():
        raise SystemExit('Minutes worker failed, rc=%s (see %s)' % (rc, log_path))
    args.minutes_pred = str(run_dir / 'minutes_pred.npy')
    print('Minutes model finished after %.1f min' % ((time.time() - started) / 60.0))


def run_internal_worker(args):
    if not args.fits or not args.run_dir:
        raise SystemExit('--internal-worker needs --fits and --run-dir')
    keys = [key.strip() for key in args.fits.split(',') if key.strip()]
    fits = [FIT_BY_KEY[key] for key in keys]
    wanted = {name.strip() for name in (args.arms or '').split(',') if name.strip()}
    arms = [
        arm for arm in ARMS
        if arm.fit in set(keys) and (not wanted or arm.name in wanted)
    ]
    print('Worker %s: fits %s' % (args.worker_index, ', '.join(keys)))
    args = resolve_args(args)
    rows, fit_stats, context = compute(fits, arms, args)
    out_path = Path(args.run_dir) / ('worker_%s.json' % args.worker_index)
    out_path.write_text(
        json.dumps({'rows': rows, 'fit_stats': fit_stats, 'context': context}, indent=2, default=str),
        encoding='utf-8',
    )
    print('Worker %s wrote %s' % (args.worker_index, out_path))


def _worker_command(fits, arms, run_dir, index, args):
    cmd = [
        worker_jobs.worker_python(), '-m', 'premierlearning.model_ab',
        '--internal-worker',
        '--fits', ','.join(fit.key for fit in fits),
        '--arms', ','.join(arm.name for arm in arms),
        '--run-dir', str(run_dir),
        '--worker-index', str(index),
        '--reps', str(args.reps),
        '--split-gw', str(args.split_gw),
        '--horizon', str(args.horizon),
    ]
    if args.holdout_season is not None:
        cmd.extend(['--holdout-season', str(args.holdout_season)])
    score_minutes = getattr(args, 'score_minutes', None)
    if score_minutes:
        cmd.extend(['--score-minutes', score_minutes])
    minutes_pred = getattr(args, 'minutes_pred', None)
    if minutes_pred:
        cmd.extend(['--minutes-pred', str(minutes_pred)])
    if args.quick:
        cmd.append('--quick')
    return cmd


def _log_tail(log_path):
    """Last non-empty line of a worker log, for the heartbeat."""
    try:
        lines = [
            line.strip()
            for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines()
            if line.strip()
        ]
    except OSError:
        return ''
    return lines[-1][:110] if lines else ''


def _await_workers(procs, run_dir, poll_seconds=5, heartbeat_seconds=120):
    """Wait for every worker, reporting finishes as they happen.

    Workers do not finish in the order they started, and a fit can run for many
    minutes without printing, so a silent wait looks identical to a hang.
    """
    started = time.time()
    pending = {index: (proc, handle, log_path) for index, proc, handle, log_path in procs}
    failures = []
    next_heartbeat = started + heartbeat_seconds
    while pending:
        for index in sorted(pending):
            proc, handle, log_path = pending[index]
            rc = proc.poll()
            if rc is None:
                continue
            del pending[index]
            worker_jobs.LIVE_WORKERS.discard(proc)
            handle.close()
            elapsed = (time.time() - started) / 60.0
            if rc != 0 or not (run_dir / ('worker_%s.json' % index)).is_file():
                failures.append(
                    'worker %s exited %s (see %s)' % (index, rc, log_path)
                )
                print('  worker %s FAILED after %.1f min, rc=%s' % (index, elapsed, rc))
            else:
                print('  worker %s finished after %.1f min' % (index, elapsed))
        if not pending:
            break
        if time.time() >= next_heartbeat:
            next_heartbeat = time.time() + heartbeat_seconds
            print('  [%.0f min] %s still running' % (
                (time.time() - started) / 60.0, len(pending)
            ))
            for index in sorted(pending):
                print('    worker %s: %s' % (index, _log_tail(pending[index][2])))
            sys.stdout.flush()
        time.sleep(poll_seconds)
    return failures


def run_with_workers(fits, arms, args, run_dir):
    """Spread the fits over child processes and merge what they write back."""
    groups = partition_fits(fits, args.workers)
    env = worker_jobs.worker_env(worker_jobs.thread_limit_env(len(groups)))
    cwd = str(API_DIR)
    print(
        'Running %s fits over %s workers (%s threads each)'
        % (len(fits), len(groups), env.get('OMP_NUM_THREADS'))
    )
    procs = []
    logs = []
    for index, group in enumerate(groups):
        log_path = run_dir / ('worker_%s.log' % index)
        cmd = _worker_command(group, arms, run_dir, index, args)
        print(
            '  worker %s: %s -> %s'
            % (index, ', '.join(fit.key for fit in group), log_path.name)
        )
        handle = log_path.open('w', encoding='utf-8')
        proc = worker_jobs.popen_worker(cmd, env=env, cwd=cwd, stdout=handle)
        procs.append((index, proc, handle, log_path))
        logs.append(log_path)

    failures = _await_workers(procs, run_dir)
    if failures:
        raise SystemExit('Workers failed:\n  ' + '\n  '.join(failures))

    rows = []
    fit_stats = {}
    context = None
    for index, _proc, _handle, _log in procs:
        payload = json.loads(
            (run_dir / ('worker_%s.json' % index)).read_text(encoding='utf-8')
        )
        rows.extend(payload['rows'])
        fit_stats.update(payload['fit_stats'])
        context = context or payload['context']
    order = {arm.name: i for i, arm in enumerate(ARMS)}
    rows.sort(key=lambda row: order.get(row['arm'], 0))
    return rows, fit_stats, context


def main():
    args = resolve_args(parse_args())
    if args.internal_minutes:
        return run_internal_minutes(args)
    if args.internal_worker:
        return run_internal_worker(args)

    fits, arms = _selected(args)
    years = _holdout_years(args)
    if args.dry_run:
        groups = partition_fits(fits, args.workers)
        print(
            'pack=%s score_minutes=%s holdout=%s'
            % (args.pack, args.score_minutes, years)
        )
        print('%s fits, %s reported arms, %s reps -> %s fits total'
              % (len(fits), len(arms), args.reps, len(fits) * args.reps))
        for fit in fits:
            print('  fit  %-16s model=%-7s loss=%-7s mw=%-5s stride=%s cost=%.2f'
                  % (fit.key, fit.model, fit.loss, fit.minutes_weight, fit.stride,
                     _fit_cost(fit)))
        for arm in arms:
            print('  arm  %-24s fit=%-16s smear=%s' % (arm.name, arm.fit, arm.smear))
        if args.workers > 1:
            for index, group in enumerate(groups):
                print('  worker %s (load %.2f): %s'
                      % (index, sum(_fit_cost(f) for f in group),
                         ', '.join(f.key for f in group)))
        return

    if args.quick and '--quick' not in sys.argv:
        sys.argv.append('--quick')

    root = RESULTS_ROOT / datetime.now().strftime('%Y%m%d_%H%M%S')
    for year in years:
        season_args = copy.copy(args)
        season_args.holdout_season = year
        season_args.minutes_pred = None
        dest = root / str(year) if len(years) > 1 and year is not None else root
        dest.mkdir(parents=True, exist_ok=True)
        print('Holdout season %s -> %s' % (year if year is not None else 'latest', dest))
        _run_one(season_args, fits, arms, dest)


def _run_one(args, fits, arms, run_dir):
    if args.workers > 1:
        atexit.register(worker_jobs.kill_live_workers)
        try:
            if args.score_minutes == 'predicted':
                run_minutes_job(args, run_dir)
            rows, fit_stats, context = run_with_workers(fits, arms, args, run_dir)
        except BaseException:
            worker_jobs.kill_live_workers()
            raise
    else:
        rows, fit_stats, context = compute(fits, arms, args, run_dir)
    stats_path = run_dir / 'minutes_stats.json'
    if stats_path.is_file():
        payload = json.loads(stats_path.read_text(encoding='utf-8'))
        stats = payload.get('stats') or payload
        context['minutes_mae'] = stats.get('holdout_minutes_mae')
        context['score_minutes'] = args.score_minutes
    write_results(run_dir, rows, fit_stats, context)


if __name__ == '__main__':
    main()
