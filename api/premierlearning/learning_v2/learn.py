"""Opt-in h1 points model. Replaces Runner.learn when --learning-v2 is set.

With --v2-blend as well, learn_blend trains the model the fantasy manager
evaluation scored best (set 35, "v2_log_minbands_blend25"): 75% v2_log with
a minutes-bands head, 25% v2_log plus a 5-gameweek net.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy

from premierlearning.learning import _parse_string_flag
from premierlearning.learning_v2.dataset import (
    MINUTES_TARGET_BANDS,
    build_dataset,
    window_rate_batch,
)
from premierlearning.learning_v2.kickoff import parse_kickoff
from premierlearning.learning_v2.plan import load_plan
from premierlearning.learning_v2.score import LossWeighting
from premierlearning.learning_v2.train import fit_and_predict, fit_points

OPENING_GAMEWEEKS = (1, 2, 3, 4, 5)
PREDICTION_TABLE_SIZE = 10

# Stored on each future fixture: these points are expected points with
# minutes risk already in them, so the optimiser uses them as they are (no
# minutes mapping, no second chance-of-playing cut).
POINTS_BASIS_RAW = 'raw'

# The evaluated blend. Same settings as simulation/model_eval/experiments.py
# ('v2_log_minbands', 'v2_log_h5') and manager_sets.py (weight 0.25).
BLEND_FLAG = '--v2-blend'
BLEND_LOSS = 'huber'          # Huber on log1p points
BLEND_SMEAR = 'global'        # one smearing factor back to the points scale
BLEND_MINUTES_WEIGHT = 0.1    # minutes-bands head loss weight (0, 1-59, 60+)
BLEND_HORIZON = 5             # the other model's window net
BLEND_OTHER_WEIGHT = 0.25


def store_points(matches, predictions):
    """Put predicted raw points where the rest of the pipeline reads them."""
    for match, points in zip(matches, predictions):
        value = float(points)
        match.points = value
        match.points_basis = POINTS_BASIS_RAW
        stats = getattr(match, 'game_stats', None)
        if stats is None:
            match.game_stats = {'total_points': value}
        else:
            stats['total_points'] = value


def store_minutes_bands(matches, bands):
    """Chance of 0, 1-59 and 60+ minutes on each fixture, from the bands head."""
    if bands is None:
        return
    for match, row in zip(matches, numpy.asarray(bands, dtype=float)):
        match.minutes_bands = tuple(float(value) for value in row[:3])


def window_points(batch, h1_points, rates, start_gameweek, horizon):
    """spread_window_rates as a pure function: the window model's fixture points.

    A player's window total is the sum of the window net's rate on each of
    his fixtures in the window; it is split across those fixtures by the h1
    predictions. Fixtures outside the window keep their h1 points.
    """
    values = numpy.asarray(h1_points, dtype=float).reshape(-1).copy()
    windows = defaultdict(list)
    for index, (owner, match, rate) in enumerate(
        zip(batch.owners, batch.matches, numpy.asarray(rates).reshape(-1))
    ):
        gameweek = getattr(match, 'gameweek', None)
        if gameweek is not None and start_gameweek <= int(gameweek) < start_gameweek + horizon:
            windows[id(owner)].append((index, max(0.0, float(rate))))
    for fixtures in windows.values():
        total = sum(rate for _index, rate in fixtures)
        shares = [max(0.0, float(values[index])) for index, _rate in fixtures]
        share_sum = sum(shares)
        for (index, _rate), share in zip(fixtures, shares):
            part = share / share_sum if share_sum > 0 else 1.0 / len(fixtures)
            values[index] = total * part
    return values


def blend_points(base, other, other_weight=BLEND_OTHER_WEIGHT):
    """(1 - w) base + w other, fixture by fixture, as manager_sets.blend_sets does."""
    base = numpy.asarray(base, dtype=float).reshape(-1)
    other = numpy.asarray(other, dtype=float).reshape(-1)
    return (1.0 - float(other_weight)) * base + float(other_weight) * other


def learn_blend(runner, plan_name=None):
    """The evaluated blend, trained on everything up to now.

    1. v2_log_minbands: the h1 net with a minutes-bands head (weight 0.1).
       Its points are the base; its bands head gives each fixture's chance
       of 0, 1-59 and 60+ minutes, kept for the report.
    2. v2_log: the same h1 net without the head, then a 5-gameweek window net
       whose totals are split across fixtures by that h1 net.
    3. Points = 75% of 1 + 25% of 2.
    """
    plan = load_plan(
        plan_name
        or getattr(runner, 'v2_model_plan', None)
        or _parse_string_flag('--model-plan', 'simple_h1')
    )
    seasons = list(getattr(runner, 'past_seasons', []) or [])
    current = runner.current_season
    seasons.append(current)
    cutoff = None
    if hasattr(runner, '_trained_through_gameweek'):
        cutoff = runner._trained_through_gameweek()
    reps = getattr(runner, 'v2_points_reps', None)
    weighting = LossWeighting('none')
    print(
        'learning_v2 blend: %.0f%% v2_log_minbands + %.0f%% v2_log_h%s (plan %s)'
        % ((1 - BLEND_OTHER_WEIGHT) * 100, BLEND_OTHER_WEIGHT * 100, BLEND_HORIZON, plan.name)
    )
    train_batch, predict_batch, vocab = build_dataset(
        seasons, plan, predict_season=current, cutoff_gameweek=cutoff,
    )
    print(
        '  %s train rows, %s predict rows, %s players, %s teams'
        % (
            len(train_batch.matches), len(predict_batch.matches),
            vocab.player_size - 1, vocab.team_size - 1,
        )
    )
    if not train_batch.matches:
        raise SystemExit('learning_v2 found no training fixtures')

    print('learning_v2 blend 1/3: v2_log_minbands (h1 net with the minutes-bands head)')
    base_fit = fit_points(
        plan, vocab, train_batch, loss_name=BLEND_LOSS, reps=reps, weighting=weighting,
        smear_mode=BLEND_SMEAR, minutes_weight=BLEND_MINUTES_WEIGHT,
        minutes_target=MINUTES_TARGET_BANDS,
    )
    base = base_fit.predict(predict_batch)
    bands = base_fit.predict_minutes_bands(predict_batch)
    del base_fit

    print('learning_v2 blend 2/3: v2_log (h1 net)')
    h1_fit = fit_points(
        plan, vocab, train_batch, loss_name=BLEND_LOSS, reps=reps, weighting=weighting,
        smear_mode=BLEND_SMEAR,
    )
    other_h1 = h1_fit.predict(predict_batch)
    del h1_fit

    print('learning_v2 blend 3/3: v2_log %s-gameweek window net' % BLEND_HORIZON)
    window_batch, groups = window_rate_batch(train_batch, BLEND_HORIZON)
    # The window rows copy h1's tensors; drop h1's before fitting again.
    del train_batch
    print('  %s window rows, %s GWs' % (len(window_batch.matches), BLEND_HORIZON))
    window_fit = fit_points(
        plan, vocab, window_batch, loss_name=BLEND_LOSS, reps=reps, weighting=weighting,
        smear_mode=BLEND_SMEAR, groups=groups,
    )
    rates = window_fit.predict(predict_batch)
    del window_fit, window_batch

    upcoming = upcoming_gameweeks(current)
    other = window_points(predict_batch, other_h1, rates, upcoming[0], BLEND_HORIZON)
    blended = blend_points(base, other)
    store_points(predict_batch.matches, blended)
    store_minutes_bands(predict_batch.matches, bands)
    print(format_opening_predictions(
        predict_batch.matches,
        gameweeks=upcoming,
        limit=PREDICTION_TABLE_SIZE,
        title='Top %s predicted points over the next %s gameweeks (GW%s-%s), blend'
        % (PREDICTION_TABLE_SIZE, len(upcoming), upcoming[0], upcoming[-1]),
    ))
    return blended


def spread_window_rates(batch, rates, start_gameweek, horizon):
    """Set each player's window total from the rates, split by h1 fixture shares.

    Every fixture in the window adds its own row's rate, so each opponent counts
    and a double counts twice. Fixtures past the window keep their h1 points.
    """
    windows = defaultdict(list)
    for owner, match, rate in zip(batch.owners, batch.matches, numpy.asarray(rates).reshape(-1)):
        gameweek = getattr(match, 'gameweek', None)
        if gameweek is not None and start_gameweek <= int(gameweek) < start_gameweek + horizon:
            windows[id(owner)].append((match, max(0.0, float(rate))))
    for fixtures in windows.values():
        total = sum(rate for _match, rate in fixtures)
        shares = [max(0.0, float(match.points)) for match, _rate in fixtures]
        share_sum = sum(shares)
        for (match, _rate), share in zip(fixtures, shares):
            match.h1_points = share
            part = share / share_sum if share_sum > 0 else 1.0 / len(fixtures)
            store_points([match], [total * part])


def learn_h1(runner, plan_name=None):
    """Predict raw points for each future fixture. Minutes stay on the history tokens.

    A historical build can set v2_model_plan, v2_weighting and v2_points_reps on
    the runner; otherwise the command-line flags decide.
    """
    plan_name = (
        plan_name
        or getattr(runner, 'v2_model_plan', None)
        or _parse_string_flag('--model-plan', 'simple_h1')
    )
    plan = load_plan(plan_name)
    seasons = list(getattr(runner, 'past_seasons', []) or [])
    current = runner.current_season
    seasons.append(current)
    cutoff = None
    if hasattr(runner, '_trained_through_gameweek'):
        cutoff = runner._trained_through_gameweek()
    print(
        'learning_v2 %s: one h1 points model '
        '(history %s, squad cap %s, form window %s)'
        % (plan.name, plan.config.history_len, plan.config.max_squad, plan.config.team_form_n)
    )
    train_batch, predict_batch, vocab = build_dataset(
        seasons, plan, predict_season=current, cutoff_gameweek=cutoff,
    )
    print(
        '  %s train rows, %s predict rows, %s players, %s teams'
        % (
            len(train_batch.matches), len(predict_batch.matches),
            vocab.player_size - 1, vocab.team_size - 1,
        )
    )
    if not train_batch.matches:
        raise SystemExit('learning_v2 found no training fixtures')
    predictions, _played = fit_and_predict(
        plan, vocab, train_batch, predict_batch,
        reps=getattr(runner, 'v2_points_reps', None),
        weighting=getattr(runner, 'v2_weighting', None),
    )
    store_points(predict_batch.matches, predictions)
    upcoming = upcoming_gameweeks(current)
    print(format_opening_predictions(
        predict_batch.matches,
        gameweeks=upcoming,
        limit=PREDICTION_TABLE_SIZE,
        title='Top %s predicted points over the next %s gameweeks (GW%s-%s)'
        % (PREDICTION_TABLE_SIZE, len(upcoming), upcoming[0], upcoming[-1]),
    ))
    return predictions


def _next_deadline_gameweek(season, now=None) -> int:
    """First gameweek whose deadline has not passed. The is_next flag goes stale without an update."""
    fantasy = getattr(season, 'fantasy_json', None) or {}
    events = fantasy.get('events') if isinstance(fantasy, dict) else None
    if not events:
        return 0
    if now is None:
        now = datetime.utcnow()
    chosen = 0
    for event in events:
        deadline = parse_kickoff(event.get('deadline_time'))
        if deadline is None or deadline < now:
            continue
        event_id = int(event.get('id') or 0)
        if event_id and (chosen == 0 or event_id < chosen):
            chosen = event_id
    return chosen


def upcoming_gameweeks(season, count=5, now=None):
    """The next `count` gameweeks, starting at the next deadline still ahead."""
    start = _next_deadline_gameweek(season, now)
    if start <= 0:
        start = int(getattr(season, 'next_gameweek', 0) or 0)
    if start <= 0:
        current = int(getattr(season, 'current_gameweek', 0) or 0)
        start = current + 1 if current else 1
    return tuple(range(start, start + count))


def _player_name(player) -> str:
    getter = getattr(player, 'get_name', None)
    if getter is not None:
        return str(getter())
    return str(getattr(player, 'id', 'player'))


def _team_name(player) -> str:
    named = getattr(player, 'team_name', None)
    if named:
        return str(named)
    team = getattr(player, 'team', None)
    return str(getattr(team, 'name', '') or '')


def format_opening_predictions(matches, gameweeks=OPENING_GAMEWEEKS, limit=5, title=None) -> str:
    """Top players by summed predicted points over the given gameweeks."""
    gameweeks = tuple(int(gw) for gw in gameweeks)
    totals = {}
    for match in matches:
        gameweek = getattr(match, 'gameweek', None)
        if gameweek is None or int(gameweek) not in gameweeks:
            continue
        player = getattr(match, 'player', None)
        if player is None:
            continue
        row = totals.get(id(player))
        if row is None:
            row = {
                'name': _player_name(player),
                'team': _team_name(player),
                'by_gw': defaultdict(float),
            }
            totals[id(player)] = row
        row['by_gw'][int(gameweek)] += float(getattr(match, 'points', 0) or 0)
    ranked = sorted(
        totals.values(),
        key=lambda row: (-sum(row['by_gw'].values()), row['name']),
    )[:limit]
    span = 'GW%s-%s' % (gameweeks[0], gameweeks[-1]) if gameweeks else 'GW'
    lines = [title or ('Top %s predicted points over %s' % (limit, span))]
    if not ranked:
        lines.append('  no predicted fixtures in %s' % span)
        return '\n'.join(lines)
    header = '  %3s  %-28s %-6s %7s' % ('#', 'Player', 'Team', 'Total')
    for gameweek in gameweeks:
        header += ' %6s' % ('GW%s' % gameweek)
    lines.append(header)
    for index, row in enumerate(ranked, start=1):
        total = sum(row['by_gw'].values())
        line = '  %3d  %-28s %-6s %7.2f' % (index, row['name'][:28], row['team'][:6], total)
        for gameweek in gameweeks:
            points = row['by_gw'].get(gameweek)
            line += ' %6s' % ('-' if points is None else '%.2f' % points)
        lines.append(line)
    return '\n'.join(lines)
