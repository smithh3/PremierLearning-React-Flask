"""Opt-in h1 points model. Replaces Runner.learn when --learning-v2 is set."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from premierlearning.learning import _parse_string_flag
from premierlearning.learning_v2.dataset import build_dataset
from premierlearning.learning_v2.kickoff import parse_kickoff
from premierlearning.learning_v2.plan import load_plan
from premierlearning.learning_v2.train import fit_and_predict

OPENING_GAMEWEEKS = (1, 2, 3, 4, 5)
PREDICTION_TABLE_SIZE = 10


def learn_h1(runner, plan_name=None):
    """Predict raw points for each future fixture. Minutes stay on the history tokens."""
    plan_name = plan_name or _parse_string_flag('--model-plan', 'simple_h1')
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
    predictions, _played = fit_and_predict(plan, vocab, train_batch, predict_batch)
    for match, points in zip(predict_batch.matches, predictions):
        value = float(points)
        match.points = value
        stats = getattr(match, 'game_stats', None)
        if stats is None:
            match.game_stats = {'total_points': value}
        else:
            stats['total_points'] = value
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
