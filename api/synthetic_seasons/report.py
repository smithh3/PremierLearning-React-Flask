"""Realism report: the saved copies next to the real season they were made from.

Also checks the rules every copy must keep (team goals = player goals plus the
opponent's own goals, clean sheet only with 60+ minutes and nothing conceded on
the pitch, the real minutes in every team-match, a keeper in every team-match,
flags at the start of every moved or added injury).
"""
from __future__ import annotations

import csv
import gzip
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .schedule import MIN_REMATCH_GAP
from .source import DATA_ROOT, _parse_row, load_source, parse_fixtures, season_name

POSITION_NAMES = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}


def load_copy_fixtures(directory: Path):
    with (directory / 'fixtures.csv').open(encoding='utf-8') as handle:
        return parse_fixtures(list(csv.DictReader(handle)))


def load_copy_rows(source, directory: Path, fixtures=None):
    fixtures = load_copy_fixtures(directory) if fixtures is None else fixtures
    with gzip.open(directory / 'player_matches.csv.gz', 'rt', encoding='utf-8') as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = [_parse_row(raw, fixtures, source.positions) for raw in raw_rows]
    return [row for row in rows if row is not None]


def load_copy_scores(directory: Path):
    with (directory / 'fixtures.csv').open(encoding='utf-8') as handle:
        return {int(r['id']): (int(r['team_h_score']), int(r['team_a_score']))
                for r in csv.DictReader(handle) if r.get('event') not in ('', None)}


def season_stats(source, rows, scores) -> dict:
    fixtures = [f for f in source.fixtures.values() if f.gameweek is not None]
    team_goals = Counter()
    results = Counter()
    goal_counts = Counter()
    for f in fixtures:
        home, away = scores[f.id]
        team_goals[f.home] += home
        team_goals[f.away] += away
        results['home' if home > away else 'away' if away > home else 'draw'] += 1
        goal_counts[min(home, 4)] += 1
        goal_counts[min(away, 4)] += 1
    matches = len(fixtures)
    played = [r for r in rows if r['minutes'] > 0]
    by_player_goals = Counter()
    season_points = Counter()
    position = {}
    for r in rows:
        by_player_goals[r['element']] += r['goals_scored']
        season_points[r['element']] += r['total_points']
        position[r['element']] = r['_position']
    clean = defaultdict(int)
    for r in played:
        if r['_position'] in (1, 2) and r['minutes'] >= 60:
            clean[(r['fixture'], r['_team'])] = max(clean[(r['fixture'], r['_team'])], r['clean_sheets'])
    team_clean = sum(1 for f in fixtures for team, conceded in ((f.home, scores[f.id][1]), (f.away, scores[f.id][0]))
                     if conceded == 0)
    stats = {
        'goals_per_match': sum(team_goals.values()) / matches,
        'nil_nil': sum(1 for f in fixtures if scores[f.id] == (0, 0)) / matches,
        'home_win': results['home'] / matches, 'draw': results['draw'] / matches, 'away_win': results['away'] / matches,
        'team_scores': {k: goal_counts[k] / (2 * matches) for k in range(5)},
        'clean_sheet_rate': team_clean / (2 * matches),
        'team_goals': dict(team_goals),
        'assists_per_goal': sum(r['assists'] for r in rows) / max(1, sum(r['goals_scored'] for r in rows)),
        'bonus_per_match': sum(r['bonus'] for r in rows) / matches,
        'points_per_appearance': {p: np.mean([r['total_points'] for r in played if r['_position'] == p]) for p in POSITION_NAMES},
        'haul_share': sum(1 for r in played if r['total_points'] >= 10) / len(played),
        'blank_share': sum(1 for r in played if r['total_points'] <= 2) / len(played),
        'top10_points': {p: np.mean(sorted((v for e, v in season_points.items() if position[e] == p), reverse=True)[:10])
                         for p in POSITION_NAMES},
        'best_player_points': max(season_points.values()),
        'player_goals': dict(by_player_goals),
        'player_points': dict(season_points),
        'hat_tricks': sum(1 for r in rows if r['goals_scored'] >= 3),
    }
    return stats


def check_copy(source, rows, scores, meta, flags, fixtures=None) -> Counter:
    """Rule breaks in one copy. fixtures: the copy's (its gameweeks and kick-offs)."""
    fixtures = source.fixtures if fixtures is None else fixtures
    problems = Counter()
    # Minutes stay with the team's date (its gameweek), whoever the opponent is.
    real_minutes = defaultdict(int)
    for r in source.rows:
        real_minutes[(r['_gw'], r['_team'])] += r['minutes']
    copy_minutes = defaultdict(int)
    minutes, keepers, goals, own = defaultdict(int), Counter(), Counter(), Counter()
    for r in rows:
        key = (r['fixture'], r['_team'])
        minutes[key] += r['minutes']
        copy_minutes[(r['_gw'], r['_team'])] += r['minutes']
        if r['_position'] == 1 and r['minutes'] > 0:
            keepers[key] += 1
        goals[key] += r['goals_scored']
        own[(r['fixture'], r['_opponent'])] += r['own_goals']
        problems['row opponent not its fixture\'s'] += int(int(float(r.get('opponent_team') or 0)) != r['_opponent'])
        if r['minutes'] > 0:
            should = int(r['minutes'] >= 60 and r['goals_conceded'] == 0)
            problems['clean sheet wrong'] += int(r['clean_sheets'] != should)
        else:
            problems['points without minutes'] += int(r['total_points'] > 0)
    for key, value in real_minutes.items():
        problems['team-gameweek minutes changed'] += int(copy_minutes.get(key, 0) != value)
    for f in fixtures.values():
        if f.gameweek is None:
            continue
        home, away = scores[f.id]
        problems['team goals != player goals + own goals'] += int(goals[(f.id, f.home)] + own[(f.id, f.home)] != home)
        problems['team goals != player goals + own goals'] += int(goals[(f.id, f.away)] + own[(f.id, f.away)] != away)
        for team in (f.home, f.away):
            problems['team-match without a keeper'] += int(keepers[(f.id, team)] == 0)
    problems.update(check_schedule(source, fixtures))
    for spell in meta['spells']:
        if spell['kind'] in ('moved', 'added'):
            # Some real injuries happen after a deadline, so the first missed
            # match can be unflagged; a later deadline in the spell must be.
            window = range(spell['to_gw'], spell.get('to_last_gw', spell['to_gw'] + spell['matches']) + 1)
            flagged = any((((flags.get(str(gw)) or {}).get(str(spell['element'])) or {}).get('status') or 'a') != 'a'
                          for gw in window)
            problems['injury with no flag during it'] += int(not flagged)
    return problems


def check_schedule(source, fixtures) -> Counter:
    """The copy's fixture list keeps the real one's shape: same fixtures, teams per gameweek, deadlines."""
    problems = Counter()
    real = {f.id: f for f in source.fixtures.values()}
    problems['fixture teams changed'] = sum(
        1 for fid, f in fixtures.items() if fid not in real or (real[fid].home, real[fid].away) != (f.home, f.away))
    problems['fixture missing'] = len(set(real) - set(fixtures))

    def per_gameweek(items):
        count, first = Counter(), {}
        for f in items:
            if f.gameweek is None:
                continue
            count[(f.gameweek, f.home)] += 1
            count[(f.gameweek, f.away)] += 1
            if f.kickoff and (f.gameweek not in first or f.kickoff < first[f.gameweek]):
                first[f.gameweek] = f.kickoff
        return count, first

    real_count, real_first = per_gameweek(real.values())
    copy_count, copy_first = per_gameweek(fixtures.values())
    problems['team matches in a gameweek changed'] = sum(
        1 for key in set(real_count) | set(copy_count) if real_count.get(key) != copy_count.get(key))
    problems['deadline moved'] = sum(1 for gw, kick in real_first.items() if copy_first.get(gw) != kick)
    problems['rematch within %d gameweeks (more than real)' % MIN_REMATCH_GAP] = max(
        0, _rematches(fixtures.values()) - _rematches(real.values()))
    return problems


def _rematches(fixtures) -> int:
    met = defaultdict(list)
    for f in fixtures:
        if f.gameweek is not None:
            met[frozenset((f.home, f.away))].append(f.gameweek)
    return sum(1 for weeks in met.values() for a, b in zip(sorted(weeks), sorted(weeks)[1:]) if b - a < MIN_REMATCH_GAP)


def build_report(years, data_root: Path = None) -> str:
    root = Path(data_root) if data_root else DATA_ROOT
    lines = ['# Synthetic seasons: realism report', '',
             'Each block compares the real season with its copies. "Copies" shows the average over the copies and,',
             'in brackets, the lowest and highest copy.', '']
    for year in years:
        source = load_source(year, root)
        real = season_stats(source, source.rows, {f.id: (f.home_goals, f.away_goals) for f in source.fixtures.values()})
        directories = sorted((root / 'synthetic' / season_name(year)).glob('copy_*'))
        copies, problems, metas = [], Counter(), []
        for directory in directories:
            fixtures = load_copy_fixtures(directory)
            rows = load_copy_rows(source, directory, fixtures)
            scores = load_copy_scores(directory)
            meta = json.loads((directory / 'meta.json').read_text(encoding='utf-8'))
            flags = json.loads((directory / 'flags.json').read_text(encoding='utf-8'))
            copies.append(season_stats(source, rows, scores))
            problems.update(check_copy(source, rows, scores, meta, flags, fixtures))
            metas.append(meta)
        if not copies:
            continue
        lines += ['## %s (%d copies)' % (source.name.replace('season_', '').replace('_', '/'), len(copies)), '']
        lines += ['| | Real | Copies |', '|---|---|---|']

        def row(label, getter, fmt='%.2f'):
            values = [getter(c) for c in copies]
            lines.append('| %s | %s | %s (%s to %s) |' % (label, fmt % getter(real), fmt % np.mean(values),
                                                        fmt % min(values), fmt % max(values)))
        row('Goals per match', lambda s: s['goals_per_match'])
        row('0-0 share', lambda s: 100 * s['nil_nil'], '%.1f%%')
        row('Home win', lambda s: 100 * s['home_win'], '%.1f%%')
        row('Draw', lambda s: 100 * s['draw'], '%.1f%%')
        row('Away win', lambda s: 100 * s['away_win'], '%.1f%%')
        row('Team clean-sheet rate', lambda s: 100 * s['clean_sheet_rate'], '%.1f%%')
        for k in range(5):
            row('Team scores %s' % ('4+' if k == 4 else k), lambda s, k=k: 100 * s['team_scores'][k], '%.1f%%')
        row('Assists per goal', lambda s: s['assists_per_goal'])
        row('Bonus points per match', lambda s: s['bonus_per_match'])
        row('Hat-tricks in the season', lambda s: s['hat_tricks'], '%.0f')
        for p, name in POSITION_NAMES.items():
            row('%s points per appearance' % name, lambda s, p=p: s['points_per_appearance'][p])
        for p, name in POSITION_NAMES.items():
            row('%s top-10 season points (avg)' % name, lambda s, p=p: s['top10_points'][p], '%.0f')
        row('Best player season points', lambda s: s['best_player_points'], '%.0f')
        row('Appearances with 10+ points', lambda s: 100 * s['haul_share'], '%.1f%%')
        row('Appearances with 2 or fewer points', lambda s: 100 * s['blank_share'], '%.1f%%')
        # Team goals: how far each team's season tally moves.
        teams = sorted(real['team_goals'])
        spread = np.mean([np.std([c['team_goals'][t] for c in copies]) for t in teams])
        diff = np.mean([abs(np.mean([c['team_goals'][t] for c in copies]) - real['team_goals'][t]) for t in teams])
        lines += ['', 'Team season goals: a team\'s tally moves by about %.1f goals between copies (standard deviation); '
                  'the copies\' average is %.1f goals from the real tally per team.' % (spread, diff), '']
        names = _names(source)
        top = sorted(real['player_goals'].items(), key=lambda kv: -kv[1])[:10]
        lines += ['Top scorers: real goals, copies average (lowest to highest), and real vs copy season points.', '',
                  '| Player | Real goals | Copy goals | Real points | Copy points |', '|---|---|---|---|---|']
        for element, goals in top:
            g = [c['player_goals'].get(element, 0) for c in copies]
            pts = [c['player_points'].get(element, 0) for c in copies]
            lines.append('| %s | %d | %.1f (%d to %d) | %d | %.0f (%d to %d) |' % (
                names.get(element, element), goals, np.mean(g), min(g), max(g),
                real['player_points'][element], np.mean(pts), min(pts), max(pts)))
        moved = np.mean([m['moved'] for m in metas])
        removed = np.mean([m['removed'] for m in metas])
        added = np.mean([m['added'] for m in metas])
        swaps = Counter()
        for m in metas:
            swaps.update(m['swaps'])
        lines += ['', 'Injuries: %d real spells (runs of matches a regular starter missed with an FPL injury/doubt flag). '
                  'Per copy on average %.0f moved, %.0f removed, %.0f added. Minute swaps per copy: %s.' % (
                      metas[0]['real_spells'], moved, removed, added,
                      ', '.join('%s %.0f' % (k, v / len(metas)) for k, v in sorted(swaps.items()))), '']
        if any('gameweeks_moved' in m for m in metas):
            lines += ['Fixture order: per copy on average %.1f gameweeks play another gameweek\'s fixtures; '
                      'schedule rule breaks left after the search %d in all copies.' % (
                          np.mean([m.get('gameweeks_moved', 0) for m in metas]),
                          sum(m.get('schedule_rule_breaks', 0) for m in metas)), '']
        lines.append('Rule checks over all copies: ' + (', '.join('%s %d' % (k, v) for k, v in sorted(problems.items()))
                                                      or 'none broken') + '.')
        lines.append('')
    return '\n'.join(lines)


def _names(source):
    names = {}
    path = source.directory / 'players_raw.csv'
    with path.open(encoding='utf-8-sig') as handle:
        for r in csv.DictReader(handle):
            names[int(float(r['id']))] = r.get('web_name') or r.get('second_name')
    return names
