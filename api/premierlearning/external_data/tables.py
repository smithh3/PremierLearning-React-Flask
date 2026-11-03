"""Build the processed CSVs in data/external/ from the cached ESPN responses.

Summaries are streamed from disk one at a time, so memory stays small on the
full 2018+ cache (tens of thousands of matches).
"""
from __future__ import annotations

import csv
import datetime as dt
import os
from collections import defaultdict

from premierlearning.external_data import competitions, espn, fpl_map, fpl_seasons, managers
from premierlearning.external_data.cache import EXTERNAL_DIR

FRIENDLY_NOTICE_DAYS = 28
FIXED_STAGE_NOTICE_DAYS = 14

MATCH_COLUMNS = [
    'event_id', 'kickoff_utc', 'season_year', 'league_id', 'league_slug', 'category', 'stage', 'status', 'completed',
    'neutral', 'has_lineups', 'fpl_season', 'fpl_fixture_id', 'home_team_id', 'home_team', 'home_fpl_team_code',
    'away_team_id', 'away_team', 'away_fpl_team_code', 'home_score', 'away_score', 'home_shootout', 'away_shootout', 'notes']
TEAM_MATCH_COLUMNS = [
    'event_id', 'kickoff_utc', 'known_from_utc', 'season_year', 'league_id', 'league_slug', 'category', 'stage', 'status',
    'completed', 'neutral', 'has_lineups', 'fpl_season', 'fpl_fixture_id', 'team_id', 'team', 'fpl_team_code',
    'opponent_id', 'opponent', 'opponent_fpl_team_code', 'home_away', 'goals_for', 'goals_against',
    'manager_id', 'manager_caretaker']
APPEARANCE_COLUMNS = [
    'event_id', 'kickoff_utc', 'league_slug', 'category', 'fpl_season', 'fpl_fixture_id', 'espn_athlete_id', 'fpl_code',
    'name', 'team_id', 'fpl_team_code', 'home_away', 'team_score', 'opponent_score', 'position', 'in_squad', 'starter',
    'minutes', 'goals', 'assists', 'own_goals', 'yellow_cards', 'red_cards', 'saves', 'shots', 'shots_on_target',
    'goals_for_on_pitch', 'goals_against_on_pitch', 'clean_sheet']


def _write(name, rows, columns):
    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    path = os.path.join(EXTERNAL_DIR, name)
    temp = path + '.tmp'
    with open(temp, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(temp, path)
    print(f'wrote {name}: {len(rows)} rows', flush=True)


def _int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def match_length(status: str) -> int:
    return 120 if ('AET' in status or 'PEN' in status or 'EXTRA' in status) else 90


def on_pitch(player, summary):
    """(start, end) on the sortable match clock, or None for an unused sub."""
    length = match_length(summary.get('status', ''))
    subs = sorted(p['minute'] for p in player['plays'] if p['sub'] and p['minute'] is not None)
    if not subs:  # older matches: fall back on key events, athletes = [on, off]
        subs = sorted(k['minute'] for k in summary['key_events']
                      if 'ubstitution' in k['text'] and k['minute'] is not None and player['athlete_id'] in k['athletes'])
    if player['starter']:
        start = 0
    elif player['subbed_in']:
        if not subs:
            return None
        start = subs[0]
    else:
        return None
    end = length + 0.99  # on at the final whistle: include stoppage-time events (90.04 etc.)
    if player['subbed_out']:
        outs = subs if player['starter'] else subs[1:]
        if outs:
            end = outs[-1]
    reds = [p['minute'] for p in player['plays'] if p['red'] and p['minute'] is not None]
    if reds:
        end = min(end, reds[0])
    start, end = max(0, min(start, length + 0.99)), max(0, min(end, length + 0.99))
    return start, max(start, end)


def minutes_played(player, summary) -> int:
    window = on_pitch(player, summary)
    if window is None:
        return 0
    return max(1, int(window[1]) - int(window[0]))


def _goal_timeline(summary):
    """[(clock, scoring team_id)] from key events, shoot-outs excluded.

    ESPN tags own goals with the team credited with the goal (QA vs FPL goals
    conceded agrees 99.7%), so no special case is needed.
    """
    teams = [c['team_id'] for c in summary['competitors']]
    goals = []
    for k in summary['key_events']:
        text = k['text'].lower()
        if not k['scoring'] or k['minute'] is None or 'shootout' in text or 'penalty - saved' in text or 'missed' in text:
            continue
        goals.append((k['minute'], k['team_id']))
    return goals if all(t in teams for _, t in goals) else None


def _known_from(rows):
    """Earliest time each team-match was surely on the schedule (conservative, for 'upcoming' lists).

    League / group / league-phase: 14 days before the team's first match of that stage.
    Friendlies: 28 days before kickoff.
    Knockouts: kickoff of the team's previous match in the same competition and
    season (you only know you're in the next round once you've won the last one),
    or 28 days before kickoff for the first round they enter.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[(row['team_id'], row['league_id'], row['season_year'])].append(row)
    for items in groups.values():
        items.sort(key=lambda r: r['kickoff_utc'])
        stage_first = {}
        for row in items:
            stage_first.setdefault(row['stage'], row['kickoff_utc'])
        previous = None
        for row in items:
            kick = dt.datetime.strptime(row['kickoff_utc'][:16], '%Y-%m-%dT%H:%M')
            fixed = row['category'] == 'premier_league' or competitions.FIXED_STAGE.search(row['stage'] or '')
            if fixed:
                first = dt.datetime.strptime(stage_first[row['stage']][:16], '%Y-%m-%dT%H:%M')
                known = first - dt.timedelta(days=FIXED_STAGE_NOTICE_DAYS)
            elif row['category'] in ('club_friendly', 'intl_friendly') or previous is None:
                known = kick - dt.timedelta(days=FRIENDLY_NOTICE_DAYS)
            else:
                known = dt.datetime.strptime(previous[:16], '%Y-%m-%dT%H:%M')
            row['known_from_utc'] = known.strftime('%Y-%m-%dT%H:%MZ')
            previous = row['kickoff_utc']


def _appearance_rows(summary, match, athlete_map, team_code):
    timeline = _goal_timeline(summary)
    for side in summary['rosters']:
        ha = side['home_away']
        for p in side['players']:
            mapped = athlete_map.get(p['athlete_id'])
            if not mapped:
                continue
            window = on_pitch(p, summary)
            minutes = minutes_played(p, summary)
            stats = p['stats']
            row = {
                'event_id': match['event_id'], 'kickoff_utc': match['kickoff_utc'], 'league_slug': match['league_slug'],
                'category': match['category'], 'fpl_season': match['fpl_season'], 'fpl_fixture_id': match['fpl_fixture_id'],
                'espn_athlete_id': p['athlete_id'], 'fpl_code': mapped['fpl_code'], 'name': p['name'],
                'team_id': side['team_id'], 'fpl_team_code': team_code.get(side['team_id'], ''), 'home_away': ha,
                'team_score': match['home_score'] if ha == 'home' else match['away_score'],
                'opponent_score': match['away_score'] if ha == 'home' else match['home_score'],
                'position': p['position'], 'in_squad': 1, 'starter': int(p['starter']), 'minutes': minutes,
                'goals': _int(stats.get('totalGoals')), 'assists': _int(stats.get('goalAssists')),
                'own_goals': _int(stats.get('ownGoals')), 'yellow_cards': _int(stats.get('yellowCards')),
                'red_cards': _int(stats.get('redCards')), 'saves': _int(stats.get('saves')),
                'shots': _int(stats.get('totalShots')), 'shots_on_target': _int(stats.get('shotsOnTarget')),
            }
            if window is not None and timeline is not None:
                start, stop = window
                row['goals_for_on_pitch'] = sum(1 for m, t in timeline if t == side['team_id'] and start < m <= stop)
                row['goals_against_on_pitch'] = sum(1 for m, t in timeline if t != side['team_id'] and start < m <= stop)
                row['clean_sheet'] = int(minutes >= 60 and row['goals_against_on_pitch'] == 0)
            yield row


def build(start=espn.FIRST_DAY, end=None):
    leagues = espn.fetch_leagues(None) if os.path.exists(espn.leagues_path()) else {}
    if not leagues:
        raise SystemExit('run the "leagues" step first')
    relevant = sorted(espn.relevant_events(leagues, start, end), key=lambda e: (e['date'], e['id']))

    seasons = fpl_seasons.load_all()
    team_map, event_map = fpl_map.map_teams(relevant, seasons)
    athlete_map = fpl_map.map_players(espn.load_summary, event_map, team_map, seasons, minutes_played)

    league_rows = []
    for lid, info in sorted(leagues.items(), key=lambda kv: int(kv[0])):
        league_rows.append({'league_id': lid, 'league_slug': info['slug'], 'league_name': info['name'],
                            'category': competitions.category(info['slug'], info['name'])})
    _write('leagues.csv', league_rows, ['league_id', 'league_slug', 'league_name', 'category'])
    _write('espn_team_map.csv',
           [{'season_year': y, 'espn_team_id': t, **info} for (y, t), info in sorted(team_map.items())],
           ['season_year', 'espn_team_id', 'espn_name', 'fpl_team_id', 'fpl_team_code', 'fpl_name', 'votes'])
    _write('espn_player_map.csv',
           [{'espn_athlete_id': a, **info} for a, info in sorted(athlete_map.items(), key=lambda kv: kv[1]['fpl_code'])],
           ['espn_athlete_id', 'fpl_code', 'espn_name', 'fpl_name', 'votes', 'total_votes', 'best_score'])

    # ESPN team id -> FPL team code (stable across seasons; a club keeps its ESPN id)
    team_code = {team_id: info['fpl_team_code'] for (year, team_id), info in team_map.items()}

    match_rows, team_rows, n_apps = [], [], 0
    app_path = os.path.join(EXTERNAL_DIR, 'appearances.csv')
    with open(app_path + '.tmp', 'w', encoding='utf-8', newline='') as handle:
        app_writer = csv.DictWriter(handle, fieldnames=APPEARANCE_COLUMNS, extrasaction='ignore')
        app_writer.writeheader()
        for e in relevant:
            league = leagues.get(e['league_id'], {})
            cat = competitions.category(league.get('slug', ''), league.get('name', ''))
            comps = {c['home_away']: c for c in e['competitors']}
            home, away = comps.get('home'), comps.get('away')
            if not home or not away:
                continue
            summary = espn.load_summary(e['id']) if e['completed'] else None
            has_lineups = bool(summary and any(side['players'] for side in summary['rosters']))
            fixture = event_map.get(e['id'])
            base = {
                'event_id': e['id'], 'kickoff_utc': e['date'], 'season_year': espn.season_of(espn._event_day(e)),
                'league_id': e['league_id'], 'league_slug': league.get('slug', ''), 'category': cat,
                'stage': e.get('season_slug', ''), 'status': e['status'], 'completed': int(e['completed']),
                'neutral': int(e['neutral']), 'has_lineups': int(has_lineups),
                'fpl_season': fixture[0] if fixture else '', 'fpl_fixture_id': fixture[1] if fixture else '',
            }
            hs, as_ = (_int(home['score']), _int(away['score'])) if e['completed'] else (None, None)
            match = {**base,
                     'home_team_id': home['team_id'], 'home_team': home['name'], 'home_fpl_team_code': team_code.get(home['team_id'], ''),
                     'away_team_id': away['team_id'], 'away_team': away['name'], 'away_fpl_team_code': team_code.get(away['team_id'], ''),
                     'home_score': hs, 'away_score': as_, 'home_shootout': home.get('shootout'), 'away_shootout': away.get('shootout'),
                     'notes': ' | '.join(e.get('notes', []))}
            match_rows.append(match)
            for side, other, ha in ((home, away, 'home'), (away, home, 'away')):
                team_rows.append({**base, 'team_id': side['team_id'], 'team': side['name'], 'fpl_team_code': team_code.get(side['team_id'], ''),
                                  'opponent_id': other['team_id'], 'opponent': other['name'],
                                  'opponent_fpl_team_code': team_code.get(other['team_id'], ''), 'home_away': ha,
                                  'goals_for': hs if ha == 'home' else as_, 'goals_against': as_ if ha == 'home' else hs})
            if has_lineups:
                for row in _appearance_rows(summary, match, athlete_map, team_code):
                    app_writer.writerow(row)
                    n_apps += 1
    os.replace(app_path + '.tmp', app_path)
    print(f'wrote appearances.csv: {n_apps} rows', flush=True)

    _known_from(team_rows)
    lookup = managers.ManagerLookup(managers.load_spells())
    for row in team_rows:
        spell = lookup.manager_at(row['fpl_team_code'], row['kickoff_utc'][:10]) if row['fpl_team_code'] != '' else None
        row['manager_id'] = spell['manager_id'] if spell else ''
        row['manager_caretaker'] = spell['caretaker'] if spell else ''
    _write('matches.csv', match_rows, MATCH_COLUMNS)
    _write('team_matches.csv', team_rows, TEAM_MATCH_COLUMNS)
