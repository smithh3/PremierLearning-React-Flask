"""Crawl ESPN's public soccer JSON API and turn it into match / appearance tables.

One `all` scoreboard request per day lists every competition ESPN covers
(club friendlies included). Match summaries, which carry lineups, subs, goals
and assists, are fetched only for matches involving a Premier League club of
that season or a men's national team.

Raw responses are trimmed before saving to keep the cache small:
  data/external/raw/espn/scoreboard/YYYY/YYYYMMDD.json.gz
  data/external/raw/espn/summary/<last 3 digits>/<event_id>.json.gz
"""
from __future__ import annotations

import datetime as dt
import os
import re

from premierlearning.external_data import competitions
from premierlearning.external_data.cache import RAW_DIR, Fetcher, read_json_gz, write_json_gz

SITE = 'https://site.api.espn.com/apis/site/v2/sports/soccer/%s/%s'
CORE_LEAGUES = 'https://sports.core.api.espn.com/v2/sports/soccer/leagues'
ESPN_DIR = os.path.join(RAW_DIR, 'espn')
PREMIER_LEAGUE_ID = '700'
FIRST_DAY = dt.date(2018, 6, 1)
REFRESH_PAST_DAYS = 10
FUTURE_DAYS = 120


def season_of(day: dt.date) -> int:
    """Football season start year. July onwards counts as the next season (pre-season)."""
    return day.year if day.month >= 7 else day.year - 1


def _league_id(uid: str) -> str:
    match = re.search(r'~l:(\d+)', uid or '')
    return match.group(1) if match else ''


# ---------------------------------------------------------------- leagues

def leagues_path():
    return os.path.join(ESPN_DIR, 'leagues.json.gz')


def fetch_leagues(fetcher: Fetcher, refresh=False) -> dict:
    path = leagues_path()
    if os.path.exists(path) and not refresh:
        return read_json_gz(path)
    index = fetcher.get_json(CORE_LEAGUES, {'limit': 1000})
    leagues = {}
    for item in index.get('items', []):
        detail = fetcher.get_json(item['$ref'].replace('http://', 'https://'))
        leagues[str(detail['id'])] = {'slug': detail.get('slug', ''), 'name': detail.get('name', '')}
    write_json_gz(path, leagues)
    print(f'leagues: {len(leagues)}', flush=True)
    return leagues


# ---------------------------------------------------------------- scoreboards

def scoreboard_path(day: dt.date):
    return os.path.join(ESPN_DIR, 'scoreboard', str(day.year), day.strftime('%Y%m%d') + '.json.gz')


def _trim_competitor(comp):
    team = comp.get('team', {})
    return {
        'team_id': str(team.get('id', comp.get('id', ''))),
        'name': team.get('displayName') or team.get('name', ''),
        'abbrev': team.get('abbreviation', ''),
        'home_away': comp.get('homeAway', ''),
        'score': comp.get('score'),
        'shootout': comp.get('shootoutScore'),
        'winner': comp.get('winner'),
    }


def trim_scoreboard_event(event):
    comp = event['competitions'][0]
    status = comp.get('status', event.get('status', {})).get('type', {})
    season = event.get('season', {})
    notes = [n.get('headline', '') for n in comp.get('notes', []) if n.get('headline')]
    return {
        'id': str(event['id']),
        'date': event.get('date'),
        'league_id': _league_id(event.get('uid', '')),
        'season_year': season.get('year'),
        'season_slug': season.get('slug', ''),
        'status': status.get('name', ''),
        'completed': bool(status.get('completed')),
        'neutral': bool(comp.get('neutralSite')),
        'notes': notes,
        'competitors': [_trim_competitor(c) for c in comp.get('competitors', [])],
    }


def fetch_scoreboard(fetcher: Fetcher, day: dt.date, today: dt.date):
    path = scoreboard_path(day)
    fresh = (today - day).days > REFRESH_PAST_DAYS
    if os.path.exists(path) and fresh:
        return read_json_gz(path), False
    data = fetcher.get_json(SITE % ('all', 'scoreboard'), {'dates': day.strftime('%Y%m%d'), 'limit': 1000})
    events = [trim_scoreboard_event(e) for e in data.get('events', [])]
    write_json_gz(path, events)
    return events, True


def crawl_scoreboards(fetcher: Fetcher, start: dt.date = FIRST_DAY, end: dt.date = None):
    today = dt.date.today()
    end = end or today + dt.timedelta(days=FUTURE_DAYS)
    day, fetched, total = start, 0, 0
    while day <= end:
        events, new = fetch_scoreboard(fetcher, day, today)
        fetched += new
        total += len(events)
        if new and (fetched % 50 == 0):
            print(f'scoreboards: up to {day} fetched {fetched} (events so far {total})', flush=True)
        day += dt.timedelta(days=1)
    print(f'scoreboards done: {fetched} fetched, {total} events {start}..{end}', flush=True)


def iter_scoreboard_events(start: dt.date = FIRST_DAY, end: dt.date = None):
    """Every cached event once, de-duplicated by id (an event can sit on two UTC days)."""
    root = os.path.join(ESPN_DIR, 'scoreboard')
    seen = set()
    for year in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        for name in sorted(os.listdir(os.path.join(root, year))):
            if not name.endswith('.json.gz'):
                continue
            day = dt.datetime.strptime(name[:8], '%Y%m%d').date()
            if day < start or (end and day > end):
                continue
            for event in read_json_gz(os.path.join(root, year, name)):
                if event['id'] in seen:
                    continue
                seen.add(event['id'])
                yield event


# ---------------------------------------------------------------- relevance

def premier_league_clubs(events) -> dict:
    """season year -> ESPN team ids that played a Premier League match that season."""
    clubs = {}
    for event in events:
        if event['league_id'] != PREMIER_LEAGUE_ID:
            continue
        year = event.get('season_year') or season_of(_event_day(event))
        for comp in event['competitors']:
            clubs.setdefault(int(year), set()).add(comp['team_id'])
    return clubs


def _event_day(event) -> dt.date:
    return dt.datetime.strptime(event['date'][:10], '%Y-%m-%d').date()


def classify(event, leagues) -> str:
    league = leagues.get(event['league_id'], {})
    return competitions.category(league.get('slug', ''), league.get('name', ''))


def is_relevant(event, leagues, pl_clubs) -> bool:
    cat = classify(event, leagues)
    if cat in ('women', 'college'):
        return False
    if event['league_id'] == PREMIER_LEAGUE_ID or competitions.is_international(cat):
        return True
    tracked = pl_clubs.get(season_of(_event_day(event)), set())
    return any(c['team_id'] in tracked for c in event['competitors'])


# ---------------------------------------------------------------- summaries

def summary_path(event_id: str):
    return os.path.join(ESPN_DIR, 'summary', event_id[-3:].zfill(3), event_id + '.json.gz')


def _minute(clock):
    """Sortable match clock: "45'+2'" -> 45.02, "90'" -> 90.0. None when missing.

    Stoppage time is kept as hundredths so 45+3 sorts before 46 and 90+4 before 91;
    int() of the value gives the base minute for counting minutes played.
    """
    text = (clock or {}).get('displayValue') or ''
    numbers = [int(n) for n in re.findall(r'\d+', text)]
    if not numbers:
        return None
    return numbers[0] + (min(numbers[1], 99) / 100 if len(numbers) > 1 else 0)


def _trim_player(entry):
    athlete = entry.get('athlete', {})
    plays = []
    for play in entry.get('plays', []) or []:
        plays.append({
            'minute': _minute(play.get('clock')),
            'sub': bool(play.get('substitution')),
            'goal': bool(play.get('didScore')),
            'own_goal': bool(play.get('ownGoal')),
            'pen': bool(play.get('penaltyKick')),
            'yellow': bool(play.get('yellowCard')),
            'red': bool(play.get('redCard')),
        })
    return {
        'athlete_id': str(athlete.get('id', '')),
        'name': athlete.get('displayName', ''),
        'full_name': athlete.get('fullName', ''),
        'last_name': athlete.get('lastName', ''),
        'starter': bool(entry.get('starter')),
        'subbed_in': bool(entry.get('subbedIn')),
        'subbed_out': bool(entry.get('subbedOut')),
        'position': (entry.get('position') or {}).get('abbreviation', ''),
        'plays': plays,
        'stats': {s['name']: s.get('value') for s in entry.get('stats', []) if 'name' in s},
    }


def trim_summary(data, event_id):
    header = data.get('header', {})
    comp = (header.get('competitions') or [{}])[0]
    league = header.get('league', {})
    key_events = []
    for item in data.get('keyEvents', []) or []:
        key_events.append({
            'type': (item.get('type') or {}).get('type') or (item.get('type') or {}).get('text', ''),
            'text': (item.get('type') or {}).get('text', ''),
            'minute': _minute(item.get('clock')),
            'period': (item.get('period') or {}).get('number'),
            'team_id': str((item.get('team') or {}).get('id', '')),
            'scoring': bool(item.get('scoringPlay')),
            'own_goal': bool(item.get('ownGoal')),
            'athletes': [str((p.get('athlete') or {}).get('id', '')) for p in item.get('participants', []) or []],
        })
    rosters = []
    for side in data.get('rosters', []) or []:
        rosters.append({
            'team_id': str((side.get('team') or {}).get('id', '')),
            'home_away': side.get('homeAway', ''),
            'formation': side.get('formation', ''),
            'players': [_trim_player(p) for p in side.get('roster', []) or []],
        })
    return {
        'id': str(event_id),
        'date': comp.get('date'),
        'league': {'id': str(league.get('id', '')), 'slug': league.get('slug', ''), 'name': league.get('name', '')},
        'status': ((comp.get('status') or {}).get('type') or {}).get('name', ''),
        'competitors': [_trim_competitor(c) for c in comp.get('competitors', [])],
        'rosters': rosters,
        'key_events': key_events,
    }


def fetch_summary(fetcher: Fetcher, event, leagues):
    path = summary_path(event['id'])
    if os.path.exists(path):
        return read_json_gz(path), False
    slug = leagues.get(event['league_id'], {}).get('slug') or 'all'
    data = fetcher.get_json(SITE % (slug, 'summary'), {'event': event['id']})
    trimmed = trim_summary(data, event['id'])
    write_json_gz(path, trimmed)
    return trimmed, True


def relevant_events(leagues, start: dt.date = FIRST_DAY, end: dt.date = None):
    """Two streaming passes over the cache (all events would not fit comfortably in RAM)."""
    pl_clubs = premier_league_clubs(e for e in iter_scoreboard_events(start, end) if e['league_id'] == PREMIER_LEAGUE_ID)
    relevant = [e for e in iter_scoreboard_events(start, end) if is_relevant(e, leagues, pl_clubs)]
    print(f'{len(relevant)} relevant events; PL clubs per season: { {y: len(c) for y, c in sorted(pl_clubs.items())} }', flush=True)
    return relevant


def load_summary(event_id: str):
    path = summary_path(event_id)
    return read_json_gz(path) if os.path.exists(path) else None


def crawl_summaries(fetcher: Fetcher, leagues, start: dt.date = FIRST_DAY, end: dt.date = None):
    todo = [e for e in relevant_events(leagues, start, end) if e['completed']]
    print(f'summaries: {len(todo)} relevant finished matches', flush=True)
    fetched = 0
    for index, event in enumerate(todo, 1):
        try:
            _, new = fetch_summary(fetcher, event, leagues)
        except Exception as error:  # one broken match must not stop the crawl
            print(f'  summary {event["id"]} failed: {error!r}', flush=True)
            continue
        fetched += new
        if new and fetched % 200 == 0:
            print(f'summaries: {index}/{len(todo)} checked, {fetched} fetched (at {event["date"][:10]})', flush=True)
    print(f'summaries done: {fetched} fetched, {len(todo)} relevant', flush=True)
