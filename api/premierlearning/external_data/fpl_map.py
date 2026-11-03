"""Link ESPN teams, events and athletes to FPL team codes, fixtures and player codes.

Premier League matches exist in both sources, so they act as the bridge:
  1. Teams: ESPN PL events and FPL fixtures with the same kickoff vote on which
     ESPN team id is which FPL team.
  2. Events: with teams known, each ESPN PL event maps to exactly one FPL fixture.
  3. Players: within one mapped fixture and one side, ESPN players who played are
     matched to FPL players who played by name (+ minutes as a tie-break). Votes
     across all their PL matches give each ESPN athlete id one FPL `code`.
The athlete id is then valid for every competition (cups, Europe, internationals).
"""
from __future__ import annotations

import datetime as dt
import difflib
import unicodedata
from collections import Counter, defaultdict

PREMIER_LEAGUE_ID = '700'


def _parse_time(text):
    """Both sources give UTC ("...T19:00Z" / "...T19:00:00Z"); minute precision is enough."""
    return dt.datetime.strptime(text[:16], '%Y-%m-%dT%H:%M')


def norm(text: str) -> str:
    text = unicodedata.normalize('NFKD', text or '')
    text = ''.join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = text.replace('ø', 'o').replace('ß', 'ss').replace('ł', 'l').replace('æ', 'ae').replace('đ', 'd').replace('ı', 'i')
    return ' '.join(''.join(ch if 'a' <= ch <= 'z' else ' ' for ch in text).split())


def _ratio(a, b):
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def name_score(espn: dict, fpl: dict) -> float:
    e_full = norm(espn.get('full_name') or espn.get('name'))
    e_disp = norm(espn.get('name'))
    e_last = norm(espn.get('last_name'))
    f_full = norm(f"{fpl['first_name']} {fpl['second_name']}")
    f_web = norm(fpl['web_name'])
    f_second = norm(fpl['second_name'])
    best = max(
        _ratio(e_full, f_full), _ratio(e_disp, f_full), _ratio(e_disp, f_web),
        _ratio(e_last, f_web), _ratio(e_last, f_second), _ratio(e_disp, f_second),
    )
    e_tokens = set(e_full.split()) | set(e_disp.split())
    f_tokens = set(f_full.split()) | set(f_web.split())
    if e_last and (e_last in f_tokens or e_last == f_web):
        best = max(best, 0.85)
    shared = {t for t in e_tokens & f_tokens if len(t) > 2}
    if shared:
        best = min(1.0, best + 0.1 * len(shared))
    return best


# ---------------------------------------------------------------- teams / events

# FPL short team names -> full club names (as ESPN and Wikipedia write them)
FPL_TEAM_FULL_NAME = {
    'Man City': 'Manchester City', 'Man Utd': 'Manchester United', 'Spurs': 'Tottenham Hotspur',
    'Wolves': 'Wolverhampton Wanderers', 'Brighton': 'Brighton & Hove Albion', "Nott'm Forest": 'Nottingham Forest',
    'Bournemouth': 'AFC Bournemouth', 'Sheffield Utd': 'Sheffield United', 'West Brom': 'West Bromwich Albion',
    'Newcastle': 'Newcastle United', 'Leeds': 'Leeds United', 'Leicester': 'Leicester City', 'West Ham': 'West Ham United',
    'Luton': 'Luton Town', 'Ipswich': 'Ipswich Town', 'Norwich': 'Norwich City', 'Cardiff': 'Cardiff City',
    'Huddersfield': 'Huddersfield Town', 'Stoke': 'Stoke City', 'Swansea': 'Swansea City', 'Hull': 'Hull City',
}


def team_name_score(espn_name: str, fpl_name: str) -> float:
    a = norm(espn_name.replace('&', 'and'))
    b = norm(FPL_TEAM_FULL_NAME.get(fpl_name, fpl_name).replace('&', 'and'))
    return 1.0 if a == b else _ratio(a, b)

def map_teams(espn_events, fpl_seasons):
    """Returns ({(season, espn_team_id): fpl team info}, {espn_event_id: (season, fixture_id)})."""
    team_map, event_map = {}, {}
    pl_events = [e for e in espn_events if e['league_id'] == PREMIER_LEAGUE_ID and len(e['competitors']) == 2]
    by_season = defaultdict(list)
    for event in pl_events:
        by_season[int(event['season_year'])].append(event)
    for year, events in sorted(by_season.items()):
        season = fpl_seasons.get(year)
        if not season:
            continue
        by_time = defaultdict(list)
        for fid, fx in season.fixtures.items():
            by_time[_parse_time(fx['kickoff'])].append(fid)
        votes = Counter()
        for event in events:
            kick = _parse_time(event['date'])
            candidates = by_time.get(kick, [])
            if not candidates:
                continue
            home = next(c for c in event['competitors'] if c['home_away'] == 'home')
            away = next(c for c in event['competitors'] if c['home_away'] == 'away')
            # Several fixtures share a 15:00 kickoff: names pick the right one,
            # the kickoff match stops a name lookalike being chosen on its own.
            scored = sorted(((team_name_score(home['name'], season.teams[season.fixtures[f]['team_h']]['name'])
                              + team_name_score(away['name'], season.teams[season.fixtures[f]['team_a']]['name']), f)
                             for f in candidates), reverse=True)
            best_score, fid = scored[0]
            if best_score < 0.9 or (len(scored) > 1 and scored[1][0] > best_score - 0.3):
                continue
            fx = season.fixtures[fid]
            votes[(home['team_id'], fx['team_h'])] += 1
            votes[(away['team_id'], fx['team_a'])] += 1
        names = {c['team_id']: c['name'] for e in events for c in e['competitors']}
        taken = set()
        for (espn_id, fpl_id), count in votes.most_common():
            if (year, espn_id) in team_map or fpl_id in taken:
                continue
            team = season.teams[fpl_id]
            team_map[(year, espn_id)] = {'fpl_team_id': fpl_id, 'fpl_team_code': team['code'], 'fpl_name': team['name'],
                                         'espn_name': names.get(espn_id, ''), 'votes': round(count, 1)}
            taken.add(fpl_id)
        fixture_by_teams = {(fx['team_h'], fx['team_a']): fid for fid, fx in season.fixtures.items()}
        for event in events:
            home = next(c for c in event['competitors'] if c['home_away'] == 'home')
            away = next(c for c in event['competitors'] if c['home_away'] == 'away')
            h, a = team_map.get((year, home['team_id'])), team_map.get((year, away['team_id']))
            if h and a:
                fid = fixture_by_teams.get((h['fpl_team_id'], a['fpl_team_id']))
                if fid is not None:
                    event_map[event['id']] = (year, fid)
        unmapped = sorted({names[t] for e in events for c in e['competitors'] for t in [c['team_id']] if (year, t) not in team_map})
        print(f'teams {year}: {sum(1 for k in team_map if k[0] == year)} mapped, events mapped '
              f'{sum(1 for e in events if e["id"] in event_map)}/{len(events)}' + (f', unmapped {unmapped}' if unmapped else ''), flush=True)
    return team_map, event_map


# ---------------------------------------------------------------- players

def map_players(load_summary, event_map, team_map, fpl_seasons, minutes_of):
    """load_summary(event_id) -> trimmed summary or None. minutes_of(player, summary) -> minutes.

    Returns {athlete_id: {fpl_code, votes, total, name, fpl_name, score}}.
    """
    played = defaultdict(list)  # (year, fixture, was_home) -> [(element, minutes)]
    for year, season in fpl_seasons.items():
        for element, fixture, minutes, was_home in season.appearances:
            if minutes > 0:
                played[(year, fixture, was_home)].append((element, minutes))

    votes = defaultdict(Counter)
    best_score = defaultdict(float)
    names = {}
    for event_id, (year, fixture) in event_map.items():
        summary = load_summary(event_id)
        if not summary:
            continue
        season = fpl_seasons[year]
        for side in summary['rosters']:
            home = side['home_away'] == 'home'
            fpl_side = played.get((year, fixture, home), [])
            espn_side = [(p, minutes_of(p, summary)) for p in side['players']]
            espn_side = [(p, m) for p, m in espn_side if m > 0 and p['athlete_id']]
            pairs = []
            for p, em in espn_side:
                for element, fm in fpl_side:
                    fpl = season.players.get(element)
                    if not fpl:
                        continue
                    score = name_score(p, fpl)
                    diff = abs(em - fm)
                    score += 0.15 if diff <= 3 else (0.05 if diff <= 10 else -0.1)
                    pairs.append((score, p['athlete_id'], element, fpl, p))
            pairs.sort(key=lambda x: -x[0])
            used_a, used_e = set(), set()
            for score, athlete, element, fpl, p in pairs:
                if athlete in used_a or element in used_e or score < 0.62:
                    continue
                used_a.add(athlete)
                used_e.add(element)
                votes[athlete][fpl['code']] += 1
                best_score[(athlete, fpl['code'])] = max(best_score[(athlete, fpl['code'])], score)
                names[athlete] = p.get('full_name') or p['name']
                names[(athlete, fpl['code'])] = f"{fpl['first_name']} {fpl['second_name']}"

    mapping = {}
    for athlete, counter in votes.items():
        code, count = counter.most_common(1)[0]
        total = sum(counter.values())
        if count / total < 0.7:
            continue
        mapping[athlete] = {'fpl_code': code, 'votes': count, 'total_votes': total, 'espn_name': names[athlete],
                            'fpl_name': names[(athlete, code)], 'best_score': round(best_score[(athlete, code)], 3)}
    print(f'players: {len(mapping)} ESPN athletes mapped to {len({m["fpl_code"] for m in mapping.values()})} FPL codes', flush=True)
    return mapping
