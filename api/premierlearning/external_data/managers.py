"""Premier League manager spells from Wikipedia, linked to FPL team codes.

Sources (Wikipedia parse API, cached raw under data/external/raw/wikipedia/):
  * "List of Premier League managers": every PL manager spell, caretakers included
    (row background #b7e8b9), with ISO dates in data-sort-value attributes.
  * "List of current Premier League and English Football League managers":
    patches recent appointments the first page has not caught up with yet.
manager_id is the Wikipedia article slug, e.g. "Mikel_Arteta", which is stable.
"""
from __future__ import annotations

import csv
import datetime as dt
import html
import os
import re
from urllib.parse import unquote

from premierlearning.external_data.cache import EXTERNAL_DIR, RAW_DIR, Fetcher, read_json_gz, write_json_gz

WIKI_API = 'https://en.wikipedia.org/w/api.php'
ALL_PAGE = 'List_of_Premier_League_managers'
CURRENT_PAGE = 'List_of_current_Premier_League_and_English_Football_League_managers'
CARETAKER_BG = 'b7e8b9'

# FPL team name -> the club name Wikipedia uses
FPL_TO_WIKI = {
    'Man City': 'Manchester City', 'Man Utd': 'Manchester United', 'Spurs': 'Tottenham Hotspur',
    'Wolves': 'Wolverhampton Wanderers', 'Brighton': 'Brighton & Hove Albion', "Nott'm Forest": 'Nottingham Forest',
    'Bournemouth': 'AFC Bournemouth', 'Sheffield Utd': 'Sheffield United', 'West Brom': 'West Bromwich Albion',
    'Newcastle': 'Newcastle United', 'Leeds': 'Leeds United', 'Leicester': 'Leicester City', 'West Ham': 'West Ham United',
    'Luton': 'Luton Town', 'Ipswich': 'Ipswich Town', 'Norwich': 'Norwich City', 'Cardiff': 'Cardiff City',
    'Huddersfield': 'Huddersfield Town', 'Stoke': 'Stoke City', 'Swansea': 'Swansea City', 'Hull': 'Hull City',
}


def _club_key(name: str) -> str:
    name = html.unescape(name).replace('&', 'and')
    name = re.sub(r'\b(F\.?C\.?|A\.?F\.?C\.?)\b', '', name)
    return re.sub(r'[^a-z]', '', name.lower())


def fetch_page(fetcher: Fetcher, page: str, refresh=False) -> str:
    path = os.path.join(RAW_DIR, 'wikipedia', page + '.json.gz')
    if os.path.exists(path) and not refresh:
        return read_json_gz(path)['html']
    data = fetcher.get_json(WIKI_API, {'action': 'parse', 'page': page, 'prop': 'text', 'format': 'json', 'formatversion': 2})
    page_html = data['parse']['text']
    write_json_gz(path, {'fetched': dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z', 'html': page_html})
    return page_html


def _rows(table_html):
    return re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, flags=re.S)


def _cells(row_html):
    return re.findall(r'<(t[hd])([^>]*)>(.*?)</t[hd]>', row_html, flags=re.S)


def _text(cell_html):
    text = re.sub(r'<sup.*?</sup>', '', cell_html, flags=re.S)
    return ' '.join(html.unescape(re.sub(r'<[^>]+>', ' ', text)).split())


def _sort_date(cell_html):
    match = re.search(r'data-sort-value="0*(\d{4}-\d\d-\d\d)', cell_html)
    if match:
        return match.group(1)
    text = _text(cell_html)
    for fmt in ('%d %B %Y', '%B %d, %Y'):
        try:
            return dt.datetime.strptime(text, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return ''


def _link(cell_html):
    match = re.search(r'<a href="/wiki/([^"#]+)"[^>]*>(.*?)</a>', cell_html, flags=re.S)
    if not match:
        return '', _text(cell_html)
    return unquote(html.unescape(match.group(1))), _text(match.group(2))


def parse_all_managers(page_html):
    table = next(t for t in re.findall(r'<table class="wikitable sortable[^"]*">(.*?)</table>', page_html, flags=re.S)
                 if '>Club' in t and '>From' in t and '>Until' in t)
    spells = []
    for row in _rows(table)[1:]:
        cells = _cells(row)
        if len(cells) < 5:
            continue
        (_, name_attr, name_html), _, (_, _, club_html), (_, _, from_html), (_, _, until_html) = cells[:5]
        manager_id, manager_name = _link(name_html)
        until_text = _text(until_html)
        spells.append({
            'manager_id': manager_id, 'manager_name': manager_name, 'club': _link(club_html)[1],
            'from_date': _sort_date(from_html),
            'until_date': '' if until_text.lower().startswith('present') else _sort_date(until_html),
            'caretaker': int(CARETAKER_BG in name_attr.lower()), 'source': ALL_PAGE,
        })
    return spells


def parse_current_managers(page_html):
    """Current PL managers: [(club, manager_id, manager_name, appointed date)]."""
    current = []
    for table in re.findall(r'<table class="wikitable[^"]*"[^>]*>(.*?)</table>', page_html, flags=re.S):
        rows = _rows(table)
        if not rows:
            continue
        columns = [_text(c[2]) for c in _cells(rows[0])]
        find = lambda *names: next((i for i, c in enumerate(columns) if any(c.startswith(n) for n in names)), None)
        i_mgr, i_club, i_date, i_div = find('Manager'), find('Club'), find('From', 'Appointed', 'Date appointed'), find('Division', 'League')
        if None in (i_mgr, i_club, i_date):
            continue
        for row in rows[1:]:
            cells = _cells(row)
            if len(cells) <= max(i_mgr, i_club, i_date):
                continue
            if i_div is not None and _text(cells[i_div][2]) != 'Premier League':
                continue
            manager_id, manager_name = _link(cells[i_mgr][2])
            current.append((_link(cells[i_club][2])[1], manager_id, manager_name, _sort_date(cells[i_date][2])))
    return current


def patch_with_current(spells, current):
    """If a club's current manager is missing from the full list, close the old open spell and add it."""
    added = 0
    for club, manager_id, manager_name, appointed in current:
        key = _club_key(club)
        open_spells = [s for s in spells if _club_key(s['club']) == key and not s['until_date']]
        if any(s['manager_id'] == manager_id for s in open_spells) or not appointed:
            continue
        for s in open_spells:
            s['until_date'] = (dt.date.fromisoformat(appointed) - dt.timedelta(days=1)).isoformat()
        spells.append({'manager_id': manager_id, 'manager_name': manager_name, 'club': club, 'from_date': appointed,
                       'until_date': '', 'caretaker': 0, 'source': CURRENT_PAGE})
        added += 1
    print(f'managers: {added} recent appointment(s) patched in from the current-managers page', flush=True)
    return spells


def club_codes(fpl_seasons) -> dict:
    """club key -> FPL team code, from FPL team names across all local seasons."""
    codes = {}
    for season in fpl_seasons.values():
        for team in season.teams.values():
            codes[_club_key(FPL_TO_WIKI.get(team['name'], team['name']))] = team['code']
    return codes


class ManagerLookup:
    """manager_at(team_code, 'YYYY-MM-DD') -> spell dict or None. Latest-starting spell wins (caretakers)."""

    def __init__(self, spells):
        self.by_code = {}
        for s in spells:
            if s.get('fpl_team_code') in ('', None):
                continue
            self.by_code.setdefault(int(s['fpl_team_code']), []).append(s)
        for items in self.by_code.values():
            items.sort(key=lambda s: s['from_date'])

    def manager_at(self, team_code, day: str):
        best = None
        for s in self.by_code.get(int(team_code), []):
            if s['from_date'] <= day and (not s['until_date'] or day <= s['until_date']):
                best = s
        return best


def build_spells(fetcher: Fetcher, fpl_seasons, refresh=False):
    spells = parse_all_managers(fetch_page(fetcher, ALL_PAGE, refresh))
    try:
        spells = patch_with_current(spells, parse_current_managers(fetch_page(fetcher, CURRENT_PAGE, refresh)))
    except Exception as error:  # the patch is a nice-to-have
        print(f'managers: current-managers page not used ({error!r})', flush=True)
    codes = club_codes(fpl_seasons)
    for s in spells:
        s['fpl_team_code'] = codes.get(_club_key(s['club']), '')
    recent = [s for s in spells if not s['until_date'] or s['until_date'] >= '2018-06-01']
    missing = sorted({s['club'] for s in recent if s['fpl_team_code'] == ''})
    if missing:
        print(f'managers: clubs without an FPL code (fine if never in local FPL data): {missing}', flush=True)
    recent.sort(key=lambda s: (str(s['fpl_team_code']), s['from_date']))
    path = os.path.join(EXTERNAL_DIR, 'manager_spells.csv')
    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['manager_id', 'manager_name', 'club', 'fpl_team_code', 'from_date',
                                                    'until_date', 'caretaker', 'source'])
        writer.writeheader()
        writer.writerows(recent)
    print(f'wrote manager_spells.csv: {len(recent)} spells since 2018-06', flush=True)
    return recent


def load_spells():
    path = os.path.join(EXTERNAL_DIR, 'manager_spells.csv')
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def write_fixture_managers(fpl_seasons, spells):
    """One row per local FPL fixture with both managers at kickoff."""
    lookup = ManagerLookup(spells)
    rows, gaps = [], 0
    for year, season in sorted(fpl_seasons.items()):
        for fid, fx in sorted(season.fixtures.items()):
            day = fx['kickoff'][:10]
            row = {'fpl_season': year, 'fpl_fixture_id': fid, 'kickoff_utc': fx['kickoff']}
            for side, team_id in (('home', fx['team_h']), ('away', fx['team_a'])):
                code = season.teams[team_id]['code']
                spell = lookup.manager_at(code, day)
                row[f'{side}_team_code'] = code
                row[f'{side}_manager_id'] = spell['manager_id'] if spell else ''
                row[f'{side}_caretaker'] = spell['caretaker'] if spell else ''
                row[f'{side}_days_in_post'] = (dt.date.fromisoformat(day) - dt.date.fromisoformat(spell['from_date'])).days if spell else ''
                gaps += spell is None
            rows.append(row)
    columns = ['fpl_season', 'fpl_fixture_id', 'kickoff_utc']
    for side in ('home', 'away'):
        columns += [f'{side}_team_code', f'{side}_manager_id', f'{side}_caretaker', f'{side}_days_in_post']
    with open(os.path.join(EXTERNAL_DIR, 'fixture_managers.csv'), 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote fixture_managers.csv: {len(rows)} fixtures, {gaps} team-fixtures with no manager found', flush=True)
