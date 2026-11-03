"""Quality checks: ESPN-derived stats vs official FPL stats on the same PL matches.

python -m premierlearning.external_data.qa   (after `build`)
Writes data/external/qa_report.txt and prints it.
"""
from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict

from premierlearning.external_data import fpl_seasons
from premierlearning.external_data.cache import EXTERNAL_DIR


def _rows(name):
    with open(os.path.join(EXTERNAL_DIR, name), encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _fpl_stats(seasons):
    """(season, fixture, fpl code) -> dict of official stats, from the per-gw CSV / JSON rows."""
    out = {}
    for year, path in fpl_seasons.season_folders():
        if year not in seasons:
            continue
        season = seasons[year]
        merged = os.path.join(path, 'gws', 'merged_gw.csv')
        if os.path.exists(merged):
            with open(merged, encoding='utf-8', errors='replace', newline='') as handle:
                for r in csv.DictReader(handle):
                    try:
                        code = season.players[int(r['element'])]['code']
                        out[(year, int(r['fixture']), code)] = {k: int(float(r[k])) for k in ('minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded')}
                    except (KeyError, ValueError):
                        continue
        else:
            import glob
            import json
            for file in glob.glob(os.path.join(path, 'players', 'player_*.json')):
                with open(file, encoding='utf-8') as handle:
                    for r in json.load(handle).get('history', []):
                        code = season.players.get(int(r['element']), {}).get('code')
                        if code:
                            out[(year, int(r['fixture']), code)] = {k: int(r[k]) for k in ('minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded')}
    return out


def report():
    seasons = fpl_seasons.load_all()
    official = _fpl_stats(seasons)
    matches = _rows('matches.csv')
    lines = []
    say = lines.append

    compared = Counter()
    minute_err, exact = [], Counter()
    have = set()
    cats = Counter()
    with open(os.path.join(EXTERNAL_DIR, 'appearances.csv'), encoding='utf-8', newline='') as handle:
        for a in csv.DictReader(handle):  # streamed: the full file is large
            if not a['fpl_fixture_id']:
                if int(a['minutes']) > 0:
                    cats[a['category']] += 1
                continue
            have.add((int(a['fpl_season']), int(a['fpl_fixture_id']), int(a['fpl_code'])))
            _compare(a, official, compared, minute_err, exact)
    _summarise(say, compared, minute_err, exact)
    _rest(say, official, matches, have, cats)
    _finish(say, lines, matches)


def _compare(a, official, compared, minute_err, exact):
    ref = official.get((int(a['fpl_season']), int(a['fpl_fixture_id']), int(a['fpl_code'])))
    if ref is None:
        compared['no_fpl_row'] += 1
        return
    compared['rows'] += 1
    m = int(a['minutes'])
    minute_err.append(abs(m - ref['minutes']))
    exact['played_agrees'] += (m > 0) == (ref['minutes'] > 0)
    exact['goals'] += (a['goals'] or '0') == str(ref['goals_scored'])
    exact['assists'] += (a['assists'] or '0') == str(ref['assists'])
    if a['clean_sheet'] != '':
        compared['cs_rows'] += 1
        exact['clean_sheet'] += int(a['clean_sheet']) == ref['clean_sheets']
    if a['goals_against_on_pitch'] != '' and m > 0:
        compared['gc_rows'] += 1
        exact['goals_conceded_on_pitch'] += int(a['goals_against_on_pitch']) == ref['goals_conceded']


def _summarise(say, compared, minute_err, exact):
    n = max(compared['rows'], 1)
    say('== ESPN vs FPL on the same Premier League matches')
    say(f"appearance rows compared: {compared['rows']} (no FPL row: {compared['no_fpl_row']})")
    if minute_err:
        minute_err.sort()
        say(f"minutes |error|: median {minute_err[len(minute_err) // 2]}, 90th pct {minute_err[int(len(minute_err) * .9)]}, "
            f"within 5 min {sum(e <= 5 for e in minute_err) / len(minute_err):.1%}")
    for k in ('played_agrees', 'goals', 'assists'):
        say(f'{k} exact: {exact[k] / n:.2%}')
    if compared['cs_rows']:
        say(f"clean_sheet exact: {exact['clean_sheet'] / compared['cs_rows']:.2%} of {compared['cs_rows']}")
    if compared['gc_rows']:
        say(f"goals conceded on pitch exact: {exact['goals_conceded_on_pitch'] / compared['gc_rows']:.2%} of {compared['gc_rows']}")


def _rest(say, official, matches, have, cats):
    # coverage: FPL players who played in mapped fixtures, are they in appearances?
    mapped_fixtures = {(int(m['fpl_season']), int(m['fpl_fixture_id'])) for m in matches if m['fpl_fixture_id']}
    played = [k for k, v in official.items() if (k[0], k[1]) in mapped_fixtures and v['minutes'] > 0]
    found = sum(k in have for k in played)
    say(f'coverage: {found}/{len(played)} FPL player-matches with minutes>0 found in ESPN appearances ({found / max(len(played), 1):.1%})')

    say('')
    say('== Matches by category and lineup availability')
    by_cat = defaultdict(Counter)
    for m in matches:
        if m['completed'] == '1':
            by_cat[m['category']]['done'] += 1
            by_cat[m['category']]['lineups'] += int(m['has_lineups'])
        else:
            by_cat[m['category']]['scheduled'] += 1
    for cat, c in sorted(by_cat.items(), key=lambda kv: -kv[1]['done']):
        say(f"{cat:22s} finished {c['done']:6d}  with lineups {c['lineups'] / max(c['done'], 1):6.1%}  scheduled {c['scheduled']}")

    say('')
    say('== Non-PL appearances of mapped FPL players by category')
    for cat, count in cats.most_common():
        say(f'{cat:22s} {count}')


def _finish(say, lines, matches):
    say('')
    say('== Club friendlies of PL clubs: lineup coverage by season')
    per = defaultdict(Counter)
    for m in matches:
        if m['category'] == 'club_friendly' and m['completed'] == '1':
            per[m['season_year']]['done'] += 1
            per[m['season_year']]['lineups'] += int(m['has_lineups'])
    for year, c in sorted(per.items()):
        say(f"{year}: {c['done']:4d} friendlies, {c['lineups'] / max(c['done'], 1):.0%} with lineups")

    weak = [r for r in _rows('espn_player_map.csv') if float(r['best_score']) < 0.8]
    say('')
    say(f'== Weakest player matches (best_score < 0.8): {len(weak)}')
    for r in sorted(weak, key=lambda r: float(r['best_score']))[:40]:
        say(f"  {r['espn_name']!r:35} -> {r['fpl_name']!r:35} score {r['best_score']} votes {r['votes']}/{r['total_votes']}")

    text = '\n'.join(lines)
    with open(os.path.join(EXTERNAL_DIR, 'qa_report.txt'), 'w', encoding='utf-8') as handle:
        handle.write(text + '\n')
    print(text)


if __name__ == '__main__':
    report()
