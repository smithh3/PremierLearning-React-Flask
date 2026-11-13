"""Write the live season in the vaastav CSV layout, so it can be replayed like a past one.

    python -m simulation.current_season_csv --season 26

Downloads FPL's bootstrap-static, fixtures and every player's element-summary
(numbers straight from FPL), keeps the raw JSON under
data/season_XX_YY/replay_api/<date>/, and writes next to the live files:

  fixtures.csv, teams.csv, players_raw.csv, player_idlist.csv,
  players/<first>_<second>_<id>/gw.csv   (one row per played fixture)
  gws/gwN.csv                            (every player's rows for gameweek N)

with the columns and value formatting vaastav/Fantasy-Premier-League uses
(None, True/False, lists as Python reprs), so simulation.historical_data
reads it as it reads 2025/26. Only finished gameweeks get gws files.

The live pipeline reads fantasy.json, fixtures.json and players/player_N.json;
none of those are touched. Run it again to refresh after the next gameweek.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from datetime import date
from pathlib import Path

API = 'https://fantasy.premierleague.com/api'
DATA_ROOT = Path(__file__).resolve().parents[1] / 'data'
POSITIONS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
# vaastav's gws/gwN.csv: these four, then the history fields in sorted order.
GW_LEAD = ('name', 'position', 'team', 'xP')


def _get(url, attempts=4):
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception:  # noqa: BLE001 - retried, then raised
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _text(value):
    """vaastav's formatting: str() of the JSON value."""
    return str(value)


def _write(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _text(row.get(key)) if key in row else '' for key in fieldnames})


def download(season_dir, delay=0.15):
    raw_dir = season_dir / 'replay_api' / date.today().isoformat()
    raw_dir.mkdir(parents=True, exist_ok=True)
    bootstrap = _get(API + '/bootstrap-static/')
    fixtures = _get(API + '/fixtures/')
    (raw_dir / 'bootstrap.json').write_text(json.dumps(bootstrap), encoding='utf-8')
    (raw_dir / 'fixtures.json').write_text(json.dumps(fixtures), encoding='utf-8')
    summaries = {}
    for count, element in enumerate(bootstrap['elements'], start=1):
        player_id = int(element['id'])
        summaries[player_id] = _get('%s/element-summary/%d/' % (API, player_id))
        if count % 100 == 0:
            print('  %d of %d players downloaded' % (count, len(bootstrap['elements'])), flush=True)
        time.sleep(delay)
    (raw_dir / 'element_summaries.json').write_text(json.dumps(summaries), encoding='utf-8')
    print('Raw FPL data saved to %s' % raw_dir, flush=True)
    return bootstrap, fixtures, summaries


def write_csvs(season_dir, bootstrap, fixtures, summaries):
    teams = {int(team['id']): team for team in bootstrap['teams']}
    elements = {int(element['id']): element for element in bootstrap['elements']}

    _write(season_dir / 'fixtures.csv', (
        'code', 'event', 'finished', 'finished_provisional', 'id', 'kickoff_time', 'minutes',
        'provisional_start_time', 'started', 'team_a', 'team_a_score', 'team_h', 'team_h_score',
        'stats', 'team_h_difficulty', 'team_a_difficulty', 'pulse_id'), fixtures)
    team_fields = list(bootstrap['teams'][0])
    _write(season_dir / 'teams.csv', team_fields, bootstrap['teams'])
    element_fields = sorted(bootstrap['elements'][0])
    _write(season_dir / 'players_raw.csv', element_fields, bootstrap['elements'])
    _write(season_dir / 'player_idlist.csv', ('first_name', 'second_name', 'id'), bootstrap['elements'])

    finished = {int(event['id']) for event in bootstrap['events'] if event.get('finished')}
    history_fields = set()
    by_gameweek = {}
    players_dir = season_dir / 'players'
    for player_id, summary in sorted(summaries.items()):
        element = elements.get(int(player_id))
        history = summary.get('history') or []
        if element is None:
            continue
        folder = ('%s_%s_%d' % (element['first_name'], element['second_name'], int(player_id))).replace(' ', '_')
        fields = sorted({key for row in history for key in row}) or ['element']
        history_fields.update(fields)
        _write(players_dir / folder / 'gw.csv', fields, history)
        for row in history:
            gameweek = int(row['round'])
            if gameweek not in finished:
                continue
            out = dict(row)
            out['name'] = '%s %s' % (element['first_name'], element['second_name'])
            out['position'] = POSITIONS.get(int(element['element_type']), '')
            out['team'] = teams[int(element['team'])]['name']
            out['xP'] = 0.0
            by_gameweek.setdefault(gameweek, []).append(out)
    gw_fields = list(GW_LEAD) + sorted(history_fields)
    for gameweek, rows in sorted(by_gameweek.items()):
        _write(season_dir / 'gws' / ('gw%d.csv' % gameweek), gw_fields, rows)
    print('Wrote %d players and gameweeks %s to %s' % (
        len(summaries), ', '.join(str(gw) for gw in sorted(by_gameweek)), season_dir), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='simulation.current_season_csv')
    parser.add_argument('--season', type=int, default=26)
    parser.add_argument('--from-raw', help='A replay_api/<date> folder to convert instead of downloading')
    args = parser.parse_args(argv)
    season = int(args.season) % 100
    season_dir = DATA_ROOT / ('season_%02d_%02d' % (season, season + 1))
    if args.from_raw:
        raw = Path(args.from_raw)
        bootstrap = json.loads((raw / 'bootstrap.json').read_text(encoding='utf-8'))
        fixtures = json.loads((raw / 'fixtures.json').read_text(encoding='utf-8'))
        summaries = {int(key): value for key, value in json.loads(
            (raw / 'element_summaries.json').read_text(encoding='utf-8')).items()}
    else:
        bootstrap, fixtures, summaries = download(season_dir)
    write_csvs(season_dir, bootstrap, fixtures, summaries)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
