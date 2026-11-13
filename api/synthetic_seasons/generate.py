"""Build synthetic copies of a real season and save them.

Each copy is saved under data/synthetic/season_XX_YY/copy_NNN/:
- player_matches.csv.gz: every player's match lines, same columns as the real
  players/*/gw.csv (prices, ownership and transfers stay real);
- fixtures.csv: the real fixtures with the drawn scores and stats, in the
  copy's gameweeks and kick-off times (schedule.py);
- schedule.json: which real gameweek's fixtures are played in which
  gameweek, and for each team's moved date the fixture it now plays;
- flags.json: {gw: {element: FPL news entry}} for the flags that differ from
  the real fpl_news/chance_of_playing files (moved, removed or added injuries);
- meta.json: seed, spells moved/removed/added, team form, swap counts.
Copy k of season S uses seed S * 1000 + k, so it can be rebuilt exactly.
"""
from __future__ import annotations

import csv
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .events import SeasonModel, calibration, draw_matches, team_form
from .injuries import Squads, apply_minutes, find_spells, flag_overrides, plan_injuries
from .schedule import SCHEDULE_FILE, apply_reshuffle, draw_reshuffle
from .source import DATA_ROOT, load_source, season_name

STATS = ('goals_scored', 'assists', 'own_goals', 'penalties_saved', 'penalties_missed',
         'yellow_cards', 'red_cards', 'saves', 'bonus', 'bps', 'defensive_contribution')


def copy_dir(year: int, index: int, data_root: Path = None) -> Path:
    root = Path(data_root) if data_root else DATA_ROOT
    return root / 'synthetic' / season_name(year) / ('copy_%03d' % int(index))


def seed_for(year: int, index: int) -> int:
    return int(year) % 100 * 1000 + int(index)


class SeasonCopier:
    """Loads a real season once and makes as many copies as asked."""

    def __init__(self, year: int, data_root: Path = None, calib: dict = None):
        self.data_root = data_root
        self.source = load_source(year, data_root)
        self.model = SeasonModel(self.source, calib or calibration(data_root))
        self.spells = find_spells(Squads(self.source.rows, self.source))

    def make(self, index: int) -> dict:
        source = self.source
        rng = np.random.default_rng(seed_for(source.year, index))
        rows = [dict(row) for row in source.rows]
        squads = Squads(rows, source)
        plan = plan_injuries(squads, self.spells, rng, self.model.proneness)
        swaps = apply_minutes(squads, plan, rng)
        overrides = flag_overrides(squads, plan)
        # Its own stream, so the fixture order can change without moving the
        # injury and match draws onto other random numbers.
        reshuffle = draw_reshuffle(source, np.random.default_rng([seed_for(source.year, index), 1]))
        apply_reshuffle(rows, reshuffle, self.model.match_xg)
        form = team_form(self.model, rng)
        scores = draw_matches(self.model, Squads(rows, source), form, rng, fixtures=reshuffle.fixtures)
        meta = {
            'source_season': source.name, 'copy': int(index), 'seed': seed_for(source.year, index),
            'real_spells': len(self.spells), 'moved': len(plan.moved), 'removed': len(plan.removed),
            'added': len(plan.added), 'swaps': swaps,
            'spells': (
                [{'kind': 'moved', 'element': s.element, 'team': s.team, 'from_gw': s.first_gw,
                  'to_gw': squads.lines[(s.element, s.team)][start]['_gw'],
                  'to_last_gw': squads.lines[(s.element, s.team)][start + s.length - 1]['_gw'], 'matches': s.length}
                 for s, start in plan.moved]
                + [{'kind': 'removed', 'element': s.element, 'team': s.team, 'from_gw': s.first_gw,
                    'matches': s.length} for s in plan.removed]
                + [{'kind': 'added', 'element': element, 'team': team, 'template': t.element,
                    'to_gw': squads.lines[(element, team)][start]['_gw'],
                    'to_last_gw': squads.lines[(element, team)][start + t.length - 1]['_gw'], 'matches': t.length}
                   for t, element, team, start in plan.added]
            ),
            'form_start': {kind: {str(team): round(float(values[1]), 3) for team, values in teams.items()}
                           for kind, teams in form.items()},
            'gameweeks_moved': sum(1 for gw, slot in reshuffle.slots.items() if gw != slot),
            'schedule_rule_breaks': reshuffle.cost,
        }
        return {'rows': rows, 'scores': scores, 'flags': overrides, 'meta': meta, 'reshuffle': reshuffle}

    def save(self, index: int, copy: dict) -> Path:
        directory = copy_dir(self.source.year, index, self.data_root)
        directory.mkdir(parents=True, exist_ok=True)
        # Forecasts made from an earlier version of this copy no longer fit it.
        stale = directory / 'forecasts.csv.gz'
        if stale.is_file():
            stale.unlink()
        self._write_rows(directory / 'player_matches.csv.gz', copy['rows'])
        self._write_fixtures(directory / 'fixtures.csv', copy['rows'], copy['scores'], copy['reshuffle'].fixtures)
        (directory / SCHEDULE_FILE).write_text(json.dumps(copy['reshuffle'].to_json()), encoding='utf-8')
        (directory / 'flags.json').write_text(json.dumps(
            {str(gw): {str(e): entry for e, entry in players.items()} for gw, players in sorted(copy['flags'].items())}
        ), encoding='utf-8')
        (directory / 'meta.json').write_text(json.dumps(copy['meta'], indent=1), encoding='utf-8')
        return directory

    def _write_rows(self, path: Path, rows):
        source = self.source
        columns = source.row_columns
        with gzip.open(path, 'wt', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in sorted(rows, key=lambda r: (r['element'], r['_kickoff'], r['fixture'])):
                out = dict(row)
                if source.has_starts:
                    out['starts'] = int(bool(row['_started']))
                if source.has_xg:
                    out['expected_goals'] = round(row.get('_xg', 0.0), 2)
                    out['expected_assists'] = round(row.get('_xa', 0.0), 2)
                    out['expected_goal_involvements'] = round(
                        row.get('_xgi', row.get('_xg', 0.0) + row.get('_xa', 0.0)), 2)
                    out['expected_goals_conceded'] = round(row.get('_xgc', 0.0), 2)
                writer.writerow([_cell(out.get(name)) for name in columns])

    def _write_fixtures(self, path: Path, rows, scores, fixtures):
        by_fixture = defaultdict(list)
        for row in rows:
            by_fixture[row['fixture']].append(row)
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.source.fixture_columns)
            writer.writeheader()
            for fixture in sorted(fixtures.values(), key=lambda f: f.id):
                raw = dict(fixture.raw)
                if fixture.id in scores:
                    raw['team_h_score'], raw['team_a_score'] = scores[fixture.id]
                    raw['stats'] = repr(_stats(fixture, by_fixture[fixture.id]))
                writer.writerow(raw)


def _stats(fixture, rows):
    result = []
    for name in STATS:
        entry = {'identifier': name, 'a': [], 'h': []}
        for row in sorted(rows, key=lambda r: -int(r.get(name) or 0)):
            value = int(row.get(name) or 0)
            if value and row['minutes'] > 0:
                entry['h' if row['_home'] else 'a'].append({'value': value, 'element': row['element']})
        result.append(entry)
    return result


def _cell(value):
    if value is None:
        return ''
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return ('%.2f' % value).rstrip('0').rstrip('.')
    return value


def generate(years, copies: int, start: int = 1, data_root: Path = None, verbose=True):
    calib = calibration(data_root)
    written = []
    for year in years:
        began = time.time()
        copier = SeasonCopier(year, data_root, calib)
        if verbose:
            print('%s: loaded, %d real injury spells (%.0fs)' % (season_name(year), len(copier.spells), time.time() - began),
                  flush=True)
        for index in range(start, start + copies):
            began = time.time()
            copy = copier.make(index)
            directory = copier.save(index, copy)
            meta = copy['meta']
            goals = sum(h + a for h, a in copy['scores'].values())
            if verbose:
                print('  copy %03d: %d goals, spells moved %d removed %d added %d, swaps %s -> %s (%.0fs)' % (
                    index, goals, meta['moved'], meta['removed'], meta['added'], meta['swaps'], directory,
                    time.time() - began), flush=True)
            written.append(directory)
    return written
