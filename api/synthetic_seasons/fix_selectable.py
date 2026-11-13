"""Re-flag which players could be picked, after the can_select fix (2026-11-11).

The 24/25 and 25/26 players_raw.csv were downloaded after those seasons, when
FPL had set can_select False for everyone. VaastavSeasonData.known_unavailable
read that as deadline news, so most players who played were stored as not
selectable late in those seasons. The forecasts themselves never used the flag
(only the manager and the replays do), so this rewrites the flag alone:

- prediction set 97 (lgbm_v6) -> a new set, same rows with the corrected
  `selectable`; set 97 itself is left as it is;
- each copy's forecasts.csv.gz, in place (the old file is kept as
  forecasts_selectbug.csv.gz).

Each step first recomputes the OLD flag the same way and checks it matches
what is stored, so only the can_select change can differ.

    python -m synthetic_seasons.fix_selectable set --from-set 97
    python -m synthetic_seasons.fix_selectable copies --seasons 23,24,25 --copies 1-10
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import shutil
import sys
from collections import defaultdict

os.environ.setdefault('PL_SIMULATION_BUILD', '1')

from simulation.historical_data import VaastavSeasonData  # noqa: E402
from squadtools.chance_to_play import ensure_gameweek_chances  # noqa: E402

SUFFIX = '-selectfix1'


def unavailable_ids(source, gameweek, chances, old_logic=False):
    """Player ids not selectable at this deadline, as HistoricalCurrentSeason decides."""
    usable = source.can_select_usable
    if old_logic:
        source.can_select_usable = True
    try:
        out = set()
        for player_id, raw in source.players_raw.items():
            chance = chances.get(player_id)
            if source.known_unavailable(raw, gameweek) or (chance is not None and chance <= 0):
                out.add(int(player_id))
        return out
    finally:
        source.can_select_usable = usable


def _real_chances(source, gameweek):
    return ensure_gameweek_chances(source.season_start_year, gameweek,
                                   deadline=source.deadline_for_gameweek(gameweek),
                                   data_root=source.data_root, allow_download=False) or {}


def _check_and_fix(rows, old_out, new_out, where):
    """Set rows' selectable from new_out after checking the stored flag equals old_out."""
    stored = {int(r['player_id']): bool(r['selectable']) for r in rows}
    wrong = [pid for pid, flag in stored.items() if flag == (pid in old_out)]
    if wrong:
        raise SystemExit('%s: stored flag differs from the old rule for %d players (e.g. %s); not touching it'
                         % (where, len(wrong), wrong[:5]))
    for row in rows:
        row['selectable'] = int(row['player_id']) not in new_out
    return sum(1 for pid, flag in stored.items() if not flag and pid not in new_out)


# ----------------------------------------------------------------- the database set

def _new_set(db, from_set):
    meta = db.get_prediction_set(from_set)
    if meta is None:
        raise SystemExit('No prediction set %s' % from_set)
    new_meta = dict(meta, name=meta['name'] + SUFFIX, model_config_hash=meta['model_config_hash'] + SUFFIX,
                    config=dict(meta['config'], selectable_fixed_from_set=int(from_set)))
    return meta, new_meta, db.get_or_create_prediction_set(new_meta)


def set_id(from_set):
    """Create (or find) the fixed set's row and return its id, without copying rows."""
    from flask import Flask
    from persistence import DB_Handler

    app = Flask(__name__)
    with app.app_context():
        db = DB_Handler()
        try:
            return _new_set(db, from_set)[2]
        finally:
            db.close_connection()


def fix_set(from_set, log=print):
    from flask import Flask
    from persistence import DB_Handler

    app = Flask(__name__)
    with app.app_context():
        db = DB_Handler()
        try:
            meta, new_meta, new_set = _new_set(db, from_set)
            log('set %s (%s) -> set %s (%s)' % (from_set, meta['name'], new_set, new_meta['name']))
            coverage = [c for c in db.prediction_coverage(from_set) if c['status'] == 'complete']
            sources = {}
            for item in coverage:
                year, gameweek = item['season_start_year'], item['as_of_gameweek']
                if year not in sources:
                    sources[year] = VaastavSeasonData(year)
                source = sources[year]
                rows = db.get_historical_predictions(from_set, year, gameweek)
                chances = _real_chances(source, gameweek)
                old_out = unavailable_ids(source, gameweek, chances, old_logic=True)
                new_out = unavailable_ids(source, gameweek, chances)
                freed = _check_and_fix(rows, old_out, new_out, '20%d/%d GW%d' % (year, year + 1, gameweek))
                for row in rows:
                    row['season_start_year'] = year
                    row['as_of_gameweek'] = gameweek
                snapshot_id, needs_build = db.begin_prediction_snapshot(
                    new_set, year, gameweek, expected_max_target_gameweek=max(r['target_gameweek'] for r in rows))
                if needs_build:
                    db.persist_historical_predictions(snapshot_id, rows)
                    db.complete_prediction_snapshot(snapshot_id, len(rows),
                                                    ['selectable re-flagged from set %s' % from_set])
                log('  20%d/%d GW%d: %d rows, %d players selectable again%s' % (
                    year, year + 1, gameweek, len(rows), freed, '' if needs_build else ' (already built)'))
            log('done: set %s' % new_set)
            return new_set
        finally:
            db.close_connection()


# ----------------------------------------------------------------- the copies

def fix_copy(year, index, log=print):
    from .forecasts import FORECAST_COLUMNS, CopySeasonData

    source = CopySeasonData(year, index)
    path = source.copy_directory / 'forecasts.csv.gz'
    backup = source.copy_directory / 'forecasts_selectbug.csv.gz'
    if backup.exists():
        log('%s: already fixed (%s exists)' % (source.label, backup.name))
        return
    with gzip.open(path, 'rt', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        rows = list(reader)
    by_week = defaultdict(list)
    for row in rows:
        row['selectable'] = row['selectable'] in ('1', 'True', 'true')
        by_week[int(row['as_of_gameweek'])].append(row)
    freed = 0
    for gameweek, week_rows in sorted(by_week.items()):
        chances = source.chances(gameweek, source.deadline_for_gameweek(gameweek))
        old_out = unavailable_ids(source, gameweek, chances, old_logic=True)
        new_out = unavailable_ids(source, gameweek, chances)
        freed += _check_and_fix(week_rows, old_out, new_out, '%s GW%d' % (source.label, gameweek))
    shutil.copy2(path, backup)
    temporary = path.with_suffix('.tmp')
    with gzip.open(temporary, 'wt', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=header or FORECAST_COLUMNS)
        writer.writeheader()
        for row in rows:
            row['selectable'] = int(row['selectable'])
            writer.writerow(row)
    os.replace(temporary, path)
    log('%s: %d player-deadlines selectable again' % (source.label, freed))


def _int_list(text):
    out = []
    for part in text.split(','):
        lo, _, hi = part.partition('-')
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m synthetic_seasons.fix_selectable')
    sub = parser.add_subparsers(dest='command', required=True)
    one = sub.add_parser('set')
    one.add_argument('--from-set', type=int, default=97)
    ids = sub.add_parser('set-id')
    ids.add_argument('--from-set', type=int, default=97)
    two = sub.add_parser('copies')
    two.add_argument('--seasons', default='23,24,25')
    two.add_argument('--copies', default='1-10')
    args = parser.parse_args(argv)
    if args.command == 'set':
        fix_set(args.from_set, log=lambda m: print(m, flush=True))
    elif args.command == 'set-id':
        print(set_id(args.from_set))
    else:
        for year in _int_list(args.seasons):
            for index in _int_list(args.copies):
                fix_copy(year, index, log=lambda m: print(m, flush=True))


if __name__ == '__main__':
    sys.exit(main())
