"""Model-ready inputs from the external tables, as they were known at each FPL gameweek deadline.

python -m premierlearning.external_data.features      (after `build`; offline, a few minutes)

Writes data/external/features/:
  gameweek_schedule.csv          one row per season x gameweek x team: fixtures (0/1/2), blank/double,
                                 cause, and when that was known
  team_fixture_features.csv      one row per FPL fixture x side (join twice: own team and opposition):
                                 manager, all-competition schedule scalars, double/blank flags
  team_upcoming.csv              long table, the club's next matches in any competition (attention list)
  player_fixture_features.csv    one row per FPL player x fixture: all-competition load / freshness
  player_international_upcoming.csv  long table, the player's national-team matches coming up

"As of" time = the deadline of the fixture's gameweek, the moment a prediction is made. Only results
before that time are used. Future matches are used only if `known_from_utc` <= that time, so nothing is
revealed that wasn't knowable. The functions take `as_of`, so other horizons can call them directly.
"""
from __future__ import annotations

import bisect
import csv
import datetime as dt
import math
import os
from collections import defaultdict

from premierlearning.external_data import competitions, fpl_seasons
from premierlearning.external_data.cache import EXTERNAL_DIR

FEATURE_DIR = os.path.join(EXTERNAL_DIR, 'features')
UPCOMING_DAYS = 35
UPCOMING_LEN = 14
CAP_DAYS = 120.0
LOOKAHEAD_GWS = 3
DEFAULT_DEADLINE_BEFORE_KICKOFF = dt.timedelta(minutes=90)
DOUBLE_NOTICE = dt.timedelta(days=21)        # FPL usually confirms doubles 3+ weeks ahead
BLANK_OTHER_NOTICE = dt.timedelta(days=3)     # postponements: assume a late surprise (conservative)
NATIONAL_TEAM_WINDOW = dt.timedelta(days=365)
SENIOR_INTL = {'intl_tournament', 'intl_qualifier', 'intl_nations_league', 'intl_friendly'}
DEAD_STATUS = ('POSTPONED', 'CANCEL', 'ABANDON')
CUP_CATEGORIES = {'domestic_cup', 'domestic_super_cup', 'champions_league', 'europa_league',
                  'conference_league', 'club_super_cup', 'club_world_cup'}


def _t(text):
    """Naive UTC datetime from '2025-08-15T19:00Z' / '2025-08-15T19:00:00Z'."""
    return dt.datetime.strptime(text[:16], '%Y-%m-%dT%H:%M') if text else None


def _iso(moment):
    return moment.strftime('%Y-%m-%dT%H:%MZ') if moment else ''


def _days(later, earlier, cap=CAP_DAYS):
    if later is None or earlier is None:
        return cap
    return round(min(cap, max(0.0, (later - earlier).total_seconds() / 86400.0)), 3)


def _rows(name):
    with open(os.path.join(EXTERNAL_DIR, name), encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _write(name, rows, columns):
    os.makedirs(FEATURE_DIR, exist_ok=True)
    path = os.path.join(FEATURE_DIR, name)
    with open(path + '.tmp', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(path + '.tmp', path)
    print(f'wrote features/{name}: {len(rows)} rows', flush=True)


# ---------------------------------------------------------------- loading

class TeamMatch:
    __slots__ = ('event_id', 'kickoff', 'known_from', 'completed', 'has_lineups', 'category', 'league_slug',
                 'stage', 'opponent_code', 'home_away', 'neutral', 'knockout', 'dead')

    def __init__(self, r):
        self.event_id = r['event_id']
        self.kickoff = _t(r['kickoff_utc'])
        self.known_from = _t(r['known_from_utc'])
        self.completed = r['completed'] == '1'
        self.has_lineups = r['has_lineups'] == '1'
        self.category = r['category']
        self.league_slug = r['league_slug']
        self.stage = r['stage']
        self.opponent_code = r['opponent_fpl_team_code']
        self.home_away = 1 if r['home_away'] == 'home' else 0
        self.neutral = int(r['neutral'] or 0)
        self.knockout = int(self.category in CUP_CATEGORIES | {'intl_tournament'}
                            and not competitions.FIXED_STAGE.search(self.stage or ''))
        self.dead = any(word in r['status'] for word in DEAD_STATUS)


class Context:
    """Everything indexed for fast 'as of' lookups."""

    def __init__(self):
        self.seasons = fpl_seasons.load_all()
        self.latest_year = max(self.seasons)
        self.now = dt.datetime.utcnow()

        self.espn_ids = defaultdict(set)  # FPL team code -> ESPN team ids
        for r in _rows('espn_team_map.csv'):
            self.espn_ids[int(r['fpl_team_code'])].add(r['espn_team_id'])

        by_team = defaultdict(list)
        for r in _rows('team_matches.csv'):
            m = TeamMatch(r)
            if m.kickoff is not None and m.known_from is not None:
                by_team[r['team_id']].append(m)
        self.team_matches = {}
        for team_id, items in by_team.items():
            items.sort(key=lambda m: m.kickoff)
            self.team_matches[team_id] = (items, [m.kickoff for m in items])

        apps = defaultdict(list)
        for r in _rows('appearances.csv'):
            apps[int(r['fpl_code'])].append((_t(r['kickoff_utc']), int(r['minutes'] or 0), int(r['starter'] or 0),
                                             r['event_id'], r['team_id'], r['category']))
        self.apps = {}
        for code, items in apps.items():
            items.sort(key=lambda a: a[0])
            self.apps[code] = (items, [a[0] for a in items], {a[3] for a in items})

        self.managers = {}
        if os.path.exists(os.path.join(EXTERNAL_DIR, 'fixture_managers.csv')):
            for r in _rows('fixture_managers.csv'):
                key = (int(r['fpl_season']), int(r['fpl_fixture_id']))
                self.managers[(key, int(r['home_team_code']))] = (r['home_manager_id'], r['home_caretaker'], r['home_days_in_post'])
                self.managers[(key, int(r['away_team_code']))] = (r['away_manager_id'], r['away_caretaker'], r['away_days_in_post'])

        self.deadlines = {year: self._deadlines(season) for year, season in self.seasons.items()}
        self.schedule = {}  # (year, gw, team_code) -> schedule row
        self.gw_order = {}  # year -> sorted gameweeks

    # ---------------------------------------------------- basics

    def _deadlines(self, season):
        first_kick = {}
        for fx in season.fixtures.values():
            if fx['event'] is None:
                continue
            kick = _t(fx['kickoff'])
            if fx['event'] not in first_kick or kick < first_kick[fx['event']]:
                first_kick[fx['event']] = kick
        out = {}
        for gw, kick in first_kick.items():
            given = season.deadlines.get(gw)
            out[gw] = _t(given) if given else kick - DEFAULT_DEADLINE_BEFORE_KICKOFF
        return out

    def as_of(self, year, fixture):
        gw = fixture['event']
        if gw is not None and gw in self.deadlines[year]:
            return self.deadlines[year][gw]
        return _t(fixture['kickoff']) - DEFAULT_DEADLINE_BEFORE_KICKOFF

    def club_matches(self, team_code):
        """Merged, sorted club matches over every ESPN id the club has had."""
        ids = self.espn_ids.get(int(team_code), ())
        if len(ids) == 1:
            return self.team_matches.get(next(iter(ids)), ([], []))
        items = sorted((m for i in ids for m in self.team_matches.get(i, ([], []))[0]), key=lambda m: m.kickoff)
        return items, [m.kickoff for m in items]

    @staticmethod
    def visible(m, as_of):
        """Was this match a known fact (result) or a known plan (fixture) at as_of?"""
        if m.dead:
            return False
        if m.kickoff < as_of:
            return m.completed
        return m.known_from <= as_of


# ---------------------------------------------------------------- gameweek schedule (double / blank)

def build_schedule(ctx: Context):
    rows = []
    for year, season in sorted(ctx.seasons.items()):
        deadlines = ctx.deadlines[year]
        gws = sorted(deadlines)
        ctx.gw_order[year] = gws
        if not gws:
            continue
        season_known = deadlines[gws[0]] - dt.timedelta(days=60)
        counts = defaultdict(list)
        for fid, fx in season.fixtures.items():
            if fx['event'] is None:
                continue
            for team_id in (fx['team_h'], fx['team_a']):
                counts[(fx['event'], season.teams[team_id]['code'])].append((_t(fx['kickoff']), fid))
        codes = sorted({t['code'] for t in season.teams.values()})
        for i, gw in enumerate(gws):
            start = deadlines[gw]
            end = deadlines[gws[i + 1]] if i + 1 < len(gws) else start + dt.timedelta(days=7)
            for code in codes:
                fixtures = sorted(counts.get((gw, code), []))
                n = len(fixtures)
                if n == 1:
                    cause, known = 'normal', season_known
                elif n >= 2:
                    cause, known = 'double', start - DOUBLE_NOTICE
                else:
                    cup = [m for m in ctx.club_matches(code)[0]
                           if m.category in CUP_CATEGORIES and start <= m.kickoff < end and not m.dead]
                    if cup:
                        cause, known = 'blank_cup', min(m.known_from for m in cup)
                    else:
                        cause, known = 'blank_other', start - BLANK_OTHER_NOTICE
                if year == ctx.latest_year and n != 1:
                    known = min(known, ctx.now)  # it is in today's FPL data, so it's known now
                row = {'fpl_season': year, 'gw': gw, 'deadline_utc': _iso(start), 'team_code': code, 'fixtures': n,
                       'is_blank': int(n == 0), 'is_double': int(n >= 2), 'cause': cause, 'known_from_utc': _iso(known),
                       'fixture_ids': ' '.join(str(fid) for _, fid in fixtures)}
                ctx.schedule[(year, gw, code)] = row
                rows.append(row)
    _write('gameweek_schedule.csv', rows,
           ['fpl_season', 'gw', 'deadline_utc', 'team_code', 'fixtures', 'is_blank', 'is_double', 'cause',
            'known_from_utc', 'fixture_ids'])


def gw_flags(ctx: Context, year, gw, code, fixture_id, as_of):
    """This gameweek's count / leg, and the next LOOKAHEAD_GWS gameweeks as known at as_of."""
    out = {}
    row = ctx.schedule.get((year, gw, code))
    if row is None:
        out.update({'gw_fixtures': 1, 'gw_double': 0, 'dgw_leg': 1})
    else:
        ids = row['fixture_ids'].split()
        out['gw_fixtures'] = row['fixtures']
        out['gw_double'] = row['is_double']
        out['dgw_leg'] = ids.index(str(fixture_id)) + 1 if str(fixture_id) in ids else 1
    gws = ctx.gw_order.get(year, [])
    pos = gws.index(gw) if gw in gws else None
    for h in range(1, LOOKAHEAD_GWS + 1):
        nxt = ctx.schedule.get((year, gws[pos + h], code)) if pos is not None and pos + h < len(gws) else None
        known = nxt is not None and nxt['cause'] != 'normal' and _t(nxt['known_from_utc']) <= as_of
        out[f'next{h}_gw_double_known'] = int(known and nxt['is_double'] == 1)
        out[f'next{h}_gw_blank_known'] = int(known and nxt['is_blank'] == 1)
    return out


# ---------------------------------------------------------------- team features

def team_schedule(ctx: Context, code, kick, as_of):
    items, kicks = ctx.club_matches(code)
    i = bisect.bisect_left(kicks, kick)
    before = [m for m in items[max(0, i - 40):i] if ctx.visible(m, as_of)]
    after = []
    j = bisect.bisect_right(kicks, kick)
    horizon = kick + dt.timedelta(days=UPCOMING_DAYS)
    while j < len(items) and items[j].kickoff <= horizon:
        if ctx.visible(items[j], as_of):
            after.append(items[j])
        j += 1
    return {
        'days_since_last_match': _days(kick, before[-1].kickoff if before else None),
        'matches_last_14d': sum(1 for m in before if m.kickoff >= kick - dt.timedelta(days=14)),
        'days_to_next_match': _days(after[0].kickoff if after else None, kick),
        'matches_next_7d': sum(1 for m in after if m.kickoff <= kick + dt.timedelta(days=7)),
    }, after[:UPCOMING_LEN]


def _manager_match_numbers(ctx: Context):
    """(year, fixture, team_code) -> how many PL matches this manager has had in charge (this one included).

    None when the spell began before the local data does (2018-07), i.e. long-serving managers.
    """
    fixtures = []
    for year, season in ctx.seasons.items():
        for fid, fx in season.fixtures.items():
            for team_id in (fx['team_h'], fx['team_a']):
                fixtures.append((_t(fx['kickoff']), year, fid, season.teams[team_id]['code']))
    fixtures.sort()
    out, counts = {}, defaultdict(int)
    for kick, year, fid, code in fixtures:
        manager_id, _, days = ctx.managers.get(((year, fid), code), ('', '', ''))
        if not manager_id or days == '':
            out[(year, fid, code)] = None
            continue
        start = kick - dt.timedelta(days=int(days))
        counts[(code, manager_id, start.date())] += 1
        out[(year, fid, code)] = counts[(code, manager_id, start.date())] if start >= dt.datetime(2018, 7, 1) else None
    return out


def build_team_features(ctx: Context):
    rows, upcoming = [], []
    ctx.team_rows = {}
    manager_numbers = _manager_match_numbers(ctx)
    for year, season in sorted(ctx.seasons.items()):
        for fid, fx in sorted(season.fixtures.items()):
            kick, as_of = _t(fx['kickoff']), ctx.as_of(year, fx)
            for side, team_id, opp_id in (('home', fx['team_h'], fx['team_a']), ('away', fx['team_a'], fx['team_h'])):
                code = season.teams[team_id]['code']
                sched, nxt = team_schedule(ctx, code, kick, as_of)
                manager_id, caretaker, days_in_post = ctx.managers.get(((year, fid), code), ('', '', ''))
                row = {'fpl_season': year, 'fpl_fixture_id': fid, 'gw': fx['event'], 'team_code': code,
                       'opponent_code': season.teams[opp_id]['code'], 'side': side, 'kickoff_utc': _iso(kick),
                       'as_of_utc': _iso(as_of), 'manager_id': manager_id, 'manager_caretaker': caretaker,
                       'manager_days_in_post_log': round(math.log1p(float(days_in_post)), 4) if days_in_post != '' else '',
                       **sched, **gw_flags(ctx, year, fx['event'], code, fid, as_of)}
                rows.append(row)
                ctx.team_rows[(year, fid, code)] = dict(row, manager_match_number=manager_numbers.get((year, fid, code)))
                for rank, m in enumerate(nxt, 1):
                    upcoming.append({'fpl_season': year, 'fpl_fixture_id': fid, 'team_code': code, 'rank': rank,
                                     'event_id': m.event_id, 'kickoff_utc': _iso(m.kickoff), 'category': m.category,
                                     'league_slug': m.league_slug, 'opponent_fpl_team_code': m.opponent_code,
                                     'home_away': m.home_away, 'neutral': m.neutral, 'knockout': m.knockout,
                                     'days_after_current': _days(m.kickoff, kick)})
    columns = ['fpl_season', 'fpl_fixture_id', 'gw', 'team_code', 'opponent_code', 'side', 'kickoff_utc', 'as_of_utc',
               'manager_id', 'manager_caretaker', 'manager_days_in_post_log', 'days_since_last_match',
               'matches_last_14d', 'days_to_next_match', 'matches_next_7d', 'gw_fixtures', 'gw_double', 'dgw_leg']
    for h in range(1, LOOKAHEAD_GWS + 1):
        columns += [f'next{h}_gw_double_known', f'next{h}_gw_blank_known']
    _write('team_fixture_features.csv', rows, columns)
    _write('team_upcoming.csv', upcoming,
           ['fpl_season', 'fpl_fixture_id', 'team_code', 'rank', 'event_id', 'kickoff_utc', 'category', 'league_slug',
            'opponent_fpl_team_code', 'home_away', 'neutral', 'knockout', 'days_after_current'])


# ---------------------------------------------------------------- player features

def player_load(ctx: Context, code, team_code, kick, as_of):
    items, kicks, events = ctx.apps.get(int(code), ([], [], set()))
    i = bisect.bisect_left(kicks, as_of)  # results only up to the deadline
    past = items[max(0, i - 60):i]
    played = [a for a in past if a[1] > 0]
    started = [a for a in past if a[2] == 1 and a[1] > 0]
    out = {
        'days_since_last_played': _days(kick, played[-1][0] if played else None),
        'days_since_last_started': _days(kick, started[-1][0] if started else None),
    }
    for days in (7, 14, 28):
        since = kick - dt.timedelta(days=days)
        out[f'minutes_last_{days}d'] = sum(a[1] for a in past if a[0] >= since)

    club, club_kicks = ctx.club_matches(team_code)
    j = bisect.bisect_left(club_kicks, as_of)
    recent = [m for m in club[max(0, j - 15):j] if m.completed and not m.dead]
    last = recent[-1] if recent else None
    out['in_squad_last_club_match'] = int(bool(last and last.has_lineups and last.event_id in events))
    window = [m for m in recent if m.kickoff >= kick - dt.timedelta(days=28)]
    out['load_known'] = int(bool(last) and last.has_lineups and all(m.has_lineups for m in window))

    national = None
    for a in reversed(past):
        if a[0] < as_of - NATIONAL_TEAM_WINDOW:
            break
        if a[5] in SENIOR_INTL:
            national = a[4]
            break
    out['national_team_id'] = national or ''
    return out


def national_upcoming(ctx: Context, national_team_id, kick, as_of):
    items, kicks = ctx.team_matches.get(national_team_id, ([], []))
    j = bisect.bisect_right(kicks, kick)
    horizon = kick + dt.timedelta(days=UPCOMING_DAYS)
    out = []
    while j < len(items) and items[j].kickoff <= horizon and len(out) < UPCOMING_LEN:
        if ctx.visible(items[j], as_of) and items[j].category in SENIOR_INTL:
            out.append(items[j])
        j += 1
    return out


def eval_slices(ctx: Context, year, fixture, code, team_code, kick, as_of):
    """0/1 evaluation slices for one player-fixture (for per-slice error tables, not model inputs).

    Slices may use the true schedule (e.g. next gameweek blank), since they only group rows for analysis.
    """
    team = ctx.team_rows.get((year, fixture['id'], team_code), {})
    club, kicks = ctx.club_matches(team_code)
    i = bisect.bisect_left(kicks, kick)
    before = [m for m in club[max(0, i - 10):i] if ctx.visible(m, as_of)]
    gaps = [(kick - m.kickoff).total_seconds() / 86400 for m in before]
    out = {
        'slice_club_match_3_4d_before': int(any(2.5 <= g <= 4.5 for g in gaps)),
        'slice_club_match_le4d': int(any(g <= 4.5 for g in gaps)),
        'slice_nonpl_match_le4d': int(any(g <= 4.5 and m.category != 'premier_league' for g, m in zip(gaps, before))),
        'slice_dgw_leg': int(team.get('gw_double', 0) == 1),
    }
    gws = ctx.gw_order.get(year, [])
    gw = fixture.get('event')
    nxt = ctx.schedule.get((year, gws[gws.index(gw) + 1], team_code)) if gw in gws and gws.index(gw) + 1 < len(gws) else None
    out['slice_gw_before_blank'] = int(bool(nxt and nxt['is_blank']))
    number = team.get('manager_match_number')
    out['slice_new_manager_first10'] = int(str(team.get('manager_caretaker')) == '1' or (number is not None and number <= 10))

    # played for his country since the club's previous match (i.e. this is his first club match after the break)
    previous = before[-1].kickoff if before else kick - dt.timedelta(days=30)
    items, app_kicks, _ = ctx.apps.get(int(code), ([], [], set()))
    lo, hi = bisect.bisect_right(app_kicks, previous), bisect.bisect_left(app_kicks, min(kick, as_of))
    out['slice_after_intl_break_played'] = int(any(a[5] in SENIOR_INTL and a[1] > 0 for a in items[lo:hi]))
    return out


SLICE_COLUMNS = ['slice_club_match_3_4d_before', 'slice_club_match_le4d', 'slice_nonpl_match_le4d', 'slice_dgw_leg',
                 'slice_gw_before_blank', 'slice_new_manager_first10', 'slice_after_intl_break_played']


def build_player_features(ctx: Context):
    rows, intl, slices = [], [], []
    for year, season in sorted(ctx.seasons.items()):
        entries = [(e, f, h) for e, f, _, h in season.appearances]
        if year == ctx.latest_year:
            entries += season.future
        seen = set()
        for element, fid, was_home in entries:
            fx = season.fixtures.get(fid)
            player = season.players.get(element)
            if fx is None or player is None or (element, fid) in seen:
                continue
            seen.add((element, fid))
            team_code = season.teams[fx['team_h'] if was_home else fx['team_a']]['code']
            kick, as_of = _t(fx['kickoff']), ctx.as_of(year, fx)
            load = player_load(ctx, player['code'], team_code, kick, as_of)
            rows.append({'fpl_season': year, 'fpl_fixture_id': fid, 'fpl_code': player['code'], 'element': element,
                         'team_code': team_code, 'kickoff_utc': _iso(kick), 'as_of_utc': _iso(as_of), **load})
            slices.append({'fpl_season': year, 'fpl_fixture_id': fid, 'fpl_code': player['code'],
                           **eval_slices(ctx, year, dict(fx, id=fid), player['code'], team_code, kick, as_of)})
            if load['national_team_id']:
                for rank, m in enumerate(national_upcoming(ctx, load['national_team_id'], kick, as_of), 1):
                    intl.append({'fpl_season': year, 'fpl_fixture_id': fid, 'fpl_code': player['code'], 'rank': rank,
                                 'national_team_id': load['national_team_id'], 'event_id': m.event_id,
                                 'kickoff_utc': _iso(m.kickoff), 'category': m.category, 'league_slug': m.league_slug,
                                 'home_away': m.home_away, 'neutral': m.neutral, 'knockout': m.knockout,
                                 'days_after_current': _days(m.kickoff, kick)})
    _write('player_fixture_features.csv', rows,
           ['fpl_season', 'fpl_fixture_id', 'fpl_code', 'element', 'team_code', 'kickoff_utc', 'as_of_utc',
            'days_since_last_played', 'days_since_last_started', 'minutes_last_7d', 'minutes_last_14d',
            'minutes_last_28d', 'in_squad_last_club_match', 'load_known', 'national_team_id'])
    _write('eval_slices.csv', slices, ['fpl_season', 'fpl_fixture_id', 'fpl_code'] + SLICE_COLUMNS)
    _write('player_international_upcoming.csv', intl,
           ['fpl_season', 'fpl_fixture_id', 'fpl_code', 'rank', 'national_team_id', 'event_id', 'kickoff_utc',
            'category', 'league_slug', 'home_away', 'neutral', 'knockout', 'days_after_current'])


def main():
    ctx = Context()
    build_schedule(ctx)
    build_team_features(ctx)
    build_player_features(ctx)


if __name__ == '__main__':
    main()
