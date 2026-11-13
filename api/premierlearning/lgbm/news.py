"""FPL's own player news as LightGBM columns (feature set 4, lgbm_v6).

FPL's news field comes from club statements and press conferences, written by
FPL: a status (a available, d doubtful, i injured, s suspended, u unavailable,
n not in squad), text such as "Hamstring injury - Expected back 12 Nov", the
time it was added and the 0/25/50/75/100 chance. The flat chance already reaches
the models as context.chance_of_playing; this adds the rest.

Data: for every deadline that has a squadtools/chance_to_play cache file, the
same Randdalf/fplcache snapshot (the last at or before the deadline) is
downloaded again and its news fields saved under
data/season_XX_YY/fpl_news/gwN.json (``python -m premierlearning.lgbm.news``).
Seasons before 2021/22 have no files: their rows get news.known = 0 and NaN
elsewhere. The live season (2026-11-09): each pipeline run saves the upcoming
deadline's news from the live bootstrap (archive_live_news) and fills any
passed deadline without a file from fplcache (backfill_live_news), the same
way as the past seasons.

Each row reads the snapshot for the deadline it was predicted from: a played
fixture its own gameweek's (as the chance does), an upcoming fixture the
runner's as-of gameweek, so a fixture four weeks ahead never sees later news.
Dates are measured against that fixture's own kickoff.
"""
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy

from premierlearning.learning_v2.kickoff import kickoff_from_match, parse_kickoff
from squadtools.chance_to_play import (
    FPLCACHE_REPO, _as_utc, _download_bootstrap, _fetch_pre_deadline_bootstrap, chance_cache_path,
    data_root_default, season_dir,
)

NEWS_DIR = 'fpl_news'
NEWS_FIELDS = ('status', 'news', 'news_added', 'chance_of_playing_this_round', 'chance_of_playing_next_round')
STATUSES = ('d', 'i', 's', 'u', 'n')
UNFIT = ('i', 's', 'u', 'n')
# Reason groups, matched in order against the text before " - ".
REASONS = (
    ('serious', r'cruciate|ligament|fractur|broken|surgery|rupture|dislocat|achilles'),
    ('suspended', r'suspend|ineligible'),
    ('left', r'loan|transfer|joined|left the club|left by|mutual consent|released|contract|permanent'),
    ('squad', r'not included|squad|not being considered|not training|unavailable'),
    ('international', r'international|africa cup|asian cup|nations'),
    ('fitness', r'match fitness|fatigue'),
    ('illness', r'illness|\bill\b|virus|covid|quarantine'),
    ('knock', r'knock|dead leg|gashed|bruis'),
    ('head', r'head|concussion|facial|jaw|eye|nose'),
    ('muscle', r'hamstring|muscle|calf|thigh|groin|glute|abdominal|strain|adductor|tendon'),
    ('joint', r'knee|ankle|foot|toe|heel|hip|back|shoulder|leg|pelvic|hernia|rib|shin|hand|wrist|arm|neck'),
    ('personal', r'personal|compassionate'),
)
REASON_NAMES = tuple(name for name, _pattern in REASONS) + ('other',)
MONTHS = {name: number for number, name in enumerate(
    ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'), start=1)}
DATE_PATTERN = re.compile(r'(?:expected back|suspended until)\s+(\d{1,2})\s+([a-z]{3})', re.IGNORECASE)
PERCENT_PATTERN = re.compile(r'(\d{1,3})%')
UNKNOWN_RETURN = re.compile(r'unknown return date', re.IGNORECASE)
NEWS_COLUMNS = (
    ('news.known',) + tuple('news.status_%s' % status for status in STATUSES)
    + tuple('news.reason_%s' % name for name in REASON_NAMES)
    + ('news.flagged', 'news.chance', 'news.days_to_return', 'news.return_unknown', 'news.days_flagged',
       'news.prev_chance', 'news.chance_change', 'news.flagged_deadlines', 'news.deadlines_since_flag',
       'news.certainty')
)


def news_cache_path(season_start_year, gameweek, data_root=None) -> Path:
    return season_dir(season_start_year, data_root) / NEWS_DIR / ('gw%d.json' % int(gameweek))


def reason_of(text):
    """The reason group of a news text ('' when there is no text)."""
    text = (text or '').strip().lower()
    if not text:
        return ''
    head = text.rsplit(' - ', 1)[0]
    for name, pattern in REASONS:
        if re.search(pattern, head):
            return name
    return 'other'


def return_date(text, reference):
    """The "Expected back" / "Suspended until" date as a naive datetime, or None.

    FPL gives day and month only; the year is the first that puts the date no
    more than two weeks before ``reference`` (when the news was added).
    """
    match = DATE_PATTERN.search(text or '')
    if match is None or reference is None:
        return None
    month = MONTHS.get(match.group(2).lower())
    if month is None:
        return None
    day = int(match.group(1))
    for year in (reference.year - 1, reference.year, reference.year + 1):
        try:
            candidate = datetime(year, month, day)
        except ValueError:
            continue
        if candidate >= reference - timedelta(days=14):
            return candidate
    return None


def flag_chance(entry):
    """FPL's chance for the coming deadline, 0-1; unflagged players are 1."""
    if not entry:
        return 1.0
    for name in ('chance_of_playing_next_round', 'chance_of_playing_this_round'):
        value = entry.get(name)
        if value not in (None, ''):
            return min(1.0, max(0.0, float(value) / 100.0))
    if (entry.get('status') or 'a') in UNFIT:
        return 0.0
    return 1.0


def is_flagged(entry):
    return bool(entry) and ((entry.get('status') or 'a') != 'a' or flag_chance(entry) < 1.0)


def certainty(status, chance, days_to_return, return_unknown):
    """How much of a sure thing it is that he is available, 0 (out) to 10 (no doubt).

    Fixed rules on FPL's own fields, written by hand (not fitted), checked
    against 2021/22-2024/25 (not the 2025/26 test season): no flag is 10; a
    doubt is FPL's percentage (75% doubts played about half the time, 50% a
    quarter, 25% under a tenth, whatever the injury, so no per-injury bonus);
    an injury dated back within 3 days of kickoff, or already past its date
    but still flagged, is 1 (about 1 in 10 of those played); a later date, an
    unknown return date, a ban, a loan or transfer, or "not in the squad" is 0.
    """
    if not status or status == 'a':
        return 10.0 * chance if chance < 1.0 else 10.0
    if status == 'd':
        return float(min(10.0, max(0.0, 10.0 * chance)))
    if status == 's':
        return 0.0
    if days_to_return is not None and not numpy.isnan(days_to_return):
        return 1.0 if days_to_return <= 3.5 else 0.0
    if return_unknown or status in UNFIT:
        return 0.0
    return 10.0 * chance


def _day(value):
    return datetime(value.year, value.month, value.day)


class NewsStore:
    """The saved news per (season, gameweek), loaded once; None where no file exists."""

    def __init__(self, data_root=None):
        self.data_root = data_root
        self._files = {}
        self._parsed = {}

    def snapshot(self, year, gameweek):
        key = (int(year), int(gameweek))
        if key not in self._files:
            path = news_cache_path(key[0], key[1], self.data_root)
            payload = None
            if path.is_file():
                payload = json.loads(path.read_text(encoding='utf-8'))
                payload['deadline_time'] = parse_kickoff(payload.get('deadline'))
            self._files[key] = payload
        return self._files[key]

    def _earlier(self, year, gameweek):
        """Snapshots of this season's earlier deadlines, newest first."""
        for earlier in range(int(gameweek) - 1, 0, -1):
            payload = self.snapshot(year, earlier)
            if payload is not None:
                yield payload

    def player(self, year, gameweek, player_id):
        """Kickoff-independent values for one player at one deadline, or None without a file."""
        key = (int(year), int(gameweek), int(player_id))
        if key in self._parsed:
            return self._parsed[key]
        payload = self.snapshot(year, gameweek)
        if payload is None:
            self._parsed[key] = None
            return None
        entry = (payload.get('players') or {}).get(str(int(player_id))) or {}
        text = (entry.get('news') or '').strip()
        status = (entry.get('status') or 'a') if entry else 'a'
        added = parse_kickoff(entry.get('news_added'))
        deadline = payload.get('deadline_time')
        reason = reason_of(text)
        chance = flag_chance(entry)
        flagged = is_flagged(entry)
        streak, since = (1 if flagged else 0), (0 if flagged else None)
        previous = None
        for steps, earlier in enumerate(self._earlier(year, gameweek), start=1):
            earlier_entry = (earlier.get('players') or {}).get(str(int(player_id)))
            earlier_flagged = is_flagged(earlier_entry)
            if previous is None:
                previous = flag_chance(earlier_entry)
            if flagged and earlier_flagged and streak == steps:
                streak += 1
            if since is None and earlier_flagged:
                since = steps
            if (not flagged or streak <= steps) and since is not None:
                break
        values = {
            'status': status, 'reason': reason, 'chance': chance, 'flagged': flagged,
            'back': return_date(text, _day(added) if added else (_day(deadline) if deadline else None)),
            'return_unknown': bool(UNKNOWN_RETURN.search(text)),
            'days_flagged': ((deadline - added).total_seconds() / 86400.0
                             if (flagged and added is not None and deadline is not None) else numpy.nan),
            'prev_chance': numpy.nan if previous is None else previous,
            'streak': streak, 'since': numpy.nan if since is None else since,
        }
        self._parsed[key] = values
        return values


def row_news(values, kickoff):
    """{NEWS_COLUMNS name: value} for one row kicking off at ``kickoff``; values None means no file."""
    row = {name: numpy.nan for name in NEWS_COLUMNS}
    row['news.known'] = 0.0
    if values is None:
        return row
    row['news.known'] = 1.0
    for status in STATUSES:
        row['news.status_%s' % status] = float(values['status'] == status)
    for name in REASON_NAMES:
        row['news.reason_%s' % name] = float(values['reason'] == name)
    days_to_return = numpy.nan
    if values['back'] is not None and kickoff is not None:
        days_to_return = (values['back'] - _day(kickoff)).total_seconds() / 86400.0
    row.update({
        'news.flagged': float(values['flagged']),
        'news.chance': values['chance'],
        'news.days_to_return': days_to_return,
        'news.return_unknown': float(values['return_unknown']),
        'news.days_flagged': values['days_flagged'],
        'news.prev_chance': values['prev_chance'],
        'news.chance_change': values['chance'] - values['prev_chance'],
        'news.flagged_deadlines': float(values['streak']),
        'news.deadlines_since_flag': values['since'],
        'news.certainty': certainty(values['status'], values['chance'], days_to_return,
                                    values['return_unknown']),
    })
    return row


def news_columns(batch, as_of_gameweek=None, store=None):
    """{column: values} for a batch. as_of_gameweek: the deadline upcoming rows were predicted from."""
    store = store or NewsStore()
    count = len(batch.matches)
    columns = {name: numpy.full(count, numpy.nan, dtype=numpy.float32) for name in NEWS_COLUMNS}
    years = list(batch.years) or [0] * count
    owners = list(batch.owners) or [None] * count
    for index, (owner, year, match) in enumerate(zip(owners, years, batch.matches)):
        player_id = getattr(owner, 'id', None)
        gameweek = as_of_gameweek or getattr(match, 'gameweek', None)
        values = None
        if player_id is not None and gameweek and year:
            values = store.player(year, gameweek, player_id)
        for name, value in row_news(values, kickoff_from_match(match)).items():
            columns[name][index] = value
    return columns


def download_season(season_start_year, data_root=None, verbose=True):
    """Save the news for every deadline that has a chance file; returns the count written."""
    written = 0
    chance_dir = chance_cache_path(season_start_year, 1, data_root).parent
    if not chance_dir.is_dir():
        return 0
    for chance_file in sorted(chance_dir.glob('gw*.json'), key=lambda path: int(path.stem[2:])):
        gameweek = int(chance_file.stem[2:])
        target = news_cache_path(season_start_year, gameweek, data_root)
        if target.is_file():
            continue
        meta = json.loads(chance_file.read_text(encoding='utf-8'))
        if meta.get('source') != FPLCACHE_REPO or not meta.get('snapshot'):
            if verbose:
                print('  GW%d: chance file is not from %s, skipped' % (gameweek, FPLCACHE_REPO))
            continue
        bootstrap = _download_bootstrap(meta['snapshot'])
        players = {}
        for element in bootstrap.get('elements') or []:
            if element.get('id') in (None, ''):
                continue
            players[str(int(element['id']))] = {name: element.get(name) for name in NEWS_FIELDS}
        payload = {'source': FPLCACHE_REPO, 'snapshot': meta['snapshot'], 'gameweek': gameweek,
                   'deadline': meta.get('deadline'), 'players': players}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + '\n',
                          encoding='utf-8')
        written += 1
        if verbose:
            print('  GW%d: %s, %d players' % (gameweek, meta['snapshot'], len(players)))
    return written


def _news_payload(source, snapshot, gameweek, deadline, elements):
    players = {}
    for element in elements or []:
        if element.get('id') in (None, ''):
            continue
        players[str(int(element['id']))] = {name: element.get(name) for name in NEWS_FIELDS}
    return {'source': source, 'snapshot': snapshot, 'gameweek': int(gameweek),
            'deadline': deadline, 'players': players}


def _write_news(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')


def _deadlines(events):
    """[(gameweek, aware UTC deadline)] in gameweek order."""
    out = []
    for event in sorted(events or [], key=lambda row: int(row.get('id') or 0)):
        if event.get('deadline_time'):
            out.append((int(event['id']), _as_utc(datetime.fromisoformat(
                str(event['deadline_time']).replace('Z', '+00:00')))))
    return out


def archive_live_news(season_start_year, bootstrap, now=None, data_root=None):
    """Save the upcoming deadline's news from a live bootstrap (the live season's as-of snapshot).

    Same file as the fplcache backfill. Rewritten on every run before the
    deadline, left alone after it. Returns the path written, or None.
    """
    now = _as_utc(now or datetime.now(timezone.utc))
    upcoming = next(((gameweek, deadline) for gameweek, deadline in _deadlines((bootstrap or {}).get('events'))
                     if deadline > now), None)
    if upcoming is None:
        return None
    gameweek, deadline = upcoming
    path = news_cache_path(season_start_year, gameweek, data_root)
    _write_news(path, _news_payload('fpl_api', now.isoformat(), gameweek, deadline.isoformat(),
                                    bootstrap.get('elements')))
    return path


def backfill_live_news(season_start_year, events, now=None, data_root=None, verbose=True):
    """Save the news for each passed deadline of the live season that has no file yet.

    Each is the last Randdalf/fplcache snapshot at or before that deadline, as
    for the past seasons, so the live season's played fixtures get the same
    columns they were trained with. A deadline fplcache cannot serve is left
    without a file (its rows get news.known = 0). Returns the count written.
    """
    now = _as_utc(now or datetime.now(timezone.utc))
    written = 0
    for gameweek, deadline in _deadlines(events):
        if deadline > now:
            break
        target = news_cache_path(season_start_year, gameweek, data_root)
        if target.is_file():
            continue
        try:
            snapshot, bootstrap = _fetch_pre_deadline_bootstrap(deadline)
        except Exception as error:  # noqa: BLE001 - network / archive gaps
            if verbose:
                print('  FPL news GW%d: no fplcache snapshot (%s)' % (gameweek, error))
            continue
        _write_news(target, _news_payload(FPLCACHE_REPO, snapshot, gameweek, deadline.isoformat(),
                                          bootstrap.get('elements')))
        written += 1
        if verbose:
            print('  FPL news GW%d: %s' % (gameweek, snapshot))
    return written


def main(argv=None):
    """Past seasons from their chance files; ``live`` backfills the live season from its events."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['live']:
        from premierlearning.config_constants import SEASON_YR

        events = json.loads((season_dir(SEASON_YR) / 'fantasy.json').read_text(encoding='utf-8')).get('events')
        print('Season %02d/%02d (live)' % (SEASON_YR, SEASON_YR + 1))
        print('  %d written' % backfill_live_news(SEASON_YR, events))
        return
    years = [int(value) for value in argv] or [21, 22, 23, 24, 25]
    for year in years:
        print('Season %02d/%02d' % (year % 100, year % 100 + 1))
        print('  %d written' % download_season(year))


if __name__ == '__main__':
    main()
