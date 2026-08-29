"""FPL transfer-deadline helpers (Python 3.8-safe, no zoneinfo)."""
import calendar
from datetime import date, datetime, timedelta, timezone


def last_sunday(year, month):
    last_day = calendar.monthrange(year, month)[1]
    last = date(year, month, last_day)
    return last - timedelta(days=(last.weekday() + 1) % 7)


def uk_is_dst(utc_dt):
    """UK DST: 01:00 UTC last Sunday of March to 01:00 UTC last Sunday of October."""
    utc_dt = utc_dt.astimezone(timezone.utc)
    year = utc_dt.year
    start = datetime(
        year, 3, last_sunday(year, 3).day, 1, 0, tzinfo=timezone.utc
    )
    end = datetime(
        year, 10, last_sunday(year, 10).day, 1, 0, tzinfo=timezone.utc
    )
    return start <= utc_dt < end


def to_london(utc_dt):
    utc_dt = utc_dt.astimezone(timezone.utc)
    offset = timedelta(hours=1 if uk_is_dst(utc_dt) else 0)
    return utc_dt.astimezone(timezone(offset))


def parse_fpl_deadline(deadline_time):
    if not deadline_time:
        return None
    return datetime.strptime(
        str(deadline_time).replace('Z', '+00:00'),
        '%Y-%m-%dT%H:%M:%S%z',
    )


def next_deadline_event(events):
    events = events or []
    next_event = next((event for event in events if event.get('is_next')), None)
    if next_event is None:
        next_event = next((event for event in events if event.get('is_current')), None)
    return next_event


def is_transfer_deadline_day(deadline_utc, now_utc=None):
    """True when the next deadline is still ahead and falls on today's UK date."""
    if deadline_utc is None:
        return False
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    deadline = deadline_utc.astimezone(timezone.utc)
    if deadline <= now:
        return False
    return to_london(deadline).date() == to_london(now).date()
