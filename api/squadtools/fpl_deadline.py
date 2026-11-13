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


def future_deadlines(events, now_utc):
    """Deadline datetimes still ahead of now, earliest first."""
    now = now_utc.astimezone(timezone.utc)
    deadlines = []
    for event in events or []:
        deadline = parse_fpl_deadline(event.get('deadline_time'))
        if deadline is not None and deadline > now:
            deadlines.append(deadline.astimezone(timezone.utc))
    deadlines.sort()
    return deadlines


def deadline_for_following_run(events, now_utc, this_run_window_hours=3, hours_before=None):
    """Deadline the next scheduled run should sit ahead of.

    A run that starts shortly before a deadline is for that deadline, so the
    following booking skips it. A run on any other day books the soonest one.
    hours_before: how long before its deadline a run is booked; a deadline
    whose run time has already passed is skipped (2026-11-06: a 5-hour-early
    run that trained for 2 hours ended 3h02 before the deadline, just outside
    the 3-hour window, tried to rebook the same deadline and booked nothing).
    """
    deadlines = future_deadlines(events, now_utc)
    if hours_before is not None:
        now = now_utc.astimezone(timezone.utc)
        deadlines = [deadline for deadline in deadlines
                     if deadline - timedelta(hours=float(hours_before)) > now]
    if not deadlines:
        return None
    soonest = deadlines[0]
    window = timedelta(hours=float(this_run_window_hours))
    if soonest - now_utc.astimezone(timezone.utc) <= window and len(deadlines) > 1:
        return deadlines[1]
    return soonest


def london_at_expression(utc_dt):
    """EventBridge one-shot expression in UK local time."""
    return 'at(%s)' % to_london(utc_dt).strftime('%Y-%m-%dT%H:%M:%S')


def is_transfer_deadline_day(deadline_utc, now_utc=None):
    """True when the next deadline is still ahead and falls on today's UK date."""
    if deadline_utc is None:
        return False
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    deadline = deadline_utc.astimezone(timezone.utc)
    if deadline <= now:
        return False
    return to_london(deadline).date() == to_london(now).date()
