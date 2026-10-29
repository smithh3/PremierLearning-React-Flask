"""Kickoff parsing shared by match objects and the v2 dataset."""
from datetime import datetime


def parse_kickoff(value):
    """Return a naive datetime, or None when the value is missing or unreadable."""
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    text = str(value).strip()
    if text.lower() in ('none', 'null'):
        return None
    text = text.replace('Z', '')
    if 'T' in text and '+' in text[10:]:
        text = text.split('+', 1)[0]
    for fmt, size in (
        ('%Y-%m-%dT%H:%M:%S', 19),
        ('%Y-%m-%d %H:%M:%S', 19),
        ('%Y-%m-%d', 10),
    ):
        try:
            return datetime.strptime(text[:size], fmt)
        except ValueError:
            continue
    return None


def kickoff_from_match(match):
    """Player kickoff, else the team fixture's kickoff_time."""
    parsed = parse_kickoff(getattr(match, 'kickoff_time', None))
    if parsed is not None:
        return parsed
    team_match = getattr(match, 'team_match', None)
    fixture = getattr(team_match, 'fixture_json', None) or {}
    if isinstance(fixture, dict):
        return parse_kickoff(fixture.get('kickoff_time'))
    return None


def days_between(later, earlier):
    """Non-negative days from earlier to later. Missing dates are 0."""
    if later is None or earlier is None:
        return 0.0
    return max(0.0, (later - earlier).total_seconds() / 86400.0)
