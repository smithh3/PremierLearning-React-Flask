import json


def read_json(file_path):
    try:
        with open(file_path) as file:
            return json.load(file)
    except Exception:
        pass


def coerce_bool(value):
    """Normalize JSON bools and vaastav CSV 'True'/'False' strings."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in ('true', '1', 'yes'):
        return True
    if s in ('false', '0', 'no', '', 'none', 'null'):
        return False
    return bool(value)


def coerce_optional_int(value):
    """Int for ids/rounds; None for blank / 'None' CSV cells."""
    if value is None:
        return None
    s = str(value).strip()
    if s == '' or s.lower() in ('none', 'null'):
        return None
    return int(float(s))


def score_is_present(value):
    """True if a fixture/player row has a real score (match has been played)."""
    return coerce_optional_int(value) is not None


def effective_minutes_for_points(mins: float) -> float:
    """Map predicted minutes before pp90 × mins / 90.

    Below 70: identity. From 70 → 80 up to 90 → 90, easing toward 90 so
    70–90 predicted minutes are not linearly haircut.
    """
    m = max(0.0, min(90.0, float(mins or 0)))
    if m < 70.0:
        return m
    t = (m - 70.0) / 20.0
    return 80.0 + 10.0 * (t ** 0.5)
