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
    """Minutes baked into persisted predicted_points. Identity so the optimiser
    can apply live step_75 (or a sim scale) from stored pp90 × minutes.
    """
    m = max(0.0, min(90.0, float(mins or 0)))
    return m


def implied_pp90(adj_points: float, minutes: float) -> float:
    """Recover predicted points per 90 from LP-adjusted points and predicted minutes."""
    eff = effective_minutes_for_points(minutes)
    if eff <= 0:
        return 0.0
    return float(adj_points or 0.0) * 90.0 / eff
