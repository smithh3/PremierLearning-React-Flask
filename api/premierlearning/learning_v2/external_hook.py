"""Plan leaves filled from the external-data features (learning_v2/external_inputs.py).

Owned by the model chat; external_inputs.py is the external-data chat's. Plan paths:
  current_team.schedule.*, current_team.gameweek.*, current_team.manager_features.*
  current_opposition.(same groups)   -> ExternalInputs.team(season, fixture, that side's club)
  player_load.*                      -> ExternalInputs.player(season, fixture, player code)
Every value is as of the fixture's gameweek deadline. Rows with no fixture id or
club get the loader's "nothing known" defaults.
"""
from __future__ import annotations

TEAM_SIDES = ('current_team', 'current_opposition')
TEAM_GROUPS = ('schedule', 'gameweek', 'manager_features')
PLAYER_ROOT = 'player_load'

_LOADED = None


def external_key(path: str):
    """('current_team' | 'current_opposition' | 'player', loader key), or None."""
    for side in TEAM_SIDES:
        for group in TEAM_GROUPS:
            prefix = '%s.%s.' % (side, group)
            if path.startswith(prefix):
                return side, path[len(side) + 1:]
    if path.startswith(PLAYER_ROOT + '.'):
        return 'player', path
    return None


def can_fill_external(path: str) -> bool:
    from premierlearning.learning_v2.external_inputs import PLAYER_SCALARS, TEAM_SCALARS

    found = external_key(path)
    if found is None:
        return False
    side, key = found
    return key in (PLAYER_SCALARS if side == 'player' else TEAM_SCALARS)


def plan_uses_external(plan) -> bool:
    return any(external_key(leaf.path) is not None for leaf in plan.leaves())


def load_external():
    """The feature tables, loaded once per process (about 5 s)."""
    global _LOADED
    if _LOADED is None:
        from premierlearning.learning_v2.external_inputs import ExternalInputs

        _LOADED = ExternalInputs.load()
    return _LOADED


def fpl_season(year) -> int:
    """The loader keys seasons by start year (2018); season objects use 18."""
    year = int(year)
    return year + 2000 if year < 100 else year


def external_value(external, path, year, fixture_id, team_code, opp_code, player_code) -> float:
    from premierlearning.learning_v2.external_inputs import PLAYER_DEFAULTS, TEAM_DEFAULTS

    side, key = external_key(path)
    if side == 'player':
        if fixture_id is None or player_code is None or year is None:
            return float(PLAYER_DEFAULTS[key])
        return float(external.player(fpl_season(year), fixture_id, player_code)[key])
    code = team_code if side == 'current_team' else opp_code
    if fixture_id is None or code is None or year is None:
        return float(TEAM_DEFAULTS[key])
    return float(external.team(fpl_season(year), fixture_id, code)[key])
