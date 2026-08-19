"""Compatibility exports for the old simulation module.

Historical replay state is now deliberately independent of live ``AbstractSquad``
and its FPL API calls.
"""

from squadtools.simulation_strategy import OwnedPlayer, SquadState


class SimSquad(SquadState):
    pass


__all__ = ['SimSquad', 'SquadState', 'OwnedPlayer']