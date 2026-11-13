"""Replay the live season as if it were a past one (2026-11-13, for the 2026/27-so-far replay).

    python -m simulation.replay_live_season chances --to-gw 10
    python -m simulation.replay_live_season build -- <simulation.prediction_builder args> --seasons 26 --to-gw 10

The code treats SEASON_YR (config_constants) as the live season: no stored
chance-of-playing, live FPL files instead of the CSVs. A replay needs the
opposite, so this process moves SEASON_YR on by one before running, which makes
the live season read exactly like 2025/26 does: the CSVs from
simulation.current_season_csv, fplcache chance and news files per deadline. Only
this process is affected; the live pipeline and the files it reads are not.
"""
from __future__ import annotations

import argparse
import sys

_PACKAGES = ('premierlearning', 'squadtools', 'simulation', 'fpl_manager', 'synthetic_seasons')
_LIVE = None


def treat_live_season_as_past():
    """Call first, before importing the rest of the project. Returns the live season."""
    global _LIVE
    import premierlearning.config_constants as constants

    if _LIVE is None:
        _LIVE = int(constants.SEASON_YR)
        constants.SEASON_YR = _LIVE + 1
    repatch()
    return _LIVE


def repatch():
    """Modules that copied SEASON_YR before the change get the new value too."""
    if _LIVE is None:
        return
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in _PACKAGES and getattr(module, 'SEASON_YR', None) == _LIVE:
            module.SEASON_YR = _LIVE + 1


def fetch_chances(season, to_gw):
    from simulation.historical_data import VaastavSeasonData
    from squadtools.chance_to_play import ensure_gameweek_chances

    source = VaastavSeasonData(season)
    for gameweek in range(1, int(to_gw) + 1):
        chances = ensure_gameweek_chances(
            season, gameweek, deadline=source.deadline_for_gameweek(gameweek),
            data_root=source.data_root, allow_download=True,
        )
        flagged = sum(1 for value in chances.values() if value is not None and value < 100)
        print('%s/%s GW%s: %s players, %s flagged' % (season, season + 1, gameweek, len(chances), flagged),
              flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    live = treat_live_season_as_past()
    if argv and argv[0] == 'build':
        rest = argv[1:]
        if rest and rest[0] == '--':
            rest = rest[1:]
        from simulation import prediction_builder

        repatch()
        # Runner reads its flags from sys.argv, so it has to match the build.
        sys.argv = ['simulation.prediction_builder'] + rest
        code, _builder = prediction_builder.run(rest)
        return code
    parser = argparse.ArgumentParser(prog='simulation.replay_live_season')
    parser.add_argument('command', choices=('chances', 'build'))
    parser.add_argument('--season', type=int, default=live)
    parser.add_argument('--to-gw', type=int, default=10)
    args = parser.parse_args(argv)
    fetch_chances(int(args.season) % 100, args.to_gw)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
