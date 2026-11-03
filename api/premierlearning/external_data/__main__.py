"""python -m premierlearning.external_data <step> [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--refresh]

Steps (run from api/):
  leagues      ESPN league id -> slug / name                  (~220 requests, once)
  scoreboards  one "all competitions" scoreboard per day       (~3,100 requests first time)
  summaries    lineups/events for relevant finished matches    (~15-20k requests first time)
  managers     Wikipedia manager spells + fixture_managers.csv (2 requests)
  build        processed CSVs in data/external/                (offline)
  features     model-ready inputs in data/external/features/   (offline)
  all          everything above, in order. Re-running only fetches what is new.
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

from premierlearning.external_data import espn, features, fpl_seasons, managers, tables
from premierlearning.external_data.cache import Fetcher


def main(argv=None):
    parser = argparse.ArgumentParser(prog='premierlearning.external_data')
    parser.add_argument('step', choices=['leagues', 'scoreboards', 'summaries', 'managers', 'build', 'features', 'all'])
    parser.add_argument('--start', type=dt.date.fromisoformat, default=espn.FIRST_DAY)
    parser.add_argument('--end', type=dt.date.fromisoformat, default=None)
    parser.add_argument('--refresh', action='store_true', help='re-download leagues / Wikipedia pages')
    parser.add_argument('--interval', type=float, default=0.25, help='minimum seconds between requests')
    args = parser.parse_args(argv)

    fetcher = Fetcher(min_interval=args.interval)
    began = time.time()
    steps = ['leagues', 'scoreboards', 'summaries', 'managers', 'build', 'features'] if args.step == 'all' else [args.step]
    leagues = None
    for step in steps:
        print(f'=== {step} ({dt.datetime.now():%Y-%m-%d %H:%M:%S})', flush=True)
        if step == 'leagues':
            leagues = espn.fetch_leagues(fetcher, refresh=args.refresh)
        elif step == 'scoreboards':
            espn.crawl_scoreboards(fetcher, args.start, args.end)
        elif step == 'summaries':
            leagues = leagues or espn.fetch_leagues(fetcher)
            espn.crawl_summaries(fetcher, leagues, args.start, args.end)
        elif step == 'managers':
            seasons = fpl_seasons.load_all()
            spells = managers.build_spells(fetcher, seasons, refresh=args.refresh)
            managers.write_fixture_managers(seasons, spells)
        elif step == 'build':
            tables.build(args.start, args.end)
        elif step == 'features':
            features.main()
    print(f'=== done in {(time.time() - began) / 60:.1f} min, {fetcher.count} HTTP requests', flush=True)


if __name__ == '__main__':
    main()
