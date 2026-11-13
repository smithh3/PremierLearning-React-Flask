"""python -m synthetic_seasons generate --seasons 21,22,23,25 --copies 5
python -m synthetic_seasons report --seasons 21,22,23,25
python -m synthetic_seasons check --seasons 22,23,25 --copies 1,2     (v6 error, real vs copies)
python -m synthetic_seasons forecast --seasons 25 --copies 1-20       (v6 forecasts.csv.gz per copy)
python -m synthetic_seasons pipeline-check --seasons 25               (copy path == real path, exit 1 if not)
python -m synthetic_seasons prepare-pools --seasons 23,24,25 --copies 1-10   (manager pool files per copy week)"""
import argparse
import sys
from pathlib import Path

from .source import DATA_ROOT


def _numbers(text):
    out = []
    for part in str(text).split(','):
        part = part.strip()
        if '-' in part:
            low, high = part.split('-')
            out.extend(range(int(low), int(high) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(prog='synthetic_seasons')
    parser.add_argument('command', choices=('generate', 'report', 'check', 'forecast', 'pipeline-check',
                                            'prepare-pools'))
    parser.add_argument('--seasons', default='21,22,23,25', help='season start years, e.g. 21,22,23,25')
    parser.add_argument('--copies', default='5',
                        help='generate: how many; check/forecast: which, e.g. 1,2 or 1-20')
    parser.add_argument('--start', type=int, default=1, help='generate: first copy number')
    args = parser.parse_args(argv)
    years = _numbers(args.seasons)
    if args.command == 'generate':
        from .generate import generate
        generate(years, int(args.copies), args.start)
    elif args.command == 'report':
        from .report import build_report
        text = build_report(years)
        path = Path(DATA_ROOT) / 'synthetic' / 'REALISM_REPORT.md'
        path.write_text(text, encoding='utf-8')
        print(text)
        print('\nSaved to %s' % path)
    elif args.command == 'check':
        from .forecasts import error_check
        error_check(years, copies=_numbers(args.copies))
    elif args.command == 'pipeline-check':
        from .pipeline_check import run
        return 0 if all(run(year) for year in years) else 1
    elif args.command == 'prepare-pools':
        from .training import prepare_pools
        prepare_pools(years, _numbers(args.copies))
    else:
        from .forecasts import SeasonForecaster
        for year in years:
            forecaster = SeasonForecaster(year)
            forecaster.fit()
            for index in _numbers(args.copies):
                forecaster.forecast_copy(index)
    return 0


if __name__ == '__main__':
    sys.exit(main())
