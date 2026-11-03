"""Error and bias on situations of interest (the external-data chat's eval slices).

Uses features/eval_slices.csv through learning_v2/external_inputs.py, keyed by
(season, fixture, player code). A player-gameweek is in a slice when any of
its fixtures that gameweek is. Only the next gameweek (lead 0) is used: the
slices describe that gameweek's deadline. Rows are those with a real result
(the player's club played), so blanks do not dilute the slice.

For each slice and model: n, actual mean, bias (predicted minus actual mean),
MAE and RMSE, next to the same for all rows. Plain differences, not
noise-scaled scores: slices are small, so read them as diagnostics.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

ALL = 'all_rows'


def _fpl_season(year) -> int:
    year = int(year)
    return year + 2000 if year < 100 else year


def slice_stats(units, models, external=None):
    """{slice: {'n', 'actual', model: {'bias', 'mae', 'rmse'}}}, or {} without slice data."""
    if external is None:
        from premierlearning.learning_v2.external_inputs import ExternalInputs

        external = ExternalInputs.load()
    columns = list(external.slice_columns)
    if not columns:
        return {}
    sums = defaultdict(lambda: defaultdict(float))
    for unit in units:
        if unit.codes is None or unit.fixtures is None:
            continue
        season = _fpl_season(unit.season)
        for row, code in enumerate(unit.codes):
            fixtures = unit.fixtures[row][0]
            if not fixtures or not unit.has_fixture[row, 0]:
                continue
            flags = {column: 0 for column in columns}
            if code:
                for fixture_id in fixtures:
                    for column, value in external.slices(season, fixture_id, int(code)).items():
                        flags[column] = max(flags[column], int(value))
            actual = float(unit.actual[row, 0])
            names = [ALL] + [column for column in columns if flags[column]]
            for name in names:
                cell = sums[name]
                cell['n'] += 1
                cell['actual'] += actual
                for model in models:
                    error = float(unit.preds[model][row, 0]) - actual
                    cell[model + ':err'] += error
                    cell[model + ':abs'] += abs(error)
                    cell[model + ':sq'] += error * error
    stats = {}
    for name in [ALL] + columns:
        cell = sums.get(name)
        if not cell or not cell['n']:
            stats[name] = {'n': 0}
            continue
        n = cell['n']
        entry = {'n': int(n), 'actual': cell['actual'] / n}
        for model in models:
            entry[model] = {
                'bias': cell[model + ':err'] / n,
                'mae': cell[model + ':abs'] / n,
                'rmse': math.sqrt(cell[model + ':sq'] / n),
            }
        stats[name] = entry
    return stats


def slice_text(stats, models) -> str:
    if not stats:
        return ''
    names = list(stats)
    lines = [
        '',
        'Slices (next gameweek only; bias = predicted minus actual points, plain units):',
        '  %-36s %7s %7s' % ('slice', 'n', 'actual') + ''.join(' %22s' % model[:22] for model in models),
        '  %-36s %7s %7s' % ('', '', '') + ''.join(' %22s' % 'bias / MAE' for _ in models),
    ]
    for name in names:
        entry = stats[name]
        if not entry.get('n'):
            lines.append('  %-36s %7s' % (name, 0))
            continue
        lines.append('  %-36s %7d %7.2f' % (name, entry['n'], entry['actual']) + ''.join(
            ' %22s' % ('%+.2f / %.2f' % (entry[model]['bias'], entry[model]['mae'])) for model in models
        ))
    return '\n'.join(lines) + '\n'


def write_slices(stats, models, out_dir) -> None:
    rows = []
    for name, entry in stats.items():
        for model in models:
            if not entry.get('n'):
                continue
            rows.append(dict(slice=name, model=model, n=entry['n'], actual=entry['actual'], **entry[model]))
    with (Path(out_dir) / 'slices.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['slice', 'model', 'n', 'actual', 'bias', 'mae', 'rmse'])
        writer.writeheader()
        writer.writerows(rows)


def add_slice_report(units, models, out_dir, log=print) -> str:
    """Write slices.csv, append the table to comparison.txt, return the text ('' when unavailable)."""
    try:
        stats = slice_stats(units, models)
    except Exception as error:  # the report is a diagnostic; never fail an evaluation over it
        log('Slice report skipped: %s' % error)
        return ''
    if not stats:
        log('Slice report skipped: no eval_slices.csv')
        return ''
    write_slices(stats, models, out_dir)
    text = slice_text(stats, models)
    with (Path(out_dir) / 'comparison.txt').open('a', encoding='utf-8') as handle:
        handle.write(text)
    return text
