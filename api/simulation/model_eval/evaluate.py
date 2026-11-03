"""Score every model against v1 on a shared panel and write the report files."""

import csv
import gzip
import json
from pathlib import Path

from simulation.model_eval import scoring
from simulation.model_eval.metrics import RankCache, compute, rank_tables


def _seasons(units):
    return sorted({unit.season for unit in units})


def non_overlapping(units, config):
    """Origins 1, 6, 11, ...: 5-GW windows that share no gameweek."""
    horizon = int(config['horizon'])
    start = int(config['origin_from'])
    return [unit for unit in units if (unit.origin - start) % horizon == 0]


def evaluate(units, models, baseline, config, log=print):
    if baseline not in models:
        raise ValueError('the baseline %s must be one of the evaluated models' % baseline)
    if not units:
        raise ValueError('no evaluation units: build the prediction sets first')
    cache = RankCache()
    log('Scoring %s units for %s models...' % (len(units), len(models)))
    metrics = {}
    for model in models:
        pooled = compute(units, model, config, cache, details=True)
        seasons = {
            season: compute([u for u in units if u.season == season], model, config, cache)
            for season in _seasons(units)
        }
        metrics[model] = {'pooled': pooled, 'seasons': seasons}
    log('Bootstrapping %s resamples (blocks of %s origins)...'
        % (config['bootstrap']['resamples'], config['bootstrap']['block_origins']))
    sigmas, draw_totals = scoring.bootstrap(units, models, baseline, config, cache)
    subset = non_overlapping(units, config)
    base = metrics[baseline]
    results = {
        'baseline': baseline,
        'models': list(models),
        'seasons': _seasons(units),
        'units': len(units),
        'sigmas': sigmas,
        'metrics': metrics,
        'scores': {},
        'season_scores': {},
        'non_overlap_scores': {},
        'intervals': {},
        'flags': {},
        'missing_predictions': {},
        'ranks': {},
    }
    base_subset = compute(subset, baseline, config, cache) if subset else None
    for model in models:
        pooled = scoring.score_model(metrics[model]['pooled'], base['pooled'], sigmas, config)
        season_totals = {
            season: scoring.score_model(
                metrics[model]['seasons'][season], base['seasons'][season], sigmas, config,
            )['total']
            for season in _seasons(units)
        }
        results['scores'][model] = pooled
        results['season_scores'][model] = season_totals
        if subset:
            results['non_overlap_scores'][model] = scoring.score_model(
                compute(subset, model, config, cache), base_subset, sigmas, config,
            )
        results['intervals'][model] = scoring.interval(draw_totals[model])
        flags = scoring.guardrails(
            metrics[model]['pooled'], base['pooled'], pooled['total'], season_totals, config,
        )
        flags.extend('no noise scale for %s; scored 0' % key for key in pooled['invalid'])
        results['flags'][model] = flags if model != baseline else []
        results['missing_predictions'][model] = sum(unit.missing.get(model, 0) for unit in units)
        results['ranks'][model] = rank_tables(units, model, config, cache)
    pair = [name for name in config.get('noise_pair') or [] if name in models]
    if len(pair) == 2:
        results['noise_floor'] = {
            'pair': pair,
            'total_gap': abs(results['scores'][pair[0]]['total'] - results['scores'][pair[1]]['total']),
        }
    return results


def ranking(results):
    return sorted(
        (model for model in results['models'] if model != results['baseline']),
        key=lambda model: -results['scores'][model]['total'],
    )


def _fmt(value, digits=3):
    if value is None:
        return '-'
    return ('%.' + str(digits) + 'f') % value


def summary_text(results, notes=None, config=None):
    notes = notes or {}
    baseline = results['baseline']
    cap = (config or {}).get('score_cap')
    lines = [
        'Model evaluation against %s, seasons %s, %s origins'
        % (baseline, ', '.join(str(s) for s in results['seasons']), results['units']),
        'Scores are in noise units: 1.0 = one bootstrap standard deviation of that metric for v1.',
        'Positive is better than v1; v1 scores 0 by definition.',
        ('Each metric is capped at +-%g before weighting; "uncapped" is the old total.' % cap
         if cap is not None else 'Metric scores are not capped.'),
        '',
        '%-16s %8s %18s %7s %8s %8s %8s %8s %8s %9s' % (
            'model', 'total', '95% interval', 'P(>0)', 'overall', 'future', 'calib', 'ranking',
            'fixtures', 'uncapped'),
    ]
    for model in ranking(results):
        score = results['scores'][model]
        span = results['intervals'][model]
        lines.append('%-16s %+8.2f %18s %7s %+8.2f %+8.2f %+8.2f %+8.2f %+8.2f %+9.2f' % (
            model, score['total'],
            '[%s, %s]' % (_fmt(span['lo'], 2), _fmt(span['hi'], 2)),
            _fmt(span['p_above_zero'], 2),
            score['categories']['overall'], score['categories']['future'],
            score['categories']['calibration'], score['categories']['ranking'],
            score['categories'].get('fixtures', 0.0),
            score.get('total_uncapped', score['total']),
        ))
    if 'noise_floor' in results:
        floor = results['noise_floor']
        lines += [
            '',
            'Training-noise floor: %s and %s differ only by seed and are %.2f apart.'
            % (floor['pair'][0], floor['pair'][1], floor['total_gap']),
            'Treat smaller differences between models as noise.',
        ]
    lines += ['', 'What drove each score:']
    for model in ranking(results):
        rows = results['scores'][model]['components']
        helped = sorted((r for r in rows if r['weighted'] > 0), key=lambda r: -r['weighted'])[:2]
        hurt = [r for r in rows if r['score'] is not None and r['score'] < -1.0]
        text = ', '.join('%s %+.2f' % (r['component'], r['weighted']) for r in helped) or 'nothing'
        line = '  %s: helped most by %s' % (model, text)
        if hurt:
            line += '; regressed on %s' % ', '.join(
                '%s (%+.2f sd)' % (r['component'], r['score']) for r in hurt)
        lines.append(line)
        if notes.get(model):
            lines.append('    %s' % notes[model])
    lines += ['', 'By season (total):']
    for model in ranking(results):
        lines.append('  %-16s %s' % (model, '  '.join(
            '%s %+.2f' % (season, total)
            for season, total in sorted(results['season_scores'][model].items()))))
    if results['non_overlap_scores']:
        lines += ['', 'Non-overlapping 5-GW origins only (sensitivity check, same sigmas):']
        for model in ranking(results):
            score = results['non_overlap_scores'][model]
            lines.append('  %-16s total %+.2f, future %+.2f, ranking %+.2f' % (
                model, score['total'], score['categories']['future'], score['categories']['ranking']))
    lines += ['', 'Guardrail flags:']
    flagged = False
    for model in ranking(results):
        for flag in results['flags'][model]:
            lines.append('  %s: %s' % (model, flag))
            flagged = True
    if not flagged:
        lines.append('  none')
    lines += ['', 'Raw metrics (pooled):']
    keys = (
        'overall_mse', 'overall_mae', 'overall_bias', '2A_actual', '2A_predicted',
        '2B_actual', '2B_predicted', 'calibration_gap', 'calibration_slope',
        'spearman_single_gw', 'spearman_five_gw',
    )
    lines.append('  %-16s ' % 'model' + ' '.join('%12s' % key[:12] for key in keys))
    for model in [baseline] + ranking(results):
        pooled = results['metrics'][model]['pooled']
        lines.append('  %-16s ' % model + ' '.join('%12s' % _fmt(pooled.get(key)) for key in keys))
    lines += [
        '',
        'Fixture swings: players who played 60+ minutes in each of the next five gameweeks,',
        'one fixture each. corr: predicted rises and falls vs real ones (higher is better).',
        'slope: real swing per point of predicted swing (1 is right; above 1, the model\'s',
        'swings are too small). mse: error on the swings.',
    ]
    swing_keys = ('fixture_swing_corr', 'fixture_swing_slope', 'fixture_swing_mse', 'fixture_swing_n')
    lines.append('  %-16s ' % 'model' + ' '.join('%12s' % key[len('fixture_swing_'):] for key in swing_keys))
    for model in [baseline] + ranking(results):
        pooled = results['metrics'][model]['pooled']
        lines.append('  %-16s ' % model + ' '.join(
            '%12s' % (_fmt(pooled.get(key)) if key != 'fixture_swing_n' else pooled.get(key, '-'))
            for key in swing_keys
        ))
    missing = {m: n for m, n in results['missing_predictions'].items() if n}
    if missing:
        lines += ['', 'Eligible player-fixtures with no stored prediction (scored as 0):']
        lines += ['  %s: %s' % (model, count) for model, count in sorted(missing.items())]
    return '\n'.join(lines) + '\n'


def _write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(results, units, out_dir, config, extra=None, write_panel=True, notes=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = results['models']
    with (out / 'config.json').open('w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    raw_rows = []
    for model in models:
        scopes = [('pooled', results['metrics'][model]['pooled'])]
        scopes += [(str(s), m) for s, m in sorted(results['metrics'][model]['seasons'].items())]
        for scope, values in scopes:
            for key, value in sorted(values.items()):
                if isinstance(value, (list, dict)):
                    continue
                raw_rows.append({'model': model, 'scope': scope, 'metric': key, 'value': value})
    _write_csv(out / 'raw_metrics.csv', raw_rows, ['model', 'scope', 'metric', 'value'])

    score_rows = []
    for model in models:
        span = results['intervals'][model]
        for row in results['scores'][model]['components']:
            score_rows.append(dict(model=model, **row))
        for category, value in results['scores'][model]['categories'].items():
            score_rows.append({'model': model, 'component': 'category:' + category, 'score': value})
        score_rows.append({
            'model': model, 'component': 'total', 'score': results['scores'][model]['total'],
            'uncapped_score': results['scores'][model].get('total_uncapped'),
            'ci_lo': span['lo'], 'ci_hi': span['hi'], 'p_above_zero': span['p_above_zero'],
        })
        for season, total in sorted(results['season_scores'][model].items()):
            score_rows.append({'model': model, 'component': 'total:season_%s' % season, 'score': total})
    _write_csv(out / 'scores.csv', score_rows, [
        'model', 'component', 'metric', 'direction', 'weight', 'model_value', 'v1_value',
        'sigma', 'score', 'uncapped_score', 'weighted', 'relative_pct', 'ci_lo', 'ci_hi', 'p_above_zero',
    ])

    bucket_rows = []
    bin_rows = []
    for model in models:
        pooled = results['metrics'][model]['pooled']
        bucket_rows += [dict(model=model, **row) for row in pooled.get('buckets', [])]
        bin_rows += [dict(model=model, scope='pooled', **row) for row in pooled.get('calibration_bins', [])]
    _write_csv(out / 'buckets.csv', bucket_rows, [
        'model', 'horizon', 'side', 'bucket', 'count', 'mae', 'mean_predicted', 'mean_actual', 'signed_error',
    ])
    _write_csv(out / 'calibration_bins.csv', bin_rows, [
        'model', 'scope', 'bin', 'n', 'lo', 'hi', 'mean_predicted', 'mean_actual', 'gap',
    ])
    single_rows = []
    five_rows = []
    for model in models:
        single, five = results['ranks'][model]
        single_rows += [dict(model=model, **row) for row in single]
        five_rows += [dict(model=model, **row) for row in five]
    _write_csv(out / 'ranking_single_gw.csv', single_rows, ['model', 'season', 'gameweek', 'n', 'spearman'])
    _write_csv(out / 'ranking_five_gw.csv', five_rows, ['model', 'season', 'origin', 'n', 'spearman'])

    order = ranking(results)
    _write_csv(out / 'comparison.csv', [
        {
            'rank': index, 'model': model,
            'total': results['scores'][model]['total'],
            'total_uncapped': results['scores'][model].get('total_uncapped'),
            'ci_lo': results['intervals'][model]['lo'],
            'ci_hi': results['intervals'][model]['hi'],
            'p_above_zero': results['intervals'][model]['p_above_zero'],
            **{'category_' + k: v for k, v in results['scores'][model]['categories'].items()},
            'flags': ' | '.join(results['flags'][model]),
        }
        for index, model in enumerate(order, start=1)
    ], ['rank', 'model', 'total', 'total_uncapped', 'ci_lo', 'ci_hi', 'p_above_zero', 'category_overall',
        'category_future', 'category_calibration', 'category_ranking', 'category_fixtures', 'flags'])

    summary = {
        key: results[key] for key in (
            'baseline', 'models', 'seasons', 'units', 'sigmas', 'season_scores',
            'intervals', 'flags', 'missing_predictions',
        )
    }
    summary['totals'] = {model: results['scores'][model]['total'] for model in models}
    summary['totals_uncapped'] = {
        model: results['scores'][model].get('total_uncapped') for model in models
    }
    summary['categories'] = {model: results['scores'][model]['categories'] for model in models}
    summary['ranking'] = order
    if 'noise_floor' in results:
        summary['noise_floor'] = results['noise_floor']
    summary.update(extra or {})
    with (out / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)
    text = summary_text(results, notes, config)
    (out / 'comparison.txt').write_text(text, encoding='utf-8')

    if write_panel:
        with gzip.open(out / 'panel.csv.gz', 'wt', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ['season', 'origin', 'target_gameweek', 'lead', 'player_id', 'has_fixture', 'actual']
                + ['pred_' + model for model in models]
            )
            for unit in units:
                for row, player_id in enumerate(unit.player_ids.tolist()):
                    for lead in range(unit.valid_leads):
                        writer.writerow(
                            [unit.season, unit.origin, unit.target(lead), lead, player_id,
                             int(unit.has_fixture[row, lead]), unit.actual[row, lead]]
                            + ['%.4f' % unit.preds[model][row, lead] for model in models]
                        )
    return text
