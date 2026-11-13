"""Make finished evaluation sets usable by the fantasy manager.

v1_h5 and v2_log are used as built. This adds the blend. v2 sets keep 0 predicted
minutes: PredictionPool.predicts_minutes reads that as a model with no minutes net,
so the optimiser and chip planner go by predicted points.

Every set stays named eval-*, so none becomes a default. A manager run names its
set with --prediction-set-id and DATABASE_PATH=/data/model_eval.db.

Run from api/ with that DATABASE_PATH: python -m simulation.model_eval.manager_sets
"""

import argparse
import hashlib
import json

from flask import Flask

from persistence import DB_Handler
from simulation.model_eval.experiments import load_manifest

DEFAULT_SEASONS = (23, 24, 25)
LAST_GAMEWEEK = 38
V1_MODEL = 'v1_h5'
BLEND_BASE = 'v2_log'
BLEND_OTHER = 'v2_log_h5'
BLEND_WEIGHT = 0.25
BLEND_NAME = 'eval-v2_log_blend25'
MINUTES_BLEND_BASE = 'v2_log_minbands'
MINUTES_BLEND_NAME = 'eval-v2_log_minbands_blend25'
ORACLE_MODEL = 'v2_log_oracle_minutes'


def _row_key(row):
    return int(row['target_gameweek']), int(row['fixture_id']), int(row['player_id'])


def _eval_set(db, prediction_set_id):
    prediction_set = db.get_prediction_set(prediction_set_id)
    if prediction_set is None or not prediction_set['name'].startswith('eval-'):
        raise ValueError('Set %s is not an eval-* set in this database' % prediction_set_id)
    return prediction_set


def _rows_by_key(db, prediction_set_id, season, gameweek):
    rows = {_row_key(row): row for row in db.get_historical_predictions(prediction_set_id, season, gameweek)}
    if not rows:
        raise ValueError('Set %s has no complete GW%s snapshot for season %s' % (prediction_set_id, gameweek, season))
    return rows


def _blend_metadata(base, other, weight, name):
    config = dict(base['config'])
    config['blend'] = {'base': base['id'], 'other': other['id'], 'other_weight': float(weight)}
    text = 'blend:%s:%s:%s' % (base['model_config_hash'], other['model_config_hash'], float(weight))
    return {
        'name': name,
        'model_config_hash': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'horizon_gameweeks': base['horizon_gameweeks'],
        'training_policy': base['training_policy'],
        'availability_mode': base['availability_mode'],
        'fixture_mode': base['fixture_mode'],
        'random_seed': base['random_seed'],
        'config': config,
    }


def blend_sets(db, base_id, other_id, weight=BLEND_WEIGHT, name=BLEND_NAME, seasons=DEFAULT_SEASONS):
    """A new set: the base's rows, with points (1 - weight) base + weight other."""
    base = _eval_set(db, base_id)
    other = _eval_set(db, other_id)
    if not name.startswith('eval-'):
        raise ValueError('A blend must stay an eval-* set, got %s' % name)
    set_id = db.get_or_create_prediction_set(_blend_metadata(base, other, weight, name))
    for season in seasons:
        for gameweek in range(1, LAST_GAMEWEEK + 1):
            snapshot_id, should_build = db.begin_prediction_snapshot(
                set_id, season, gameweek, expected_max_target_gameweek=LAST_GAMEWEEK,
            )
            if not should_build:
                continue
            base_rows = _rows_by_key(db, base_id, season, gameweek)
            other_rows = _rows_by_key(db, other_id, season, gameweek)
            if set(base_rows) != set(other_rows):
                raise ValueError('Sets differ in rows at season %s GW%s' % (season, gameweek))
            rows = []
            for key, row in base_rows.items():
                row = dict(row, season_start_year=season, as_of_gameweek=gameweek)
                row['predicted_points'] = (
                    (1.0 - weight) * float(row['predicted_points'])
                    + weight * float(other_rows[key]['predicted_points'])
                )
                rows.append(row)
            db.persist_historical_predictions(snapshot_id, rows)
            warnings = ['%s%% set %s points blended into set %s.' % (weight * 100, other_id, base_id)]
            db.complete_prediction_snapshot(snapshot_id, len(rows), warnings)
    return set_id


def readiness(db, prediction_set_id, seasons=DEFAULT_SEASONS):
    """Per season: complete snapshots, rows, and the share of rows with minutes."""
    report = {}
    for season in seasons:
        db.cursor.execute(db._sql(
            '''select count(distinct s.id), count(*), sum(case when p.predicted_minutes > 0 then 1 else 0 end)
               from prediction_snapshots s
               join historical_player_predictions p on p.prediction_snapshot_id = s.id
               where s.prediction_set_id = ? and s.season_start_year = ? and s.status = 'complete' '''
        ), (int(prediction_set_id), int(season)))
        snapshots, rows, with_minutes = db.cursor.fetchone()
        report[int(season)] = {
            'snapshots': int(snapshots or 0),
            'rows': int(rows or 0),
            'minutes_share': float(with_minutes or 0) / float(rows) if rows else 0.0,
        }
    return report


def prepare(db, manifest, seasons=DEFAULT_SEASONS):
    """Returns {model name: set id} for the three manager sets."""
    v1_id = int(manifest[V1_MODEL]['prediction_set_id'])
    base_id = int(manifest[BLEND_BASE]['prediction_set_id'])
    other_id = int(manifest[BLEND_OTHER]['prediction_set_id'])
    blend_id = blend_sets(db, base_id, other_id, seasons=seasons)
    return {V1_MODEL: v1_id, BLEND_BASE: base_id, BLEND_NAME[len('eval-'):]: blend_id}


def prepare_next(db, manifest, seasons=DEFAULT_SEASONS):
    """Returns {model name: set id} for the minutes blend and the cheating oracle."""
    base_id = int(manifest[MINUTES_BLEND_BASE]['prediction_set_id'])
    other_id = int(manifest[BLEND_OTHER]['prediction_set_id'])
    blend_id = blend_sets(db, base_id, other_id, name=MINUTES_BLEND_NAME, seasons=seasons)
    return {
        MINUTES_BLEND_NAME[len('eval-'):]: blend_id,
        ORACLE_MODEL: int(manifest[ORACLE_MODEL]['prediction_set_id']),
    }


def prepare_blend(db, manifest, base_names, seasons=DEFAULT_SEASONS):
    """Each named experiment blended 25% with v2_log_h5, as the minutes blend (set 35) was."""
    other_id = int(manifest[BLEND_OTHER]['prediction_set_id'])
    sets = {}
    for base_name in base_names:
        name = 'eval-%s_blend25' % base_name
        base_id = int(manifest[base_name]['prediction_set_id'])
        sets[name[len('eval-'):]] = blend_sets(db, base_id, other_id, name=name, seasons=seasons)
    return sets


def prepare_mix(db, manifest, pairs, weight, seasons=DEFAULT_SEASONS):
    """Each (base, other) pair of built experiments blended: (1 - weight) base + weight other.

    Named eval-<base>_mix<percent>_<other>, e.g. lgbm_v1_mix50_v2_b0_seed1.
    """
    sets = {}
    for base_name, other_name in pairs:
        name = 'eval-%s_mix%d_%s' % (base_name, int(round(weight * 100)), other_name)
        base_id = int(manifest[base_name]['prediction_set_id'])
        other_id = int(manifest[other_name]['prediction_set_id'])
        sets[name[len('eval-'):]] = blend_sets(db, base_id, other_id, weight=weight, name=name, seasons=seasons)
    return sets


def main(argv=None):
    from simulation.prediction_builder import _parse_seasons

    parser = argparse.ArgumentParser(description='Prepare evaluation sets for the fantasy manager')
    parser.add_argument('--seasons', type=_parse_seasons, default=DEFAULT_SEASONS)
    parser.add_argument('--next', action='store_true', help='The minutes blend and the oracle')
    parser.add_argument(
        '--blend', action='append', metavar='EXPERIMENT',
        help='Blend this built experiment 25%% with v2_log_h5 (repeatable)',
    )
    parser.add_argument(
        '--mix', nargs=2, action='append', metavar=('BASE', 'OTHER'),
        help='Blend two built experiments, OTHER at --mix-weight (repeatable)',
    )
    parser.add_argument('--mix-weight', type=float, default=0.5)
    args = parser.parse_args(argv)
    app = Flask(__name__)
    with app.app_context():
        db = DB_Handler()
        try:
            if args.mix:
                sets = prepare_mix(db, load_manifest(), args.mix, args.mix_weight, args.seasons)
            elif args.blend:
                sets = prepare_blend(db, load_manifest(), args.blend, args.seasons)
            else:
                sets = (prepare_next if args.next else prepare)(db, load_manifest(), args.seasons)
            complete = True
            for name, set_id in sets.items():
                report = readiness(db, set_id, args.seasons)
                print('%s: set %s %s' % (name, set_id, json.dumps(report)))
                complete = complete and all(
                    entry['snapshots'] == LAST_GAMEWEEK for entry in report.values()
                )
        finally:
            db.close_connection()
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
