"""Check the copy forecast path end to end before a long forecast build.

    python -m synthetic_seasons pipeline-check --seasons 25

1. The real season, forecast the original way (HistoricalPredictionBuilder's
   runner, every past season re-read): the reference.
2. The same real season fed in as a copy (copy 900: real rows, no injuries
   moved, no reshuffle), forecast through SeasonForecaster.predict with the
   past-season cache, after the cache has already served a reshuffled copy
   and other deadlines. Must match 1 exactly: this covers the cache, the copy
   source, and the rebuilt all-competition and schedule inputs.
3. The reshuffled copy (901): each moved fixture is forecast in its copy
   gameweek.
Copies 900 and 901 are removed afterwards. Exit code 1 if anything differs.
"""
from __future__ import annotations

import gc
import shutil
import time

from .forecasts import SeasonForecaster, copy_inputs

GAMEWEEKS = (6, 20)
KEYS = ('player_id', 'fixture_id', 'target_gameweek')
VALUES = ('predicted_minutes', 'predicted_pp90', 'predicted_points', 'chance_of_playing', 'selectable', 'price')


def _real_as_copy(copier):
    from .schedule import Reshuffle

    source = copier.source
    rows = [dict(row) for row in source.rows]
    for row in rows:
        row['_xg'] = row.get('expected_goals', 0.0)
        row['_xa'] = row.get('expected_assists', 0.0)
        row['_xgc'] = row.get('expected_goals_conceded', 0.0)
        # FPL's own figure: xG + xA rounded is not always the same number.
        row['_xgi'] = row.get('expected_goal_involvements', 0.0)
    return {'rows': rows, 'scores': {f.id: (f.home_goals, f.away_goals) for f in source.fixtures.values()},
            'flags': {}, 'meta': {'spells': []}, 'reshuffle': Reshuffle({}, dict(source.fixtures), {}, {}, 0)}


def _fresh(forecaster, gameweek):
    """Rows the way the prediction-set build makes them (no cache)."""
    from simulation.prediction_builder import _learning_v2_argv, _prediction_rows, _simulation_environment

    runner = forecaster.builder._new_runner(forecaster.year, gameweek)
    with _simulation_environment(), _learning_v2_argv(True), copy_inputs(forecaster.real, {}):
        forecaster.fixed.predict(runner, already_prepared=False)
    rows = _prediction_rows(runner.current_season, forecaster.year, gameweek)
    del runner
    gc.collect()
    return rows


def _compare(reference, rows):
    index = {tuple(r[k] for k in KEYS): r for r in reference}
    got = {tuple(r[k] for k in KEYS): r for r in rows}
    worst = 0.0
    for key in set(index) & set(got):
        for name in VALUES:
            a, b = index[key].get(name), got[key].get(name)
            if a is None or b is None:
                worst = max(worst, 0.0 if a is b else float('inf'))
            else:
                worst = max(worst, abs(float(a) - float(b)))
    return len(index), len(got), len(set(index) ^ set(got)), worst


def run(year: int, log=print) -> bool:
    from .generate import SeasonCopier, copy_dir

    copier = SeasonCopier(year)
    real_dir = copier.save(900, _real_as_copy(copier))
    shuffled = copier.make(901)
    shuffled_dir = copier.save(901, shuffled)
    del copier
    ok = True
    try:
        forecaster = SeasonForecaster(year, log=log)
        forecaster.fit()
        reference = {}
        for gameweek in GAMEWEEKS:
            began = time.time()
            reference[gameweek] = _fresh(forecaster, gameweek)
            log('reference GW%d: %d rows, %.0f s (no cache)' % (gameweek, len(reference[gameweek]), time.time() - began))
        source901, loads901 = forecaster.copy_source(901)
        began = time.time()
        forecaster.predict(source901, 3, loads901)
        log('warm-up: copy 901 GW3 %.0f s' % (time.time() - began))
        source900, loads900 = forecaster.copy_source(900)
        for gameweek in GAMEWEEKS:
            for label, source, loads in (('real season', forecaster.real, {}), ('real as copy 900', source900, loads900)):
                began = time.time()
                rows = forecaster.predict(source, gameweek, loads)
                n_ref, n_got, unmatched, worst = _compare(reference[gameweek], rows)
                good = unmatched == 0 and worst < 1e-9
                ok &= good
                log('%s GW%d: %d rows vs %d reference, %d unmatched, largest difference %.3g, %.0f s (cache) -> %s' % (
                    label, gameweek, n_got, n_ref, unmatched, worst, time.time() - began, 'OK' if good else 'DIFFERENT'))
        moved = {fid: f.gameweek for fid, f in shuffled['reshuffle'].fixtures.items()
                 if fid in shuffled['reshuffle'].kickoff_text}
        rows = forecaster.predict(source901, GAMEWEEKS[0], loads901)
        wrong = sum(1 for r in rows if r['fixture_id'] in moved and r['target_gameweek'] != moved[r['fixture_id']])
        placed = sum(1 for r in rows if r['fixture_id'] in moved)
        ok &= wrong == 0 and placed > 0
        log('reshuffled copy 901 GW%d: %d rows, %d on moved fixtures, %d in the wrong gameweek' % (
            GAMEWEEKS[0], len(rows), placed, wrong))
    finally:
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(shuffled_dir, ignore_errors=True)
    log('PIPELINE CHECK %s' % ('PASSED' if ok else 'FAILED'))
    return ok
