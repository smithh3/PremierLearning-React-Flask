"""The live Fargate manager replayed over historical seasons.

    python -m fpl_manager.live_baseline --prediction-set-id 105 --seasons 23,24,25 --workers 3 \
        --out fpl_manager_runs/live_baseline_set105

Each week runs what squadtools/live_strategy.py plan_live does on Fargate:
ChipPlanner(live_chip_config(season)) picks the chip calendar, then
planner.project gives this week's transfers (sweep trial SIM_TRIAL, six weeks
ahead, live bench weights, players already owned always keepable). The plan is
scored by the same SeasonEpisode as the trained policy's evaluation runs (same
opening squad, known fixture calendar, real points with autosubs, 4-point
hits), so the totals compare directly with eval.csv.

Writes <out>/summary.csv (one row per season) and <out>/weeks_<season>.csv
(chip, transfers, hits and points each week).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import pulp

from squadtools import simulation_strategy
from squadtools.chip_strategy import ChipPlanner
from squadtools.live_strategy import live_chip_config
from squadtools.simulation_strategy import PredictionPool, apply_transfer_plan, copy_squad_state

from fpl_manager.percentiles import season_beaten_percent


def _solve_with_highs(model):
    """squadtools' CBC solve, done by HiGHS inside this process.

    Live plan_live solves the same PuLP models with CBC, which starts a
    program for every model; a replayed week runs hundreds of them, about
    ten minutes a week. Writing the model out and solving it here gives the
    same optimum (only equal-points ties can differ) in a fraction of the
    time. Anything HiGHS does not prove optimal goes back to CBC.
    """
    import highspy

    fd, path = tempfile.mkstemp(suffix='.mps')
    os.close(fd)
    try:
        model.writeMPS(path)
        highs = highspy.Highs()
        highs.setOptionValue('output_flag', False)
        highs.setOptionValue('mip_rel_gap', 0.0)
        # The models carry a 1e-9-per-id tie-break; proving it exactly takes
        # HiGHS seconds for no difference in points (fpl_manager/highs_solver.py).
        highs.setOptionValue('mip_abs_gap', 1e-4)
        highs.setOptionValue('time_limit', 120.0)
        highs.readModel(path)
        if model.sense == pulp.LpMaximize:
            highs.changeObjectiveSense(highspy.ObjSense.kMaximize)
        highs.run()
        if highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return _CBC_SOLVE(model)
        values = dict(zip(highs.getLp().col_names_, highs.getSolution().col_value))
    except Exception:  # noqa: BLE001 - CBC is always there as the fallback
        return _CBC_SOLVE(model)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    for variable in model.variables():
        value = float(values.get(variable.name, 0.0))
        if variable.cat == pulp.LpInteger:
            value = float(round(value))
        variable.varValue = value
    model.status = pulp.LpStatusOptimal
    return model.status


_CBC_SOLVE = simulation_strategy._solve_model


def use_highs():
    simulation_strategy._solve_model = _solve_with_highs


class LiveManager:
    """One week at a time, as plan_live decides it on Fargate."""

    def __init__(self, season_start_year):
        self.season_start_year = int(season_start_year)
        self.planner = ChipPlanner(live_chip_config(self.season_start_year))

    def decide(self, squad, pool, gameweek, played):
        gameweek = int(gameweek)
        pool = _owned_keepable(pool, squad.player_ids)
        state = copy_squad_state(squad)
        result = self.planner.plan(state, pool, gameweek, played)
        projected, _trailing = self.planner.project(
            copy_squad_state(squad), pool, gameweek, gameweek, result.calendar,
        )
        return projected[gameweek].plan, result.calendar.chip_at(gameweek)


def _owned_keepable(pool, owned_ids):
    """plan_live's _ensure_pool_has_owned, without changing the episode's pool."""
    players = dict(pool.players)
    for player_id in owned_ids:
        player = players.get(int(player_id))
        if player is not None and not player.selectable:
            players[int(player_id)] = dataclasses.replace(player, selectable=True)
    return PredictionPool(players)


def play_season(args):
    prediction_set_id, year, out_dir, last, highs = args
    from flask import Flask

    from persistence import DB_Handler
    from fpl_manager.season_source import load_managed_season

    out_dir = Path(out_dir)
    app = Flask(__name__)
    began = time.time()
    with app.app_context():
        db = DB_Handler()
        try:
            season = load_managed_season(
                db, prediction_set_id, year, 1, last, observation_version='v3', calendar='known',
            )
            # The opening squad is built first, with CBC, so it is the same
            # squad the policy's runs start from.
            episode = season.episode()
            if highs:
                use_highs()
            manager = LiveManager(year)
            weeks_path = out_dir / ('weeks_%s.csv' % year)
            rows = []
            while not episode.done:
                gameweek = episode.gameweek
                squad_before = set(episode.squad.player_ids)
                before = copy_squad_state(episode.squad)
                plan, chip = manager.decide(episode.squad, episode.pool, gameweek, episode.played)
                pool = episode.pool
                _obs, _reward, _done, info = episode.apply_plan(plan, chip)
                names = lambda ids: '; '.join(
                    pool[player_id].name if player_id in pool else str(player_id) for player_id in ids
                )
                bought = [t.player_in_id for t in plan.transfers]
                sold = [t.player_out_id for t in plan.transfers]
                rows.append({
                    'gameweek': gameweek,
                    'chip': chip or '',
                    'transfers': len(plan.transfers),
                    'hits': info['hits'],
                    'points': info['points'],
                    'total': episode.points_total,
                    'out': names(sold),
                    'in': names(bought),
                    'squad': names(sorted(
                        apply_transfer_plan(before, plan).player_ids if plan.transfers else squad_before)),
                })
                print('%s/%s GW%s %s %s transfers, hits %s, %s points (total %s)' % (
                    year, year + 1, gameweek, chip or '-', len(plan.transfers), info['hits'],
                    info['points'], episode.points_total), flush=True)
            _write(weeks_path, rows)
        finally:
            db.close_connection()
    beaten = season_beaten_percent(year, episode.points_total)
    return {
        'season': '%s/%s' % (year, year + 1),
        'points': round(float(episode.points_total), 1),
        'percentile': None if beaten is None else round(float(beaten), 1),
        'chips': '; '.join('%s GW%s' % (name, week) for name, weeks in (
            ('wildcard', episode.played.wildcards), ('free_hit', episode.played.free_hits),
            ('bench_boost', episode.played.bench_boosts),
            ('triple_captain', episode.played.triple_captains)) for week in weeks),
        'minutes': round((time.time() - began) / 60, 1),
    }


def _write(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='fpl_manager.live_baseline')
    parser.add_argument('--prediction-set-id', type=int, required=True)
    parser.add_argument('--seasons', default='23,24,25')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--out', required=True)
    parser.add_argument('--to-gw', type=int, default=38, help='Stop early (for a quick check)')
    parser.add_argument('--cbc', action='store_true', help='Solve with CBC as live does (slow)')
    args = parser.parse_args(argv)
    years = [int(part) % 100 for part in args.seasons.split(',') if part.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.prediction_set_id, year, str(out_dir), args.to_gw, not args.cbc) for year in years]
    if args.workers > 1:
        with Pool(min(args.workers, len(jobs))) as pool:
            results = pool.map(play_season, jobs)
    else:
        results = [play_season(job) for job in jobs]
    _write(out_dir / 'summary.csv', results)
    for row in results:
        print('Live manager %(season)s: %(points)s points, percentile %(percentile)s, '
              'chips %(chips)s, %(minutes)s min' % row, flush=True)
    print('Live manager total: %s points' % round(sum(row['points'] for row in results), 1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
