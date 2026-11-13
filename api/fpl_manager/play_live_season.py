"""Play a trained policy (top choice) and the rules manager over the live season so far.

    python -m fpl_manager.play_live_season --set-name eval-lgbm_v6-2627-replay --last 10 \
        --checkpoint fpl_manager_runs\\v3_lgbm_v6_copies_fixed --out fpl_manager_runs\\season_2627_replay

2026-11-13, Hal asked what the current run's policy would have scored in
2026/27 so far, with its team, chips and transfers each week. The forecasts
are a replayed prediction set (simulation.replay_live_season build), made at
each deadline from what was known then. The live season is read as a past one
(simulation.replay_live_season), on the final calendar (no 2026/27 fixture had
moved as of GW10). Scored like eval.csv: real points with autosubs, 4-point
hits. The policy file is only read.

Writes <out>/<manager>_weeks.csv (squad, XI, bench, captain, transfers, chip,
points each week) and <out>/summary.json.
"""
from __future__ import annotations

from simulation.replay_live_season import repatch, treat_live_season_as_past

LIVE = treat_live_season_as_past()

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

POSITIONS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}


def _team_names(season):
    from simulation.historical_data import VaastavSeasonData

    return {int(float(row['id'])): row['short_name'] for row in VaastavSeasonData(season).teams_rows}


def play(db, set_id, season, last, choose, label, out_dir, calendar):
    from fpl_manager.actions import ACTIONS
    from fpl_manager.lineup import greedy_lineup
    from fpl_manager.season_source import ManagedSeason
    from squadtools.simulation_strategy import apply_transfer_plan, copy_squad_state

    managed = ManagedSeason(db, set_id, season, 1, last, observation_version='v3', calendar=calendar)
    episode = managed.episode()
    teams = _team_names(season)
    captured = {}
    execute = episode.optimiser.execute

    def recording(action, before, pool, gameweek):
        plan, chip = execute(action, before, pool, gameweek)
        captured['plan'] = plan
        return plan, chip

    episode.optimiser.execute = recording
    rows = []
    while not episode.done:
        observation = episode.observation()
        mask = episode.mask()
        gameweek = episode.gameweek
        pool = episode.pool
        before = copy_squad_state(episode.squad)
        action = choose(observation, mask)
        _obs, _reward, _done, info = episode.step(action)
        plan = captured['plan']
        squad = apply_transfer_plan(before, plan).player_ids if plan.transfers else before.player_ids
        predicted = lambda player_id: float(pool[player_id].points(gameweek))
        starters, bench, captain, vice = greedy_lineup(squad, predicted, lambda pid: pool[pid].position)

        def name(player_id):
            player = pool[player_id]
            return '%s (%s %s %.1f)' % (player.name, teams.get(player.team_id, '?'),
                                        POSITIONS[player.position], player.price / 10.0)

        rows.append({
            'gameweek': gameweek,
            'action': ACTIONS[action],
            'chip': info.get('chip') or '',
            'free_transfers_before': before.free_transfers,
            'transfers': len(plan.transfers),
            'out': '; '.join(name(t.player_out_id) for t in plan.transfers),
            'in': '; '.join(name(t.player_in_id) for t in plan.transfers),
            'hits': info['hits'],
            'points': info['points'],
            'total': episode.points_total,
            'captain': pool[captain].name,
            'vice': pool[vice].name,
            'xi': '; '.join(name(pid) for pid in starters),
            'bench': '; '.join(name(pid) for pid in bench),
            'bank': episode.squad.bank / 10.0,
            'predicted_xi_points': round(sum(predicted(pid) for pid in starters) + predicted(captain), 1),
        })
        print('%s GW%s %-14s %s transfers, %s points (total %s), captain %s' % (
            label, gameweek, ACTIONS[action], len(plan.transfers), info['points'],
            episode.points_total, pool[captain].name), flush=True)
    out = Path(out_dir) / ('%s_weeks.csv' % label)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return float(episode.points_total)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='fpl_manager.play_live_season')
    parser.add_argument('--set-name')
    # 2026-11-13: also plays a past season week by week (e.g. --set-id 105 --season 23
    # --last 38 --calendar known, as the training run's real-season checks).
    parser.add_argument('--set-id', type=int)
    parser.add_argument('--managers', default='policy,rules')
    parser.add_argument('--season', type=int, default=LIVE)
    parser.add_argument('--last', type=int, default=10)
    parser.add_argument('--checkpoint', default=r'fpl_manager_runs\v3_lgbm_v6_copies_fixed')
    parser.add_argument('--calendar', default='final')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    from flask import Flask

    from persistence import DB_Handler
    from synthetic_seasons.manager_check import policy_chooser, rules_chooser

    repatch()
    policy, steps = policy_chooser(args.checkpoint)
    print('Policy %s at step %s' % (args.checkpoint, steps), flush=True)
    app = Flask(__name__)
    summary = {'policy_steps': steps}
    with app.app_context():
        db = DB_Handler()
        try:
            if args.set_id is not None:
                set_id = int(args.set_id)
            else:
                db.cursor.execute(db._sql('select max(id) from prediction_sets where name = ?'), (args.set_name,))
                found = db.cursor.fetchone()
                if not found or found[0] is None:
                    raise SystemExit('No prediction set named %s' % args.set_name)
                set_id = int(found[0])
            summary['prediction_set_id'] = set_id
            print('Prediction set %s is %s' % (args.set_name, set_id), flush=True)
            managers = [value.strip() for value in args.managers.split(',') if value.strip()]
            for label, choose in (('policy', policy), ('rules', rules_chooser())):
                if label not in managers:
                    continue
                summary[label] = play(db, set_id, int(args.season) % 100, args.last, choose, label,
                                      args.out, args.calendar)
        finally:
            db.close_connection()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
