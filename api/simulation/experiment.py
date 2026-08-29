"""Run one historical replay and append options plus total score to a log.

Tweak the constants below, then launch this module. No CLI flags.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask

from persistence import DB_Handler
from simulation.sim_runner import SimRunner, SimulationConfig


# --- edit these ---
LABEL = 'chip-strategy'
SEASON = 25
PREDICTION_SET_ID = None  # None = latest complete set for SEASON
FROM_GW = 1
TO_GW = 38
HORIZON = 5
MIN_CURRENT_GAIN = 0.0
MIN_HORIZON_GAIN = 2.0
MAX_TRANSFERS_PER_GW = 5
RELAX_THRESHOLDS_AT_MAX_FT = False
CHIP_STRATEGY = True
FH_THRESHOLD = 11.0
WC_THRESHOLD = 30.0
FH_RECOVERY_THRESHOLD = 8.0
TC_ABOVE_AVERAGE_THRESHOLD = 1.3
TC_WINDOW = 10
# Pass 1: leftover players worth this fraction of a starter.
# Pass 2: freeze the XI and refill the bench with slot weights.
MINUTES_SCALE = None
BENCH_FILL_WEIGHT = 0.3
BENCH_WEIGHTS = (0.5, 0.3, 0.1)
RESERVE_KEEPER_WEIGHT = 0.2
# Used only when CHIP_STRATEGY is False.
WILDCARD_GAMEWEEKS = (6, 24)
FREE_HIT_GAMEWEEKS = (35,)
# After each wildcard, plus GW16 if a bench boost remains after half-season limits.
# 25/26 has one BB per half, so this resolves to GW7 and GW25.
BENCH_BOOST_GAMEWEEKS = (7, 16, 25)
TRIPLE_CAPTAIN_GAMEWEEKS = (13, 30)
# --- end ---

RESULTS_FILE = Path(__file__).resolve().parents[1] / 'run_metrics' / 'simulations' / 'run_log.jsonl'


def run_options():
    options = {
        'label': LABEL,
        'season': SEASON,
        'prediction_set_id': PREDICTION_SET_ID,
        'from_gw': FROM_GW,
        'to_gw': TO_GW,
        'horizon': HORIZON,
        'min_current_gain': MIN_CURRENT_GAIN,
        'min_horizon_gain': MIN_HORIZON_GAIN,
        'max_transfers_per_gw': MAX_TRANSFERS_PER_GW,
        'relax_at_max_ft': RELAX_THRESHOLDS_AT_MAX_FT,
        'chip_strategy': CHIP_STRATEGY,
        'fh_threshold': FH_THRESHOLD,
        'wc_threshold': WC_THRESHOLD,
        'fh_recovery_threshold': FH_RECOVERY_THRESHOLD,
        'tc_above_average_threshold': TC_ABOVE_AVERAGE_THRESHOLD,
        'tc_window': TC_WINDOW,
        'minutes_scale': MINUTES_SCALE,
        'bench_fill_weight': BENCH_FILL_WEIGHT,
        'bench_weights': list(BENCH_WEIGHTS),
        'reserve_keeper_weight': RESERVE_KEEPER_WEIGHT,
    }
    if not CHIP_STRATEGY:
        options.update({
            'wildcard_gameweeks': list(WILDCARD_GAMEWEEKS),
            'free_hit_gameweeks': list(FREE_HIT_GAMEWEEKS),
            'bench_boost_gameweeks': list(BENCH_BOOST_GAMEWEEKS),
            'triple_captain_gameweeks': list(TRIPLE_CAPTAIN_GAMEWEEKS),
        })
    return options


def _append_result(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(row, sort_keys=True) + '\n')


def main():
    options = run_options()
    print('Run options:')
    print(json.dumps(options, indent=2, sort_keys=True))

    season = int(SEASON)
    if season >= 2000:
        season %= 100

    app = Flask(__name__)
    with app.app_context():
        db_handler = DB_Handler()
        try:
            prediction_set_id = PREDICTION_SET_ID
            if prediction_set_id is None:
                prediction_set_id = db_handler.latest_prediction_set_id(season)
            if prediction_set_id is None:
                raise SystemExit(
                    f'No complete prediction set exists for {season}/{season + 1}'
                )
            config_kwargs = dict(
                prediction_set_id=prediction_set_id,
                season_start_year=season,
                gameweek_from=FROM_GW,
                gameweek_to=TO_GW,
                horizon_gameweeks=HORIZON,
                min_current_gw_gain=MIN_CURRENT_GAIN,
                min_horizon_gain=MIN_HORIZON_GAIN,
                max_transfers_per_gameweek=MAX_TRANSFERS_PER_GW,
                relax_thresholds_at_max_ft=RELAX_THRESHOLDS_AT_MAX_FT,
                chip_strategy=CHIP_STRATEGY,
                fh_threshold=FH_THRESHOLD,
                wc_threshold=WC_THRESHOLD,
                fh_recovery_threshold=FH_RECOVERY_THRESHOLD,
                tc_above_average_threshold=TC_ABOVE_AVERAGE_THRESHOLD,
                tc_window=TC_WINDOW,
                minutes_scale=MINUTES_SCALE,
                bench_fill_weight=BENCH_FILL_WEIGHT,
                bench_weights=BENCH_WEIGHTS,
                reserve_keeper_weight=RESERVE_KEEPER_WEIGHT,
            )
            if not CHIP_STRATEGY:
                config_kwargs.update(
                    wildcard_gameweeks=WILDCARD_GAMEWEEKS,
                    free_hit_gameweeks=FREE_HIT_GAMEWEEKS,
                    bench_boost_gameweeks=BENCH_BOOST_GAMEWEEKS,
                    triple_captain_gameweeks=TRIPLE_CAPTAIN_GAMEWEEKS,
                )
            summary = SimRunner(
                SimulationConfig(**config_kwargs),
                db_handler=db_handler,
            ).run()
        finally:
            db_handler.close_connection()

    row = {
        'finished_at': datetime.now(timezone.utc).isoformat(),
        'season': summary['season'],
        'total_points': summary['total_points'],
        'transfers_made': summary['transfers_made'],
        'simulation_run_id': summary['simulation_run_id'],
        'prediction_set_id': summary['prediction_set_id'],
        'options': {
            **options,
            'prediction_set_id': summary['prediction_set_id'],
        },
    }
    _append_result(RESULTS_FILE, row)
    print(json.dumps(row, indent=2, sort_keys=True))
    print(f'Appended to {RESULTS_FILE}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
