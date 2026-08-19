"""Offline gameweek-by-gameweek historical season replay."""

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from persistence import DB_Handler
from simulation.actual_scoring import score_lineup
from simulation.historical_data import VaastavSeasonData
from squadtools.simulation_strategy import (
    FreeTransferRules,
    OwnedPlayer,
    PredictionPool,
    SimulationStrategy,
    SquadState,
    StrategyConfig,
    apply_transfer_plan,
    pick_lineup,
    selling_price,
)


@dataclass
class SimulationConfig:
    prediction_set_id: int
    season_start_year: int
    gameweek_from: int = 1
    gameweek_to: int = 38
    horizon_gameweeks: int = 5
    min_current_gw_gain: float = 0.0
    min_horizon_gain: float = 0.0
    max_transfers_per_gameweek: int = 5
    initial_budget: int = 1000
    output_dir: Optional[str] = None

    def hash(self):
        values = asdict(self)
        values.pop('output_dir', None)
        payload = json.dumps(values, sort_keys=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class SimRunner:
    def __init__(self, config: SimulationConfig, db_handler: Optional[DB_Handler] = None):
        if not (
            1
            <= int(config.gameweek_from)
            <= int(config.gameweek_to)
            <= 38
        ):
            raise ValueError('gameweek range must satisfy 1 <= from <= to <= 38')
        if not 1 <= int(config.horizon_gameweeks) <= 38:
            raise ValueError('strategy horizon must be between 1 and 38 gameweeks')
        self.config = config
        self.source = VaastavSeasonData(config.season_start_year)
        self.db = db_handler or DB_Handler()
        self._owns_db = db_handler is None
        self.strategy = SimulationStrategy(StrategyConfig(
            horizon_gameweeks=config.horizon_gameweeks,
            min_current_gw_gain=config.min_current_gw_gain,
            min_horizon_gain=config.min_horizon_gain,
            max_transfers_per_gameweek=config.max_transfers_per_gameweek,
        ))
        self.ft_rules = FreeTransferRules(config.season_start_year)
        self.gameweek_logs = []
        self.run_id = None

    def _pool(self, gameweek):
        rows = self.db.get_historical_predictions(
            self.config.prediction_set_id,
            self.config.season_start_year,
            gameweek,
        )
        if not rows:
            raise ValueError(
                f'Prediction set {self.config.prediction_set_id} has no complete '
                f'{self.config.season_start_year}/{self.config.season_start_year + 1} '
                f'GW{gameweek} snapshot'
            )
        return PredictionPool.from_rows(rows)

    def _initial_state(self, pool, gameweek):
        player_ids = self.strategy.build_initial_squad(
            pool,
            as_of_gameweek=gameweek,
            budget=self.config.initial_budget,
        )
        spent = sum(pool[player_id].price for player_id in player_ids)
        return SquadState(
            owned={
                player_id: OwnedPlayer(
                    player_id=player_id,
                    purchase_price=pool[player_id].price,
                )
                for player_id in player_ids
            },
            bank=int(self.config.initial_budget - spent),
            free_transfers=0,
        )

    def _expected_horizon(self, player_ids, pool, gameweek):
        gameweeks = range(
            int(gameweek),
            min(39, int(gameweek) + self.config.horizon_gameweeks),
        )
        return sum(
            pick_lineup(player_ids, pool, target_gw).expected_points
            for target_gw in gameweeks
        )

    def run(self):
        prediction_set = self.db.get_prediction_set(self.config.prediction_set_id)
        if prediction_set is None:
            if self._owns_db:
                self.db.close_connection()
            raise ValueError(f'Unknown prediction set {self.config.prediction_set_id}')
        prediction_scope = prediction_set['config'].get('prediction_scope')
        if (
            prediction_scope != 'rest_of_season'
            and self.config.horizon_gameweeks > prediction_set['horizon_gameweeks']
        ):
            if self._owns_db:
                self.db.close_connection()
            raise ValueError(
                f'Simulation horizon {self.config.horizon_gameweeks} exceeds '
                f'prediction horizon {prediction_set["horizon_gameweeks"]}'
            )
        self.run_id = self.db.create_simulation_run(
            self.config.prediction_set_id,
            self.config.season_start_year,
            self.config.hash(),
            asdict(self.config),
        )
        state = None
        cumulative = 0.0
        total_transfers = 0
        availability_warning_count = 0
        final_pool = None
        try:
            for gameweek in range(
                self.config.gameweek_from,
                self.config.gameweek_to + 1,
            ):
                pool = self._pool(gameweek)
                final_pool = pool
                if state is None:
                    state = self._initial_state(pool, gameweek)
                    free_before = 15
                    expected_hold_current = 0.0
                    expected_hold_horizon = 0.0
                    expected_selected_current = pick_lineup(
                        state.player_ids, pool, gameweek
                    ).expected_points
                    expected_selected_horizon = self._expected_horizon(
                        state.player_ids, pool, gameweek
                    )
                    transfers = ()
                else:
                    state.free_transfers = self.ft_rules.at_start_of_gameweek(
                        gameweek, state.free_transfers
                    )
                    free_before = state.free_transfers
                    plan = self.strategy.plan(state, pool, gameweek)
                    state = apply_transfer_plan(state, plan)
                    transfers = plan.transfers
                    expected_hold_current = plan.expected_hold_current
                    expected_hold_horizon = plan.expected_hold_horizon
                    expected_selected_current = plan.expected_selected_current
                    expected_selected_horizon = plan.expected_selected_horizon

                lineup = pick_lineup(state.player_ids, pool, gameweek)
                actual_score = score_lineup(
                    lineup,
                    pool,
                    self.source.actuals_for_gameweek(gameweek),
                )
                cumulative += actual_score.points
                total_transfers += len(transfers)
                availability_warning_count += (
                    actual_score.unexpected_zero_minute_starters
                )
                free_after = self.ft_rules.after_gameweek(
                    gameweek,
                    free_before if gameweek > 1 else 0,
                    len(transfers),
                )
                state.free_transfers = free_after
                transfer_rows = [
                    {
                        'player_out_id': transfer.player_out_id,
                        'player_in_id': transfer.player_in_id,
                        'sell_price': transfer.sell_price,
                        'buy_price': transfer.buy_price,
                    }
                    for transfer in transfers
                ]
                gw_row = {
                    'gameweek': gameweek,
                    'free_transfers_before': free_before,
                    'free_transfers_after': free_after,
                    'bank': state.bank,
                    'expected_hold_current': expected_hold_current,
                    'expected_selected_current': expected_selected_current,
                    'expected_hold_horizon': expected_hold_horizon,
                    'expected_selected_horizon': expected_selected_horizon,
                    'actual_points': actual_score.points,
                    'cumulative_points': cumulative,
                    'captain_id': lineup.captain_id,
                    'vice_captain_id': lineup.vice_captain_id,
                    'captain_scored_id': actual_score.captain_scored_id,
                    'starters': list(lineup.starters),
                    'bench': list(lineup.bench),
                    'autosubs': list(actual_score.autosubs),
                    'autosub_count': len(actual_score.autosubs),
                    'captain_fallback': int(
                        actual_score.captain_scored_id != lineup.captain_id
                    ),
                    'unexpected_zero_minute_starters': (
                        actual_score.unexpected_zero_minute_starters
                    ),
                    'transfer_count': len(transfer_rows),
                    'transfers': transfer_rows,
                    'current_gain': (
                        expected_selected_current - expected_hold_current
                    ),
                    'horizon_gain': (
                        expected_selected_horizon - expected_hold_horizon
                    ),
                }
                self.db.persist_simulation_gameweek(self.run_id, gw_row)
                self.db.persist_simulation_transfers(
                    self.run_id, gameweek, transfer_rows
                )
                self.db.connection.commit()
                self.gameweek_logs.append(gw_row)
                print(
                    f'GW{gameweek}: {actual_score.points:.0f} pts '
                    f'({cumulative:.0f} total), {len(transfers)} transfer(s), '
                    f'bank £{state.bank / 10:.1f}m, FT next {free_after}'
                )

            final_value = int(state.bank)
            for player_id, owned in state.owned.items():
                final_value += selling_price(
                    owned.purchase_price,
                    final_pool[player_id].price,
                )
            autosub_recoveries = sum(
                len(row['autosubs']) for row in self.gameweek_logs
            )
            availability_affected_gameweeks = sum(
                1
                for row in self.gameweek_logs
                if row['unexpected_zero_minute_starters'] > 0
            )
            zero_minute_captains = sum(
                1
                for row in self.gameweek_logs
                if row['captain_scored_id'] != row['captain_id']
            )
            self.db.complete_simulation_run(
                self.run_id,
                cumulative,
                total_transfers,
                state.bank,
                final_value,
                availability_warning_count,
                autosub_recoveries,
                availability_affected_gameweeks,
                zero_minute_captains,
            )
            summary = {
                'simulation_run_id': self.run_id,
                'prediction_set_id': self.config.prediction_set_id,
                'prediction_scope': prediction_scope or 'fixed_horizon',
                'season': (
                    f'{self.config.season_start_year}/'
                    f'{self.config.season_start_year + 1}'
                ),
                'total_points': cumulative,
                'transfers_made': total_transfers,
                'hold_gameweeks': sum(
                    1
                    for row in self.gameweek_logs[1:]
                    if not row['transfers']
                ),
                'final_bank': state.bank,
                'final_squad_value': final_value,
                'unexpected_zero_minute_starters': availability_warning_count,
                'autosub_recoveries': autosub_recoveries,
                'availability_affected_gameweeks': availability_affected_gameweeks,
                'zero_minute_captains': zero_minute_captains,
                'availability_caveat': (
                    'Weekly historical injury percentages were unavailable; '
                    'predicted minutes were used as the availability proxy.'
                ),
                'fixture_caveat': (
                    'Predictions use the final season fixture calendar, including '
                    'reschedules that may not have been announced at the deadline.'
                ),
                'config': asdict(self.config),
            }
            self._write_reports(summary)
            return summary
        except Exception:
            self.db.fail_simulation_run(self.run_id)
            raise
        finally:
            if self._owns_db:
                self.db.close_connection()

    def _write_reports(self, summary):
        if self.config.output_dir:
            output_dir = Path(self.config.output_dir)
        else:
            output_dir = (
                Path(__file__).resolve().parents[1]
                / 'run_metrics'
                / 'simulations'
                / f'run_{self.run_id}'
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'summary.json').write_text(
            json.dumps(summary, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        (output_dir / 'gameweeks.json').write_text(
            json.dumps(self.gameweek_logs, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        csv_columns = [
            'gameweek',
            'actual_points',
            'cumulative_points',
            'free_transfers_before',
            'free_transfers_after',
            'bank',
            'transfer_count',
            'current_gain',
            'horizon_gain',
            'autosub_count',
            'captain_fallback',
            'unexpected_zero_minute_starters',
        ]
        with (output_dir / 'gameweeks.csv').open(
            'w', newline='', encoding='utf-8'
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writeheader()
            for row in self.gameweek_logs:
                writer.writerow({column: row[column] for column in csv_columns})
        print(f'Wrote simulation reports to {output_dir}')
