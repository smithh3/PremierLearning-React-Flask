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
from simulation.overall_rank import estimate_overall_rank, format_rank_line
from squadtools.chip_strategy import (
    EARLIEST_CHIP_GAMEWEEK,
    FH_AFTER_WILDCARD_WEEKS,
    DGW_H2_MULT_DEFAULT,
    TC_EASY_UNTIL_WEEK,
    TC_MIN_CAPTAIN_POINTS,
    TC_TAPER_FLOOR,
    WC_EARLY_UNTIL_GAMEWEEK,
    ChipPlanner,
    ChipStrategyConfig,
    add_played_chip,
    visible_chip_plan,
)
from squadtools.simulation_strategy import (
    ChipCalendar,
    FreeTransferRules,
    OwnedPlayer,
    PredictionPool,
    SimulationStrategy,
    SquadState,
    StrategyConfig,
    apply_transfer_plan,
    copy_squad_state,
    expected_gameweek_points,
    flat_expected_points,
    pick_lineup,
    predicted_points_line,
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
    min_horizon_gain: float = 2.0
    max_transfers_per_gameweek: int = 5
    relax_thresholds_at_max_ft: bool = False
    wildcard_gameweeks: tuple = (6, 24)
    free_hit_gameweeks: tuple = (35,)
    bench_boost_gameweeks: tuple = (7, 16, 25)
    triple_captain_gameweeks: tuple = (13, 30)
    chip_strategy: bool = False
    fh_threshold: float = 11.0
    wc_threshold: float = 30.0
    fh_recovery_threshold: float = 8.0
    tc_above_average_threshold: float = 1.3
    tc_window: int = 10
    initial_budget: int = 1000
    minutes_scale: Optional[str] = None
    bench_fill_weight: Optional[float] = None
    bench_weights: tuple = ()
    reserve_keeper_weight: Optional[float] = None
    earliest_chip_gameweek: int = 5
    tc_taper_floor: float = TC_TAPER_FLOOR
    tc_min_captain_points: float = TC_MIN_CAPTAIN_POINTS
    tc_easy_until_week: int = TC_EASY_UNTIL_WEEK
    fh_after_wildcard_weeks: int = FH_AFTER_WILDCARD_WEEKS
    output_dir: Optional[str] = None
    roll_free_transfers: bool = False
    fh_dgw_weight: float = 0.0
    wc_dgw_weight: float = 0.0
    dgw_h2_mult: float = DGW_H2_MULT_DEFAULT
    wc_rotation_weight: float = 0.0
    wc_early_weight: float = 0.0
    wc_early_until_gameweek: int = WC_EARLY_UNTIL_GAMEWEEK

    _FIXED_CHIP_WEEK_FIELDS = (
        'wildcard_gameweeks',
        'free_hit_gameweeks',
        'bench_boost_gameweeks',
        'triple_captain_gameweeks',
    )

    def to_dict(self, include_output_dir=True):
        values = asdict(self)
        if not include_output_dir:
            values.pop('output_dir', None)
        if self.chip_strategy:
            for key in self._FIXED_CHIP_WEEK_FIELDS:
                values.pop(key, None)
        return values

    def hash(self):
        payload = json.dumps(self.to_dict(include_output_dir=False), sort_keys=True)
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
        self.ft_rules = FreeTransferRules(config.season_start_year)
        if config.chip_strategy:
            self.chips = ChipCalendar()
            self.chip_planner = ChipPlanner(ChipStrategyConfig(
                season_start_year=config.season_start_year,
                horizon_gameweeks=config.horizon_gameweeks,
                fh_threshold=config.fh_threshold,
                wc_threshold=config.wc_threshold,
                fh_recovery_threshold=config.fh_recovery_threshold,
                tc_above_average_threshold=config.tc_above_average_threshold,
                tc_window=config.tc_window,
                min_current_gw_gain=config.min_current_gw_gain,
                min_horizon_gain=config.min_horizon_gain,
                max_transfers_per_gameweek=config.max_transfers_per_gameweek,
                relax_thresholds_at_max_ft=config.relax_thresholds_at_max_ft,
                bench_fill_weight=config.bench_fill_weight,
                bench_weights=config.bench_weights,
                reserve_keeper_weight=config.reserve_keeper_weight,
                earliest_chip_gameweek=config.earliest_chip_gameweek,
                tc_taper_floor=config.tc_taper_floor,
                tc_min_captain_points=config.tc_min_captain_points,
                tc_easy_until_week=config.tc_easy_until_week,
                fh_after_wildcard_weeks=config.fh_after_wildcard_weeks,
                roll_free_transfers=config.roll_free_transfers,
                fh_dgw_weight=config.fh_dgw_weight,
                wc_dgw_weight=config.wc_dgw_weight,
                dgw_h2_mult=config.dgw_h2_mult,
                wc_rotation_weight=config.wc_rotation_weight,
                wc_early_weight=config.wc_early_weight,
                wc_early_until_gameweek=config.wc_early_until_gameweek,
            ))
        else:
            self.chips = ChipCalendar.resolve(
                config.season_start_year,
                config.wildcard_gameweeks,
                config.free_hit_gameweeks,
                config.bench_boost_gameweeks,
                config.triple_captain_gameweeks,
            )
            self.chip_planner = None
        self.played_chips = ChipCalendar()
        self.chip_plan_reason = {}
        self.free_hit_plans = {}
        self.wildcard_plans = {}
        self.strategy = SimulationStrategy(StrategyConfig(
            horizon_gameweeks=config.horizon_gameweeks,
            min_current_gw_gain=config.min_current_gw_gain,
            min_horizon_gain=config.min_horizon_gain,
            max_transfers_per_gameweek=config.max_transfers_per_gameweek,
            max_free_transfers=self.ft_rules.max_bank,
            relax_thresholds_at_max_ft=config.relax_thresholds_at_max_ft,
            chips=self.chips,
            bench_fill_weight=config.bench_fill_weight,
            bench_weights=config.bench_weights,
            reserve_keeper_weight=config.reserve_keeper_weight,
            season_start_year=config.season_start_year,
            roll_free_transfers=config.roll_free_transfers,
        ))
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
        return PredictionPool.from_rows(rows, minutes_scale=self.config.minutes_scale)

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

    def _refresh_chip_plan(self, state, pool, gameweek):
        """Replan unused chips from this week's squad. Only played chips stay locked."""
        if self.chip_planner is None:
            self.chip_plan_reason = {}
            self.free_hit_plans = {}
            self.wildcard_plans = {}
            return
        result = self.chip_planner.plan(
            state, pool, gameweek, self.played_chips
        )
        self.chips = result.calendar
        self.strategy.config.chips = self.chips
        self.chip_plan_reason = result.reason
        self.free_hit_plans = result.free_hit_plans or {}
        self.wildcard_plans = result.wildcard_plans or {}

    def _this_week_transfer_plan(self, state, pool, gameweek, chip):
        """One ordinary transfer MIP, unless this week already rebuilt as FH/WC."""
        if chip == 'free_hit':
            stored = self.free_hit_plans.get(int(gameweek))
            if stored is not None:
                return stored
        if chip == 'wildcard':
            stored = self.wildcard_plans.get(int(gameweek))
            if stored is not None:
                return stored
        return self.strategy.plan(state, pool, gameweek)

    def _chip_for_gameweek(self, gameweek):
        earliest = int(self.config.earliest_chip_gameweek or EARLIEST_CHIP_GAMEWEEK)
        if int(gameweek) < earliest:
            return None
        return self.chips.chip_at(gameweek)

    def _expected_horizon(self, player_ids, pool, gameweek):
        return sum(
            expected_gameweek_points(
                player_ids,
                pool,
                target_gw,
                bench_boost=self.chips.is_bench_boost(target_gw),
                triple_captain=self.chips.is_triple_captain(target_gw),
                bench_weights=self.config.bench_weights,
                reserve_keeper_weight=self.config.reserve_keeper_weight,
            )
            for target_gw in self.strategy._gameweeks(gameweek)
        )

    def run(self):
        from squadtools.simulation_strategy import ensure_quiet_cbc

        ensure_quiet_cbc()
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
            self.config.to_dict(),
        )
        if self.config.chip_strategy:
            print(
                'Chips this season: rolling strategy (replanned each gameweek); '
                'wc_threshold=%s fh_threshold=%s tc_above_average=%s '
                'min_horizon_gain=%s tc_taper_floor=%s tc_min_captain=%s '
                'fh_after_wildcard_weeks=%s fh_dgw=%s wc_dgw=%s dgw_h2=%s '
                'wc_rot=%s wc_early=%s earliest=%s'
                % (
                    self.config.wc_threshold,
                    self.config.fh_threshold,
                    self.config.tc_above_average_threshold,
                    self.config.min_horizon_gain,
                    self.config.tc_taper_floor,
                    self.config.tc_min_captain_points,
                    self.config.fh_after_wildcard_weeks,
                    self.config.fh_dgw_weight,
                    self.config.wc_dgw_weight,
                    self.config.dgw_h2_mult,
                    self.config.wc_rotation_weight,
                    self.config.wc_early_weight,
                    self.config.earliest_chip_gameweek,
                )
            )
        else:
            print(f'Chips this season: {json.dumps(self.chips.as_dict(), sort_keys=True)}')
        fill = self.strategy._pass1_fill_weight()
        print(
            'Scoring: minutes_scale=%s bench_fill=%s bench_slots=%s reserve_keeper=%s'
            % (
                self.config.minutes_scale or 'raw',
                fill if fill else 'unused',
                list(self.config.bench_weights) if self.config.bench_weights else 'unused',
                self.config.reserve_keeper_weight
                if self.config.reserve_keeper_weight is not None
                else 'unused',
            )
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
                play_state = None
                if state is None:
                    print('Planning GW%s (initial squad)...' % gameweek, flush=True)
                    state = self._initial_state(pool, gameweek)
                    print('Planning GW%s (chips)...' % gameweek, flush=True)
                    squad_before_transfers = copy_squad_state(state)
                    self._refresh_chip_plan(state, pool, gameweek)
                    chip = self._chip_for_gameweek(gameweek)
                    play_state = state
                    free_before = 15
                    expected_hold_current = 0.0
                    expected_hold_horizon = 0.0
                    expected_selected_current = expected_gameweek_points(
                        state.player_ids,
                        pool,
                        gameweek,
                        bench_boost=self.chips.is_bench_boost(gameweek),
                        triple_captain=self.chips.is_triple_captain(gameweek),
                        bench_weights=self.config.bench_weights,
                        reserve_keeper_weight=self.config.reserve_keeper_weight,
                    )
                    expected_selected_horizon = self._expected_horizon(
                        state.player_ids, pool, gameweek
                    )
                    transfers = ()
                    transfers_used = 0
                else:
                    print('Planning GW%s (chips + transfers)...' % gameweek, flush=True)
                    state.free_transfers = self.ft_rules.at_start_of_gameweek(
                        gameweek, state.free_transfers
                    )
                    self._refresh_chip_plan(state, pool, gameweek)
                    chip = self._chip_for_gameweek(gameweek)
                    free_before = state.free_transfers
                    squad_before_transfers = copy_squad_state(state)
                    plan = self._this_week_transfer_plan(
                        state, pool, gameweek, chip
                    )
                    play_state = apply_transfer_plan(state, plan)
                    transfers = plan.transfers
                    expected_hold_current = plan.expected_hold_current
                    expected_hold_horizon = plan.expected_hold_horizon
                    expected_selected_current = plan.expected_selected_current
                    expected_selected_horizon = plan.expected_selected_horizon
                    if chip == 'free_hit':
                        state = squad_before_transfers
                        transfers_used = 0
                    else:
                        state = play_state
                        transfers_used = (
                            0 if chip == 'wildcard' else len(transfers)
                        )

                lineup = pick_lineup(play_state.player_ids, pool, gameweek)
                actuals = self.source.actuals_for_gameweek(gameweek)
                actual_score = score_lineup(
                    lineup,
                    pool,
                    actuals,
                    bench_boost=chip == 'bench_boost',
                    triple_captain=chip == 'triple_captain',
                )
                expected_xi = flat_expected_points(
                    lineup,
                    pool,
                    gameweek,
                    bench_boost=chip == 'bench_boost',
                    triple_captain=chip == 'triple_captain',
                )
                free_hit_points = (
                    actual_score.points if chip == 'free_hit' else None
                )
                cumulative += actual_score.points
                if chip != 'free_hit':
                    total_transfers += len(transfers)
                availability_warning_count += (
                    actual_score.unexpected_zero_minute_starters
                )
                free_after = self.ft_rules.after_gameweek(
                    gameweek,
                    free_before if gameweek > 1 else 0,
                    transfers_used,
                    chip=chip,
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
                    'chip': chip,
                    'free_transfers_before': free_before,
                    'free_transfers_after': free_after,
                    'bank': state.bank,
                    'expected_hold_current': expected_hold_current,
                    'expected_selected_current': expected_selected_current,
                    'expected_hold_horizon': expected_hold_horizon,
                    'expected_selected_horizon': expected_selected_horizon,
                    'expected_xi': expected_xi,
                    'actual_points': actual_score.points,
                    'free_hit_points': free_hit_points,
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
                    'players_scored': actual_score.players_scored,
                    'transfer_count': len(transfer_rows),
                    'transfers': transfer_rows,
                    'current_gain': (
                        expected_selected_current - expected_hold_current
                    ),
                    'horizon_gain': (
                        expected_selected_horizon - expected_hold_horizon
                    ),
                    'planned_chips': visible_chip_plan(
                        self.chips, self.chip_plan_reason
                    ),
                    'chip_reason': dict(self.chip_plan_reason),
                }
                self.db.persist_simulation_gameweek(self.run_id, gw_row)
                self.db.persist_simulation_transfers(
                    self.run_id, gameweek, transfer_rows
                )
                self.db.connection.commit()
                self.played_chips = add_played_chip(
                    self.played_chips, gameweek, chip
                )
                self.gameweek_logs.append(gw_row)
                chip_label = f', {chip}' if chip else ''
                print(
                    f'GW{gameweek}: {actual_score.points:.0f} pts '
                    f'({cumulative:.0f} total), exp {expected_xi:.0f}, '
                    f'{actual_score.players_scored} scored, '
                    f'{len(transfers)} transfer(s), '
                    f'bank £{state.bank / 10:.1f}m, FT next {free_after}'
                    f'{chip_label}',
                    flush=True,
                )
                print(predicted_points_line(lineup, pool, gameweek), flush=True)
                if self.config.chip_strategy:
                    print(
                        f'  chip plan: {json.dumps(visible_chip_plan(self.chips, self.chip_plan_reason), sort_keys=True)}',
                        flush=True,
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
            overall_rank = self._estimate_overall_rank(cumulative)
            print(format_rank_line(overall_rank))
            summary = {
                'simulation_run_id': self.run_id,
                'prediction_set_id': self.config.prediction_set_id,
                'prediction_scope': prediction_scope or 'fixed_horizon',
                'season': (
                    f'{self.config.season_start_year}/'
                    f'{self.config.season_start_year + 1}'
                ),
                'total_points': cumulative,
                'overall_rank': overall_rank,
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
                    'Predicted pp90 and minutes are stored raw. Chance-to-play '
                    'from live FPL or Randdalf/fplcache is applied in squadtools '
                    'when scoring the linear optimiser.'
                ),
                'fixture_caveat': (
                    'Predictions use the final season fixture calendar, including '
                    'reschedules that may not have been announced at the deadline.'
                ),
                'config': self.config.to_dict(),
            }
            self._write_reports(summary)
            return summary
        except Exception:
            self.db.fail_simulation_run(self.run_id)
            raise
        finally:
            if self._owns_db:
                self.db.close_connection()

    def _estimate_overall_rank(self, points):
        try:
            return estimate_overall_rank(self.config.season_start_year, points)
        except Exception:
            return None

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
            'chip',
            'actual_points',
            'free_hit_points',
            'cumulative_points',
            'free_transfers_before',
            'free_transfers_after',
            'bank',
            'transfer_count',
            'current_gain',
            'horizon_gain',
            'autosub_count',
            'players_scored',
            'captain_fallback',
            'unexpected_zero_minute_starters',
        ]
        with (output_dir / 'gameweeks.csv').open(
            'w', newline='', encoding='utf-8'
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writeheader()
            for row in self.gameweek_logs:
                writer.writerow({
                    column: '' if row.get(column) is None else row[column]
                    for column in csv_columns
                })
        print(f'Wrote simulation reports to {output_dir}')
