"""Hold, and the squad tools' own chip planner plus transfer solver.

Both are scored by the same episode, so hits, free-hit restore and the weekly
rank use the same rules as the learned policy. The planner may transfer and
play a chip in one week. A single policy action cannot.

Terminal reward and the sum of weekly rewards are reported separately.
"""
from __future__ import annotations

from squadtools.chip_strategy import ChipPlanner, ChipStrategyConfig
from squadtools.simulation_strategy import (
    ChipCalendar,
    SimulationStrategy,
    StrategyConfig,
    copy_squad_state,
)

from fpl_manager.episode import SeasonEpisode, hold_action_index


def summarise(episode: SeasonEpisode) -> dict:
    return {
        'points': float(episode.points_total),
        'terminal_reward': float(episode.terminal),
        'weekly_reward': float(episode.weekly_reward_total),
    }


def play_hold(episode: SeasonEpisode) -> dict:
    while not episode.done:
        episode.step(hold_action_index())
    return summarise(episode)


class ExistingManager:
    """One week at a time from ChipPlanner and SimulationStrategy.plan."""

    def __init__(self, season_start_year: int = 25, **config):
        self.season_start_year = int(season_start_year)
        self.earliest = int(config.pop('earliest_chip_gameweek', 5))
        planner_config = dict(
            season_start_year=self.season_start_year,
            horizon_gameweeks=int(config.get('horizon_gameweeks', 5)),
            min_current_gw_gain=float(config.get('min_current_gw_gain', 0.0)),
            min_horizon_gain=float(config.get('min_horizon_gain', 2.0)),
            max_transfers_per_gameweek=int(config.get('max_transfers_per_gameweek', 5)),
            earliest_chip_gameweek=self.earliest,
            one_optimise_per_kind=bool(config.get('one_optimise_per_kind', False)),
            forecast_future_halves=bool(config.get('forecast_future_halves', True)),
        )
        self.planner = ChipPlanner(ChipStrategyConfig(**planner_config))
        self.strategy = SimulationStrategy(StrategyConfig(
            horizon_gameweeks=planner_config['horizon_gameweeks'],
            min_current_gw_gain=planner_config['min_current_gw_gain'],
            min_horizon_gain=planner_config['min_horizon_gain'],
            max_transfers_per_gameweek=planner_config['max_transfers_per_gameweek'],
            season_start_year=self.season_start_year,
            chips=ChipCalendar(),
        ))

    def decide(self, squad, pool, gameweek: int, played: ChipCalendar):
        state = copy_squad_state(squad)
        result = self.planner.plan(state, pool, int(gameweek), played)
        chip = None
        if int(gameweek) >= self.earliest:
            chip = result.calendar.chip_at(gameweek)
        self.strategy.config.chips = result.calendar
        if chip == 'free_hit' and int(gameweek) in result.free_hit_plans:
            return result.free_hit_plans[int(gameweek)], chip
        if chip == 'wildcard' and int(gameweek) in result.wildcard_plans:
            return result.wildcard_plans[int(gameweek)], chip
        return self.strategy.plan(state, pool, int(gameweek)), chip


def play_existing(episode: SeasonEpisode, manager: ExistingManager = None) -> dict:
    manager = manager or ExistingManager(episode.season_start_year)
    while not episode.done:
        plan, chip = manager.decide(
            episode.squad, episode.pool, episode.gameweek, episode.played,
        )
        episode.apply_plan(plan, chip)
    return summarise(episode)
