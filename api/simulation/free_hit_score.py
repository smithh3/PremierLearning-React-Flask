"""Score the 15-man squad a free-hit rebuild would have taken this gameweek."""

from squadtools.simulation_strategy import (
    ChipCalendar,
    SimulationStrategy,
    StrategyConfig,
    apply_transfer_plan,
    pick_lineup,
)

from simulation.actual_scoring import score_lineup


def score_selected_squad(player_ids, pool, gameweek, actuals) -> float:
    """Actual FPL points for an already-chosen 15, this gameweek only (no BB/TC)."""
    lineup = pick_lineup(player_ids, pool, gameweek)
    return score_lineup(lineup, pool, actuals).points


def actual_free_hit_points(state, pool, gameweek, actuals, strategy_config=None):
    """Build a one-week free-hit squad from ``state`` and score it on actuals.

    The MIP is this gameweek only (ChipCalendar free-hit week). Budget is bank
    plus sell-on value of the current 15, same as the live/sim planner.
    """
    base = strategy_config or StrategyConfig()
    strategy = SimulationStrategy(StrategyConfig(
        horizon_gameweeks=1,
        min_current_gw_gain=0.0,
        min_horizon_gain=0.0,
        max_transfers_per_gameweek=15,
        max_free_transfers=base.max_free_transfers,
        relax_thresholds_at_max_ft=False,
        chips=ChipCalendar(free_hits=(int(gameweek),)),
        bench_fill_weight=getattr(base, 'bench_fill_weight', None),
        bench_weights=base.bench_weights,
        reserve_keeper_weight=base.reserve_keeper_weight,
    ))
    plan = strategy.plan(state, pool, gameweek)
    apply_transfer_plan(state, plan)
    return score_selected_squad(plan.selected_player_ids, pool, gameweek, actuals)


def printed_free_hit_points(
    chip,
    played_points,
    gameweek,
    pool,
    actuals,
    stored_plan=None,
    state=None,
    strategy_config=None,
):
    """Actuals for the FH print: reuse the decision squad, or rebuild if missing.

    Playing the chip already selected a squad this week — print that score.
    Otherwise reuse the 1-GW probe from the chip planner when this week was
    evaluated. Only solve a new MIP when there was no probe (early GWs, or
    the chip was already spent).
    """
    if chip == 'free_hit':
        return played_points
    if stored_plan is not None:
        return score_selected_squad(
            stored_plan.selected_player_ids, pool, gameweek, actuals
        )
    if state is None:
        return None
    return actual_free_hit_points(
        state, pool, gameweek, actuals, strategy_config=strategy_config
    )
