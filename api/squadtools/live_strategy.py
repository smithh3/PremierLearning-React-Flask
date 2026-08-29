"""Chip planner + transfer MIP for the live Fargate / squadtools path."""
import os
from dataclasses import dataclass, field

from premierlearning.config_constants import SEASON_YR
from premierlearning.fpl_apis import load_selectable_player_ids
from premierlearning.transfer import Transfer
from squadtools.availability import (
    LIVE_OUTFIELD_BENCH_WEIGHTS,
    match_optimiser_points,
)
from squadtools.chip_strategy import (
    CHIP_BAR_REFERENCE_HORIZON,
    ChipPlanner,
    ChipStrategyConfig,
)
from squadtools.simulation_strategy import (
    CHIP_BENCH_BOOST,
    CHIP_FREE_HIT,
    CHIP_TRIPLE_CAPTAIN,
    CHIP_WILDCARD,
    ChipCalendar,
    INITIAL_BUDGET,
    MIN_GAIN_REFERENCE_HORIZON,
    OwnedPlayer,
    PlayerProjection,
    PredictionPool,
    SimulationStrategy,
    SquadState,
    StrategyConfig,
)
from squadtools.squad import WrappedPlayer


FPL_CHIP_NAMES = {
    'wildcard': CHIP_WILDCARD,
    'freehit': CHIP_FREE_HIT,
    'free_hit': CHIP_FREE_HIT,
    'bboost': CHIP_BENCH_BOOST,
    'bench_boost': CHIP_BENCH_BOOST,
    '3xc': CHIP_TRIPLE_CAPTAIN,
    'triplecaptain': CHIP_TRIPLE_CAPTAIN,
    'triple_captain': CHIP_TRIPLE_CAPTAIN,
}

CHIP_LABELS = {
    CHIP_WILDCARD: 'Wildcard',
    CHIP_FREE_HIT: 'Free Hit',
    CHIP_BENCH_BOOST: 'Bench Boost',
    CHIP_TRIPLE_CAPTAIN: 'Triple Captain',
}

HORIZON_EMAIL_WEEKS = 8
PRE_GW1_MAX_TRANSFERS = 15

# Chip and transfer knobs below are the best trial of the last historical sweep:
# 39_stuck_rot, 2168 points on 25/26 for a top 9.9% finish. It planned six weeks
# ahead and every bar it quoted was an absolute total over those six weeks.
SIM_TRIAL = '39_stuck_rot'
LIVE_PLAN_HORIZON = 6


def _requoted(sim_total, reference=CHIP_BAR_REFERENCE_HORIZON):
    """Restate one of SIM_TRIAL's bars over the window the config quotes bars in.

    The planner stretches a quoted bar back out onto horizon_gameweeks, so
    copying the trial's numbers across would demand LIVE_PLAN_HORIZON / reference
    times what the trial actually cleared. Every additive bar term scales
    together, so they all come through here.
    """
    return float(sim_total) * float(reference) / float(LIVE_PLAN_HORIZON)


def _skip_bench_fill_pass() -> bool:
    raw = os.environ.get('SKIP_BENCH_FILL', '0')
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


@dataclass
class LivePlan:
    calendar: ChipCalendar = field(default_factory=ChipCalendar)
    reason: dict = field(default_factory=dict)
    transfers: dict = None
    horizon: list = field(default_factory=list)
    built_from_scratch: bool = False
    recommended_squad: list = field(default_factory=list)
    this_week_scores: dict = field(default_factory=dict)


def live_chip_config(season_start_year=None, pre_gw1=False):
    """Chip knobs from the best historical sweep trial, SIM_TRIAL.

    Live leftover bench weight is 0: pass 1 scores the rotating XI + captain
    (Bench Boost still counts all 15). Pass 2 values the first two outfield
    bench slots at 0.25 each; later bench slots are 0. Set SKIP_BENCH_FILL=1
    for XI only. Historical sims keep fill 0.3 and slots 0.5 / 0.3 / 0.1.

    Before GW1, FPL allows unlimited free transfers, so the planner may rebuild
    the whole 15 rather than stopping at five moves. Both gain floors drop to 0
    there: there is no squad yet for a transfer to have to beat.
    """
    skip_fill = _skip_bench_fill_pass()
    print(
        'Live strategy: sweep trial %s, planning %s weeks ahead'
        % (SIM_TRIAL, LIVE_PLAN_HORIZON)
    )
    if skip_fill:
        print('Live strategy: skipping bench fill pass (SKIP_BENCH_FILL)')
    else:
        print('Live strategy: first two bench slots 0.25, other bench slots 0')
    return ChipStrategyConfig(
        season_start_year=int(
            SEASON_YR if season_start_year is None else season_start_year
        ),
        horizon_gameweeks=LIVE_PLAN_HORIZON,
        fh_threshold=_requoted(10.0),
        wc_threshold=_requoted(45.0),
        fh_recovery_threshold=_requoted(10.0),
        # Wait out a likely double rather than firing into an uneven week.
        fh_dgw_weight=_requoted(2.0),
        wc_dgw_weight=_requoted(4.0),
        dgw_h2_mult=2.0,
        # A rotting squad lowers the wildcard bar; an early one raises it.
        wc_rotation_weight=_requoted(6.0),
        wc_early_weight=_requoted(15.0),
        earliest_chip_gameweek=3,
        roll_free_transfers=True,
        tc_above_average_threshold=1.6,
        tc_window=8,
        min_current_gw_gain=0.0 if pre_gw1 else 1.0,
        min_horizon_gain=(
            0.0 if pre_gw1
            else _requoted(5.0, MIN_GAIN_REFERENCE_HORIZON)
        ),
        max_transfers_per_gameweek=(
            PRE_GW1_MAX_TRANSFERS if pre_gw1 else 5
        ),
        relax_thresholds_at_max_ft=False,
        bench_fill_weight=0.0,
        bench_weights=() if skip_fill else LIVE_OUTFIELD_BENCH_WEIGHTS,
        reserve_keeper_weight=None,
    )


def _int_price(value):
    return int(round(float(value or 0)))


def prediction_pool_from_players(player_dict, selectable_ids=None):
    if selectable_ids is None:
        try:
            selectable_ids = load_selectable_player_ids()
        except Exception:
            selectable_ids = set()
    players = {}
    for player in (player_dict or {}).values():
        player_id = int(getattr(player, 'id', 0) or 0)
        team_id = int(getattr(player, 'team_id', 0) or 0)
        position = int(getattr(player, 'position', 0) or 0)
        if player_id <= 0 or team_id <= 0 or position not in (1, 2, 3, 4):
            continue
        name = player.get_name() if hasattr(player, 'get_name') else (
            '%s %s' % (
                getattr(player, 'first_name', '') or '',
                getattr(player, 'last_name', '') or '',
            )
        ).strip()
        selectable = True if not selectable_ids else player_id in selectable_ids
        projection = PlayerProjection(
            player_id=player_id,
            player_code=int(getattr(player, 'code', 0) or 0),
            name=name,
            position=position,
            team_id=team_id,
            price=_int_price(getattr(player, 'current_cost', 0)),
            selectable=selectable,
        )
        for match in getattr(player, 'future_matches', None) or []:
            gameweek = getattr(match, 'gameweek', None)
            if gameweek is None:
                continue
            gameweek = int(gameweek)
            projection.points_by_gameweek[gameweek] = (
                projection.points_by_gameweek.get(gameweek, 0.0)
                + match_optimiser_points(match)
            )
            minutes = getattr(match, 'minutes', None)
            stats = getattr(match, 'game_stats', None) or {}
            try:
                stats_mins = float(stats.get('minutes', 0) or 0)
            except (TypeError, ValueError):
                stats_mins = 0.0
            if stats_mins > 0:
                minutes = stats_mins
            projection.minutes_by_gameweek[gameweek] = (
                projection.minutes_by_gameweek.get(gameweek, 0.0)
                + float(minutes or 0)
            )
            fixture_id = getattr(match, 'fixture_id', None)
            if fixture_id in (None, '', 0):
                fixture_id = getattr(match, 'id', None)
            if fixture_id not in (None, '', 0):
                projection.fixture_ids_by_gameweek.setdefault(gameweek, set()).add(
                    int(fixture_id)
                )
        players[player_id] = projection
    return PredictionPool(players)


def played_chips_from_user_json(user_json, season_start_year=None):
    season = int(SEASON_YR if season_start_year is None else season_start_year)
    picks = ((user_json or {}).get('fpl_api') or {}).get('picks') or {}
    wildcards, free_hits, bench_boosts, triple_captains = [], [], [], []
    for chip in picks.get('chips') or []:
        name = FPL_CHIP_NAMES.get(str(chip.get('name') or '').lower())
        if not name:
            continue
        played_weeks = list(chip.get('played_by_entry') or [])
        if not played_weeks and str(chip.get('status_for_entry') or '').lower() == 'played':
            event = chip.get('event') or chip.get('played_event')
            if event:
                played_weeks = [event]
        for gameweek in played_weeks:
            try:
                week = int(gameweek)
            except (TypeError, ValueError):
                continue
            if name == CHIP_WILDCARD:
                wildcards.append(week)
            elif name == CHIP_FREE_HIT:
                free_hits.append(week)
            elif name == CHIP_BENCH_BOOST:
                bench_boosts.append(week)
            elif name == CHIP_TRIPLE_CAPTAIN:
                triple_captains.append(week)
    if not any((wildcards, free_hits, bench_boosts, triple_captains)):
        return ChipCalendar()
    return ChipCalendar.resolve(
        season, wildcards, free_hits, bench_boosts, triple_captains
    )


def squad_state_from_squad(squad):
    owned = {}
    for player in squad.squad:
        wrapped = squad.wrapped_player_dict[player.id]
        owned[int(player.id)] = OwnedPlayer(
            player_id=int(player.id),
            purchase_price=_int_price(wrapped.purchase_price),
        )
    return SquadState(
        owned=owned,
        bank=_int_price(getattr(squad, 'bank', 0)),
        free_transfers=int(getattr(squad, 'free_transfers', 0) or 0),
    )


def _ensure_pool_has_owned(pool, squad):
    """Keep currently owned players in the pool even if they are flagged unfit."""
    players = dict(pool.players)
    for player in squad.squad:
        player_id = int(player.id)
        existing = players.get(player_id)
        if existing is None:
            continue
        if not existing.selectable:
            existing.selectable = True
    return PredictionPool(players)


def _transfer_from_planned(squad, planned):
    player_out = squad.player_dict[planned.player_out_id]
    player_in = squad.player_dict[planned.player_in_id]
    wrapped_out = squad.wrapped_player_dict.get(planned.player_out_id)
    if wrapped_out is None:
        wrapped_out = WrappedPlayer(player_out, planned.sell_price)
        wrapped_out.selling_price = planned.sell_price
    wrapped_in = squad.wrapped_player_dict.get(planned.player_in_id)
    if wrapped_in is None:
        wrapped_in = WrappedPlayer(player_in, planned.buy_price)
    transfer = Transfer(wrapped_out, wrapped_in)
    transfer.selling_price = int(planned.sell_price)
    transfer.purchase_price = int(planned.buy_price)
    return transfer


def _plan_scores(plan):
    if plan is None:
        return {
            'hold_current': 0.0,
            'selected_current': 0.0,
            'hold_horizon': 0.0,
            'selected_horizon': 0.0,
            'current_gain': 0.0,
            'horizon_gain': 0.0,
        }
    return {
        'hold_current': float(plan.expected_hold_current),
        'selected_current': float(plan.expected_selected_current),
        'hold_horizon': float(plan.expected_hold_horizon),
        'selected_horizon': float(plan.expected_selected_horizon),
        'current_gain': float(plan.current_gain),
        'horizon_gain': float(plan.horizon_gain),
    }


def _horizon_from_projection(squad, chips, projected_weeks):
    horizon = []
    transfer_objects = []
    scheme = []
    this_week_scores = _plan_scores(None)
    first = True
    for gameweek in sorted(projected_weeks):
        week = projected_weeks[gameweek]
        planned = tuple((week.plan.transfers if week.plan else ()) or ())
        converted = [_transfer_from_planned(squad, item) for item in planned]
        scores = _plan_scores(week.plan)
        if first:
            this_week_scores = scores
            first = False
        row = {
            'gameweek': int(gameweek),
            'chip': chips.chip_at(gameweek),
            'transfers': converted,
        }
        row.update(scores)
        horizon.append(row)
        scheme.append(len(converted))
        transfer_objects.extend(converted)
    return horizon, {
        'scheme': tuple(scheme) if scheme else (0,),
        'transfer_tuple': tuple(transfer_objects),
        'points': float(this_week_scores.get('horizon_gain') or 0),
    }, this_week_scores


def plan_live(squad, user_json=None, season_start_year=None):
    season = int(SEASON_YR if season_start_year is None else season_start_year)
    pre_gw1 = False
    if hasattr(squad, 'is_pre_gw1'):
        try:
            pre_gw1 = bool(squad.is_pre_gw1())
        except Exception:
            pre_gw1 = False
    if pre_gw1:
        print(
            'Pre-GW1: unlimited free transfers; allowing a full %s-player rebuild'
            % PRE_GW1_MAX_TRANSFERS
        )
        squad.free_transfers = PRE_GW1_MAX_TRANSFERS
    config = live_chip_config(season, pre_gw1=pre_gw1)
    planner = ChipPlanner(config)
    pool = _ensure_pool_has_owned(
        prediction_pool_from_players(squad.player_dict),
        squad,
    )
    as_of = int(squad.next_gameweek)
    built_from_scratch = bool(getattr(squad, 'built_from_scratch', False))
    has_full_squad = len(squad.squad) == 15
    if pre_gw1 and has_full_squad:
        built_from_scratch = False
        squad.built_from_scratch = False
    elif built_from_scratch or not has_full_squad:
        strategy = SimulationStrategy(StrategyConfig(
            horizon_gameweeks=config.horizon_gameweeks,
            min_current_gw_gain=config.min_current_gw_gain,
            min_horizon_gain=config.min_horizon_gain,
            max_transfers_per_gameweek=config.max_transfers_per_gameweek,
            max_free_transfers=planner.ft_rules.max_bank,
            relax_thresholds_at_max_ft=config.relax_thresholds_at_max_ft,
            bench_fill_weight=config.bench_fill_weight,
            bench_weights=config.bench_weights,
            reserve_keeper_weight=config.reserve_keeper_weight,
        ))
        player_ids = strategy.build_initial_squad(
            pool, as_of_gameweek=as_of, budget=INITIAL_BUDGET
        )
        squad.populate_squad(player_ids)
        spent = sum(_int_price(player.current_cost) for player in squad.squad)
        squad.bank = max(0, INITIAL_BUDGET - spent)
        squad.free_transfers = 0 if as_of <= 1 else 1
        if pre_gw1:
            squad.free_transfers = PRE_GW1_MAX_TRANSFERS
        squad.built_from_scratch = True
        built_from_scratch = True
        for player in squad.squad:
            squad.wrapped_player_dict[player.id] = WrappedPlayer(
                player, player.current_cost
            )

    state = squad_state_from_squad(squad)
    played = played_chips_from_user_json(user_json, season)
    result = planner.plan(state, pool, as_of, played)
    end_gw = min(as_of + HORIZON_EMAIL_WEEKS - 1, 38)
    projected, _trailing = planner.project(
        state, pool, as_of, end_gw, result.calendar
    )
    horizon, transfers, this_week_scores = _horizon_from_projection(
        squad, result.calendar, projected
    )
    recommended = [player.get_name() for player in squad.squad]
    print(
        'Chip plan: %s' % result.calendar.as_dict()
    )
    return LivePlan(
        calendar=result.calendar,
        reason=result.reason,
        transfers=None if built_from_scratch else transfers,
        horizon=horizon,
        built_from_scratch=built_from_scratch,
        recommended_squad=recommended,
        this_week_scores=this_week_scores,
    )


def apply_live_plan(opt):
    plan = plan_live(opt.squad, getattr(opt, 'user_json', None))
    opt.chip_calendar = plan.calendar
    opt.chip_reason = plan.reason
    opt.horizon_plan = plan.horizon
    opt.recommended_squad = plan.recommended_squad
    opt.this_week_scores = plan.this_week_scores
    if plan.built_from_scratch:
        opt.transfers = None
        print(
            'Initial squad from chip-strategy builder: %s'
            % ', '.join(plan.recommended_squad)
        )
        return plan
    opt.transfers = plan.transfers
    this_week = plan.horizon[0]['transfers'] if plan.horizon else []
    if this_week:
        print('Suggested transfers for GW %s:' % opt.squad.next_gameweek)
        for transfer in this_week:
            print('  %s' % transfer)
    else:
        print('No transfers suggested for this gameweek')
    return plan
