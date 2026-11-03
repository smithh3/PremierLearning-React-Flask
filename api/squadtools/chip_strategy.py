"""Rolling chip planner for historical replays.

Each gameweek projects the squad forward with free transfers, scores free-hit
and wildcard resets, then returns a ChipCalendar the existing transfer MIP
already understands. Already-played chips are locked; future chip weeks are
always replanned from this week's squad and predictions.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from squadtools.simulation_strategy import (
    CHIP_BENCH_BOOST,
    CHIP_FREE_HIT,
    CHIP_TRIPLE_CAPTAIN,
    CHIP_WILDCARD,
    FIRST_HALF_LAST_GAMEWEEK,
    OPTIMISER_MIN_AVG_POINTS,
    ChipCalendar,
    FreeTransferRules,
    SimulationStrategy,
    SquadState,
    StrategyConfig,
    TransferPlan,
    apply_transfer_plan,
    copy_squad_state,
    expected_gameweek_points,
    pick_lineup,
)


LAST_GAMEWEEK = 38
EARLIEST_CHIP_GAMEWEEK = 5
# 1-indexed week of the half. TC uses the base gap until this week, then
# tapers linearly to TC_TAPER_FLOOR at the last GW of the half.
TC_EASY_UNTIL_WEEK = 10
TC_TAPER_FLOOR = 0.8
TC_MIN_CAPTAIN_POINTS = 8.0
# Free hit cannot land on the wildcard week, the BB week, or the next
# FH_AFTER_WILDCARD_WEEKS weeks after the wildcard.
FH_AFTER_WILDCARD_WEEKS = 3
# Second-half DGW-wait multiplier on the fixture-imbalance term.
DGW_H2_MULT_DEFAULT = 2.0
# Extra WC bar from the first legal week, fading linearly to 0 at this GW.
WC_EARLY_UNTIL_GAMEWEEK = 6
# After a wildcard rebuild, score the branch with at most this many FTs/week.
WC_EVAL_MAX_TRANSFERS = 2
LOW_MINUTE_CUTOFF = 60.0
# Chip gaps are summed over horizon_gameweeks, so one bar demands a different
# gain per week at each horizon length: 30 over 5 weeks asks for 6 a week, but
# over 3 weeks it asks for 10. Bars are quoted over this many weeks and rescaled
# to the horizon in use, so changing the horizon no longer moves the bar.
CHIP_BAR_REFERENCE_HORIZON = 5


def earliest_chip_gameweek(as_of_gameweek: int = 1, earliest=None) -> int:
    floor = EARLIEST_CHIP_GAMEWEEK if earliest is None else int(earliest)
    return max(int(as_of_gameweek), floor)


def blocked_chip_gameweeks(earliest=None):
    floor = EARLIEST_CHIP_GAMEWEEK if earliest is None else int(earliest)
    return set(range(1, floor))


def half_bounds(gameweek: int) -> Tuple[int, int]:
    if int(gameweek) <= FIRST_HALF_LAST_GAMEWEEK:
        return 1, FIRST_HALF_LAST_GAMEWEEK
    return FIRST_HALF_LAST_GAMEWEEK + 1, LAST_GAMEWEEK


def chip_window(season_start_year: int, chip: str, gameweek: int) -> Tuple[int, int]:
    """Weeks in which this chip can still be played, as of this gameweek.

    Wildcards have always been one per half. Free hit / bench boost / triple
    captain only became per-half in 2025/26; before that the season's single
    chip stays live until GW38, so the halfway deadline must not push it out.
    """
    half_start, half_end = half_bounds(gameweek)
    if chip == CHIP_WILDCARD or int(season_start_year) >= 25:
        return half_start, half_end
    return 1, LAST_GAMEWEEK


def remaining_fixtures_by_team(pool, as_of_gameweek: int) -> Dict[int, float]:
    """Remaining matches per club from the fixture list on the snapshot.

    Distinct fixture IDs with target GW >= as_of, unioned across a club's
    players. Blanks contribute 0, singles 1, doubles 2. Mean absolute
    deviation of these counts is the DGW-pressure proxy. Minutes are not
    used: a promoted club with a fixture and no history still has a match.
    """
    as_of = int(as_of_gameweek)
    fixtures_by_team: Dict[int, set] = {}
    counts: Dict[int, float] = {}
    for player in pool.players.values():
        team_id = int(player.team_id)
        counts.setdefault(team_id, 0.0)
        team_fixtures = fixtures_by_team.setdefault(team_id, set())
        by_gw = getattr(player, 'fixture_ids_by_gameweek', None) or {}
        for gameweek, fixture_ids in by_gw.items():
            if int(gameweek) < as_of:
                continue
            for fixture_id in fixture_ids:
                try:
                    value = int(fixture_id)
                except (TypeError, ValueError):
                    continue
                if value:
                    team_fixtures.add(value)
    for team_id, fixture_ids in fixtures_by_team.items():
        counts[team_id] = float(len(fixture_ids))
    return counts


def fixture_imbalance(pool, as_of_gameweek: int) -> float:
    """Mean |remaining fixtures − league average| across clubs."""
    counts = list(remaining_fixtures_by_team(pool, as_of_gameweek).values())
    if len(counts) < 2:
        return 0.0
    mean = sum(counts) / float(len(counts))
    return sum(abs(count - mean) for count in counts) / float(len(counts))


def dgw_half_multiplier(as_of_gameweek: int, h2_mult) -> float:
    if int(as_of_gameweek) <= FIRST_HALF_LAST_GAMEWEEK:
        return 1.0
    return max(0.0, float(h2_mult))


def low_minute_owned_count(
    state: SquadState,
    pool,
    as_of_gameweek: int,
    cutoff=None,
) -> int:
    """Owned players predicted under `cutoff` minutes in the next match.

    A pool with no predicted minutes counts those under the optimiser's points
    floor instead.
    """
    limit = LOW_MINUTE_CUTOFF if cutoff is None else float(cutoff)
    use_minutes = pool.predicts_minutes
    as_of = int(as_of_gameweek)
    count = 0
    for player_id in state.player_ids:
        player = pool.players.get(int(player_id))
        if player is None:
            continue
        if use_minutes:
            low = player.minutes(as_of) < limit
        else:
            low = player.points(as_of) < OPTIMISER_MIN_AVG_POINTS
        if low:
            count += 1
    return count


def adjust_chip_thresholds(
    fh_threshold: float,
    wc_threshold: float,
    *,
    fixture_imbalance_value: float,
    half_multiplier: float,
    fh_dgw_weight: float,
    wc_dgw_weight: float,
    low_minute_count: int,
    wc_rotation_weight: float,
) -> Tuple[float, float]:
    """Raise FH/WC bars when fixtures are uneven; ease WC when the squad is rotting.

    Higher threshold = harder to fire now. DGW pressure is added so a likely
    later double makes us wait. Low-minute owned players are subtracted from
    the WC bar so a dead squad is easier to rebuild.
    """
    dgw = float(fixture_imbalance_value) * float(half_multiplier)
    fh = float(fh_threshold) + float(fh_dgw_weight) * dgw
    wc = (
        float(wc_threshold)
        + float(wc_dgw_weight) * dgw
        - float(wc_rotation_weight) * float(low_minute_count)
    )
    return fh, wc


def wc_early_delay_factor(gameweek, earliest=None, until=None) -> float:
    """1.0 at the first legal WC week, 0 from `until` onward. Linear in between.

    With earliest=3 and until=6: GW3=1, GW4=2/3, GW5=1/3, GW6+=0.
    """
    start = EARLIEST_CHIP_GAMEWEEK if earliest is None else int(earliest)
    stop = WC_EARLY_UNTIL_GAMEWEEK if until is None else int(until)
    gw = int(gameweek)
    if stop <= start or gw >= stop:
        return 0.0
    if gw <= start:
        return 1.0
    return (stop - gw) / float(stop - start)


def wc_early_threshold_bonus(gameweek, weight, earliest=None, until=None) -> float:
    """Points added to the WC bar before GW6. 0 weight keeps the old bar."""
    if float(weight) <= 0:
        return 0.0
    return float(weight) * wc_early_delay_factor(
        gameweek, earliest=earliest, until=until
    )


@dataclass(frozen=True)
class ChipInventory:
    wildcard: bool = True
    free_hit: bool = True
    bench_boost: bool = True
    triple_captain: bool = True

    def count(self) -> int:
        return (
            int(self.wildcard)
            + int(self.free_hit)
            + int(self.bench_boost)
            + int(self.triple_captain)
        )

    def without(self, chip: str) -> 'ChipInventory':
        return ChipInventory(
            wildcard=self.wildcard and chip != CHIP_WILDCARD,
            free_hit=self.free_hit and chip != CHIP_FREE_HIT,
            bench_boost=self.bench_boost and chip != CHIP_BENCH_BOOST,
            triple_captain=self.triple_captain and chip != CHIP_TRIPLE_CAPTAIN,
        )


def inventory_for(
    season_start_year: int,
    as_of_gameweek: int,
    played: ChipCalendar,
) -> ChipInventory:
    half_start, half_end = half_bounds(as_of_gameweek)
    per_half = int(season_start_year) >= 25

    def used(weeks: Sequence[int], chip_per_half: bool) -> bool:
        if chip_per_half:
            return any(half_start <= int(gameweek) <= half_end for gameweek in weeks)
        return any(1 <= int(gameweek) <= LAST_GAMEWEEK for gameweek in weeks)

    return ChipInventory(
        wildcard=not used(played.wildcards, True),
        free_hit=not used(played.free_hits, per_half),
        bench_boost=not used(played.bench_boosts, per_half),
        triple_captain=not used(played.triple_captains, per_half),
    )


def held_chips(inventory: ChipInventory) -> Tuple[str, ...]:
    return tuple(
        chip
        for chip, held in (
            (CHIP_WILDCARD, inventory.wildcard),
            (CHIP_FREE_HIT, inventory.free_hit),
            (CHIP_BENCH_BOOST, inventory.bench_boost),
            (CHIP_TRIPLE_CAPTAIN, inventory.triple_captain),
        )
        if held
    )


def expiring_inventory(
    season_start_year: int,
    inventory: ChipInventory,
    as_of_gameweek: int,
) -> ChipInventory:
    """Held chips whose window closes with the current half.

    Only these have to be squeezed in before the halfway deadline. Pre-2025/26
    that is the wildcard alone in the first half.
    """
    _half_start, half_end = half_bounds(as_of_gameweek)
    expiring = {
        chip
        for chip in held_chips(inventory)
        if chip_window(season_start_year, chip, as_of_gameweek)[1] <= half_end
    }
    return ChipInventory(
        wildcard=CHIP_WILDCARD in expiring,
        free_hit=CHIP_FREE_HIT in expiring,
        bench_boost=CHIP_BENCH_BOOST in expiring,
        triple_captain=CHIP_TRIPLE_CAPTAIN in expiring,
    )


def chips_sharing_window(
    season_start_year: int,
    inventory: ChipInventory,
    as_of_gameweek: int,
    chip: str,
) -> int:
    """Held chips competing for the same weeks as this one."""
    window = chip_window(season_start_year, chip, as_of_gameweek)
    return sum(
        1
        for other in held_chips(inventory)
        if chip_window(season_start_year, other, as_of_gameweek) == window
    )


def effective_threshold(
    base: float,
    as_of_gameweek: int,
    chips_left: int,
    window_start: int,
    window_end: int,
) -> float:
    """Relax the bar as the chip's own window runs out of spare weeks.

    ``window_start`` / ``window_end`` bound the weeks the chip can still be
    played in, and ``chips_left`` counts the held chips competing for them.
    """
    weeks_left = int(window_end) - int(as_of_gameweek) + 1
    slack = weeks_left - int(chips_left)
    if int(chips_left) <= 0:
        return float(base)
    if slack <= 0:
        return 0.0
    window_length = int(window_end) - int(window_start) + 1
    max_slack = max(1, window_length - int(chips_left))
    return float(base) * slack / max_slack


def triple_captain_threshold(
    gameweek: int, base: float, floor=None, easy_until=None
) -> float:
    """Base gap until week 10 of the half, then linear to a floor at the last GW.

    First half: 1.3 through GW10, 0.8 at GW19. Second half: the same on GW20–38.
    """
    if floor is None:
        floor = TC_TAPER_FLOOR
    if easy_until is None:
        easy_until = TC_EASY_UNTIL_WEEK
    floor = max(0.0, float(floor))
    base = max(floor, float(base))
    easy_until = max(1, int(easy_until))
    half_start, half_end = half_bounds(gameweek)
    week_in_half = int(gameweek) - int(half_start) + 1
    half_length = int(half_end) - int(half_start) + 1
    if week_in_half < easy_until:
        return base
    span = half_length - easy_until
    if span <= 0:
        return base
    progress = (week_in_half - easy_until) / float(span)
    return max(floor, base + (floor - base) * progress)


def free_hit_blocked_after_wildcard(fh_gameweek, wildcard_gameweek, weeks=None) -> bool:
    if wildcard_gameweek is None:
        return False
    fh = int(fh_gameweek)
    wildcard = int(wildcard_gameweek)
    gap = FH_AFTER_WILDCARD_WEEKS if weeks is None else int(weeks)
    return wildcard <= fh <= wildcard + gap


def force_remaining_chips(
    as_of_gameweek: int,
    half_end: int,
    inventory: ChipInventory,
    occupied=None,
    earliest=None,
) -> Dict[int, str]:
    """Place unused chips one per GW before the half ends, WC then BB then FH then TC."""
    occupied = set(int(gameweek) for gameweek in (occupied or ()))
    occupied.update(blocked_chip_gameweeks(earliest))
    assigned: Dict[int, str] = {}
    start = earliest_chip_gameweek(as_of_gameweek, earliest)
    half_end = int(half_end)

    def take_week(from_gw: int) -> Optional[int]:
        gameweek = int(from_gw)
        while gameweek <= half_end and gameweek in occupied:
            gameweek += 1
        if gameweek > half_end:
            return None
        return gameweek

    if inventory.wildcard and inventory.bench_boost:
        pair = None
        for wildcard_gw in range(start, half_end):
            bench_gw = wildcard_gw + 1
            if (
                wildcard_gw not in occupied
                and bench_gw not in occupied
                and bench_gw <= half_end
            ):
                pair = (wildcard_gw, bench_gw)
                break
        if pair:
            assigned[pair[0]] = CHIP_WILDCARD
            assigned[pair[1]] = CHIP_BENCH_BOOST
            occupied.add(pair[0])
            occupied.add(pair[1])
        else:
            wildcard_gw = take_week(start)
            if wildcard_gw is not None:
                assigned[wildcard_gw] = CHIP_WILDCARD
                occupied.add(wildcard_gw)
    elif inventory.wildcard:
        wildcard_gw = take_week(start)
        if wildcard_gw is not None:
            assigned[wildcard_gw] = CHIP_WILDCARD
            occupied.add(wildcard_gw)
    elif inventory.bench_boost:
        bench_gw = take_week(start)
        if bench_gw is not None:
            assigned[bench_gw] = CHIP_BENCH_BOOST
            occupied.add(bench_gw)

    if inventory.free_hit:
        free_hit_gw = take_week(start)
        if free_hit_gw is not None:
            assigned[free_hit_gw] = CHIP_FREE_HIT
            occupied.add(free_hit_gw)
    if inventory.triple_captain:
        triple_gw = take_week(start)
        if triple_gw is not None:
            assigned[triple_gw] = CHIP_TRIPLE_CAPTAIN
            occupied.add(triple_gw)
    return assigned


def add_played_chip(
    played: ChipCalendar,
    gameweek: int,
    chip: Optional[str],
) -> ChipCalendar:
    if not chip:
        return played
    gameweek = int(gameweek)
    wildcards = list(played.wildcards)
    free_hits = list(played.free_hits)
    bench_boosts = list(played.bench_boosts)
    triple_captains = list(played.triple_captains)
    if chip == CHIP_WILDCARD:
        wildcards.append(gameweek)
    elif chip == CHIP_FREE_HIT:
        free_hits.append(gameweek)
    elif chip == CHIP_BENCH_BOOST:
        bench_boosts.append(gameweek)
    elif chip == CHIP_TRIPLE_CAPTAIN:
        triple_captains.append(gameweek)
    return ChipCalendar(
        wildcards=tuple(wildcards),
        free_hits=tuple(free_hits),
        bench_boosts=tuple(bench_boosts),
        triple_captains=tuple(triple_captains),
    )


def _calendar_from_assignments(
    season_start_year: int,
    played: ChipCalendar,
    assignments: Dict[int, str],
    earliest=None,
) -> ChipCalendar:
    wildcards = list(played.wildcards)
    free_hits = list(played.free_hits)
    bench_boosts = list(played.bench_boosts)
    triple_captains = list(played.triple_captains)
    floor = EARLIEST_CHIP_GAMEWEEK if earliest is None else int(earliest)
    for gameweek, chip in sorted(assignments.items()):
        gameweek = int(gameweek)
        if gameweek < floor:
            continue
        if chip == CHIP_WILDCARD:
            wildcards.append(int(gameweek))
        elif chip == CHIP_FREE_HIT:
            free_hits.append(int(gameweek))
        elif chip == CHIP_BENCH_BOOST:
            bench_boosts.append(int(gameweek))
        elif chip == CHIP_TRIPLE_CAPTAIN:
            triple_captains.append(int(gameweek))
    return ChipCalendar.resolve(
        season_start_year,
        wildcards,
        free_hits,
        bench_boosts,
        triple_captains,
    )


@dataclass
class ChipStrategyConfig:
    season_start_year: int
    horizon_gameweeks: int = 5
    fh_threshold: float = 11.0
    wc_threshold: float = 30.0
    fh_recovery_threshold: float = 8.0
    tc_above_average_threshold: float = 1.3
    tc_window: int = 10
    forecast_future_halves: bool = True
    min_current_gw_gain: float = 0.0
    min_horizon_gain: float = 0.0
    max_transfers_per_gameweek: int = 5
    relax_thresholds_at_max_ft: bool = False
    bench_fill_weight: Optional[float] = None
    bench_weights: Tuple[float, ...] = ()
    reserve_keeper_weight: Optional[float] = None
    # Simulation: at most one FH rebuild, one WC rebuild, and no weekly
    # transfer rolls inside the chip planner. Live Fargate keeps False.
    one_optimise_per_kind: bool = False
    earliest_chip_gameweek: int = EARLIEST_CHIP_GAMEWEEK
    tc_taper_floor: float = TC_TAPER_FLOOR
    tc_min_captain_points: float = TC_MIN_CAPTAIN_POINTS
    tc_easy_until_week: int = TC_EASY_UNTIL_WEEK
    fh_after_wildcard_weeks: int = FH_AFTER_WILDCARD_WEEKS
    roll_free_transfers: bool = False
    # Points added to the FH/WC gap bar per unit of fixture imbalance.
    # 0 keeps the old threshold. Second half multiplies the DGW term.
    fh_dgw_weight: float = 0.0
    wc_dgw_weight: float = 0.0
    dgw_h2_mult: float = DGW_H2_MULT_DEFAULT
    # Points subtracted from the WC bar per owned player under 60 minutes.
    wc_rotation_weight: float = 0.0
    # Extra WC bar at the first legal week, fading to 0 by wc_early_until.
    wc_early_weight: float = 0.0
    wc_early_until_gameweek: int = WC_EARLY_UNTIL_GAMEWEEK
    wc_eval_max_transfers: int = WC_EVAL_MAX_TRANSFERS
    # Weeks the FH/WC/recovery bars above are quoted over. Every bar-unit knob
    # (rotation, early, DGW wait) is quoted over the same span.
    chip_bar_reference_horizon: int = CHIP_BAR_REFERENCE_HORIZON


@dataclass
class ChipPlanResult:
    calendar: ChipCalendar
    reason: dict = field(default_factory=dict)
    # gameweek -> TransferPlan from the 1-GW free-hit probe (not JSON; used to
    # score actuals without solving that MIP again for the sim print).
    free_hit_plans: dict = field(default_factory=dict)
    wildcard_plans: dict = field(default_factory=dict)


@dataclass
class _ProjectedWeek:
    state_at_start: SquadState
    play_state: SquadState
    score: float
    plan: Optional[TransferPlan] = None


class ChipPlanner:
    def __init__(self, config: ChipStrategyConfig):
        self.config = config
        self.ft_rules = FreeTransferRules(config.season_start_year)

    def plan(
        self,
        state: SquadState,
        pool,
        as_of_gameweek: int,
        played: Optional[ChipCalendar] = None,
    ) -> ChipPlanResult:
        as_of = int(as_of_gameweek)
        played = played or ChipCalendar()
        current = self._plan_half(state, pool, as_of, played)
        if not self.config.forecast_future_halves:
            return current
        return self._with_placeholder_next_half(current, as_of)

    def _with_placeholder_next_half(self, current: ChipPlanResult, as_of: int):
        """Cap the transfer horizon at the next half without extra MIP rolls."""
        if not self.config.forecast_future_halves:
            return current
        _half_start, half_end = half_bounds(as_of)
        next_half_start = half_end + 1
        if next_half_start > LAST_GAMEWEEK:
            return current
        reason = dict(current.reason)
        reason['second_half'] = {
            'placeholder_wildcard': next_half_start,
            'deferred': True,
        }
        calendar = ChipCalendar.resolve(
            self.config.season_start_year,
            wildcards=tuple(current.calendar.wildcards) + (next_half_start,),
            free_hits=current.calendar.free_hits,
            bench_boosts=current.calendar.bench_boosts,
            triple_captains=current.calendar.triple_captains,
        )
        return ChipPlanResult(
            calendar=calendar,
            reason=reason,
            free_hit_plans=current.free_hit_plans,
            wildcard_plans=current.wildcard_plans,
        )

    def _earliest_chip(self) -> int:
        value = self.config.earliest_chip_gameweek
        if value is None:
            return EARLIEST_CHIP_GAMEWEEK
        return max(1, int(value))

    def _wildcard_bar(self, base_threshold):
        weight = float(self.config.wc_early_weight)
        earliest = self._earliest_chip()
        until = int(self.config.wc_early_until_gameweek)

        def bar(gameweek):
            return float(base_threshold) + wc_early_threshold_bonus(
                gameweek,
                weight,
                earliest=earliest,
                until=until,
            )

        return bar

    def _chip_candidate_weeks(self, as_of: int, half_end: int) -> Tuple[int, ...]:
        """FH/WC probe weeks: this week through the transfer horizon, this half."""
        last = min(
            int(as_of) + int(self.config.horizon_gameweeks) - 1,
            int(half_end),
            LAST_GAMEWEEK,
        )
        earliest = self._earliest_chip()
        return tuple(
            gameweek
            for gameweek in range(int(as_of), last + 1)
            if gameweek >= earliest
        )

    def _forecast_end(self, as_of: int) -> int:
        """Hold-path length: two horizons, so a WC in week 5 still has 5 weeks to score."""
        return min(
            LAST_GAMEWEEK,
            int(as_of) + 2 * int(self.config.horizon_gameweeks) - 1,
        )

    def _plan_half(
        self,
        state: SquadState,
        pool,
        as_of_gameweek: int,
        played: ChipCalendar,
    ) -> ChipPlanResult:
        as_of = int(as_of_gameweek)
        played = played or ChipCalendar()
        inventory = inventory_for(
            self.config.season_start_year, as_of, played
        )
        season = self.config.season_start_year
        _half_start, half_end = half_bounds(as_of)
        chips_left = inventory.count()
        weeks_left = half_end - as_of + 1
        assignments: Dict[int, str] = {}
        earliest = self._earliest_chip()
        occupied = blocked_chip_gameweeks(earliest)
        reason = {
            'as_of_gameweek': as_of,
            'chips_left': chips_left,
            'weeks_left': weeks_left,
            'forced': False,
        }

        if inventory.bench_boost and self._wildcard_already_played(
            played, as_of
        ):
            if as_of not in occupied and as_of <= half_end:
                assignments[as_of] = CHIP_BENCH_BOOST
                occupied.add(as_of)
                inventory = inventory.without(CHIP_BENCH_BOOST)
                reason['bench_boost_after_played_wildcard'] = as_of

        expiring = expiring_inventory(season, inventory, as_of)
        expiring_left = expiring.count()
        reason['expiring_chips_left'] = expiring_left

        if expiring_left > 0 and weeks_left <= expiring_left:
            forced = force_remaining_chips(
                as_of, half_end, expiring, occupied, earliest=earliest
            )
            assignments.update(forced)
            reason['forced'] = True
            reason['forced_assignments'] = {
                str(gameweek): chip for gameweek, chip in sorted(forced.items())
            }
            return ChipPlanResult(
                calendar=_calendar_from_assignments(
                    self.config.season_start_year,
                    played,
                    assignments,
                    earliest=earliest,
                ),
                reason=reason,
                free_hit_plans={},
            )

        fh_window = chip_window(season, CHIP_FREE_HIT, as_of)
        wc_window = chip_window(season, CHIP_WILDCARD, as_of)
        fh_rivals = chips_sharing_window(season, inventory, as_of, CHIP_FREE_HIT)
        wc_rivals = chips_sharing_window(season, inventory, as_of, CHIP_WILDCARD)
        fh_base = effective_threshold(
            self.config.fh_threshold, as_of, fh_rivals, *fh_window
        )
        wc_base = effective_threshold(
            self.config.wc_threshold, as_of, wc_rivals, *wc_window
        )
        recovery_threshold = effective_threshold(
            self.config.fh_recovery_threshold, as_of, fh_rivals, *fh_window
        )
        imbalance = fixture_imbalance(pool, as_of)
        half_mult = dgw_half_multiplier(as_of, self.config.dgw_h2_mult)
        low_mins = low_minute_owned_count(state, pool, as_of)
        fh_threshold, wc_threshold = adjust_chip_thresholds(
            fh_base,
            wc_base,
            fixture_imbalance_value=imbalance,
            half_multiplier=half_mult,
            fh_dgw_weight=self.config.fh_dgw_weight,
            wc_dgw_weight=self.config.wc_dgw_weight,
            low_minute_count=low_mins,
            wc_rotation_weight=self.config.wc_rotation_weight,
        )
        bar_scale = self._bar_scale()
        reason['effective_fh_threshold'] = fh_threshold * bar_scale
        reason['effective_wc_threshold'] = wc_threshold * bar_scale
        reason['effective_recovery_threshold'] = recovery_threshold * bar_scale
        reason['chip_bar_scale'] = bar_scale
        reason['effective_tc_threshold'] = self._tc_threshold(as_of)
        reason['tc_taper_floor'] = self.config.tc_taper_floor
        reason['tc_min_captain_points'] = self.config.tc_min_captain_points
        reason['fh_after_wildcard_weeks_blocked'] = (
            self.config.fh_after_wildcard_weeks
        )
        reason['fh_threshold_base'] = fh_base
        reason['wc_threshold_base'] = wc_base
        reason['fixture_imbalance'] = imbalance
        reason['dgw_half_multiplier'] = half_mult
        reason['low_minute_owned'] = low_mins
        reason['wc_early_weight'] = self.config.wc_early_weight
        reason['wc_early_until_gameweek'] = self.config.wc_early_until_gameweek

        this_week_only = bool(self.config.one_optimise_per_kind)
        if this_week_only:
            candidates = (
                (as_of,) if as_of >= earliest else ()
            )
            project_end = as_of
            start_states = {as_of: copy_squad_state(state)}
            score_until = self._forecast_end(as_of)
            base_scores = self._hold_scores(state, pool, as_of, score_until)
            owned_by_gw = {
                gameweek: state.player_ids
                for gameweek in range(as_of, score_until + 1)
            }
        else:
            candidates = self._chip_candidate_weeks(as_of, half_end)
            project_end = self._forecast_end(as_of)
            no_chip, _ = self._roll(state, pool, as_of, project_end, ChipCalendar())
            start_states = {
                gameweek: projected.state_at_start
                for gameweek, projected in no_chip.items()
            }
            base_scores = {
                gameweek: projected.score
                for gameweek, projected in no_chip.items()
            }
            owned_by_gw = {
                gameweek: projected.play_state.player_ids
                for gameweek, projected in no_chip.items()
            }

        fh_evals = {}
        if inventory.free_hit:
            fh_evals = self._evaluate_free_hits(
                pool,
                candidates,
                start_states,
                base_scores,
                roll_transfers=not this_week_only,
            )
        fh_plans = {gameweek: plan for gameweek, (_diff, plan) in fh_evals.items()}
        wc_evals = {}
        if inventory.wildcard:
            wc_evals = self._evaluate_wildcards(
                pool,
                candidates,
                start_states,
                base_scores,
                roll_transfers=not this_week_only,
            )
        wc_plans = {gameweek: plan for gameweek, (_diff, plan) in wc_evals.items()}
        reason['fh_differentials'] = [
            {'gameweek': gameweek, 'differential': diff}
            for gameweek, (diff, _plan) in sorted(fh_evals.items())
        ]
        reason['wc_differentials'] = [
            {'gameweek': gameweek, 'differential': diff}
            for gameweek, (diff, _plan) in sorted(wc_evals.items())
        ]

        played_wc = self._wildcard_week_this_half(played, as_of)
        best_fh = self._best_week(
            fh_evals, fh_threshold, self._fh_occupied(occupied, played_wc)
        )
        best_wc = self._best_week(
            wc_evals, self._wildcard_bar(wc_threshold), occupied
        )
        planned_fh = None
        planned_wc = None

        if (
            best_fh is not None
            and best_wc is not None
            and best_fh == best_wc
        ):
            if this_week_only:
                planned_wc = best_wc
                planned_fh = None
                reason['recovery'] = {
                    'gameweek': best_fh,
                    'chose_free_hit': False,
                    'skipped_roll': True,
                }
            else:
                recovered = self._recovery_prefers_free_hit(
                    pool,
                    best_fh,
                    start_states[best_fh],
                    wc_evals[best_wc][1],
                    recovery_threshold,
                )
                reason['recovery'] = {
                    'gameweek': best_fh,
                    'chose_free_hit': recovered,
                }
                if recovered:
                    planned_fh = best_fh
                    planned_wc = self._best_week(
                        wc_evals,
                        self._wildcard_bar(wc_threshold),
                        occupied | {best_fh},
                    )
                else:
                    planned_wc = best_wc
                    planned_fh = self._best_week(
                        fh_evals, fh_threshold, occupied | {best_wc}
                    )
        else:
            planned_fh = best_fh
            planned_wc = best_wc

        if planned_wc is not None and inventory.bench_boost:
            if planned_wc == half_end and half_end - 1 >= earliest_chip_gameweek(
                as_of, earliest
            ):
                if half_end - 1 not in occupied:
                    planned_wc = half_end - 1
            bench_gw = planned_wc + 1
            if bench_gw > half_end or bench_gw in occupied:
                bench_gw = None
        else:
            bench_gw = None

        if planned_wc is not None and planned_fh is not None:
            if planned_fh == planned_wc or free_hit_blocked_after_wildcard(
                planned_fh,
                planned_wc,
                weeks=self.config.fh_after_wildcard_weeks,
            ):
                planned_fh = None
        if bench_gw is not None and planned_fh == bench_gw:
            planned_fh = self._best_week(
                fh_evals,
                fh_threshold,
                self._fh_occupied(occupied | {planned_wc, bench_gw}, planned_wc),
            )

        post_wc_weeks = []
        wc_reprojected = False
        if planned_wc is not None:
            assignments[planned_wc] = CHIP_WILDCARD
            occupied.add(planned_wc)
            wc_plan = wc_evals.get(planned_wc)
            if this_week_only:
                if wc_plan is not None:
                    for gameweek in range(
                        int(planned_wc), min(half_end, LAST_GAMEWEEK) + 1
                    ):
                        owned_by_gw[gameweek] = wc_plan[1].selected_player_ids
            else:
                (
                    start_states,
                    base_scores,
                    owned_by_gw,
                    rolled_plans,
                ) = self._project_with_chips(
                    state,
                    pool,
                    as_of,
                    project_end,
                    self._planned_calendar(planned_wc, None, bench_gw),
                )
                wc_reprojected = True
                if planned_wc in rolled_plans:
                    wc_plans[int(planned_wc)] = rolled_plans[planned_wc]
                if inventory.free_hit:
                    later = tuple(
                        gameweek
                        for gameweek in candidates
                        if gameweek not in occupied
                        and not free_hit_blocked_after_wildcard(
                            gameweek,
                            planned_wc,
                            weeks=self.config.fh_after_wildcard_weeks,
                        )
                    )
                    post_wc_weeks = list(later)
                    fh_evals = self._evaluate_free_hits(
                        pool,
                        later,
                        start_states,
                        base_scores,
                        roll_transfers=True,
                    )
                    for gameweek, (_diff, plan) in fh_evals.items():
                        fh_plans[int(gameweek)] = plan
                    reason['fh_after_wildcard_weeks'] = list(later)
                    reason['fh_differentials_after_wildcard'] = [
                        {'gameweek': gameweek, 'differential': diff}
                        for gameweek, (diff, _plan) in sorted(fh_evals.items())
                    ]
                    if planned_fh is None or planned_fh > planned_wc:
                        planned_fh = self._best_week(
                            fh_evals,
                            fh_threshold,
                            self._fh_occupied(occupied, planned_wc),
                        )

        if bench_gw is not None:
            assignments[bench_gw] = CHIP_BENCH_BOOST
            occupied.add(bench_gw)
            reason['bench_boost_after_wildcard'] = bench_gw
        if planned_wc is not None:
            reason['wildcard_gameweek'] = planned_wc
        if planned_fh is not None and planned_fh not in occupied:
            assignments[planned_fh] = CHIP_FREE_HIT
            occupied.add(planned_fh)
            reason['free_hit_gameweek'] = planned_fh

        if not this_week_only and (
            planned_wc is not None or planned_fh is not None
        ):
            already_final = wc_reprojected and planned_fh is None
            if not already_final:
                (
                    start_states,
                    base_scores,
                    owned_by_gw,
                    rolled_plans,
                ) = self._project_with_chips(
                    state,
                    pool,
                    as_of,
                    project_end,
                    self._planned_calendar(planned_wc, planned_fh, bench_gw),
                )
                if planned_wc is not None and planned_wc in rolled_plans:
                    wc_plans[int(planned_wc)] = rolled_plans[planned_wc]
                if planned_fh is not None and planned_fh in rolled_plans:
                    fh_plans[int(planned_fh)] = rolled_plans[planned_fh]
            reason['committed_chip_roll'] = True

        if inventory.triple_captain:
            triple = self._pick_triple_captain(
                pool,
                as_of,
                min(project_end, half_end),
                owned_by_gw,
                occupied,
            )
            if triple is not None:
                assignments[triple['gameweek']] = CHIP_TRIPLE_CAPTAIN
                occupied.add(triple['gameweek'])
                reason['triple_captain'] = triple

        reason['fh_after_wildcard_weeks'] = reason.get(
            'fh_after_wildcard_weeks', post_wc_weeks
        )
        return ChipPlanResult(
            calendar=_calendar_from_assignments(
                self.config.season_start_year,
                played,
                assignments,
                earliest=earliest,
            ),
            reason=reason,
            free_hit_plans=fh_plans,
            wildcard_plans=wc_plans,
        )

    def _planned_calendar(self, planned_wc, planned_fh, bench_gw) -> ChipCalendar:
        return ChipCalendar(
            wildcards=(int(planned_wc),) if planned_wc is not None else (),
            free_hits=(int(planned_fh),) if planned_fh is not None else (),
            bench_boosts=(int(bench_gw),) if bench_gw is not None else (),
        )

    def _project_with_chips(
        self,
        state: SquadState,
        pool,
        as_of: int,
        project_end: int,
        chips: ChipCalendar,
    ):
        """Re-roll transfers from this week with committed chips on the calendar."""
        rolled, _ = self._roll(state, pool, as_of, project_end, chips)
        start_states = {
            gameweek: projected.state_at_start
            for gameweek, projected in rolled.items()
        }
        base_scores = {
            gameweek: projected.score for gameweek, projected in rolled.items()
        }
        owned_by_gw = {
            gameweek: projected.play_state.player_ids
            for gameweek, projected in rolled.items()
        }
        plans = {
            gameweek: projected.plan
            for gameweek, projected in rolled.items()
            if projected.plan is not None
        }
        return start_states, base_scores, owned_by_gw, plans

    def _state_at_start_of(
        self,
        state: SquadState,
        pool,
        as_of: int,
        target: int,
        chips: ChipCalendar,
    ) -> SquadState:
        if int(target) <= int(as_of):
            return copy_squad_state(state)
        _weeks, trailing = self._roll(
            state, pool, as_of, int(target) - 1, chips
        )
        return trailing

    def _wildcard_already_played(self, played: ChipCalendar, as_of: int) -> bool:
        week = self._wildcard_week_this_half(played, as_of)
        return week is not None and week < int(as_of)

    def _wildcard_week_this_half(self, played: ChipCalendar, as_of: int):
        half_start, half_end = half_bounds(as_of)
        weeks = [
            int(gameweek)
            for gameweek in played.wildcards
            if half_start <= int(gameweek) <= half_end
        ]
        return weeks[0] if weeks else None

    def _tc_threshold(self, gameweek: int) -> float:
        return triple_captain_threshold(
            gameweek,
            self.config.tc_above_average_threshold,
            floor=self.config.tc_taper_floor,
            easy_until=self.config.tc_easy_until_week,
        )

    def _fh_occupied(self, occupied, wildcard_gameweek):
        blocked = set(occupied)
        if wildcard_gameweek is None:
            return blocked
        gap = int(self.config.fh_after_wildcard_weeks)
        blocked.update(
            range(
                int(wildcard_gameweek),
                int(wildcard_gameweek) + gap + 1,
            )
        )
        return blocked

    def project(
        self,
        state: SquadState,
        pool,
        start_gw: int,
        end_gw: int,
        chips: ChipCalendar,
    ) -> Tuple[Dict[int, _ProjectedWeek], SquadState]:
        """Roll the squad forward week by week under a chip calendar."""
        return self._roll(state, pool, start_gw, end_gw, chips)

    def _strategy(
        self,
        chips: ChipCalendar,
        horizon=None,
        max_transfers=None,
        roll_free_transfers=None,
    ) -> SimulationStrategy:
        return SimulationStrategy(StrategyConfig(
            horizon_gameweeks=int(
                self.config.horizon_gameweeks if horizon is None else horizon
            ),
            min_current_gw_gain=self.config.min_current_gw_gain,
            min_horizon_gain=self.config.min_horizon_gain,
            max_transfers_per_gameweek=int(
                self.config.max_transfers_per_gameweek
                if max_transfers is None
                else max_transfers
            ),
            max_free_transfers=self.ft_rules.max_bank,
            relax_thresholds_at_max_ft=self.config.relax_thresholds_at_max_ft,
            chips=chips,
            bench_fill_weight=self.config.bench_fill_weight,
            bench_weights=self.config.bench_weights,
            reserve_keeper_weight=self.config.reserve_keeper_weight,
            season_start_year=self.config.season_start_year,
            roll_free_transfers=(
                self.config.roll_free_transfers
                if roll_free_transfers is None
                else bool(roll_free_transfers)
            ),
        ))

    def _expected_points(self, player_ids, pool, gameweek, **kwargs):
        kwargs.setdefault('bench_weights', self.config.bench_weights)
        kwargs.setdefault('reserve_keeper_weight', self.config.reserve_keeper_weight)
        return expected_gameweek_points(player_ids, pool, gameweek, **kwargs)

    def _hold_scores(self, state, pool, start_gw, end_gw):
        """Predicted XI points for the current 15, no transfer MIPs."""
        scores = {}
        player_ids = state.player_ids
        for gameweek in range(int(start_gw), min(int(end_gw), LAST_GAMEWEEK) + 1):
            scores[int(gameweek)] = self._expected_points(
                player_ids, pool, gameweek
            )
        return scores

    def _roll(
        self,
        state: SquadState,
        pool,
        start_gw: int,
        end_gw: int,
        chips: ChipCalendar,
        max_transfers=None,
        roll_free_transfers=None,
    ) -> Tuple[Dict[int, _ProjectedWeek], SquadState]:
        current = copy_squad_state(state)
        if int(end_gw) < int(start_gw) or int(start_gw) > LAST_GAMEWEEK:
            return {}, current
        strategy = self._strategy(
            chips,
            max_transfers=max_transfers,
            roll_free_transfers=roll_free_transfers,
        )
        weeks = {}
        for gameweek in range(int(start_gw), min(int(end_gw), LAST_GAMEWEEK) + 1):
            current.free_transfers = self.ft_rules.at_start_of_gameweek(
                gameweek, current.free_transfers
            )
            state_at_start = copy_squad_state(current)
            chip = chips.chip_at(gameweek)
            try:
                plan = strategy.plan(current, pool, gameweek)
            except ValueError:
                hold = self._expected_points(
                    current.player_ids, pool, gameweek
                )
                plan = SimulationStrategy._hold_plan(current, hold, hold)
            play = apply_transfer_plan(current, plan)
            score = self._expected_points(
                play.player_ids,
                pool,
                gameweek,
                bench_boost=chip == CHIP_BENCH_BOOST,
                triple_captain=chip == CHIP_TRIPLE_CAPTAIN,
            )
            weeks[gameweek] = _ProjectedWeek(state_at_start, play, score, plan)
            if chip == CHIP_FREE_HIT:
                current = copy_squad_state(state_at_start)
                used = 0
            else:
                current = play
                used = 0 if chip == CHIP_WILDCARD else len(plan.transfers)
            current.free_transfers = self.ft_rules.after_gameweek(
                gameweek,
                state_at_start.free_transfers if gameweek > 1 else 0,
                used,
                chip=chip,
            )
        return weeks, current

    def _reset_plan(self, state, pool, gameweek, *, free_hit: bool):
        if free_hit:
            chips = ChipCalendar(free_hits=(int(gameweek),))
        else:
            chips = ChipCalendar(wildcards=(int(gameweek),))
        return self._strategy(chips).plan(state, pool, gameweek)

    def _evaluate_free_hits(
        self,
        pool,
        weeks,
        start_states,
        base_scores,
        roll_transfers=True,
    ):
        """Horizon rebuild vs hold. Skip when the half's free hit is already used."""
        results = {}
        horizon = int(self.config.horizon_gameweeks)
        for gameweek in weeks:
            state = start_states.get(gameweek)
            if state is None:
                continue
            try:
                plan = self._reset_plan(state, pool, gameweek, free_hit=True)
            except ValueError:
                continue
            if roll_transfers:
                score = self._free_hit_horizon_score(state, pool, gameweek, plan)
                base = sum(
                    float(base_scores.get(target, 0.0))
                    for target in range(gameweek, gameweek + horizon)
                )
            else:
                score = float(plan.expected_selected_current)
                base = float(base_scores.get(gameweek, 0.0))
            results[int(gameweek)] = (float(score) - float(base), plan)
        return results

    def _free_hit_horizon_score(self, state, pool, gameweek, plan) -> float:
        """FH week with the rebuilt 15, then the reverted squad for the rest."""
        horizon = int(self.config.horizon_gameweeks)
        total = float(plan.expected_selected_current)
        start = int(gameweek) + 1
        end = int(gameweek) + horizon - 1
        if start > end:
            return total
        reverted = copy_squad_state(state)
        reverted.free_transfers = self.ft_rules.after_gameweek(
            gameweek,
            state.free_transfers if int(gameweek) > 1 else 0,
            0,
            chip=CHIP_FREE_HIT,
        )
        rolled, _ = self._roll(reverted, pool, start, end, ChipCalendar())
        return total + sum(projected.score for projected in rolled.values())

    def _evaluate_wildcards(
        self,
        pool,
        weeks,
        start_states,
        base_scores,
        roll_transfers=True,
    ):
        results = {}
        horizon = int(self.config.horizon_gameweeks)
        for gameweek in weeks:
            state = start_states.get(gameweek)
            if state is None:
                continue
            try:
                plan = self._reset_plan(state, pool, gameweek, free_hit=False)
            except ValueError:
                continue
            base = sum(
                float(base_scores.get(target, 0.0))
                for target in range(gameweek, gameweek + horizon)
            )
            if roll_transfers and int(self.config.wc_eval_max_transfers) > 0:
                score = self._wildcard_roll_score(state, pool, gameweek, plan)
            else:
                score = float(plan.expected_selected_horizon)
            diff = score - base
            results[int(gameweek)] = (float(diff), plan)
        return results

    def _wildcard_roll_score(self, state, pool, gameweek, plan) -> float:
        """WC week rebuild, then up to 2 rolling transfers for the rest of the horizon."""
        horizon = int(self.config.horizon_gameweeks)
        play = apply_transfer_plan(state, plan)
        play.free_transfers = self.ft_rules.after_gameweek(
            gameweek,
            state.free_transfers if int(gameweek) > 1 else 0,
            0,
            chip=CHIP_WILDCARD,
        )
        total = float(plan.expected_selected_current)
        start = int(gameweek) + 1
        end = int(gameweek) + horizon - 1
        if start > end:
            return total
        rolled, _ = self._roll(
            play,
            pool,
            start,
            end,
            ChipCalendar(),
            max_transfers=int(self.config.wc_eval_max_transfers),
            roll_free_transfers=True,
        )
        return total + sum(projected.score for projected in rolled.values())

    def _bar_scale(self) -> float:
        """Stretch a quoted bar onto the horizon the gaps are measured over."""
        reference = int(self.config.chip_bar_reference_horizon)
        if reference <= 0:
            return 1.0
        return float(self.config.horizon_gameweeks) / float(reference)

    def _week_bar(self, threshold, gameweek) -> float:
        base = threshold(int(gameweek)) if callable(threshold) else threshold
        return float(base) * self._bar_scale()

    def _best_week(self, evals, threshold, occupied):
        qualifying = [
            (diff, -int(gameweek), int(gameweek))
            for gameweek, (diff, _plan) in evals.items()
            if diff + 1e-8 >= self._week_bar(threshold, gameweek)
            and int(gameweek) not in occupied
        ]
        if not qualifying:
            return None
        qualifying.sort(reverse=True)
        return qualifying[0][2]

    def _recovery_prefers_free_hit(
        self,
        pool,
        gameweek: int,
        pre_state: SquadState,
        wc_plan,
        recovery_threshold: float,
    ) -> bool:
        horizon = int(self.config.horizon_gameweeks)
        start = int(gameweek) + 1
        end = int(gameweek) + horizon
        fh_path = sum(
            projected.score
            for projected in self._roll(
                pre_state, pool, start, end, ChipCalendar()
            )[0].values()
        )
        wc_state = apply_transfer_plan(pre_state, wc_plan)
        wc_state.free_transfers = self.ft_rules.after_gameweek(
            gameweek,
            pre_state.free_transfers if gameweek > 1 else 0,
            0,
            chip=CHIP_WILDCARD,
        )
        wc_path = sum(
            projected.score
            for projected in self._roll(
                wc_state, pool, start, end, ChipCalendar()
            )[0].values()
        )
        bar = float(recovery_threshold) * self._bar_scale()
        return wc_path - fh_path <= bar + 1e-8

    def _pick_triple_captain(
        self,
        pool,
        as_of: int,
        half_end: int,
        owned_by_gw,
        occupied,
    ):
        window = [
            gameweek
            for gameweek in range(
                int(as_of),
                min(
                    int(as_of) + int(self.config.tc_window),
                    int(half_end) + 1,
                    LAST_GAMEWEEK + 1,
                ),
            )
        ]
        if not window:
            return None
        qualifying = []
        for gameweek in window:
            if gameweek in occupied:
                continue
            player_ids = owned_by_gw.get(gameweek)
            if not player_ids:
                continue
            try:
                lineup = pick_lineup(player_ids, pool, gameweek)
            except ValueError:
                continue
            player_id = lineup.captain_id
            points = pool[player_id].points(gameweek)
            if points + 1e-8 < float(self.config.tc_min_captain_points):
                continue
            mean = sum(
                pool[player_id].points(target) for target in window
            ) / len(window)
            needed = self._tc_threshold(gameweek)
            if points + 1e-8 < mean + needed:
                continue
            qualifying.append(
                (points, -int(gameweek), player_id, int(gameweek), mean, needed)
            )
        if not qualifying:
            return None
        qualifying.sort(reverse=True)
        points, _, player_id, gameweek, mean, needed = qualifying[0]
        return {
            'gameweek': gameweek,
            'player_id': player_id,
            'points': float(points),
            'mean': float(mean),
            'threshold': float(needed),
        }


def visible_chip_plan(calendar: ChipCalendar, reason=None) -> dict:
    """Chip weeks chosen this half, omitting the deferred next-half WC cap."""
    data = calendar.as_dict()
    second_half = (reason or {}).get('second_half') or {}
    placeholder = second_half.get('placeholder_wildcard')
    if placeholder and second_half.get('deferred'):
        data['wildcards'] = [
            gameweek for gameweek in data['wildcards'] if gameweek != placeholder
        ]
    return data
