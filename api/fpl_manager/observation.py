"""Fixed observation order. The same scales are used in training and later inference.

Points and transfer gains are divided by POINT_SCALE. Money is stored by the
engine in tenths of a million, converted here to millions, then divided by
MONEY_SCALE. There is no postponement feed, so fixtures still to be
rescheduled stay 0.

A player is weak when their predicted points over the window average below
WEAK_POINTS_PER_GAMEWEEK. That threshold is not itself a feature.
"""
from __future__ import annotations

from squadtools.chip_strategy import chip_window, inventory_for
from squadtools.simulation_strategy import (
    CHIP_BENCH_BOOST,
    CHIP_FREE_HIT,
    CHIP_TRIPLE_CAPTAIN,
    CHIP_WILDCARD,
    ChipCalendar,
    FreeTransferRules,
    SquadState,
)

from fpl_manager.lineup import greedy_lineup, score_lineup

WEAK_POINTS_PER_GAMEWEEK = 2.0
POINT_SCALE = 100.0
MONEY_SCALE = 20.0
BLANK_HORIZON = 8
LAST_GAMEWEEK = 38

OBSERVATION_NAMES = (
    'current_gameweek',
    'gameweeks_remaining',
    'season_fraction_remaining',
    'gameweeks_until_blank',
    'gameweeks_until_double',
    'blank_within_horizon',
    'double_within_horizon',
    'blanks_next_3',
    'blanks_next_6',
    'doubles_next_3',
    'doubles_next_6',
    'fixtures_to_reschedule',
    'squad_xp_1',
    'squad_xp_3',
    'squad_xp_5',
    'squad_xp_8',
    'xi_next',
    'bench_next',
    'captain_contribution',
    'weak_next',
    'weak_next_3',
    'bank',
    'squad_value',
    'free_transfers',
    'wildcard_available',
    'free_hit_available',
    'bench_boost_available',
    'triple_captain_available',
    'gameweeks_until_wildcard_deadline',
    'gameweeks_until_free_hit_deadline',
    'gameweeks_until_bench_boost_deadline',
    'gameweeks_until_triple_captain_deadline',
    'best_1_transfer_gain_3gw',
    'best_1_transfer_gain_6gw',
    'best_2_transfer_gain_3gw',
    'best_2_transfer_gain_6gw',
    'best_3_transfer_gain_3gw',
    'best_3_transfer_gain_6gw',
    'best_4_transfer_gain_3gw',
    'best_4_transfer_gain_6gw',
    'best_5_transfer_gain_3gw',
    'best_5_transfer_gain_6gw',
)


def _clip_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _fixture_count(player, gameweek: int) -> int:
    fixtures = getattr(player, 'fixture_ids_by_gameweek', None) or {}
    return len(fixtures.get(int(gameweek), ()) or ())


def _event_counts(pool, squad: SquadState, start: int, length: int):
    blanks = 0
    doubles = 0
    first_blank = None
    first_double = None
    for offset in range(length):
        gameweek = int(start) + offset
        if gameweek > LAST_GAMEWEEK:
            break
        week_blank = False
        week_double = False
        for player_id in squad.player_ids:
            count = _fixture_count(pool[player_id], gameweek)
            if count == 0:
                blanks += 1
                week_blank = True
            elif count >= 2:
                doubles += 1
                week_double = True
        if week_blank and first_blank is None:
            first_blank = offset + 1
        if week_double and first_double is None:
            first_double = offset + 1
    return blanks, doubles, first_blank, first_double


def _xi_points(squad: SquadState, pool, gameweek: int) -> float:
    starters, bench, captain_id, _vice = greedy_lineup(
        squad.player_ids,
        lambda player_id: pool[player_id].points(gameweek),
        lambda player_id: pool[player_id].position,
    )
    return score_lineup(
        lambda player_id: pool[player_id].points(gameweek),
        starters,
        bench,
        captain_id,
        bench_boost=False,
        triple_captain=False,
    )


def _window_points(squad, pool, start, length) -> float:
    total = 0.0
    for offset in range(length):
        gameweek = int(start) + offset
        if gameweek > LAST_GAMEWEEK:
            break
        total += _xi_points(squad, pool, gameweek)
    return total


def _weak_count(squad, pool, start, length) -> int:
    weeks = [
        int(start) + offset
        for offset in range(length)
        if int(start) + offset <= LAST_GAMEWEEK
    ]
    if not weeks:
        return 0
    count = 0
    for player_id in squad.player_ids:
        average = sum(float(pool[player_id].points(gameweek)) for gameweek in weeks) / float(len(weeks))
        if average < WEAK_POINTS_PER_GAMEWEEK:
            count += 1
    return count


def _deadline_feature(season_start_year, gameweek, chip, available: bool) -> float:
    if not available:
        return 0.0
    _start, end = chip_window(season_start_year, chip, gameweek)
    return _clip_unit((int(end) - int(gameweek)) / float(LAST_GAMEWEEK))


def build_observation(
    season_start_year: int,
    gameweek: int,
    squad: SquadState,
    played: ChipCalendar,
    pool,
    transfer_gains=None,
) -> tuple:
    """Forecasts only. Realised outcomes for the gameweek being played are not an input."""
    rules = FreeTransferRules(season_start_year)
    free_transfers = rules.at_start_of_gameweek(gameweek, squad.free_transfers)
    remaining = max(0, LAST_GAMEWEEK - int(gameweek) + 1)
    _blanks, _doubles, first_blank, first_double = _event_counts(
        pool, squad, gameweek, BLANK_HORIZON,
    )
    blanks_3, doubles_3, _, _ = _event_counts(pool, squad, gameweek, 3)
    blanks_6, doubles_6, _, _ = _event_counts(pool, squad, gameweek, 6)
    inventory = inventory_for(season_start_year, gameweek, played)
    starters, bench, captain_id, _vice = greedy_lineup(
        squad.player_ids,
        lambda player_id: pool[player_id].points(gameweek),
        lambda player_id: pool[player_id].position,
    )
    bench_points = sum(float(pool[player_id].points(gameweek)) for player_id in bench)
    captain_points = float(pool[captain_id].points(gameweek))
    squad_value = sum(int(pool[player_id].price) for player_id in squad.player_ids) / 10.0
    gains = transfer_gains or {}
    values = [
        _clip_unit((int(gameweek) - 1) / float(LAST_GAMEWEEK - 1)),
        _clip_unit(remaining / float(LAST_GAMEWEEK)),
        _clip_unit(remaining / float(LAST_GAMEWEEK)),
        _clip_unit((first_blank or BLANK_HORIZON) / float(BLANK_HORIZON)),
        _clip_unit((first_double or BLANK_HORIZON) / float(BLANK_HORIZON)),
        1.0 if first_blank is not None else 0.0,
        1.0 if first_double is not None else 0.0,
        blanks_3 / 15.0,
        blanks_6 / 15.0,
        doubles_3 / 15.0,
        doubles_6 / 15.0,
        0.0,
        _window_points(squad, pool, gameweek, 1) / POINT_SCALE,
        _window_points(squad, pool, gameweek, 3) / POINT_SCALE,
        _window_points(squad, pool, gameweek, 5) / POINT_SCALE,
        _window_points(squad, pool, gameweek, 8) / POINT_SCALE,
        _xi_points(squad, pool, gameweek) / POINT_SCALE,
        bench_points / POINT_SCALE,
        (2.0 * captain_points) / POINT_SCALE,
        _weak_count(squad, pool, gameweek, 1) / 15.0,
        _weak_count(squad, pool, gameweek, 3) / 15.0,
        (int(squad.bank) / 10.0) / MONEY_SCALE,
        squad_value / (MONEY_SCALE * 5.0),
        free_transfers / float(rules.max_bank),
        1.0 if inventory.wildcard else 0.0,
        1.0 if inventory.free_hit else 0.0,
        1.0 if inventory.bench_boost else 0.0,
        1.0 if inventory.triple_captain else 0.0,
        _deadline_feature(season_start_year, gameweek, CHIP_WILDCARD, inventory.wildcard),
        _deadline_feature(season_start_year, gameweek, CHIP_FREE_HIT, inventory.free_hit),
        _deadline_feature(season_start_year, gameweek, CHIP_BENCH_BOOST, inventory.bench_boost),
        _deadline_feature(season_start_year, gameweek, CHIP_TRIPLE_CAPTAIN, inventory.triple_captain),
    ]
    for count in (1, 2, 3, 4, 5):
        for horizon in (3, 6):
            values.append(float(gains.get((count, horizon), 0.0)) / POINT_SCALE)
    if len(values) != len(OBSERVATION_NAMES):
        raise RuntimeError('observation length drifted from its documented names')
    return tuple(float(value) for value in values)
