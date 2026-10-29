"""One season, one decision per gameweek.

`forecast_at(gameweek)` supplies that week's saved predictions. Without it,
`pool` stays as it was passed in. Points awarded for the week come from
`outcomes`, which may be the real game or a sampled one. Free Hit scores the
temporary squad and then puts the old squad back. Wildcard and Free Hit do
not pay hit points. Any other transfers beyond the free allowance cost 4 each.
"""
from __future__ import annotations

import time

from squadtools.simulation_strategy import (
    CHIP_FREE_HIT,
    CHIP_WILDCARD,
    ChipCalendar,
    FreeTransferRules,
    SquadState,
    apply_transfer_plan,
    copy_squad_state,
)

from squadtools.chip_strategy import chip_window, inventory_for

from fpl_manager.actions import (
    ACTIONS,
    BENCH_BOOST,
    FREE_HIT,
    HOLD,
    TRANSFER_1,
    TRANSFER_2,
    TRANSFER_3,
    TRANSFER_4,
    TRANSFER_5,
    TRIPLE_CAPTAIN,
    WILDCARD,
    legal_mask,
)
from fpl_manager.lineup import greedy_lineup, hit_penalty, score_lineup
from fpl_manager.observation import LAST_GAMEWEEK, build_observation
from fpl_manager.optimiser import GreedyOptimiser
from fpl_manager.percentiles import describe_season
from fpl_manager.reward import terminal_reward, weekly_reward


class SeasonEpisode:
    def __init__(
        self,
        pool,
        squad: SquadState,
        outcomes,
        season_start_year: int = 25,
        first_gameweek: int = 1,
        last_gameweek: int = LAST_GAMEWEEK,
        optimiser=None,
        reference_scores=None,
        forecast_at=None,
    ):
        self.pool = pool
        self.forecast_at = forecast_at
        self._forecast_gameweek = None
        self.squad = copy_squad_state(squad)
        self.outcomes = outcomes
        self.season_start_year = int(season_start_year)
        self.gameweek = int(first_gameweek)
        self.last_gameweek = int(last_gameweek)
        self.optimiser = optimiser or GreedyOptimiser()
        self.reference_scores = reference_scores or (lambda _gameweek: ())
        self._cache_at_start = _cache_counts(self.optimiser)
        self._season_started = time.perf_counter()
        self.solved_count = 0
        self.cached_count = 0
        self.season_seconds = 0.0
        self._sync_forecast()
        self.played = ChipCalendar()
        self.points_total = 0.0
        self.weekly_reward_total = 0.0
        self.terminal = 0.0
        self.done = False
        self.rules = FreeTransferRules(self.season_start_year)

    def mask(self) -> tuple:
        return legal_mask(self.season_start_year, self.gameweek, self.played, self.squad)

    def _sync_forecast(self):
        if self.forecast_at is None or self._forecast_gameweek == self.gameweek:
            return
        pool, snapshot_id = self.forecast_at(self.gameweek)
        self.pool = pool
        self._forecast_gameweek = self.gameweek
        if snapshot_id is not None:
            self.optimiser.snapshot_id = int(snapshot_id)

    def observation(self) -> tuple:
        self._sync_forecast()
        gains = {}
        reporter = getattr(self.optimiser, 'transfer_gains', None)
        if reporter is not None:
            gains = reporter(self.squad, self.pool, self.gameweek)
        return build_observation(
            self.season_start_year,
            self.gameweek,
            self.squad,
            self.played,
            self.pool,
            gains,
        )

    def step(self, action_index: int):
        if self.done:
            raise RuntimeError('season already finished')
        self._sync_forecast()
        if action_index < 0 or action_index >= len(ACTIONS) or not self.mask()[action_index]:
            raise ValueError('illegal action %s' % action_index)
        action = ACTIONS[action_index]
        before = copy_squad_state(self.squad)
        plan, chip = self.optimiser.execute(action, before, self.pool, self.gameweek)
        return self._commit(action, before, plan, chip)

    def apply_plan(self, plan, chip):
        """Apply a plan from the existing manager.

        That manager can transfer and play a chip in the same week. One PPO
        action cannot. The chip still has to be legal, and a free hit still
        puts the old squad back.
        """
        if self.done:
            raise RuntimeError('season already finished')
        self._sync_forecast()
        if chip is not None and not self._chip_allowed(chip):
            raise ValueError('illegal chip %s' % chip)
        before = copy_squad_state(self.squad)
        return self._commit(_label(plan, chip), before, plan, chip)

    def _chip_allowed(self, chip: str) -> bool:
        inventory = inventory_for(self.season_start_year, self.gameweek, self.played)
        held = {
            CHIP_WILDCARD: inventory.wildcard,
            CHIP_FREE_HIT: inventory.free_hit,
            'bench_boost': inventory.bench_boost,
            'triple_captain': inventory.triple_captain,
        }
        if not held.get(chip, False):
            return False
        start, end = chip_window(self.season_start_year, chip, self.gameweek)
        return start <= int(self.gameweek) <= end

    def _commit(self, action, before, plan, chip):
        gameweek = self.gameweek
        scoring = apply_transfer_plan(before, plan) if plan.transfers else before
        hits = hit_penalty(
            len(plan.transfers),
            before.free_transfers,
            chip in (CHIP_WILDCARD, CHIP_FREE_HIT),
        )
        played_points = self._score(scoring.player_ids, gameweek, chip)
        net = played_points - hits
        self.points_total += net
        week_reward = weekly_reward(net, self.reference_scores(gameweek))
        self.weekly_reward_total += week_reward
        self._advance(before, scoring, plan, chip, gameweek)
        done = self.gameweek > self.last_gameweek
        self.done = done
        reward = week_reward
        if done:
            self.terminal = terminal_reward(self.points_total)
            reward += self.terminal
            solved, cached = _cache_counts(self.optimiser)
            self.solved_count = solved - self._cache_at_start[0]
            self.cached_count = cached - self._cache_at_start[1]
            self.season_seconds = time.perf_counter() - self._season_started
            print(
                'Season score %s%s. Squad optimisations run %s, from cache %s. %s.'
                % (
                    _format_points(self.points_total),
                    _percentile_clause(describe_season(
                        self.season_start_year, self.points_total,
                    )),
                    solved - self._cache_at_start[0],
                    cached - self._cache_at_start[1],
                    _format_duration(time.perf_counter() - self._season_started),
                ),
                flush=True,
            )
        info = {
            'action': action,
            'chip': chip,
            'gameweek': gameweek,
            'points': net,
            'hits': hits,
            'transfers': len(plan.transfers),
            'weekly_reward': week_reward,
            'terminal_reward': self.terminal if done else 0.0,
            'squad': self.squad.player_ids,
        }
        return self.observation() if not done else tuple(), float(reward), done, info

    def _score(self, player_ids, gameweek, chip) -> float:
        # The XI and captain are chosen from this week's forecast. The points
        # added afterwards are what those players actually scored.
        predicted = lambda player_id: float(self.pool[player_id].points(gameweek))
        actual = lambda player_id: float(self.outcomes(player_id, gameweek))
        position_of = lambda player_id: self.pool[player_id].position
        starters, bench, captain_id, _vice = greedy_lineup(
            player_ids, predicted, position_of,
        )
        return score_lineup(
            actual,
            starters,
            bench,
            captain_id,
            bench_boost=chip == 'bench_boost',
            triple_captain=chip == 'triple_captain',
        )

    def _advance(self, before, scoring, plan, chip, gameweek):
        if chip:
            self.played = _with_chip(self.played, chip, gameweek)
        used = 0 if chip in (CHIP_WILDCARD, CHIP_FREE_HIT) else len(plan.transfers)
        next_transfers = self.rules.after_gameweek(
            gameweek, before.free_transfers, used, chip,
        )
        if chip == CHIP_FREE_HIT:
            self.squad = SquadState(
                owned=dict(before.owned),
                bank=int(before.bank),
                free_transfers=next_transfers,
            )
        else:
            self.squad = SquadState(
                owned=dict(scoring.owned),
                bank=int(scoring.bank),
                free_transfers=next_transfers,
            )
        self.gameweek = int(gameweek) + 1


def _with_chip(played: ChipCalendar, chip: str, gameweek: int) -> ChipCalendar:
    gameweek = int(gameweek)
    wildcards = played.wildcards
    free_hits = played.free_hits
    bench_boosts = played.bench_boosts
    triple_captains = played.triple_captains
    if chip == 'wildcard':
        wildcards = wildcards + (gameweek,)
    elif chip == 'free_hit':
        free_hits = free_hits + (gameweek,)
    elif chip == 'bench_boost':
        bench_boosts = bench_boosts + (gameweek,)
    elif chip == 'triple_captain':
        triple_captains = triple_captains + (gameweek,)
    return ChipCalendar(
        wildcards=wildcards,
        free_hits=free_hits,
        bench_boosts=bench_boosts,
        triple_captains=triple_captains,
    )


def _cache_counts(optimiser):
    cache = getattr(optimiser, 'cache', None)
    if cache is None:
        return 0, 0
    return int(cache.misses), int(cache.hits)


def _percentile_clause(text) -> str:
    text = str(text or '').strip()
    if not text:
        return ''
    return ', ' + text


def _format_duration(seconds) -> str:
    remaining = max(0, int(round(float(seconds))))
    hours, remaining = divmod(remaining, 3600)
    minutes, secs = divmod(remaining, 60)
    if hours:
        return '%dh %dm %ds' % (hours, minutes, secs)
    if minutes:
        return '%dm %ds' % (minutes, secs)
    return '%ds' % secs


def _format_points(value) -> str:
    value = float(value)
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return '%.1f' % value


def hold_action_index() -> int:
    return ACTIONS.index(HOLD)


def _label(plan, chip) -> str:
    if chip == CHIP_WILDCARD:
        return WILDCARD
    if chip == CHIP_FREE_HIT:
        return FREE_HIT
    if chip == 'bench_boost':
        return BENCH_BOOST
    if chip == 'triple_captain':
        return TRIPLE_CAPTAIN
    count = len(plan.transfers)
    if count <= 0:
        return HOLD
    names = (TRANSFER_1, TRANSFER_2, TRANSFER_3, TRANSFER_4, TRANSFER_5)
    if count > len(names):
        return names[-1]
    return names[count - 1]
