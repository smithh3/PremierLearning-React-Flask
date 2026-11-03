"""Pure offline squad and transfer optimisation for historical replays."""

import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

from pulp import (
    LpMaximize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    PulpSolverError,
    lpSum,
)

from squadtools.availability import (
    SIM_BENCH_FILL_WEIGHT,
    slot_bench_weights,
    row_optimiser_points,
)


_original_popen = None
_quiet_cbc_installed = False


def _command_basename(cmd) -> str:
    if isinstance(cmd, (list, tuple)) and cmd:
        raw = cmd[0]
    elif isinstance(cmd, str) and cmd.strip():
        raw = cmd.split()[0]
    else:
        return ''
    return os.path.basename(str(raw)).lower()


def _is_cbc_command(cmd) -> bool:
    name = _command_basename(cmd)
    return name == 'cbc' or name == 'cbc.exe'


def _quiet_cbc_kwargs(kwargs):
    """Point CBC at NUL so it cannot inherit debugger pipes.

    On Windows, close_fds=True plus CREATE_NO_WINDOW was killing process-pool
    workers (BrokenProcessPool). DEVNULL stdio is enough to avoid the hang.
    """
    quiet = dict(kwargs)
    quiet['stdin'] = subprocess.DEVNULL
    quiet['stdout'] = subprocess.DEVNULL
    quiet['stderr'] = subprocess.DEVNULL
    if os.name == 'nt':
        quiet['close_fds'] = False
        quiet['creationflags'] = (
            quiet.get('creationflags', 0)
            | getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
        )
    else:
        quiet['close_fds'] = True
    return quiet


def _restore_unpatched_createprocess():
    """Undo pydevd's Windows CreateProcess hook if it is installed."""
    if os.name != 'nt':
        return False
    try:
        import _winapi
    except ImportError:
        return False
    original = getattr(_winapi, 'original_CreateProcess', None)
    if original is None or _winapi.CreateProcess is original:
        return False
    _winapi.CreateProcess = original
    return True


def _pydev_monkey_module():
    try:
        from _pydev_bundle import pydev_monkey
        return pydev_monkey
    except ImportError:
        pass
    for module in list(sys.modules.values()):
        if module is None:
            continue
        name = str(getattr(module, '__name__', '') or '')
        if name.endswith('pydev_monkey'):
            return module
    return None


def _popen_for_cbc():
    """The real Popen, not a debugger wrapper that injects into cbc.exe."""
    monkey = _pydev_monkey_module()
    if monkey is not None:
        for attr in ('original_Popen', 'original_subprocess_Popen'):
            recorded = getattr(monkey, attr, None)
            if callable(recorded):
                return recorded
        skip = getattr(monkey, 'skip_subprocess_arg_patch', None)
        if skip is not None:
            def _skipped(*args, **kwargs):
                with skip():
                    return _original_popen(*args, **kwargs)
            return _skipped
    return _original_popen


# v1, v2 and the fast run each spawn CBC from every worker. Letting all of
# those start at once exhausts the Windows paging file (WinError 1455) and
# the worker dies, which the parent then sees as a broken pipe.
_CBC_SLOTS = 4
_CBC_ATTEMPTS = 3
_CBC_MEMORY_WINERRORS = (8, 1450, 1455)


def _memory_error(error) -> bool:
    winerror = getattr(error, 'winerror', None)
    return winerror in _CBC_MEMORY_WINERRORS


class _CbcSlot:
    """One of a few machine-wide permissions to run cbc.exe."""

    def __init__(self):
        self._handle = None

    def acquire(self):
        directory = Path(tempfile.gettempdir()) / 'premier-cbc-slots'
        directory.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + 600
        while True:
            for index in range(_CBC_SLOTS):
                handle = open(directory / ('slot-%s.lock' % index), 'a+b')
                if handle.seek(0, os.SEEK_END) == 0:
                    handle.write(b'\0')
                    handle.flush()
                handle.seek(0)
                try:
                    _lock_file(handle)
                except OSError:
                    handle.close()
                    continue
                self._handle = handle
                return
            if time.time() > deadline:
                raise OSError('No CBC slot became free')
            time.sleep(0.25)

    def release(self):
        handle = self._handle
        self._handle = None
        if handle is not None:
            _unlock_file(handle)


def _lock_file(handle):
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle):
    try:
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


# A healthy solve takes seconds: the slowest of 55,000 saved in
# model_eval.db took 6.4s. The bundled CBC sometimes hangs at full CPU and
# never finishes, which froze a training run for 75 minutes.
_CBC_WALL_SECONDS = 120


class CbcTimedOut(ValueError):
    """CBC ran past the wall-clock limit and was stopped."""


class _ReleasingPopen:
    """Releases the CBC slot when PuLP finishes waiting on the process.

    PuLP waits with no time limit. A wait without one is capped at
    `_CBC_WALL_SECONDS`, after which CBC is killed.
    """

    def __init__(self, process, slot):
        self._process = process
        self._slot = slot

    def wait(self, timeout=None):
        try:
            if timeout is not None:
                return self._process.wait(timeout)
            try:
                return self._process.wait(_CBC_WALL_SECONDS)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
                _remove_pulp_files(getattr(self._process, 'args', None))
                raise CbcTimedOut(
                    'CBC was still running after %ss and was stopped' % _CBC_WALL_SECONDS
                )
        finally:
            slot = self._slot
            self._slot = None
            if slot is not None:
                slot.release()

    def __getattr__(self, name):
        return getattr(self._process, name)


def _remove_pulp_files(args):
    """Delete the model and solution files a killed solve leaves in temp."""
    if isinstance(args, str):
        args = args.split()
    for arg in args or ():
        text = str(arg)
        if text.endswith(('-pulp.mps', '-pulp.lp', '-pulp.sol', '-pulp.mst')):
            try:
                os.remove(text)
            except OSError:
                pass


def _cbc_aware_popen(*args, **kwargs):
    cmd = args[0] if args else kwargs.get('args')
    if not _is_cbc_command(cmd):
        return _original_popen(*args, **kwargs)
    _restore_unpatched_createprocess()
    kwargs = _quiet_cbc_kwargs(kwargs)
    last_error = None
    for attempt in range(_CBC_ATTEMPTS):
        slot = _CbcSlot()
        slot.acquire()
        try:
            process = _popen_for_cbc()(*args, **kwargs)
        except OSError as error:
            slot.release()
            last_error = error
            if not _memory_error(error) or attempt + 1 == _CBC_ATTEMPTS:
                raise
            time.sleep(1.0 * (attempt + 1))
            continue
        return _ReleasingPopen(process, slot)
    raise last_error


def ensure_quiet_cbc():
    """Make CBC subprocesses ignore inherited stdio. Safe to call many times."""
    global _quiet_cbc_installed, _original_popen
    _restore_unpatched_createprocess()
    if _quiet_cbc_installed:
        return
    _original_popen = subprocess.Popen
    subprocess.Popen = _cbc_aware_popen
    _quiet_cbc_installed = True


def _cbc_solver():
    """Solve with CBC. Never pass `threads`.

    The CBC binary bundled with PuLP deadlocks on Windows as soon as branch
    and bound enters the tree: it finishes the root node, then sits forever
    at constant CPU and never writes a solution file. Even `threads 1` does
    it, and single-threaded is already the default, so the option only buys
    a hang. Models that solve at the root node survive it, which is why a
    run can complete GW1 and then stall on the first branching model.
    """
    ensure_quiet_cbc()
    return PULP_CBC_CMD(msg=0)


def _solve_model(model):
    """Run CBC, retrying a crash so a full training worker does not die.

    A solve that hangs is not retried: CBC hangs on the same model again.
    """
    last_error = None
    for attempt in range(_CBC_ATTEMPTS):
        try:
            return model.solve(_cbc_solver())
        except CbcTimedOut as error:
            print('Squad optimisation skipped: %s.' % error, flush=True)
            raise
        except (PulpSolverError, OSError) as error:
            last_error = error
            time.sleep(0.2 * (attempt + 1))
    raise ValueError('CBC failed to solve %s (%s)' % (model.name, last_error))


SQUAD_POSITION_COUNTS = {1: 2, 2: 5, 3: 5, 4: 3}
LINEUP_POSITION_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
LINEUP_POSITION_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
MAX_PLAYERS_PER_TEAM = 3
INITIAL_BUDGET = 1000
# Drop rotation risks / non-assets before the MIP, except owned players and a
# couple of cheap enablers per position so budget / 3-per-club still works.
OPTIMISER_MIN_AVG_MINUTES = 50.0
OPTIMISER_MIN_AVG_POINTS = 2.0
CHEAP_ENABLERS_PER_POSITION = 2
# Weeks that min_horizon_gain is quoted over. See StrategyConfig.
MIN_GAIN_REFERENCE_HORIZON = 5


@dataclass
class PlayerProjection:
    player_id: int
    player_code: int
    name: str
    position: int
    team_id: int
    price: int
    selectable: bool
    points_by_gameweek: Dict[int, float] = field(default_factory=dict)
    minutes_by_gameweek: Dict[int, float] = field(default_factory=dict)
    fixture_ids_by_gameweek: Dict[int, set] = field(default_factory=dict)

    def points(self, gameweek: int) -> float:
        if not self.selectable:
            return 0.0
        return float(self.points_by_gameweek.get(int(gameweek), 0.0))

    def minutes(self, gameweek: int) -> float:
        if not self.selectable:
            return 0.0
        return float(self.minutes_by_gameweek.get(int(gameweek), 0.0))


class PredictionPool:
    def __init__(self, players: Dict[int, PlayerProjection]):
        self.players = players

    @property
    def predicts_minutes(self) -> bool:
        """False for a model with no minutes net (learning_v2): its points carry
        minutes risk, so anything gated on minutes uses predicted points instead."""
        return any(
            float(minutes or 0.0) > 0.0
            for player in self.players.values()
            for minutes in player.minutes_by_gameweek.values()
        )

    @classmethod
    def from_rows(cls, rows, minutes_scale=None):
        players: Dict[int, PlayerProjection] = {}
        for row in rows:
            player_id = int(row['player_id'])
            player = players.get(player_id)
            if player is None:
                player = PlayerProjection(
                    player_id=player_id,
                    player_code=int(row['player_code']),
                    name=str(row['player_name']),
                    position=int(row['position']),
                    team_id=int(row['team_id']),
                    price=int(round(float(row['price']))),
                    selectable=bool(row['selectable']),
                )
                players[player_id] = player
            gameweek = int(row['target_gameweek'])
            player.points_by_gameweek[gameweek] = (
                player.points_by_gameweek.get(gameweek, 0.0)
                + row_optimiser_points(row, minutes_scale=minutes_scale)
            )
            player.minutes_by_gameweek[gameweek] = (
                player.minutes_by_gameweek.get(gameweek, 0.0)
                + float(row['predicted_minutes'] or 0.0)
            )
            raw_fixture = row.get('fixture_id')
            if raw_fixture not in (None, ''):
                player.fixture_ids_by_gameweek.setdefault(gameweek, set()).add(
                    int(raw_fixture)
                )
        return cls(players)

    def __getitem__(self, player_id):
        return self.players[int(player_id)]

    def __contains__(self, player_id):
        return int(player_id) in self.players


def _mean_over_gameweeks(by_gameweek, gameweeks) -> float:
    if not gameweeks:
        return 0.0
    total = 0.0
    for gameweek in gameweeks:
        total += float(by_gameweek.get(int(gameweek), 0.0) or 0.0)
    return total / float(len(gameweeks))


def _owned_team_counts(pool, current_ids):
    counts = {}
    for player_id in current_ids:
        player = pool.players.get(int(player_id))
        if player is None:
            continue
        counts[player.team_id] = counts.get(player.team_id, 0) + 1
    return counts


def _cheap_enabler_ids(pool, current_ids, selectable_ids):
    """Two cheapest non-owned options per position, different clubs.

    Prefer clubs that are not already at three in the current squad so the
    cheap option can actually be transferred in.
    """
    team_counts = _owned_team_counts(pool, current_ids)
    selectable_set = set(int(player_id) for player_id in selectable_ids)
    owned = set(int(player_id) for player_id in current_ids)
    kept = set()
    for position in SQUAD_POSITION_COUNTS:
        candidates = [
            player_id
            for player_id in selectable_set
            if player_id not in owned
            and pool[player_id].position == position
            and pool[player_id].selectable
        ]
        candidates.sort(
            key=lambda player_id: (
                int(
                    team_counts.get(pool[player_id].team_id, 0)
                    >= MAX_PLAYERS_PER_TEAM
                ),
                pool[player_id].price,
                player_id,
            )
        )
        picked_teams = set()
        for player_id in candidates:
            team_id = pool[player_id].team_id
            if team_id in picked_teams:
                continue
            kept.add(player_id)
            picked_teams.add(team_id)
            if len(picked_teams) >= CHEAP_ENABLERS_PER_POSITION:
                break
    return kept


def optimiser_candidate_ids(pool, current_ids, gameweeks):
    """Players the transfer / squad MIP is allowed to consider.

    Drops anyone averaging under 50 predicted minutes or 2 points across the
    optimiser window, but always keeps the current squad and two cheapest
    different-club options per position (preferring clubs we can still sign).
    A pool with no predicted minutes is gated on points alone.
    """
    current_ids = {int(player_id) for player_id in (current_ids or ())}
    gameweeks = tuple(int(gameweek) for gameweek in gameweeks)
    use_minutes = pool.predicts_minutes
    selectable_ids = [
        int(player_id)
        for player_id, player in pool.players.items()
        if player.selectable or player_id in current_ids
    ]
    kept = set()
    for player_id in selectable_ids:
        if player_id in current_ids:
            kept.add(player_id)
            continue
        player = pool[player_id]
        if not player.selectable:
            continue
        avg_minutes = _mean_over_gameweeks(player.minutes_by_gameweek, gameweeks)
        avg_points = _mean_over_gameweeks(player.points_by_gameweek, gameweeks)
        if (
            (avg_minutes >= OPTIMISER_MIN_AVG_MINUTES or not use_minutes)
            and avg_points >= OPTIMISER_MIN_AVG_POINTS
        ):
            kept.add(player_id)
    kept.update(_cheap_enabler_ids(pool, current_ids, selectable_ids))
    return sorted(kept)


@dataclass
class OwnedPlayer:
    player_id: int
    purchase_price: int


@dataclass
class SquadState:
    owned: Dict[int, OwnedPlayer]
    bank: int
    free_transfers: int

    @property
    def player_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self.owned))


@dataclass
class Lineup:
    starters: Tuple[int, ...]
    bench: Tuple[int, ...]
    captain_id: int
    vice_captain_id: int
    expected_points: float


@dataclass
class PlannedTransfer:
    player_out_id: int
    player_in_id: int
    sell_price: int
    buy_price: int


# Free-transfer dumps in the 2-week roll never exceed this (no hits).
ROLL_FREE_TRANSFER_CAP = 4


@dataclass
class TransferPlan:
    transfers: Tuple[PlannedTransfer, ...]
    selected_player_ids: Tuple[int, ...]
    expected_hold_current: float
    expected_selected_current: float
    expected_hold_horizon: float
    expected_selected_horizon: float
    roll_path: Optional[str] = None

    @property
    def current_gain(self):
        return self.expected_selected_current - self.expected_hold_current

    @property
    def horizon_gain(self):
        return self.expected_selected_horizon - self.expected_hold_horizon


CHIP_WILDCARD = 'wildcard'
CHIP_FREE_HIT = 'free_hit'
CHIP_BENCH_BOOST = 'bench_boost'
CHIP_TRIPLE_CAPTAIN = 'triple_captain'
FIRST_HALF_LAST_GAMEWEEK = 19
RESET_CHIPS = {CHIP_WILDCARD, CHIP_FREE_HIT}


def _season_half(gameweek: int) -> int:
    return 1 if int(gameweek) <= FIRST_HALF_LAST_GAMEWEEK else 2


@dataclass(frozen=True)
class ChipCalendar:
    """Fixed chip weeks, already filtered to the season's inventory."""

    wildcards: Tuple[int, ...] = ()
    free_hits: Tuple[int, ...] = ()
    bench_boosts: Tuple[int, ...] = ()
    triple_captains: Tuple[int, ...] = ()

    @classmethod
    def resolve(
        cls,
        season_start_year: int,
        wildcards: Sequence[int] = (6, 24),
        free_hits: Sequence[int] = (35,),
        bench_boosts: Sequence[int] = (7, 16, 25),
        triple_captains: Sequence[int] = (13, 30),
    ) -> 'ChipCalendar':
        season = int(season_start_year)
        two_sets = season >= 25
        wildcards = _take_chip_weeks(wildcards, limit=2, one_per_half=True)
        free_hits = _take_chip_weeks(
            free_hits,
            limit=2 if two_sets else 1,
            one_per_half=two_sets,
        )
        bench_boosts = _take_chip_weeks(
            bench_boosts,
            limit=2 if two_sets else 1,
            one_per_half=two_sets,
        )
        triple_captains = _take_chip_weeks(
            triple_captains,
            limit=2 if two_sets else 1,
            one_per_half=two_sets,
        )
        occupied = set(wildcards)
        kept_free_hits = []
        for gameweek in free_hits:
            if gameweek in occupied:
                continue
            kept_free_hits.append(gameweek)
            occupied.add(gameweek)
        kept_bench_boosts = []
        for gameweek in bench_boosts:
            if gameweek in occupied:
                continue
            kept_bench_boosts.append(gameweek)
            occupied.add(gameweek)
        kept_triple_captains = []
        for gameweek in triple_captains:
            if gameweek in occupied:
                continue
            kept_triple_captains.append(gameweek)
            occupied.add(gameweek)
        return cls(
            wildcards=tuple(wildcards),
            free_hits=tuple(kept_free_hits),
            bench_boosts=tuple(kept_bench_boosts),
            triple_captains=tuple(kept_triple_captains),
        )

    def chip_at(self, gameweek: int) -> Optional[str]:
        gameweek = int(gameweek)
        if gameweek in self.wildcards:
            return CHIP_WILDCARD
        if gameweek in self.free_hits:
            return CHIP_FREE_HIT
        if gameweek in self.bench_boosts:
            return CHIP_BENCH_BOOST
        if gameweek in self.triple_captains:
            return CHIP_TRIPLE_CAPTAIN
        return None

    def is_wildcard(self, gameweek: int) -> bool:
        return int(gameweek) in self.wildcards

    def is_free_hit(self, gameweek: int) -> bool:
        return int(gameweek) in self.free_hits

    def is_bench_boost(self, gameweek: int) -> bool:
        return int(gameweek) in self.bench_boosts

    def is_triple_captain(self, gameweek: int) -> bool:
        return int(gameweek) in self.triple_captains

    def captain_extra(self, gameweek: int) -> int:
        """Extra copies of the captain on top of the XI: 1=double, 2=triple."""
        return 2 if self.is_triple_captain(gameweek) else 1

    def next_wildcard_after(self, gameweek: int) -> Optional[int]:
        for wildcard in self.wildcards:
            if int(wildcard) > int(gameweek):
                return int(wildcard)
        return None

    def as_dict(self):
        return {
            'wildcards': list(self.wildcards),
            'free_hits': list(self.free_hits),
            'bench_boosts': list(self.bench_boosts),
            'triple_captains': list(self.triple_captains),
        }

    def planning_gameweeks(self, as_of_gameweek: int, horizon_gameweeks: int):
        """Weeks the transfer MIP scores, starting at ``as_of_gameweek``.

        A planned free-hit week is skipped from other weeks' horizons because
        that 15 is different. The free-hit rebuild itself still includes this
        week and looks the same number of gameweeks ahead as a wildcard.
        Planning stops before the next wildcard.
        """
        as_of = int(as_of_gameweek)
        horizon = int(horizon_gameweeks)
        limit = self.next_wildcard_after(as_of) or 39
        weeks = []
        gameweek = as_of
        while gameweek < limit and gameweek <= 38 and len(weeks) < horizon:
            if gameweek in self.free_hits and gameweek != as_of:
                gameweek += 1
                continue
            weeks.append(gameweek)
            gameweek += 1
        return tuple(weeks)


def _take_chip_weeks(requested, limit, one_per_half):
    chosen = []
    used_halves = set()
    for gameweek in requested:
        gameweek = int(gameweek)
        if not 1 <= gameweek <= 38 or gameweek in chosen:
            continue
        if one_per_half:
            half = _season_half(gameweek)
            if half in used_halves:
                continue
            used_halves.add(half)
        chosen.append(gameweek)
        if len(chosen) >= int(limit):
            break
    return tuple(chosen)


@dataclass
class StrategyConfig:
    horizon_gameweeks: int = 5
    min_current_gw_gain: float = 0.0
    min_horizon_gain: float = 0.0
    # min_horizon_gain is a total over the weeks being planned, so it is quoted
    # over this many weeks and rescaled to the window in use. Without it, a
    # longer horizon quietly lowers the gain demanded per week.
    min_gain_reference_horizon: int = MIN_GAIN_REFERENCE_HORIZON
    max_transfers_per_gameweek: int = 5
    max_free_transfers: int = 5
    relax_thresholds_at_max_ft: bool = False
    chips: ChipCalendar = field(default_factory=ChipCalendar)
    # Pass 1: uniform leftover weight on (selected - start). None with
    # bench_weights set uses SIM_BENCH_FILL_WEIGHT. 0 and empty slots = XI only.
    bench_fill_weight: Optional[float] = None
    bench_weights: Tuple[float, ...] = ()
    reserve_keeper_weight: Optional[float] = None
    season_start_year: int = 25
    roll_free_transfers: bool = False


def selling_price(purchase_price: int, current_price: int) -> int:
    purchase_price = int(purchase_price)
    current_price = int(current_price)
    price_change = current_price - purchase_price
    if price_change <= 0:
        return current_price
    return purchase_price + math.floor(price_change / 2)


class FreeTransferRules:
    """Season-aware free-transfer state, excluding hits and chips."""

    def __init__(self, season_start_year: int):
        self.season_start_year = int(season_start_year)
        self.max_bank = 2 if self.season_start_year <= 23 else 5

    def at_start_of_gameweek(self, gameweek: int, available: int) -> int:
        if self.season_start_year == 25 and int(gameweek) == 16:
            return 5
        return min(self.max_bank, max(0, int(available)))

    def after_gameweek(
        self,
        gameweek: int,
        available: int,
        used: int,
        chip: Optional[str] = None,
    ) -> int:
        if chip in RESET_CHIPS:
            used = 0
        if int(gameweek) == 1:
            next_available = 1
        else:
            next_available = min(
                self.max_bank,
                max(0, int(available) - int(used)) + 1,
            )
        return self.at_start_of_gameweek(int(gameweek) + 1, next_available)


def _team_counts(player_ids: Iterable[int], pool: PredictionPool):
    counts = {}
    for player_id in player_ids:
        team_id = pool[player_id].team_id
        counts[team_id] = counts.get(team_id, 0) + 1
    return counts


def _club_excess(player_ids: Iterable[int], pool: PredictionPool) -> int:
    """How many sells are required to restore the 3-per-club rule."""
    return sum(
        max(0, count - MAX_PLAYERS_PER_TEAM)
        for count in _team_counts(player_ids, pool).values()
    )


def _validate_squad(
    player_ids: Iterable[int],
    pool: PredictionPool,
    enforce_club_limit=True,
):
    ids = tuple(player_ids)
    if len(ids) != 15 or len(set(ids)) != 15:
        raise ValueError('A squad must contain 15 unique players')
    position_counts = {position: 0 for position in SQUAD_POSITION_COUNTS}
    for player_id in ids:
        position_counts[pool[player_id].position] = (
            position_counts.get(pool[player_id].position, 0) + 1
        )
    if position_counts != SQUAD_POSITION_COUNTS:
        raise ValueError(f'Invalid squad position counts: {position_counts}')
    if enforce_club_limit:
        team_counts = _team_counts(ids, pool)
        if any(count > MAX_PLAYERS_PER_TEAM for count in team_counts.values()):
            raise ValueError(f'Invalid squad club counts: {team_counts}')


def _remaining_position_counts(starter_ids, pool):
    remaining = {
        position: int(count) for position, count in SQUAD_POSITION_COUNTS.items()
    }
    for player_id in starter_ids:
        remaining[pool[player_id].position] -= 1
    if any(count < 0 for count in remaining.values()):
        raise ValueError('Frozen XI exceeds squad position counts')
    if sum(remaining.values()) != 4:
        raise ValueError('Frozen XI does not leave four bench places')
    return remaining


def fill_bench(
    starter_ids: Sequence[int],
    pool: PredictionPool,
    gameweeks: Sequence[int],
    *,
    budget: int,
    current_state: Optional[SquadState] = None,
    max_transfers: int = 15,
    bench_weights: Sequence[float] = (),
    reserve_keeper_weight: Optional[float] = None,
    chips: Optional[ChipCalendar] = None,
):
    """Pick four bench players for a frozen XI. Returns a 15-id squad or None."""
    slots = slot_bench_weights(bench_weights, reserve_keeper_weight)
    if not slots:
        return None
    starters = tuple(int(player_id) for player_id in starter_ids)
    if len(starters) != 11 or len(set(starters)) != 11:
        return None
    try:
        remaining_positions = _remaining_position_counts(starters, pool)
    except ValueError:
        return None
    starter_set = set(starters)
    current_ids = set(current_state.player_ids if current_state else ())
    chips = chips or ChipCalendar()
    gameweeks = tuple(int(gameweek) for gameweek in gameweeks)
    if not gameweeks:
        return None
    candidates = [
        player_id
        for player_id in optimiser_candidate_ids(pool, current_ids, gameweeks)
        if player_id not in starter_set
    ]
    if len(candidates) < 4:
        return None
    remaining_club = {
        team_id: MAX_PLAYERS_PER_TEAM - count
        for team_id, count in _team_counts(starters, pool).items()
    }
    model = LpProblem(name='simulation_bench_fill', sense=LpMaximize)
    picked = {
        player_id: LpVariable(f'bench_pick_{player_id}', cat='Binary')
        for player_id in candidates
    }
    model += lpSum(picked.values()) == 4
    for position, required in remaining_positions.items():
        model += lpSum(
            picked[player_id] * int(pool[player_id].position == position)
            for player_id in candidates
        ) == required
    team_ids = sorted({pool[player_id].team_id for player_id in candidates})
    for team_id in team_ids:
        model += lpSum(
            picked[player_id] * int(pool[player_id].team_id == team_id)
            for player_id in candidates
        ) <= remaining_club.get(team_id, MAX_PLAYERS_PER_TEAM)
    if current_state is None:
        starter_cost = sum(pool[player_id].price for player_id in starters)
        model += (
            lpSum(picked[player_id] * pool[player_id].price for player_id in candidates)
            + starter_cost
            <= int(budget)
        )
    else:
        outgoing = lpSum(
            0 if player_id in starter_set else (1 - picked[player_id])
            for player_id in current_ids
        )
        model += outgoing <= int(max_transfers)
        incoming_cost = lpSum(
            pool[player_id].price
            for player_id in starters
            if player_id not in current_ids
        ) + lpSum(
            picked[player_id] * pool[player_id].price
            for player_id in candidates
            if player_id not in current_ids
        )
        sale_proceeds = lpSum(
            (1 - picked[player_id])
            * selling_price(
                current_state.owned[player_id].purchase_price,
                pool[player_id].price,
            )
            for player_id in current_ids
            if player_id not in starter_set
        )
        model += incoming_cost <= int(current_state.bank) + sale_proceeds
    outfield = [
        player_id for player_id in candidates if pool[player_id].position != 1
    ]
    keepers = [
        player_id for player_id in candidates if pool[player_id].position == 1
    ]
    score = []
    for gameweek in gameweeks:
        if chips.is_bench_boost(gameweek):
            score.append(lpSum(
                picked[player_id] * pool[player_id].points(gameweek)
                for player_id in candidates
            ))
            continue
        assign = {
            slot: {
                player_id: LpVariable(
                    f'bench_slot_{gameweek}_{slot}_{player_id}', cat='Binary'
                )
                for player_id in outfield
            }
            for slot in range(3)
        }
        for player_id in outfield:
            model += lpSum(
                assign[slot][player_id] for slot in range(3)
            ) == picked[player_id]
        for slot in range(3):
            model += lpSum(assign[slot].values()) == 1
        score.append(lpSum(
            slots[slot] * assign[slot][player_id] * pool[player_id].points(gameweek)
            for slot in range(3)
            for player_id in outfield
        ))
        if keepers:
            score.append(lpSum(
                slots[3] * picked[player_id] * pool[player_id].points(gameweek)
                for player_id in keepers
            ))
    model += lpSum(score) + lpSum(
        picked[player_id] * (1e-9 * (10000 - player_id))
        for player_id in candidates
    )
    try:
        _solve_model(model)
    except ValueError:
        return None
    if LpStatus[model.status] != 'Optimal':
        return None
    bench = tuple(sorted(
        player_id for player_id in candidates if picked[player_id].value() == 1
    ))
    if len(bench) != 4:
        return None
    result = tuple(sorted(starters + bench))
    try:
        _validate_squad(result, pool)
    except ValueError:
        return None
    return result


def pick_lineup(
    player_ids: Sequence[int],
    pool: PredictionPool,
    gameweek: int,
) -> Lineup:
    """Pick a legal XI, captain, vice-captain and deterministic bench order."""
    ids = tuple(int(player_id) for player_id in player_ids)
    if len(ids) != 15:
        raise ValueError('Lineup selection requires a 15-player squad')
    model = LpProblem(name=f'lineup_gw_{gameweek}', sense=LpMaximize)
    starts = {
        player_id: LpVariable(f'start_{gameweek}_{player_id}', cat='Binary')
        for player_id in ids
    }
    model += lpSum(starts.values()) == 11
    for position in SQUAD_POSITION_COUNTS:
        count = lpSum(
            starts[player_id] * int(pool[player_id].position == position)
            for player_id in ids
        )
        model += count >= LINEUP_POSITION_MIN[position]
        model += count <= LINEUP_POSITION_MAX[position]
    model += lpSum(
        starts[player_id]
        * (
            pool[player_id].points(gameweek)
            + (1e-9 * (10000 - player_id))
        )
        for player_id in ids
    )
    _solve_model(model)
    if LpStatus[model.status] != 'Optimal':
        raise ValueError(f'Could not select legal GW{gameweek} lineup')
    starter_ids = tuple(sorted(
        player_id for player_id in ids if starts[player_id].value() == 1
    ))
    ranked_starters = sorted(
        starter_ids,
        key=lambda player_id: (
            pool[player_id].points(gameweek),
            -player_id,
        ),
        reverse=True,
    )
    captain_id = ranked_starters[0]
    vice_captain_id = ranked_starters[1]
    bench_ids = [player_id for player_id in ids if player_id not in starter_ids]
    outfield_bench = sorted(
        (
            player_id
            for player_id in bench_ids
            if pool[player_id].position != 1
        ),
        key=lambda player_id: (
            pool[player_id].points(gameweek),
            -player_id,
        ),
        reverse=True,
    )
    keeper_bench = [
        player_id for player_id in bench_ids if pool[player_id].position == 1
    ]
    bench = tuple(outfield_bench + keeper_bench)
    expected = (
        sum(pool[player_id].points(gameweek) for player_id in starter_ids)
        + pool[captain_id].points(gameweek)
    )
    return Lineup(
        starters=starter_ids,
        bench=bench,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        expected_points=float(expected),
    )


def predicted_points_line(lineup: Lineup, pool: PredictionPool, gameweek: int) -> str:
    """One wrapping line: all 15 predicted scores, highest first, bench included."""
    bench = set(lineup.bench)
    rows = []
    for player_id in list(lineup.starters) + list(lineup.bench):
        player = pool[player_id]
        marker = ''
        if player_id == lineup.captain_id:
            marker = 'C'
        elif player_id in bench:
            marker = 'b'
        rows.append((player.points(gameweek), player.name, marker))
    rows.sort(key=lambda row: (-row[0], row[1]))
    parts = []
    for points, name, marker in rows:
        label = f'{name} {points:.1f}'
        if marker:
            label += marker
        parts.append(label)
    return 'pred: ' + ', '.join(parts)


def _bench_expected_points(
    lineup, pool, gameweek, bench_weights, bench_boost, reserve_keeper_weight=None,
):
    if bench_boost:
        return sum(pool[player_id].points(gameweek) for player_id in lineup.bench)
    total = 0.0
    slots = slot_bench_weights(bench_weights, reserve_keeper_weight)
    for index, weight in enumerate(slots):
        if index >= len(lineup.bench):
            break
        total += float(weight) * pool[lineup.bench[index]].points(gameweek)
    return total


def expected_gameweek_points(
    player_ids: Sequence[int],
    pool: PredictionPool,
    gameweek: int,
    bench_boost: bool = False,
    triple_captain: bool = False,
    bench_weights: Sequence[float] = (),
    reserve_keeper_weight: Optional[float] = None,
) -> float:
    lineup = pick_lineup(player_ids, pool, gameweek)
    points = float(lineup.expected_points)
    points += _bench_expected_points(
        lineup,
        pool,
        gameweek,
        bench_weights,
        bench_boost,
        reserve_keeper_weight,
    )
    if triple_captain:
        points += pool[lineup.captain_id].points(gameweek)
    return points


def flat_expected_points(
    lineup: Lineup,
    pool: PredictionPool,
    gameweek: int,
    bench_boost: bool = False,
    triple_captain: bool = False,
) -> float:
    """XI + captain, plus bench on BB and a second extra on TC. No bench weights."""
    points = float(lineup.expected_points)
    if bench_boost:
        points += sum(pool[player_id].points(gameweek) for player_id in lineup.bench)
    if triple_captain:
        points += pool[lineup.captain_id].points(gameweek)
    return points


def copy_squad_state(state: SquadState) -> SquadState:
    return SquadState(
        owned={
            player_id: OwnedPlayer(owned.player_id, owned.purchase_price)
            for player_id, owned in state.owned.items()
        },
        bank=int(state.bank),
        free_transfers=int(state.free_transfers),
    )


class SimulationStrategy:
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()

    def _gameweeks(self, as_of_gameweek):
        return self.config.chips.planning_gameweeks(
            as_of_gameweek,
            self.config.horizon_gameweeks,
        )

    def _expected_scores(self, player_ids, pool, gameweeks):
        return {
            gameweek: expected_gameweek_points(
                player_ids,
                pool,
                gameweek,
                bench_boost=self.config.chips.is_bench_boost(gameweek),
                triple_captain=self.config.chips.is_triple_captain(gameweek),
                bench_weights=self.config.bench_weights,
                reserve_keeper_weight=self.config.reserve_keeper_weight,
            )
            for gameweek in gameweeks
        }

    def build_initial_squad(
        self,
        pool: PredictionPool,
        as_of_gameweek: int = 1,
        budget: int = INITIAL_BUDGET,
    ) -> Tuple[int, ...]:
        return self._optimise_squad(
            pool=pool,
            as_of_gameweek=as_of_gameweek,
            current_state=None,
            max_transfers=15,
            budget=int(budget),
            hold_current=0.0,
            hold_horizon=0.0,
            require_transfer=False,
            enforce_thresholds=False,
        )

    def plan(
        self,
        state: SquadState,
        pool: PredictionPool,
        as_of_gameweek: int,
    ) -> TransferPlan:
        spend = self._plan_spend(state, pool, as_of_gameweek)
        if not self.config.roll_free_transfers:
            spend.roll_path = 'spend'
            return spend
        chip = self.config.chips.chip_at(as_of_gameweek)
        if chip in RESET_CHIPS:
            spend.roll_path = 'spend'
            return spend
        return self._plan_with_roll(state, pool, as_of_gameweek, spend)

    def _window_score(self, player_ids, pool, gameweeks):
        scores = self._expected_scores(player_ids, pool, gameweeks)
        return sum(scores.values()), scores

    def _ft_after_holds(self, as_of, free_transfers, hold_n):
        rules = FreeTransferRules(self.config.season_start_year)
        available = int(free_transfers)
        gameweek = int(as_of)
        for _ in range(int(hold_n)):
            available = rules.after_gameweek(
                gameweek,
                available if gameweek > 1 else 0,
                0,
            )
            gameweek += 1
        return int(available)

    def _reset_blocks_wait(self, as_of, hold_n):
        for offset in range(1, int(hold_n) + 1):
            chip = self.config.chips.chip_at(int(as_of) + offset)
            if chip in RESET_CHIPS:
                return True
        return False

    def _plan_with_roll(self, state, pool, as_of_gameweek, spend):
        as_of = int(as_of_gameweek)
        window = self._gameweeks(as_of)
        hold_ids = state.player_ids
        hold_total, hold_scores = self._window_score(hold_ids, pool, window)
        spend_total, _ = self._window_score(
            spend.selected_player_ids, pool, window
        )
        candidates = [('spend', spend_total, spend)]
        for hold_n in (1, 2):
            if self._reset_blocks_wait(as_of, hold_n):
                continue
            spend_week = as_of + hold_n
            remain = tuple(week for week in window if int(week) >= spend_week)
            if not remain:
                continue
            held = tuple(week for week in window if int(week) < spend_week)
            ft_at_spend = min(
                ROLL_FREE_TRANSFER_CAP,
                self._ft_after_holds(as_of, state.free_transfers, hold_n),
            )
            if ft_at_spend <= 0:
                continue
            future_state = copy_squad_state(state)
            future_state.free_transfers = ft_at_spend
            later = self._plan_spend(
                future_state,
                pool,
                spend_week,
                gameweeks=remain,
                min_current_gw_gain=0.0,
                min_horizon_gain=0.0,
                require_transfer=False,
                enforce_thresholds=False,
            )
            later_total, _ = self._window_score(
                later.selected_player_ids, pool, remain
            )
            held_total = sum(hold_scores.get(week, 0.0) for week in held)
            path_total = held_total + later_total
            hold_plan = self._hold_plan(
                state,
                hold_scores.get(as_of, 0.0),
                hold_total,
            )
            hold_plan.expected_selected_horizon = float(path_total)
            hold_plan.roll_path = 'wait_%s' % hold_n
            candidates.append(('wait_%s' % hold_n, path_total, hold_plan))
        best_name, best_total, best_plan = max(
            candidates,
            key=lambda row: (row[1], 0 if row[0] == 'spend' else -int(row[0][-1])),
        )
        best_plan.roll_path = best_name
        if best_name == 'spend':
            best_plan.expected_selected_horizon = float(best_total)
        return best_plan

    def _plan_spend(
        self,
        state: SquadState,
        pool: PredictionPool,
        as_of_gameweek: int,
        gameweeks=None,
        min_current_gw_gain=None,
        min_horizon_gain=None,
        require_transfer=None,
        enforce_thresholds=None,
    ) -> TransferPlan:
        # A held squad can briefly exceed 3-per-club after a real-world transfer.
        # That is legal until this deadline, but the selected squad after planning
        # must be restored to 15 / positions / 3-per-club.
        _validate_squad(state.player_ids, pool, enforce_club_limit=False)
        club_excess = _club_excess(state.player_ids, pool)
        chip = self.config.chips.chip_at(as_of_gameweek)
        reset_squad = chip in RESET_CHIPS
        at_max_ft = (
            club_excess == 0
            and not reset_squad
            and bool(self.config.relax_thresholds_at_max_ft)
            and int(state.free_transfers) >= int(self.config.max_free_transfers)
        )
        min_current = (
            0.0
            if at_max_ft or reset_squad
            else float(self.config.min_current_gw_gain)
        )
        min_horizon = (
            0.0
            if at_max_ft or reset_squad
            else float(self.config.min_horizon_gain)
        )
        if min_current_gw_gain is not None:
            min_current = float(min_current_gw_gain)
        if min_horizon_gain is not None:
            min_horizon = float(min_horizon_gain)
        if gameweeks is None:
            gameweeks = self._gameweeks(as_of_gameweek)
        else:
            gameweeks = tuple(int(week) for week in gameweeks)
        # Scale here, not at the constraint: the gain is checked again after the
        # solve, and the two checks have to agree on what the bar is.
        reference = int(self.config.min_gain_reference_horizon)
        if reference > 0:
            min_horizon *= float(len(gameweeks)) / float(reference)
        hold_scores = self._expected_scores(state.player_ids, pool, gameweeks)
        hold_current = hold_scores.get(int(as_of_gameweek), 0.0)
        hold_horizon = sum(hold_scores.values())
        if reset_squad:
            max_transfers = 15
        else:
            max_transfers = min(
                max(0, int(state.free_transfers)),
                max(0, int(self.config.max_transfers_per_gameweek)),
                ROLL_FREE_TRANSFER_CAP,
            )
        if club_excess > 0:
            max_transfers = max(
                max_transfers,
                min(club_excess, int(self.config.max_transfers_per_gameweek)),
            )
        if max_transfers <= 0:
            return self._hold_plan(state, hold_current, hold_horizon)
        must_transfer = not reset_squad if require_transfer is None else bool(require_transfer)
        must_enforce = (
            club_excess == 0 and not reset_squad
            if enforce_thresholds is None
            else bool(enforce_thresholds)
        )
        try:
            selected = self._optimise_squad(
                pool=pool,
                as_of_gameweek=as_of_gameweek,
                current_state=state,
                max_transfers=max_transfers,
                budget=state.bank,
                hold_current=hold_current,
                hold_horizon=hold_horizon,
                require_transfer=must_transfer,
                enforce_thresholds=must_enforce,
                min_current_gw_gain=min_current,
                min_horizon_gain=min_horizon,
                gameweeks=gameweeks,
            )
        except ValueError:
            if club_excess > 0:
                raise
            return self._hold_plan(state, hold_current, hold_horizon)

        selected_scores = self._expected_scores(selected, pool, gameweeks)
        selected_current = selected_scores.get(int(as_of_gameweek), 0.0)
        selected_horizon = sum(selected_scores.values())
        current_gain = selected_current - hold_current
        horizon_gain = selected_horizon - hold_horizon
        if club_excess == 0 and (
            current_gain + 1e-8 < min_current
            or horizon_gain + 1e-8 < min_horizon
            or (
                min_current <= 0
                and min_horizon <= 0
                and horizon_gain <= 1e-8
            )
        ):
            return self._hold_plan(state, hold_current, hold_horizon)
        transfers = self._pair_transfers(state, selected, pool)
        return TransferPlan(
            transfers=tuple(transfers),
            selected_player_ids=tuple(sorted(selected)),
            expected_hold_current=float(hold_current),
            expected_selected_current=float(selected_current),
            expected_hold_horizon=float(hold_horizon),
            expected_selected_horizon=float(selected_horizon),
        )

    @staticmethod
    def _hold_plan(state, hold_current, hold_horizon):
        return TransferPlan(
            transfers=(),
            selected_player_ids=state.player_ids,
            expected_hold_current=float(hold_current),
            expected_selected_current=float(hold_current),
            expected_hold_horizon=float(hold_horizon),
            expected_selected_horizon=float(hold_horizon),
        )

    def _optimise_squad(
        self,
        pool,
        as_of_gameweek,
        current_state,
        max_transfers,
        budget,
        hold_current,
        hold_horizon,
        require_transfer,
        enforce_thresholds,
        min_current_gw_gain=None,
        min_horizon_gain=None,
        gameweeks=None,
        hit_cost=0.0,
        free_transfers=0,
    ):
        if gameweeks is None:
            gameweeks = self._gameweeks(as_of_gameweek)
        else:
            gameweeks = tuple(int(week) for week in gameweeks)
        current_ids = set(current_state.player_ids if current_state else ())
        valid_ids = optimiser_candidate_ids(pool, current_ids, gameweeks)
        if len(valid_ids) < 15:
            raise ValueError('Not enough selectable players to build a squad')
        model = LpProblem(name=f'simulation_squad_gw_{as_of_gameweek}', sense=LpMaximize)
        selected = {
            player_id: LpVariable(f'selected_{player_id}', cat='Binary')
            for player_id in valid_ids
        }
        starts = {
            gameweek: {
                player_id: LpVariable(
                    f'start_{gameweek}_{player_id}', cat='Binary'
                )
                for player_id in valid_ids
            }
            for gameweek in gameweeks
        }
        captains = {
            gameweek: {
                player_id: LpVariable(
                    f'captain_{gameweek}_{player_id}', cat='Binary'
                )
                for player_id in valid_ids
            }
            for gameweek in gameweeks
        }

        model += lpSum(selected.values()) == 15
        for position, required in SQUAD_POSITION_COUNTS.items():
            model += lpSum(
                selected[player_id]
                * int(pool[player_id].position == position)
                for player_id in valid_ids
            ) == required
        team_ids = sorted({pool[player_id].team_id for player_id in valid_ids})
        for team_id in team_ids:
            model += lpSum(
                selected[player_id]
                * int(pool[player_id].team_id == team_id)
                for player_id in valid_ids
            ) <= MAX_PLAYERS_PER_TEAM

        bench_fill = self._pass1_fill_weight()
        gw_score_expressions = {}
        for gameweek in gameweeks:
            model += lpSum(starts[gameweek].values()) == 11
            model += lpSum(captains[gameweek].values()) == 1
            for player_id in valid_ids:
                model += starts[gameweek][player_id] <= selected[player_id]
                model += captains[gameweek][player_id] <= starts[gameweek][player_id]
            for position in SQUAD_POSITION_COUNTS:
                lineup_count = lpSum(
                    starts[gameweek][player_id]
                    * int(pool[player_id].position == position)
                    for player_id in valid_ids
                )
                model += lineup_count >= LINEUP_POSITION_MIN[position]
                model += lineup_count <= LINEUP_POSITION_MAX[position]
            captain_extra = self.config.chips.captain_extra(gameweek)
            if self.config.chips.is_bench_boost(gameweek):
                gw_score_expressions[gameweek] = lpSum(
                    (
                        selected[player_id]
                        + captain_extra * captains[gameweek][player_id]
                    )
                    * pool[player_id].points(gameweek)
                    for player_id in valid_ids
                )
                continue
            starter_score = lpSum(
                (
                    starts[gameweek][player_id]
                    + captain_extra * captains[gameweek][player_id]
                )
                * pool[player_id].points(gameweek)
                for player_id in valid_ids
            )
            if bench_fill <= 0:
                gw_score_expressions[gameweek] = starter_score
                continue
            leftover = lpSum(
                (selected[player_id] - starts[gameweek][player_id])
                * pool[player_id].points(gameweek)
                for player_id in valid_ids
            )
            gw_score_expressions[gameweek] = starter_score + bench_fill * leftover

        if current_state is None:
            model += lpSum(
                selected[player_id] * pool[player_id].price
                for player_id in valid_ids
            ) <= int(budget)
        else:
            outgoing = lpSum(
                1 - selected[player_id] for player_id in current_ids
            )
            model += outgoing <= int(max_transfers)
            if require_transfer:
                model += outgoing >= 1
            incoming_cost = lpSum(
                selected[player_id] * pool[player_id].price
                for player_id in valid_ids
                if player_id not in current_ids
            )
            sale_proceeds = lpSum(
                (1 - selected[player_id])
                * selling_price(
                    current_state.owned[player_id].purchase_price,
                    pool[player_id].price,
                )
                for player_id in current_ids
            )
            model += incoming_cost <= int(current_state.bank) + sale_proceeds

        horizon_score = lpSum(gw_score_expressions.values())
        # Paid transfers have to cost points inside the objective. A cap alone
        # makes the solver spend every allowed move, because extra players
        # never reduce the score.
        paid_transfers = None
        if (
            current_state is not None
            and float(hit_cost) > 0
            and int(max_transfers) > int(free_transfers)
        ):
            paid_transfers = LpVariable(
                'paid_transfers',
                lowBound=0,
                upBound=int(max_transfers),
                cat='Integer',
            )
            model += paid_transfers >= outgoing - int(free_transfers)
        if enforce_thresholds:
            min_current = (
                float(self.config.min_current_gw_gain)
                if min_current_gw_gain is None
                else float(min_current_gw_gain)
            )
            min_horizon = (
                float(self.config.min_horizon_gain)
                if min_horizon_gain is None
                else float(min_horizon_gain)
            )
            model += (
                gw_score_expressions[int(as_of_gameweek)]
                >= float(hold_current) + min_current
            )
            model += (
                horizon_score
                >= float(hold_horizon) + min_horizon
            )
        objective = horizon_score + lpSum(
            selected[player_id] * (1e-9 * (10000 - player_id))
            for player_id in valid_ids
        )
        if paid_transfers is not None:
            objective = objective - (float(hit_cost) * paid_transfers)
        model += objective
        _solve_model(model)
        if LpStatus[model.status] != 'Optimal':
            raise ValueError(
                f'No feasible squad for GW{as_of_gameweek}: {LpStatus[model.status]}'
            )
        result = tuple(sorted(
            player_id
            for player_id in valid_ids
            if selected[player_id].value() == 1
        ))
        _validate_squad(result, pool)
        lineup_week = int(as_of_gameweek)
        if lineup_week not in starts:
            lineup_week = int(gameweeks[0])
        starter_ids = tuple(sorted(
            player_id
            for player_id in valid_ids
            if starts[lineup_week][player_id].value() == 1
        ))
        refilled = fill_bench(
            starter_ids,
            pool,
            gameweeks,
            budget=int(budget),
            current_state=current_state,
            max_transfers=max_transfers,
            bench_weights=self.config.bench_weights,
            reserve_keeper_weight=self.config.reserve_keeper_weight,
            chips=self.config.chips,
        )
        if refilled is not None:
            result = refilled
            _validate_squad(result, pool)
        return result

    def _pass1_fill_weight(self) -> float:
        if self.config.bench_fill_weight is not None:
            return float(self.config.bench_fill_weight)
        if self.config.bench_weights:
            return float(SIM_BENCH_FILL_WEIGHT)
        return 0.0

    @staticmethod
    def _pair_transfers(state, selected, pool):
        current_ids = set(state.player_ids)
        selected_ids = set(selected)
        outgoing = sorted(current_ids - selected_ids)
        incoming = sorted(selected_ids - current_ids)
        by_position_out = {}
        by_position_in = {}
        for player_id in outgoing:
            by_position_out.setdefault(pool[player_id].position, []).append(player_id)
        for player_id in incoming:
            by_position_in.setdefault(pool[player_id].position, []).append(player_id)
        transfers = []
        for position in SQUAD_POSITION_COUNTS:
            position_out = by_position_out.get(position, [])
            position_in = by_position_in.get(position, [])
            if len(position_out) != len(position_in):
                raise ValueError(
                    f'Transfer position mismatch at position {position}: '
                    f'{position_out} -> {position_in}'
                )
            for player_out_id, player_in_id in zip(position_out, position_in):
                transfers.append(PlannedTransfer(
                    player_out_id=player_out_id,
                    player_in_id=player_in_id,
                    sell_price=selling_price(
                        state.owned[player_out_id].purchase_price,
                        pool[player_out_id].price,
                    ),
                    buy_price=pool[player_in_id].price,
                ))
        return transfers


def apply_transfer_plan(
    state: SquadState,
    plan: TransferPlan,
) -> SquadState:
    owned = dict(state.owned)
    bank = int(state.bank)
    for transfer in plan.transfers:
        if transfer.player_out_id not in owned:
            raise ValueError(f'Cannot sell unowned player {transfer.player_out_id}')
        del owned[transfer.player_out_id]
        bank += int(transfer.sell_price)
    for transfer in plan.transfers:
        if transfer.player_in_id in owned:
            raise ValueError(f'Cannot buy owned player {transfer.player_in_id}')
        if bank < int(transfer.buy_price):
            raise ValueError('Transfer plan exceeds available bank')
        bank -= int(transfer.buy_price)
        owned[transfer.player_in_id] = OwnedPlayer(
            player_id=transfer.player_in_id,
            purchase_price=int(transfer.buy_price),
        )
    return SquadState(
        owned=owned,
        bank=bank,
        free_transfers=max(0, int(state.free_transfers) - len(plan.transfers)),
    )
