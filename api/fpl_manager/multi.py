"""One training run played across several seasons.

v1 and v2: the policy does not receive a season's points when that season
ends. Those points and finish percentiles are summed, and that sum is the
reward on the last gameweek of the run.

v3 (`per_season=True`): each season is its own learning episode. The season's
own reward (weekly points, then points and percentile at its end) is paid as
it happens, and the episode ends with the season. The run still logs one row
for all its seasons. That keeps a decision's credit within one season of it,
where the old end-of-run payment reached 2023/24 decisions at under 1%.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from fpl_manager.percentiles import season_beaten_percent, season_label
from fpl_manager.reward import multi_season_reward, season_end_reward

EVALUATION_LOG = 'eval.csv'


_RUN_IDS = 0


def _next_run_id(directory=None) -> int:
    if directory is None:
        global _RUN_IDS
        _RUN_IDS += 1
        return _RUN_IDS
    return _allocate_run_id(directory)


def note_completed_runs(directory, completed: int):
    """Do not hand out a run number the finished runs already used."""
    completed = int(completed)
    if completed <= 0:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'next_run_id.txt'
    handle = _lock(directory)
    try:
        current = _read_counter(path)
        if completed > current:
            path.write_text(str(completed))
    finally:
        _unlock(handle)


def _allocate_run_id(directory) -> int:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'next_run_id.txt'
    handle = _lock(directory)
    try:
        current = _read_counter(path) + 1
        path.write_text(str(current))
        return current
    finally:
        _unlock(handle)


def _read_counter(path: Path) -> int:
    if not path.is_file():
        return 0
    text = path.read_text().strip()
    return int(text) if text else 0


def _lock(directory: Path):
    path = Path(directory) / '.run.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unbuffered, so a write refused below leaves nothing waiting to be flushed.
    handle = open(path, 'a+b', buffering=0)
    if handle.seek(0, os.SEEK_END) == 0:
        try:
            handle.write(b'\0')
        except PermissionError:
            # The other worker wrote the byte first and already holds the lock.
            pass
    handle.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _unlock(handle):
    handle.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


class MultiSeasonRun:
    def __init__(self, seasons, log_path=None, per_season=False):
        self.seasons = list(seasons)
        if not self.seasons:
            raise ValueError('a multi-season run needs at least one season')
        self.log_path = None if log_path is None else Path(log_path)
        self.run_id = _next_run_id(None if self.log_path is None else self.log_path.parent)
        self.per_season = bool(per_season)
        # An evaluation run plays the policy's top choice and logs to eval.csv.
        self.evaluation = False
        self.index = 0
        self.episode = None
        self.points = []
        self.percentiles = []
        self.solved = []
        self.cached = []
        self.seconds = []
        self.done = False
        self._waiting = False
        self._begin()

    @property
    def at_start(self) -> bool:
        """No decision has been made in this run yet."""
        return (
            self.index == 0
            and not self.points
            and self.episode is not None
            and int(self.episode.gameweek) == int(self.seasons[0].first)
        )

    def start_next_season(self):
        """Per-season runs: begin the next season after the last one's episode ended."""
        if not self._waiting:
            raise RuntimeError('the current season has not finished')
        self._waiting = False
        self.index += 1
        self._begin()

    def observation(self):
        return self.episode.observation()

    def mask(self):
        return self.episode.mask()

    def step(self, action_index: int):
        if self.done:
            raise RuntimeError('run already finished')
        if self._waiting:
            raise RuntimeError('start the next season before stepping')
        observation, reward, finished, info = self.episode.step(action_index)
        if not finished:
            return observation, reward, False, info
        if self.per_season:
            return self._finish_season(reward, info)
        reward -= float(info.get('terminal_reward') or 0.0)
        points = float(self.episode.points_total)
        beaten = season_beaten_percent(self.episode.season_start_year, points)
        percentile = 0.0 if beaten is None else float(beaten)
        self.points.append(points)
        self.percentiles.append(percentile)
        self.solved.append(int(getattr(self.episode, 'solved_count', 0)))
        self.cached.append(int(getattr(self.episode, 'cached_count', 0)))
        self.seconds.append(float(getattr(self.episode, 'season_seconds', 0.0)))
        info = dict(info)
        info['run_id'] = self.run_id
        info['season_points'] = points
        info['season_percentile'] = percentile
        if self.index + 1 < len(self.seasons):
            self.index += 1
            self._begin()
            return self.episode.observation(), float(reward), False, info
        self.done = True
        combined = multi_season_reward(self.points, self.percentiles)
        reward += combined
        info['terminal_reward'] = combined
        _print_finished(self.run_id, self.points, self.percentiles, combined)
        _append_run_log(self)
        return tuple(), float(reward), True, info

    def _finish_season(self, reward, info):
        """Per-season: the episode's own reward stands and the episode ends here."""
        self._record_season()
        info = dict(info)
        info['run_id'] = self.run_id
        info['season_points'] = self.points[-1]
        info['season_percentile'] = self.percentiles[-1]
        if self.index + 1 < len(self.seasons):
            self._waiting = True
            info['run_finished'] = False
            return tuple(), float(reward), True, info
        self.done = True
        info['run_finished'] = True
        _print_finished(self.run_id, self.points, self.percentiles, self.run_reward(),
                        label='Evaluation run' if self.evaluation else 'Run')
        _append_run_log(self)
        return tuple(), float(reward), True, info

    def _record_season(self):
        points = float(self.episode.points_total)
        beaten = season_beaten_percent(self.episode.season_start_year, points)
        self.points.append(points)
        self.percentiles.append(0.0 if beaten is None else float(beaten))
        self.solved.append(int(getattr(self.episode, 'solved_count', 0)))
        self.cached.append(int(getattr(self.episode, 'cached_count', 0)))
        self.seconds.append(float(getattr(self.episode, 'season_seconds', 0.0)))

    def run_reward(self) -> float:
        """The end-of-season payments summed, as logged in the reward column."""
        if self.per_season:
            return sum(
                season_end_reward(points, percentile)
                for points, percentile in zip(self.points, self.percentiles)
            )
        return multi_season_reward(self.points, self.percentiles)

    def _begin(self):
        season = self.seasons[self.index]
        print(
            '%s %s, season %s/%s, %s'
            % (
                'Evaluation run' if self.evaluation else 'Run',
                self.run_id,
                self.index + 1,
                len(self.seasons),
                season_label(season.season),
            ),
            flush=True,
        )
        self.episode = season.episode()


class SeasonByRun:
    """Episode factory for per-season runs.

    Each call is one environment reset. It continues the current run with
    its next season, or starts a new run once the last one is over. A reset
    in the middle of a season abandons that run.
    """

    def __init__(self, seasons, log_path=None):
        self.seasons = list(seasons)
        self.log_path = log_path
        self.run = None

    def __call__(self):
        run = self.run
        if run is not None and not run.done and run._waiting:
            run.start_next_season()
            return run
        self.run = MultiSeasonRun(self.seasons, log_path=self.log_path, per_season=True)
        return self.run


def _print_finished(run_id, points, percentiles, reward, label='Run'):
    point_text = ' + '.join(_number(value) for value in points)
    rank_text = ' + '.join(str(int(round(value))) for value in percentiles)
    print(
        '%s %s finished. Points %s = %s. Percentiles %s = %s. Reward %.2f.'
        % (
            label,
            run_id,
            point_text,
            _number(sum(points)),
            rank_text,
            int(round(sum(percentiles))),
            float(reward),
        ),
        flush=True,
    )


def _append_run_log(run: MultiSeasonRun):
    if run.log_path is None:
        return
    log_path = run.log_path.parent / EVALUATION_LOG if run.evaluation else run.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['run']
    # Runs on a mix of real seasons and synthetic copies say which version each was.
    labelled = all(hasattr(season, 'data_label') for season in run.seasons)
    for season in run.seasons:
        year = int(season.season)
        if labelled:
            fields.append('%s_data' % year)
        fields.extend([
            '%s_points' % year,
            '%s_percentile' % year,
            '%s_optimisations' % year,
            '%s_from_cache' % year,
            '%s_seconds' % year,
        ])
    fields.extend([
        'total_points', 'total_percentile', 'reward',
        'optimisations', 'from_cache', 'seconds',
    ])
    row = {'run': run.run_id}
    for season, points, percentile, solved, cached, seconds in zip(
        run.seasons, run.points, run.percentiles, run.solved, run.cached, run.seconds,
    ):
        year = int(season.season)
        if labelled:
            row['%s_data' % year] = season.data_label
        row['%s_points' % year] = _number(points)
        row['%s_percentile' % year] = _number(percentile)
        row['%s_optimisations' % year] = int(solved)
        row['%s_from_cache' % year] = int(cached)
        row['%s_seconds' % year] = round(float(seconds), 1)
    row['total_points'] = _number(sum(run.points))
    row['total_percentile'] = _number(sum(run.percentiles))
    row['reward'] = '%.2f' % float(run.run_reward())
    row['optimisations'] = int(sum(run.solved))
    row['from_cache'] = int(sum(run.cached))
    row['seconds'] = round(float(sum(run.seconds)), 1)
    lock = _lock(log_path.parent)
    try:
        new_file = not log_path.is_file()
        with log_path.open('a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    finally:
        _unlock(lock)


def _number(value) -> str:
    value = float(value)
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return '%.1f' % value
