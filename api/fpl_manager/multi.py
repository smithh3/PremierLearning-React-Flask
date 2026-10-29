"""One training run played across several seasons.

The policy does not receive a season's points when that season ends. Those
points and finish percentiles are summed, and that sum is the reward on the
last gameweek of the run.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from fpl_manager.percentiles import season_beaten_percent, season_label
from fpl_manager.reward import multi_season_reward


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
    handle = open(path, 'a+b')
    if handle.seek(0, os.SEEK_END) == 0:
        handle.write(b'\0')
        handle.flush()
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
    def __init__(self, seasons, log_path=None):
        self.seasons = list(seasons)
        if not self.seasons:
            raise ValueError('a multi-season run needs at least one season')
        self.log_path = None if log_path is None else Path(log_path)
        self.run_id = _next_run_id(None if self.log_path is None else self.log_path.parent)
        self.index = 0
        self.episode = None
        self.points = []
        self.percentiles = []
        self.solved = []
        self.cached = []
        self.seconds = []
        self.done = False
        self._begin()

    def observation(self):
        return self.episode.observation()

    def mask(self):
        return self.episode.mask()

    def step(self, action_index: int):
        if self.done:
            raise RuntimeError('run already finished')
        observation, reward, finished, info = self.episode.step(action_index)
        if not finished:
            return observation, reward, False, info
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

    def _begin(self):
        season = self.seasons[self.index]
        print(
            'Run %s, season %s/%s, %s'
            % (
                self.run_id,
                self.index + 1,
                len(self.seasons),
                season_label(season.season),
            ),
            flush=True,
        )
        self.episode = season.episode()


def _print_finished(run_id, points, percentiles, reward):
    point_text = ' + '.join(_number(value) for value in points)
    rank_text = ' + '.join(str(int(round(value))) for value in percentiles)
    print(
        'Run %s finished. Points %s = %s. Percentiles %s = %s. Reward %.2f.'
        % (
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
    run.log_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['run']
    for season in run.seasons:
        year = int(season.season)
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
        row['%s_points' % year] = _number(points)
        row['%s_percentile' % year] = _number(percentile)
        row['%s_optimisations' % year] = int(solved)
        row['%s_from_cache' % year] = int(cached)
        row['%s_seconds' % year] = round(float(seconds), 1)
    row['total_points'] = _number(sum(run.points))
    row['total_percentile'] = _number(sum(run.percentiles))
    row['reward'] = '%.2f' % float(multi_season_reward(run.points, run.percentiles))
    row['optimisations'] = int(sum(run.solved))
    row['from_cache'] = int(sum(run.cached))
    row['seconds'] = round(float(sum(run.seconds)), 1)
    lock = _lock(run.log_path.parent)
    try:
        new_file = not run.log_path.is_file()
        with run.log_path.open('a', newline='') as handle:
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
