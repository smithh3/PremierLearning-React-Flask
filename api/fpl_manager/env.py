"""Masked season environment.

The observation is already scaled. Do not wrap this in a second normaliser:
training and evaluation have to see the same 42 fields.
"""
from __future__ import annotations

import numpy as np
from gymnasium import Env, spaces

from fpl_manager.actions import ACTIONS
from fpl_manager.observation import OBSERVATION_NAMES


class ManagerEnv(Env):
    def __init__(self, factory):
        super().__init__()
        self.factory = factory
        self.episode = None
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(OBSERVATION_NAMES),),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode = self.factory()
        return self._observation(), {}

    def step(self, action):
        observation, reward, done, info = self.episode.step(int(action))
        if done:
            observation = np.zeros(len(OBSERVATION_NAMES), dtype=np.float32)
        else:
            observation = np.asarray(observation, dtype=np.float32)
        return observation, float(reward), bool(done), False, info

    def action_masks(self):
        return np.asarray(self.episode.mask(), dtype=bool)

    def __getstate__(self):
        state = dict(self.__dict__)
        # The live episode holds the outcome callable for this rollout.
        # A checkpoint only needs the factory, which builds a new one.
        state['episode'] = None
        return state

    def _observation(self):
        return np.asarray(self.episode.observation(), dtype=np.float32)
