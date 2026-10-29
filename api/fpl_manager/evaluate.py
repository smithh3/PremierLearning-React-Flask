"""Deterministic actions for a saved policy, on a shared outcome stream."""
from __future__ import annotations

import numpy as np

from fpl_manager.baselines import summarise


def deterministic_action(model, episode) -> int:
    observation = np.asarray(episode.observation(), dtype=np.float32)
    mask = np.asarray(episode.mask(), dtype=bool)
    action, _state = model.predict(
        observation,
        action_masks=mask,
        deterministic=True,
    )
    return int(np.asarray(action).reshape(-1)[0])


def play_policy(episode, model) -> dict:
    while not episode.done:
        episode.step(deterministic_action(model, episode))
    return summarise(episode)
