"""Staged points: player minutes bands, a team-level scoreline in goal bands, points from both.

All three stages train together in one network (no out-of-fold stacking). The
scoreline stage reads only team-level inputs: team and opponent ids, their
fading averages and their match histories through one reader shared by both
sides. By default squads stay out of it (a player's teammates list leaves that
player out, so it differs between teammates); they reach points through the
player's own attention. With scoreline_squads the teammates list keeps the
player, and one shared squad summary (projection, masked mean and max) joins
each side. With score_xg_weight the bands' means are also fitted to match xG.
One pair net gives goals for as f(us, them, home) and goals against
as f(them, us, away), so a fixture's two sides are mirror images and every
player in a team gets the same scoreline. Its loss weights each team fixture
once (dataset.score_targets(weighted=True)).
"""
from __future__ import annotations

import tensorflow as tf

# 0, 1, 2, 3 and 4+ goals. 4+ counts as 4.4 goals in the mean.
GOAL_BANDS = 5
GOAL_BAND_VALUES = (0.0, 1.0, 2.0, 3.0, 4.4)
# FPL: -1 per 2 goals conceded. floor(goals / 2) per band, 4+ as 2.
CONCEDED_STEPS = (0.0, 0.0, 1.0, 1.0, 2.0)
# Position index 0 unknown, 1 GK, 2 DEF, 3 MID, 4 FWD.
CLEAN_SHEET_POINTS = (0.0, 4.0, 4.0, 1.0, 0.0)
CONCEDED_PENALTY = (0.0, 1.0, 1.0, 0.0, 0.0)


def masked_goal_band_ce(y_true, y_pred):
    """Cross-entropy on the goals-for and goals-against bands.

    y_true is [goals for, goals against, weight]; rows with a negative score
    are skipped. y_pred is the two band softmaxes side by side.
    """
    known = tf.cast(y_true[:, 0] >= 0.0, y_pred.dtype)
    weight = y_true[:, 2] * known
    top = float(GOAL_BANDS - 1)
    goals_for = tf.cast(tf.clip_by_value(y_true[:, 0], 0.0, top), tf.int32)
    goals_against = tf.cast(tf.clip_by_value(y_true[:, 1], 0.0, top), tf.int32)
    log_for = tf.math.log(y_pred[:, :GOAL_BANDS] + 1e-7)
    log_against = tf.math.log(y_pred[:, GOAL_BANDS:] + 1e-7)
    picked = (
        tf.reduce_sum(tf.one_hot(goals_for, GOAL_BANDS) * log_for, axis=-1)
        + tf.reduce_sum(tf.one_hot(goals_against, GOAL_BANDS) * log_against, axis=-1)
    )
    return -0.5 * picked * weight


def band_means(y_pred):
    """Expected goals for and against from the two band softmaxes, [n, 2]."""
    values = tf.constant(GOAL_BAND_VALUES, dtype=y_pred.dtype)
    goals_for = tf.reduce_sum(y_pred[:, :GOAL_BANDS] * values, axis=-1)
    goals_against = tf.reduce_sum(y_pred[:, GOAL_BANDS:] * values, axis=-1)
    return tf.stack([goals_for, goals_against], axis=-1)


def goal_band_loss(xg_weight=0.0):
    """masked_goal_band_ce, plus a Poisson deviance between each side's band mean and its xG.

    With xg_weight above 0, y_true is [goals for, goals against, weight,
    xG for, xG against, xG known]: rows without xG (FPL has it from 2022/23)
    add nothing to that term. xG is far less noisy than goals, so it tells
    the scoreline more about which side is likely to create chances.
    """
    xg_weight = float(xg_weight or 0.0)
    if xg_weight <= 0:
        return masked_goal_band_ce

    def loss(y_true, y_pred):
        base = masked_goal_band_ce(y_true[:, :3], y_pred)
        known = tf.cast(y_true[:, 0] >= 0.0, y_pred.dtype) * y_true[:, 5]
        weight = y_true[:, 2] * known
        mean = tf.maximum(band_means(y_pred), 1e-4)
        xg = tf.maximum(y_true[:, 3:5], 0.0)
        deviance = mean - xg + tf.math.xlogy(xg, xg / mean)
        return base + xg_weight * 0.5 * tf.reduce_sum(deviance, axis=-1) * weight

    loss.__name__ = 'goal_band_xg_loss'
    return loss


class BandScorelinePoints(tf.keras.layers.Layer):
    """Expected points from minutes-band chances and goal-band chances.

    With P0 the chance of conceding none, C the expected conceded steps
    (floor(goals against / 2)) and F the expected goals for:
      60+   : 2 + clean sheet points x P0 - conceded penalty x C + r x F + other_60
      1-59  : 1 + s x r x F + other_sub
    r and the others are softplus, s a sigmoid, from four free parts.
    """

    def __init__(self, log_output=True, **kwargs):
        super().__init__(**kwargs)
        self.log_output = bool(log_output)
        self.clean_sheet = tf.constant(CLEAN_SHEET_POINTS, dtype=tf.float32)
        self.penalty = tf.constant(CONCEDED_PENALTY, dtype=tf.float32)
        self.band_values = tf.constant(GOAL_BAND_VALUES, dtype=tf.float32)
        self.conceded_steps = tf.constant(CONCEDED_STEPS, dtype=tf.float32)

    def call(self, inputs):
        minutes, goals, parts, position = inputs
        position = tf.reshape(tf.cast(position, tf.int32), [-1])
        position = tf.clip_by_value(position, 0, len(CLEAN_SHEET_POINTS) - 1)
        clean_sheet = tf.gather(self.clean_sheet, position)[:, None]
        penalty = tf.gather(self.penalty, position)[:, None]
        for_bands = goals[:, :GOAL_BANDS]
        against_bands = goals[:, GOAL_BANDS:]
        goals_for = tf.reduce_sum(for_bands * self.band_values, axis=-1, keepdims=True)
        no_goals = against_bands[:, 0:1]
        steps = tf.reduce_sum(against_bands * self.conceded_steps, axis=-1, keepdims=True)
        attack = tf.nn.softplus(parts[:, 0:1])
        sub_share = tf.sigmoid(parts[:, 1:2])
        other_full = tf.nn.softplus(parts[:, 2:3])
        other_sub = tf.nn.softplus(parts[:, 3:4])
        full = 2.0 + clean_sheet * no_goals - penalty * steps + attack * goals_for + other_full
        sub = 1.0 + sub_share * attack * goals_for + other_sub
        expected = tf.nn.relu(minutes[:, 1:2] * sub + minutes[:, 2:3] * full)
        if self.log_output:
            return tf.math.log1p(expected)
        return expected

    def get_config(self):
        config = super().get_config()
        config.update({'log_output': self.log_output})
        return config
