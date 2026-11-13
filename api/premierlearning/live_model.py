"""Which model the live pipeline trains (run_pipeline.py and the email both ask here).

Since 2026-11-09 (Hal asked) the default is LightGBM v6 (`--learning-v2
--lightgbm`, premierlearning/lgbm/live.py): best over 2023/24-2025/26 and in the
fantasy manager runs. PREMIERLEARNING_ARGS can still pick another:
`--v2-blend` for the previous default (75% v2_log with minutes bands + 25%
v2_log h5), `--lgbm-version v3` for the previous LightGBM, `--v1` for the old
model, or a v1 recipe such as --h1-poisson.
"""
import os
import shlex

V1_FLAG = '--v1'
LIGHTGBM_FLAG = '--lightgbm'
BLEND_FLAG = '--v2-blend'
DEFAULT_ARGS = ['--learning-v2', LIGHTGBM_FLAG]
BLEND_ARGS = ['--learning-v2', BLEND_FLAG]
# Args that choose a model themselves; the default is only added without them.
MODEL_CHOICES = {
    V1_FLAG, '--learning-v2', LIGHTGBM_FLAG, BLEND_FLAG, '--h1-poisson', '--horizon-only',
    '--minutes-only', '--team-only', '--linear',
}
V2_ONLY = (LIGHTGBM_FLAG, BLEND_FLAG)


def with_model(extra):
    """extra with the default model added when it names none (and --learning-v2 where needed)."""
    extra = list(extra)
    if not MODEL_CHOICES.intersection(extra):
        return DEFAULT_ARGS + extra
    if '--learning-v2' not in extra and any(flag in extra for flag in V2_ONLY):
        return ['--learning-v2'] + extra
    return extra


def configured_args():
    return with_model(shlex.split(os.environ.get('PREMIERLEARNING_ARGS', '')))


def uses_lightgbm(args=None):
    args = configured_args() if args is None else with_model(args)
    return '--learning-v2' in args and LIGHTGBM_FLAG in args


def lightgbm_version(args=None):
    from premierlearning.lgbm.live import live_version

    return live_version(configured_args() if args is None else with_model(args))
