"""The initial experiments and where their prediction sets were recorded.

Every built experiment is trained once per season at the GW1 cutoff and predicts
every snapshot with that season's fixed weights, v1 included, so no model gets
fresher training labels than another. v2 uses one fit per season: a fit peaks
around 3 GB, and three did not fit in 8 GB of RAM. v2_huber_seed2 repeats
v2_huber with another seed, and the gap between the two is the training-noise
floor for reading score differences.
"""

import json
import os
from datetime import datetime
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[2]
# PL_MODEL_EVAL_DIR moves results and the manifest, e.g. for a throwaway smoke test.
RESULTS_ROOT = Path(os.environ.get('PL_MODEL_EVAL_DIR') or API_DIR / 'run_metrics' / 'model_eval')
MANIFEST = RESULTS_ROOT / 'experiments.json'

BASELINE = 'v1'
_V2 = ['--learning-v2', '--train-once-per-season', '--points-reps', '1']
# For experiments added from here on: a shorter early-stop patience saves about 7
# minutes a fit. 9, not 8: ReduceLROnPlateau also waits 8, and at 8 a fit stops the
# moment the rate would drop (v2_huber's best round came right after that drop).
# The ones below stay at the default so they compare like for like, and predictions
# we act on always use the default.
_V2_TEST = _V2 + ['--v2-patience-floor', '9']

EXPERIMENTS = {
    # Single-GW v1 is the baseline, since every v2 variant is a single-GW model;
    # v1_h5's 5-GW totals carry smoothing a one-GW model does not have.
    'v1': {
        'args': ['--train-once-per-season', '--no-horizon-model'],
        'note': 'v1 baseline, single GW: pp90 x minutes net, chance applied as the optimiser does.',
    },
    'v1_h5': {
        'args': ['--train-once-per-season'],
        'note': 'v1 as used live: a 5-GW total model, split across fixtures by the single-GW points.',
    },
    'v2_huber': {
        'args': _V2 + ['--v2-weighting', 'none'],
        'note': 'v2, Huber on raw points, every fixture weighted equally.',
    },
    'v2_tier': {
        'args': _V2 + ['--v2-weighting', 'tier'],
        'note': 'v2, the current blended score-band weights.',
    },
    'v2_smooth': {
        'args': _V2 + ['--v2-weighting', 'smooth'],
        'note': 'v2, smooth weights 1 + 1.4 (points / 10)^1.5.',
    },
    'v2_huber_seed2': {
        'args': _V2 + ['--v2-weighting', 'none', '--seed', '43'],
        'note': 'v2_huber with another seed: the training-noise reference.',
    },
    # Round 2: loss weighting and loss shape.
    'v1_h5_tier': {
        'args': ['--train-once-per-season', '--points-tier-weight'],
        'note': 'v1_h5 with the blended score bands as training weights (h1 and horizon nets).',
    },
    'v2_smooth_cap': {
        'args': _V2 + ['--v2-weighting', 'smooth', '--v2-smooth-cap', '2.4'],
        'note': 'v2, smooth weights capped at 2.4: the tier bands without steps.',
    },
    'v2_hump': {
        'args': _V2 + ['--v2-weighting', 'hump'],
        'note': 'v2, smooth weights peaking at 1.7 on 8 points and easing off for hauls.',
    },
    'v2_plateau': {
        'args': _V2 + ['--v2-weighting', 'plateau'],
        'note': 'v2, tier bands with 10+ held at the 6-9 weight (1 / 1.2 / 1.6 / 1.6).',
    },
    'v2_huber_d3': {
        'args': _V2 + ['--v2-weighting', 'none', '--huber-delta', '3'],
        'note': 'v2, no weights, Huber threshold 3 (default 1.8).',
    },
    'v2_huber_d5': {
        'args': _V2 + ['--v2-weighting', 'none', '--huber-delta', '5'],
        'note': 'v2, no weights, Huber threshold 5.',
    },
    'v2_mse': {
        'args': _V2 + ['--v2-weighting', 'none', '--v2-loss', 'mse'],
        'note': 'v2, no weights, squared error: fits the mean directly.',
    },
    'v2_hump_fast': {
        'args': _V2_TEST + [
            '--v2-weighting', 'hump', '--v2-hump-peak', '6', '--v2-hump-height', '2',
            '--v2-hump-power', '1', '--v2-hump-width', '15',
        ],
        'note': 'v2, hump fitted to be level-unbiased on 2023/24-24/25: 3x by 6 points, 1.8x at 20.',
    },
    'v2_hump_soft': {
        'args': _V2 + [
            '--v2-weighting', 'hump', '--v2-hump-peak', '6', '--v2-hump-height', '1.2',
            '--v2-hump-power', '1', '--v2-hump-width', '10',
        ],
        'note': 'v2_hump_fast eased: blanks lose less weight and hauls wind down sooner, default patience.',
    },
    'v2_log': {
        'args': _V2 + ['--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global'],
        'note': 'v2, no weights, trained on log1p points, one smearing factor (level only).',
    },
    'v2_log_cond': {
        'args': _V2 + ['--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'conditional'],
        'note': 'v2, no weights, trained on log1p points, smearing by predicted level (can reorder).',
    },
    'v2_log_d1': {
        'args': _V2 + ['--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global', '--huber-delta', '1'],
        'note': 'v2_log with Huber threshold 1 on the log scale, where 1.8 is nearly squared error.',
    },
    'v2_log_cap': {
        'args': _V2 + [
            '--v2-weighting', 'smooth', '--v2-smooth-cap', '2.4', '--v2-loss', 'huber', '--v2-smear', 'global',
        ],
        'note': 'v2_log with the smooth weights capped at 2.4.',
    },
    'v2_log_h5': {
        'args': _V2 + ['--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global', '--v2-horizon', '5'],
        'note': 'v2_log plus a 5-GW net on points per fixture; window totals split by h1 shares.',
    },
    'v2_log_minbands': {
        'args': _V2 + [
            '--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global',
            '--v2-minutes-head', '0.1', '--v2-minutes-target', 'bands',
        ],
        'note': 'v2_log plus a training-only cross-entropy head on minutes bands (0, 1-59, 60+).',
    },
    'v2_log_minlog': {
        'args': _V2 + [
            '--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global',
            '--v2-minutes-head', '1', '--v2-minutes-target', 'log',
        ],
        'note': 'v2_log plus a training-only Huber head on log minutes scaled to 0-1.',
    },
    # Cheats: an upper bound on what better minutes data could add. Never usable.
    'v2_log_oracle_minutes': {
        'args': _V2 + [
            '--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global',
            '--model-plan', 'simple_h1_oracle_minutes',
        ],
        'note': 'CHEATS: v2_log fed each fixture\'s real minutes, to size the value of perfect minutes news.',
    },
    # Team and opponent match histories (114 matches each, across seasons, real xG
    # from 2022/23) in transformers, their fading averages, and team identities
    # split into time blocks (newest 38 gameweeks, then 76-gameweek blocks).
    # Otherwise v2_log_minbands. Training steps take about 5x as long.
    'v2_log_minbands_teams': {
        'args': _V2 + [
            '--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global',
            '--v2-minutes-head', '0.1', '--v2-minutes-target', 'bands',
            '--model-plan', 'simple_h1_teams',
        ],
        'note': 'v2_log_minbands plus 3-season team/opponent history transformers, fading xG averages, team time blocks.',
    },
    'v2_log_minbands_teams_pblocks': {
        'args': _V2 + [
            '--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global',
            '--v2-minutes-head', '0.1', '--v2-minutes-target', 'bands',
            '--model-plan', 'simple_h1_teams_pblocks',
        ],
        'note': 'v2_log_minbands_teams with player identities split into the same time blocks.',
    },
    # The same histories read by one attention pass from the player (as the squads
    # are) instead of a transformer: about 2.5x the current step time, not 5x.
    'v2_log_minbands_teams_attn': {
        'args': _V2 + [
            '--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global',
            '--v2-minutes-head', '0.1', '--v2-minutes-target', 'bands',
            '--model-plan', 'simple_h1_teams_attn',
        ],
        'note': 'v2_log_minbands_teams with the team histories pooled by player attention, not transformers.',
    },
    # Scoreline: the team's goals for and against in the fixture, from a small net
    # on the team, opponent and match encodings (Poisson, loss weight 0.1).
    'v2_log_minbands_teams_score': {
        'args': _V2 + [
            '--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global',
            '--v2-minutes-head', '0.1', '--v2-minutes-target', 'bands',
            '--model-plan', 'simple_h1_teams_score',
        ],
        'note': 'v2_log_minbands_teams plus a training-only head on the team\'s goals for and against.',
    },
    # The same scoreline built into points: band chances x points per band, with
    # clean sheets from e^-(goals against) and attacking returns scaled by goals for.
    'v2_log_minbands_teams_scoreline': {
        'args': _V2 + [
            '--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global',
            '--v2-minutes-head', '0.1', '--v2-minutes-target', 'bands',
            '--model-plan', 'simple_h1_teams_scoreline',
        ],
        'note': 'v2_log_minbands_teams with points built from minutes-band chances and expected goals for/against.',
    },
    # Recency pooling: every transformer history (the player's matches, past
    # seasons, both team histories) is summed up by attention whose weights fade
    # with each item's age at rates the heads learn, plus a match count, instead
    # of a flat average. With and without the fixed fading averages.
    'v2_log_minbands_teams_recency_nodecay': {
        'args': _V2 + [
            '--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global',
            '--v2-minutes-head', '0.1', '--v2-minutes-target', 'bands',
            '--model-plan', 'simple_h1_teams_recency_nodecay',
        ],
        'note': 'v2_log_minbands_teams with recency-weighted attention pooling and no fixed fading averages.',
    },
    'v2_log_minbands_teams_recency': {
        'args': _V2 + [
            '--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global',
            '--v2-minutes-head', '0.1', '--v2-minutes-target', 'bands',
            '--model-plan', 'simple_h1_teams_recency',
        ],
        'note': 'v2_log_minbands_teams with recency-weighted attention pooling, fading averages kept.',
    },
    # Three stages in one network: player minutes bands, a team-level scoreline
    # in goal bands (0-4+, one reader shared by both sides, each team fixture
    # weighted once), and points built from both. Its plan file can take on the
    # pooling and fading-average choices from the runs before it.
    'v2_log_minbands_teams_staged': {
        'args': _V2 + [
            '--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global',
            '--v2-minutes-head', '0.1', '--v2-minutes-target', 'bands',
            '--model-plan', 'simple_h1_teams_staged',
        ],
        'note': 'v2_log_minbands_teams as three stages: minutes bands, team-level goal bands, points built from both.',
    },
    # Base for the external-data wave, rebuilt after the 2021/22 data swap (full
    # vaastav CSV in place of the mid-season snapshot). simple_h1_b0 is a copy of
    # the wave A winner's plan, written by pick_base. Two seeds: the base, and
    # the training-noise floor for reading the gaps between later runs.
    'v2_b0_seed1': {
        'args': _V2 + [
            '--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global',
            '--v2-minutes-head', '0.1', '--v2-minutes-target', 'bands',
            '--model-plan', 'simple_h1_b0', '--seed', '42',
        ],
        'note': 'Wave A winner retrained on the full 2021/22 data, seed 42: base for the external-data runs.',
    },
    'v2_b0_seed2': {
        'args': _V2 + [
            '--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global',
            '--v2-minutes-head', '0.1', '--v2-minutes-target', 'bands',
            '--model-plan', 'simple_h1_b0', '--seed', '43',
        ],
        'note': 'v2_b0_seed1 with seed 43: the training-noise floor.',
    },
    # Wave B, each on B0 (simple_h1_bN derived from simple_h1_b0 by derive_plan).
    'v2_b1_context': {
        'args': _V2 + [
            '--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global',
            '--v2-minutes-head', '0.1', '--v2-minutes-target', 'bands',
            '--model-plan', 'simple_h1_b1', '--seed', '42',
        ],
        'note': 'B0 plus all-competition rest / congestion, double / blank gameweek flags and player minutes load.',
    },
    # The staged structure again, now the fixture and minutes context is in.
    'v2_b1_staged': {
        'args': _V2 + [
            '--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global',
            '--v2-minutes-head', '0.1', '--v2-minutes-target', 'bands',
            '--model-plan', 'simple_h1_b1_staged', '--seed', '42',
        ],
        'note': 'v2_b1_context with a team-level scoreline in goal bands feeding points (minutes, scoreline, points).',
    },
    'v2_b1_staged_v2': {
        'args': _V2 + [
            '--v2-weighting', 'none', '--v2-loss', 'huber', '--v2-smear', 'global',
            '--v2-minutes-head', '0.1', '--v2-minutes-target', 'bands',
            '--model-plan', 'simple_h1_b1_staged_v2', '--seed', '42',
        ],
        'note': 'v2_b1_staged whose scoreline also reads both squads and fits match xG as well as goals.',
    },
    # LightGBM (gradient-boosted trees, not a net) on the rows and inputs the net
    # with that plan sees, flattened into columns by premierlearning/lgbm. Squared
    # error on raw points, so it predicts expected points directly. LIGHTGBM_PLAN.md.
    'lgbm_v1': {
        'args': ['--lightgbm', '--train-once-per-season', '--lgbm-plan', 'simple_h1'],
        'note': 'LightGBM v1: the original v2 inputs (player history, past seasons, squads, team form) as columns.',
    },
    'lgbm_v2': {
        'args': ['--lightgbm', '--train-once-per-season', '--lgbm-plan', 'simple_h1_b1'],
        'note': 'LightGBM v2: B1 inputs (adds 3-season team histories with xG, rest / congestion, DGW flags, minutes load).',
    },
    'lgbm_v2_rules': {
        'args': ['--lightgbm', '--train-once-per-season', '--lgbm-plan', 'simple_h1_b1', '--lgbm-rules', 'safe'],
        'note': 'lgbm_v2 with the safe monotone rules (e.g. more recent xG or a weaker opponent never lowers points).',
    },
    # Feature set 2: many windows on every history, trends, same-venue form,
    # fixture matchup columns and xG shares; 3 seeds averaged.
    'lgbm_v3': {
        'args': ['--lightgbm', '--train-once-per-season', '--lgbm-plan', 'simple_h1_b1',
                 '--lgbm-features', '2', '--lgbm-no-ids', '--lgbm-seeds', '3'],
        'note': 'LightGBM v3: B1 inputs, feature set 2 (more windows, trends, venue, matchups), no team ids, 3 seeds.',
    },
    'lgbm_v3_ids': {
        'args': ['--lightgbm', '--train-once-per-season', '--lgbm-plan', 'simple_h1_b1',
                 '--lgbm-features', '2', '--lgbm-seeds', '3'],
        'note': 'lgbm_v3 with the team and opponent id columns kept: isolates the ids.',
    },
    'lgbm_v3_simple': {
        'args': ['--lightgbm', '--train-once-per-season', '--lgbm-plan', 'simple_h1',
                 '--lgbm-features', '2', '--lgbm-no-ids', '--lgbm-seeds', '3'],
        'note': 'lgbm_v3 on the v1 inputs (no team histories, rest or load): is the best LightGBM the simple one?',
    },
    # v3 variants (2026-11-05): ranking objective, linear leaves, and ways to
    # let the team ids back in without the trees memorising stale identities.
    'lgbm_v3_rank': {
        'args': ['--lightgbm', '--train-once-per-season', '--lgbm-plan', 'simple_h1_b1',
                 '--lgbm-features', '2', '--lgbm-no-ids', '--lgbm-seeds', '3', '--lgbm-objective', 'lambdarank'],
        'note': 'lgbm_v3 trained to order players within each gameweek (lambdarank); scores mapped to points by percentile.',
    },
    'lgbm_v3_linear': {
        'args': ['--lightgbm', '--train-once-per-season', '--lgbm-plan', 'simple_h1_b1',
                 '--lgbm-features', '2', '--lgbm-no-ids', '--lgbm-seeds', '3', '--lgbm-linear'],
        'note': 'lgbm_v3 with linear leaves (each leaf a small line, not a constant).',
    },
    'lgbm_v3_id_stage': {
        'args': ['--lightgbm', '--train-once-per-season', '--lgbm-plan', 'simple_h1_b1',
                 '--lgbm-features', '2', '--lgbm-seeds', '3', '--lgbm-id-stage'],
        'note': 'lgbm_v3 plus a separate team-id correction that must earn its rounds on the held-out season.',
    },
    'lgbm_v3_recent': {
        'args': ['--lightgbm', '--train-once-per-season', '--lgbm-plan', 'simple_h1_b1',
                 '--lgbm-features', '2', '--lgbm-no-ids', '--lgbm-seeds', '3', '--lgbm-recency', '2'],
        'note': 'lgbm_v3 with older seasons down-weighted (weight halves every 2 seasons back).',
    },
    'lgbm_v3_id_stage_recent': {
        'args': ['--lightgbm', '--train-once-per-season', '--lgbm-plan', 'simple_h1_b1',
                 '--lgbm-features', '2', '--lgbm-seeds', '3', '--lgbm-id-stage', '--lgbm-recency', '2'],
        'note': 'lgbm_v3_id_stage with recency weights, so the ids reflect recent seasons more.',
    },
    'v2_poisson': {
        'args': _V2_TEST + ['--v2-weighting', 'none', '--v2-loss', 'poisson'],
        'note': 'v2, no weights, Poisson loss: fits the mean, softplus output, negative scores floored at 0.',
    },
}

# Retraining every 5 gameweeks (2026-11-05, Hal asked): refit at GW1, 6, ..., 36,
# each fit trained on every label before its gameweek. For the net this also
# rebuilds the vocab (new players get embeddings) and slides the team time
# blocks, so block 0 is always the 38 gameweeks before the fit. Same settings as
# the once-per-season originals otherwise, so the schedule is the only change.
EXPERIMENTS['v2_b1_every5'] = {
    'args': EXPERIMENTS['v2_b1_context']['args'] + ['--retrain-every', '5'],
    'note': 'v2_b1_context refitted every 5 GWs: recent-season team blocks and player embeddings kept current.',
}
# Identity look-ahead (2026-11-05, Hal asked): a lookup embedding is learned from
# all of a player's (or team's) training rows, later ones included, so on older
# rows it "knows" his future level, and the random validation split cannot see
# that. These B0 copies drop the lookups, so identity comes only from the
# history encoders, which read matches before each row. Same settings as
# v2_b0_seed1 / seed2 (the noise floor), only the plan differs.
for _name, _plan, _note in (
        ('v2_b0_nopid', 'simple_h1_b0_nopid', 'B0 without player id embeddings (player, teammates, opponents).'),
        ('v2_b0_noids', 'simple_h1_b0_noids', 'B0 without any id embeddings: players and teams only via their histories.'),
        # 2026-11-06, Hal asked: the no-id B0 with the scoreline built into points
        # (minutes bands -> team goal bands from both histories -> points), as v2_b1_staged.
        ('v2_b0_noids_staged', 'simple_h1_b0_noids_staged',
         'v2_b0_noids with a staged scoreline: minutes bands and team goal bands (from both histories) feed points.')):
    _args = list(EXPERIMENTS['v2_b0_seed1']['args'])
    _args[_args.index('--model-plan') + 1] = _plan
    EXPERIMENTS[_name] = {'args': _args, 'note': _note}
# 2026-11-07, Hal asked for the best net, preferring one that uses more data, over
# three seasons: v2_b1_context's inputs (B0 plus all-competition rest, load and
# DGW flags) without the id embeddings that held B0 back. Plan made from
# simple_h1_b1 by dropping every *_id_embedding key, as simple_h1_b0_noids was.
_args = list(EXPERIMENTS['v2_b1_context']['args'])
_args[_args.index('--model-plan') + 1] = 'simple_h1_b1_noids'
EXPERIMENTS['v2_b1_noids'] = {
    'args': _args,
    'note': 'B1 (B0 plus all-competition rest / congestion, DGW flags, minutes load) without any id embeddings.',
}
for _name in [name for name in EXPERIMENTS if name.startswith('lgbm_v3')]:
    EXPERIMENTS[_name + '_every5'] = {
        'args': EXPERIMENTS[_name]['args'] + ['--retrain-every', '5'],
        'note': '%s refitted every 5 GWs (hold-out: the newest 38 GWs before each fit).' % _name,
    }
# 2026-11-07, Hal asked for all three: aimed at LightGBM's overrated top picks
# (its top 5 by prediction score ~2.7 fewer points over 5 GWs than predicted)
# and its flat fixture swings. Defined after the _every5 loop: no twins.
EXPERIMENTS['lgbm_v3_tweedie'] = {
    'args': EXPERIMENTS['lgbm_v3']['args'] + ['--lgbm-objective', 'tweedie'],
    'note': 'lgbm_v3 with the Tweedie objective (power 1.3; suits mostly-small, sometimes-large points).',
}
_args = list(EXPERIMENTS['lgbm_v3']['args'])
_args[_args.index('--lgbm-features') + 1] = '3'
EXPERIMENTS['lgbm_v4'] = {
    'args': _args,
    'note': 'lgbm_v3 with feature set 3: team ratings (time-decayed Poisson, goals and xG), what the opponent '
            'concedes to each position, shrunk per-90s and competition for his place.',
}
EXPERIMENTS['lgbm_v5'] = {
    'args': EXPERIMENTS['lgbm_v4']['args'] + ['--lgbm-staged'],
    'note': 'lgbm_v4 inputs in stages: minutes bands, clean sheet and the rest of the points, combined with FPL scoring.',
}
# 2026-11-07, Hal asked: FPL's own player news beyond the flat chance (status,
# injury type, days to the expected return, how long flagged, a 0-10 certainty).
# Data: python -m premierlearning.lgbm.news (21/22-25/26 deadlines).
_args = list(EXPERIMENTS['lgbm_v4']['args'])
_args[_args.index('--lgbm-features') + 1] = '4'
EXPERIMENTS['lgbm_v6'] = {
    'args': _args,
    'note': 'lgbm_v4 with feature set 4: FPL player news (status, injury type, days to expected return, '
            'days flagged, chance change, 0-10 certainty).',
}
EXPERIMENTS['lgbm_v6_staged'] = {
    'args': EXPERIMENTS['lgbm_v6']['args'] + ['--lgbm-staged'],
    'note': 'lgbm_v5 (staged) with feature set 4: the news columns feed the minutes stage too.',
}

DERIVED = {
    # The same calibration on v1, so v2_smooth_cal is not the only model given it.
    'v1_cal': {
        'base': 'v1',
        'note': 'v1 plus the same rolling linear calibration.',
    },
    'v1_h5_cal': {
        'base': 'v1_h5',
        'note': 'v1_h5 plus the same rolling linear calibration.',
    },
    'v2_huber_cal': {
        'base': 'v2_huber',
        'note': 'v2_huber plus the same rolling linear calibration (unflagged players only).',
    },
    'v2_tier_cal': {
        'base': 'v2_tier',
        'note': 'v2_tier plus the same rolling linear calibration (unflagged players only).',
    },
    'v2_smooth_cal': {
        'base': 'v2_smooth',
        'note': 'v2_smooth plus a rolling linear calibration fitted before each deadline.',
    },
}
for _base in ('v1_h5_tier', 'v2_smooth_cap', 'v2_hump', 'v2_plateau', 'v2_huber_d3', 'v2_huber_d5', 'v2_mse',
              'v2_log', 'v2_log_cond', 'v2_hump_fast', 'v2_poisson', 'v2_hump_soft', 'v2_log_d1', 'v2_log_cap', 'v2_log_h5'):
    DERIVED[_base + '_cal'] = {
        'base': _base,
        'note': '%s plus the same rolling linear calibration.' % _base,
    }


def load_manifest(path=MANIFEST):
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def record_build(name, prediction_set_id, seasons, args, path=MANIFEST):
    path = Path(path)
    manifest = load_manifest(path)
    entry = manifest.get(name, {})
    if entry.get('prediction_set_id') != int(prediction_set_id):
        # A code or config change hashes to a new set; its old seasons are elsewhere.
        entry = {}
    built = sorted(set(entry.get('seasons', [])) | {int(season) for season in seasons})
    manifest[name] = {
        'prediction_set_id': int(prediction_set_id),
        'seasons': built,
        'args': list(args),
        'built_at': datetime.now().isoformat(timespec='seconds'),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest
