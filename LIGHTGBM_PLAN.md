# LightGBM points model: plan

Status: v1 and v2 built 2026-11-04 (Hal asked) and queued for evaluation
(`api/simulation/model_eval/run_lgbm_comparison.cmd`, run-id `lgbm_1`).
Dixon-Coles team ratings added to the plan 2026-11-04, not built yet.

Results 2026-11-04 (2025/26, `run_metrics/model_eval/lgbm_rules/comparison.txt`):
lgbm_v1 +1.59 (net B0 +1.67, best blend +1.82): best MSE (6.94) and single-GW
ranking (spearman 0.572) of any model, weak on fixture swings (corr 0.124 vs
0.150) and overpredicts its top-5 picks. lgbm_v2 +1.29 and lgbm_v2_rules +1.18:
the team-history columns made fixture swings worse (corr 0.07), while the
team / opponent id columns' share of gain doubled. Suspect: the ids learn
stale team identities across seasons. Next: v2 without team ids.

v3 started 2026-11-04 19:45 (`run_lgbm_v3_comparison.cmd`, run-id `lgbm_v3`).
Feature set 2 (`--lgbm-features 2`), from Hal's ask for several windows on
every history so the trees can weigh recent form against long-run level:
- player: means over 1/2/3/5/10/20/50 matches, faded half-lives 1.5/3/10/25,
  per-90 over 3/5/10/20/50, appearance and 60+ shares over 3/5/10/20/50,
  pre-match team form over 1/3/5/10, trends (last 3 minus last 10, last 5
  minus last 50), form at the same venue as this fixture (last 10)
- team and opponent: means over 1/3/5/10/19/38/76/114 matches, faded
  3/10/38, trend (last 5 minus last 38), same-venue form
- fixture matchups: his team's attack + opponent's defence (and the reverse)
  over several windows, how this fixture compares to the team's usual level,
  player xG+xA per 90 and points scaled by that, his share of team xG / xA
Also: `--lgbm-no-ids` (the v2 suspect) and `--lgbm-seeds 3` (averaging three
refits, against the over-confident top picks). Runs: lgbm_v3 (all of it),
lgbm_v3_ids (ids kept), lgbm_v3_simple (v1 inputs), plus 50/50 mixes of
lgbm_v1 / v3 with the B0 net.

v3 results (`run_metrics/model_eval/lgbm_v3/comparison.txt`): lgbm_v3 +1.87
and the 50/50 mix with the B0 net +2.01, both above the best net blend
(+1.82). lgbm_v3 has the best MSE (6.895) and single-GW ranking (0.579), fixture
swing corr 0.149 (as good as the nets) and no top-pick guardrail flag. Team ids
were the v2 problem: lgbm_v3_ids +1.52 (swing corr 0.093). With ids gone, the
team histories help: lgbm_v3_simple +1.43.

v3 variants started 2026-11-05 15:30 (`run_lgbm_v3_variants.cmd`, run-id
`lgbm_v3_variants`), all on the v3 inputs with 3 seeds:
- lgbm_v3_rank: trained to order players within each gameweek (lambdarank,
  top 300 of ~650 rows, linear gain on points 0-30). Scores become points by
  gameweek percentile, a curve learned on the held-out season. (Top-50 scoring
  was tried first on test data: it stops after ~8 rounds and ignores everyone
  below the top 50.)
- lgbm_v3_linear: linear leaves.
- lgbm_v3_id_stage: the main model never sees team ids; a second small model
  (ids + position only, 4 leaves, 2,000-row leaves, strong smoothing) fits what
  the ids add on top, and its rounds are picked on the held-out season. If
  "this is Arsenal" doesn't help on an unseen season, it gets 0 rounds. The fit
  log prints its rounds and the held-out RMSE with and without it.
- lgbm_v3_recent: older seasons weigh less (half every 2 seasons back), so
  team and player patterns reflect recent seasons.
- lgbm_v3_id_stage_recent: both.

v3 variants results (`run_metrics/model_eval/lgbm_v3_variants`): nothing beats
lgbm_v3 (+1.87). The team-id stage earned 0 rounds (ids add nothing on an unseen
season), so it is identical to v3. Linear leaves +1.87 (slightly worse fixture
swings, 0.145). Recency weights +1.76 (fixture swings 0.137). Ranking +0.33: the
percentile-to-points mapping wrecks calibration, and its ranking score is no
better (+2.20 vs +2.36). Best mixes: linear + B0 +2.02, v3 + B0 +2.01.

## Versions (like v1 -> v2 for the net: more data, then more complex data)

| Version | Inputs | Shape | Status |
|---|---|---|---|
| lgbm_v1 | plan `simple_h1` flattened: player's last 50 matches (means over 1/3/5/10/50, faded means, per-90 rates, appearance and 60+ shares, matches since last appearance), last two seasons, both squads (sum / mean / top 11 / max), team form, chance of playing, team ids | one model, squared error on raw points | built, queued |
| lgbm_v2 | plan `simple_h1_b1` flattened: adds both teams' last 114 matches (goals and xG means over 5/10/38, faded), all-competition rest and congestion, double / blank flags, player minutes load | as v1 | built, queued |
| lgbm_v2_rules | as lgbm_v2, with monotone rules (`SAFE_RULES` in `lgbm/model.py`): the player's recent output, appearance shares, chance of playing, his team's attack and the opponent's leakiness never lower points; time since playing, his team's and his own goals conceded, and the opponent's attack never raise them. Squads, rest and minutes load stay free | as v1 | built, queued |
| lgbm_v3 | v2 inputs, feature set 2 (see Results above), no team ids, 3 seeds; variants lgbm_v3_ids and lgbm_v3_simple | as v1 | +1.87, best single model; mix with B0 net +2.01 |
| lgbm_v3_tweedie | v3 | Tweedie objective (power 1.3), negative points train as 0 | built 2026-11-07, queued (`run_lgbm_v4.cmd`, run-id lgbm_v4) |
| lgbm_v4 | v3 plus feature set 3: team ratings (time-decayed Poisson on goals and on xG, `lgbm/extras.py`; no low-score correction), what the opponent concedes to each position (points / xG / xA over 5 / 10 / 38, and relative to the league), shrunk per-90 rates (270 minutes at the position rate), competition for his place (same-position teammates' recent minutes vs his and the usual starters). Rivals' chance of playing is not in the batch, so left out | as v1 | built 2026-11-07, queued |
| lgbm_v5 | v4 | staged (`lgbm/staged.py`): minutes classifier (0 / 1-59 / 60+), clean-sheet classifier on 60+ GK / DEF / MID rows (the ratings plus corrections), rest of points given he plays (regression with a 60+ flag), combined with FPL scoring; also stores minutes bands | built 2026-11-07, queued |
| lgbm_v6 | v4 | feature set 4: FPL's player news from the archived pre-deadline snapshots (`lgbm/news.py`): status, reason group (knock, muscle, joint, serious, illness, fitness, ban, left, squad, international), days from the expected return to kickoff, days flagged, change since the last deadline, flag streak and deadlines since the last flag, and a hand-written 0-10 certainty | built 2026-11-07, queued |
| lgbm_v6_staged | v5 | v5 staged model on feature set 4 | built 2026-11-07, queued |
| later | manager tenure, upcoming all-competition fixtures, objective / tuning sweep (Tweedie, Huber), blend with the v2 net | | ideas |

Code: `api/premierlearning/lgbm/features.py` (flattening), `lgbm/model.py`
(fit: the latest training season picks the number of rounds by early stopping,
then a refit on every season), `api/simulation/season_model_lgbm.py` (trained
once per season at GW1, like the v2 evaluation), `--lightgbm`, `--lgbm-plan`
and `--lgbm-objective` in `simulation/prediction_builder.py`. Tests:
`api/tests/test_lgbm.py`. The fit prints the held-out error and the top 30
columns by gain, which shows which data matters.

## Why

- LightGBM is not a neural net. It is gradient-boosted decision trees: many small
  flowcharts ("rest days < 4 -> ...") added up, each one correcting the errors of
  the trees before it.
- It tends to match or beat neural nets on table-shaped data of our size
  (about 25k player-gameweeks a season, about 200k rows over 8 seasons) and with
  noisy targets like FPL points.
- It lets us prescribe relationships with monotone constraints, e.g. a higher
  clean-sheet chance never lowers a defender's points.
- It trains in minutes on the CPU and shows which features matter, so it is also
  a cheap check of whether the new external data helps.
- Today the only non-neural models in the repo are the Ridge baselines
  (`learning.py`, `team_expectancy.py`). `sklearn.ensemble.HistGradientBoosting*`
  is already installed and is a close alternative if we don't want the
  `lightgbm` package.

## Tree-only tricks to try (2026-11-04)

1. **Retrain every gameweek.** A fit takes minutes, not hours, so live use can
   retrain each week on the latest results. The eval trains once per season for
   fairness with the nets; add a per-gameweek variant to size the gain.
2. **Explanations per prediction** (`pred_contrib=True`, SHAP values): each
   player's points split into "+1.1 from the opponent, -0.6 from rotation risk".
   A check on the model, and something the app could show.
3. **Ranking objective** (`lambdarank`, rows grouped by gameweek): trains on
   ordering players within a gameweek, which is what picking a team needs.
4. **Haul chances** (quantile objective, or a classifier on 10+ points): a
   ceiling as well as the average, for captaincy.
5. **Interaction constraints:** e.g. team-strength columns may only combine
   with position, which forces sensible structure with little data.
6. **Linear leaves** (`linear_tree`): each leaf fits a small line instead of a
   constant, so it extrapolates better for elite players at the edge of the data.
7. **Cheap sweeps and seed averaging:** settings, objectives (Tweedie, Huber)
   and recency weights for older seasons, many runs per night.

## Rows and leak safety

- One row per player-fixture, as in the v2 net.
- Build the flat columns from the same lists that `learning_v2/dataset.py`
  already produces for the net, not from a second pipeline. Both models then see
  the same information as of each deadline: a fair comparison and no new leak
  risk.
- Evaluate on 2025/26 only, with the same metrics and slices as `model_eval`.

## Flattening the lists into columns

**Player's own match history (50 matches)**
- Means over the last 1, 3, 5 and 10 matches, the season and the career, for
  points, minutes, xG, xA, goals, assists and defcon.
- Faded means with half-lives of 3 and 10 matches.
- Trend: last 3 minus last 10.
- Rotation signals: starts in the last 5, matches since the last start, started
  the last match, days since he last played.
- Per-90 rates shrunk toward the position's average when minutes are few.

**Team and opponent match histories (114 matches)**
- Attack and defence ratings for each team as of each date, from the Dixon-Coles
  model below.
- The existing faded goals and xG columns, last-5 form, and a home/away split.

**Teammates and opposition players (attention lists)**
- The player's share of team xG and xA over the last 10 (penalty takers, focal
  strikers).
- Competition for his place: same-position teammates with more recent minutes,
  his minutes rank in his position, whether a rival is injured or doubtful
  (their chance of playing).
- Squad totals: the team's attacking total from likely starters, and the
  opponent's defence (defenders' summed defcon, keeper).

**Last seasons:** fixed columns for last season and the season before
(seasons_before, played, points/90, xgi/90, minutes).

**External data (`learning_v2/external_inputs.py`)**
- `team()` / `player()` scalars as they are: schedule, player_load, double/blank
  flags, manager tenure.
- Upcoming list as counts: matches in the next 3, 7, 14 and 35 days by category
  (Europe, domestic cup, international), days to the next match, and whether the
  next match is a European knockout.
- Minutes in the last 7 and 14 days across all competitions.

**IDs**
- Team, opponent, manager and position as native LightGBM categorical columns.
- Leave the player id out: about 1,500+ values with few rows each would overfit,
  and his stats already identify him.

## Team ratings (Dixon-Coles), shared by LightGBM and the net

Added 2026-11-04. The v2 staged runs (B1 staged, `run_metrics/model_eval/b1_staged`)
size fixture swings well (slope 0.87, 1 is right) but rank them worst (corr
0.120 vs about 0.150 for B0 and v1). The likely cause is data: the scoreline
learns team strength from about 2,700 matches (7 seasons), not the 178k player
rows, and goals are noisy. A rating model with few parameters suits that.

- **What:** per team an attack and a defence rating plus home advantage (about
  60 numbers), with the Dixon-Coles low-score correction. Older matches count
  less (time decay, half-life to tune). Expected goals for a fixture are
  exp(attack + opposing defence + home).
- **Leak safety:** refit as of each deadline from matches before it only, like
  the rest of the as-of-deadline data.
- **Targets to try:** goals, FPL team xG (22/23+), or a mix. xG is far less
  noisy, so likely better.
- **Matches:** Premier League only first; then all competitions from the new
  match data (more matches, but cup and European opponents need their own
  ratings or a strength offset).
- **Module:** one standalone module (say `premierlearning/team_ratings.py`) with
  no TensorFlow, fast enough to refit every gameweek in seconds. It writes per
  team-fixture: expected goals for and against, clean-sheet chance, attack and
  defence ratings.

**Uses, in order:**
1. **Benchmark first:** score it on 2025/26 matches against the v2 staged
   scoreline (goal log-likelihood, correlation with match xG, fixture swing
   corr). This needs the staged model's scoreline predictions exported; check
   whether the prediction run saves them. If Dixon-Coles matches or beats the
   net's scoreline, the net's team side is not earning its place.
2. **LightGBM:** its columns go into every row (shape A), and its expected goals
   are the baseline for the team-goals stage of shape B, which then only learns
   corrections (rest, congestion, injuries, manager change).
3. **The v2 net:** the same columns as a team input (`team_features`) or as the
   scoreline's starting point, so the net learns corrections only. If the
   corrections are worthless the net can learn to leave them near zero.

**Not used:** bookmaker odds or other people's predictions as inputs. The user
rules them out: the models must make their own forecasts from match data.
Numbers straight from FPL (xG, FDR, chance of playing, prices) are fine.

The existing `premierlearning/team_expectancy.py` (v1) predicts team goals with
Ridge on raw form averages and FDR. It does not adjust for opponents, so it is
not a substitute.

## Model shapes to compare

- **A. Single model:** one LightGBM straight on points (Tweedie or plain
  regression). The baseline.
- **B. Staged,** mirroring the v2 `staged` structure (`learning_v2/staged.py`):
  - minutes: 0 / 1-59 / 60+ classifier
  - team goals: one row per team-fixture, Poisson, giving goals for, goals
    against and the clean-sheet chance
  - player goals and assists (Poisson)
  - bonus and defcon (regression)
  - combined with the FPL scoring rules, with monotone constraints on each part.

## Retraining every 5 gameweeks (queued 2026-11-05)

`run_retrain_every5.cmd` (PID 9324, behind the variants): `pick_lgbm.py` picks
the best single LightGBM from runs lgbm_v3 and lgbm_v3_variants and builds its
`_every5` twin; then v2_b1_every5, the B1 net refitted at GW1, 6, ..., 36
(about 26 h). Mid-season fits train on this season's played GWs too; the net
gets a fresh vocab and its team block 0 slides to the 38 GWs before each fit.
Run-ids: retrain_every5_lgbm (interim), retrain_every5 (final, with 50/50 mixes
on both schedules).

## Net knowledge in LightGBM: verdict (2026-11-05, Hal asked to be critical)

Raw embeddings as columns: **no**, for three separate reasons.
1. Leak: a net trained on the same rows has partly memorised their labels in
   its player and team vectors, so they look far more predictive in training
   than they are. Honest versions need walk-forward nets (one per training
   season, about 3 h each).
2. Coordinates mean nothing: each net fit rotates and flips its vectors, so
   coordinate 7 from one fit is not coordinate 7 from another. Walk-forward
   fits can't be stacked into one column.
3. Trees split one column at a time. A matchup in embedding space is a dot
   product (attacker type x defender type), which trees approximate badly
   with our data.

Better ways to get the same knowledge:
- Matchup columns built explicitly, as of each deadline: what the opponent has
  conceded to each position (FPL points, xG, xA to FWD / MID / DEF / GK over the
  last 5 / 10 / 38), and the player's share of team attack. These are
  tree-friendly and leak-free, and they are the interaction the embeddings were
  meant to find. Add to lgbm_v4.
- Player identity as an as-of residual: his points minus what his xG / xA /
  minutes imply, over his last 38 matches, shrunk toward 0 for few minutes
  (finishing, bonus and set-piece skill). The tree version of a player embedding.
- The net's predictions (points, minutes bands, team goals from the score head),
  not its vectors: a stable scale, so stacking works. It needs walk-forward net
  predictions for the training seasons; our sets only cover 23-25, so first
  try a learned blend (weights by position and minutes certainty) on held-out
  predictions, the cheap version of the 50/50 mix that already gains.

## Learned embeddings (later, only if LightGBM looks competitive)

- Feeding the net's embeddings or history summaries into LightGBM is possible,
  with two traps:
  - **Leaks:** a net trained on the same rows has memorised their points. It must
    be trained only on seasons before the one being predicted, which means extra
    net trainings at about 3.4 GB and hours each.
  - **Unstable coordinates:** each net retrain gives different embeddings, so
    LightGBM must be retrained with it.
- Cheaper ways to get most of the benefit:
  - learned but stable features like the Poisson team ratings
  - blending the net's prediction with LightGBM's output afterwards.

## Open decisions

1. Build A and B, or A first?
2. Player id left out (suggested)?
3. Net embeddings: later (suggested), or from the start?
4. Dixon-Coles fitted to goals, xG or both? Premier League only, or all
   competitions?

## Cost

- About 200k rows x 200 columns, about 160 MB in memory.
- Minutes to train on the CPU once the machine is free.
- `pip install lightgbm` into the venv (Python 3.8 wheels exist), or use
  sklearn's HistGradientBoosting.
