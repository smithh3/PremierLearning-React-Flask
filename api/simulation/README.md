# Historical prediction cache and replay

Run commands from `api/` with the project virtual environment active.

Build the shared strict point-in-time cache:

```bash
python -m simulation.prediction_builder --seasons 25 \
  --train-once-per-season \
  --name strict-historical-25-26-adamw-once
```

Each launch replaces the previous historical cache (predictions and replay
runs) and rebuilds from scratch. Only the latest set is kept. Replay with no
`--prediction-set-id` uses that set. Use `--coverage-only` to print coverage
for the current cache without rebuilding.

This mode fits the normal Premier Learning AdamW team, fixture-points, minutes
and horizon-5 models once from data available before 2025/26. Every GW then
rebuilds causal form features from results through the prior GW and reuses those
same fitted weights. The horizon model is fitted once; “5” is its GW window,
not a repetition count.

Each snapshot predicts and stores every remaining player fixture from its
deadline through GW38. `--horizon` on the replay only selects how much of that
same point-in-time snapshot the transfer optimiser scores; it never reads a
later GW snapshot to fill the horizon.

The builder's `--horizon` controls the near-term multi-GW calibration model
(default five GWs), not how many future fixtures are persisted. Later fixtures
still receive point-in-time per-fixture points/minutes predictions.

Replay all three seasons with the default zero-gain thresholds:

```bash
python -m simulation --prediction-set-id "$PREDICTION_SET_ID" --seasons 23,24,25
```

Run the example threshold experiment:

```bash
python -m simulation --prediction-set-id "$PREDICTION_SET_ID" --seasons 23,24,25 \
  --min-current-gain 2 --min-horizon-gain 4
```

Prediction snapshots and simulation results are durable database tables and are
not removed by the live pipeline reset. Replay summaries are also written under
`api/run_metrics/simulations/`.

Historical caveats:

- Each player-fixture row stores predicted pp90, expected minutes, and
  chance-to-play (Randdalf/fplcache at that deadline). Squadtools applies
  chance when scoring the linear optimiser, so availability policy can change
  without rebuilding the cache. Reports still count zero-minute
  starters/captains, affected GWs and successful autosubs.
- Fixtures use the final season calendar. A replay can therefore know a later
  reschedule earlier than it was announced.
- Players who had barely featured by the deadline are not predicted: no
  minutes in the last five GWs and, over last season plus this season so far,
  3 or fewer appearances or 20 or fewer points. Players with no last-season
  record are predicted for the first five GWs. Until 2026-10 this rule used
  whole-season totals (look-ahead); sets built before then still carry it.
  2022/23 has no CSV export, so every 2023/24 player is predicted for GW1-5.
- Hits and chips are intentionally not included in this baseline.

## Model evaluation (v2 against v1)

`simulation.model_eval` builds one prediction set per experiment through the
builder above, then scores every v2 variant against v1 on the same snapshots.

```bash
python -m simulation.model_eval list
python -m simulation.model_eval build-all --seasons 25   # smoke-test season first
python -m simulation.model_eval evaluate --seasons 25
python -m simulation.model_eval build-all                # 23, 24, 25
python -m simulation.model_eval evaluate
```

Experiments (`model_eval/experiments.py`): `v1` (single GW, the baseline),
`v1_h5` (v1 as used live, with the 5-GW total model), `v2_huber`, `v2_tier` (the
current blended bands), `v2_smooth`, and `v2_huber_seed2`. `v1_cal` and
`v2_smooth_cal` are derived at evaluation time, not built. Every built experiment trains once per
season at the GW1 cutoff; v2 uses one fit (three ran out of 8 GB RAM). Pass `--learning-v2` to the
builder directly for one-off v2 caches; v2 settings are part of the cache hash,
so v1 and v2 never share a set. Experiment sets are named `eval-*` and are
never picked as the "latest" set, so replays and the fantasy manager run without
`--prediction-set-id` keep using the cache they used before.

Scoring, all in `model_eval/config.py`:

- Eligible players are fixed at each deadline and shared by every model:
  minutes in the previous five gameweeks, or 900+ last-season minutes while the
  season is under five gameweeks old, less anyone the builder skips at that
  deadline (above). A missing prediction counts as 0.
- v1 is scored as the optimiser sees it (pp90 x minutes x chance-to-play); v2's
  raw points stand, since chance-to-play is one of its inputs. v2 training rows
  carry the FPL flag published before their own deadline (fplcache, 2021/22 on,
  plus a `chance_known` input for older rows). v2 sets built before that saw a
  constant flag and are scored with chance applied, as for v1.
- The live pipeline saves each deadline's flags to
  `data/season_XX_YY/chance_of_playing/gwN.json` (local runs only; Fargate's
  data folder is not kept), so future seasons need no backfill.
- Overall (25%): single-GW MSE. MAE is reported, not scored: it rewards
  predicting the median, and the optimiser needs the mean.
- Future high performers (25%): bucketed 5-GW MAE (2A, 60%) and next-GW MAE
  (2B, 40%), bucket weights 5/10/15/30/40 up to the top five players. Half the
  weight uses buckets by realised 5-GW points, half by the model's own 5-GW
  prediction, so neither inflating nor shrinking the top end pays.
- Calibration (15%): mean gap over five equal-count bins of predicted points.
- Ranking (35%): mean per-GW Spearman and per-origin 5-GW Spearman.
- Every component is scored as improvement over v1 divided by v1's
  block-bootstrap standard deviation for that metric, so the weights mean what
  they say. Each metric is capped at +-3 of those units (`score_cap`), so a
  metric v1 barely varies on, like the calibration gap, cannot swamp the rest;
  the uncapped total is still reported. Totals come with a bootstrap 95% interval; `v2_huber` vs
  `v2_huber_seed2` is the training-noise floor.

Results land in `api/run_metrics/model_eval/<run_id>/`; start with
`comparison.txt`. Set `PL_MODEL_EVAL_DIR` (and `DATABASE_PATH`) to keep a trial
run away from the real manifest and cache.
