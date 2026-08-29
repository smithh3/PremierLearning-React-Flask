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
- Hits and chips are intentionally not included in this baseline.
