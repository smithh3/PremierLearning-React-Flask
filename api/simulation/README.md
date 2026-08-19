# Historical prediction cache and replay

Run commands from `api/` with the project virtual environment active.

Build (or resume) the shared strict point-in-time cache:

```bash
python -m simulation.prediction_builder --seasons 23,24,25 --linear
```

The builder commits one season/GW snapshot at a time. Re-running the same
command skips complete snapshots and rebuilds failed or partial snapshots.
Use `--coverage-only` to print the coverage report.

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

- Weekly injury/status snapshots are unavailable. Predicted minutes are the
  availability proxy, while only dated, definitely unavailable players are
  excluded. Reports count zero-minute starters/captains, affected GWs and
  successful autosubs so this limitation stays visible.
- Fixtures use the final season calendar. A replay can therefore know a later
  reschedule earlier than it was announced.
- Hits and chips are intentionally not included in this baseline.
