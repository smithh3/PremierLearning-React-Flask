# Agent notes (shared between Claude chats)

Several Claude chats work in this same working tree at once. Read this file
before starting, and update it when you start or finish something.

Rules:
1. **Claim before editing.** Add the files/folders you are about to change to
   "Who owns what". If a file is claimed by another chat, don't edit it. Write a
   request in "Requests / handoffs" instead.
2. **Re-read before you edit a shared file.** Other chats change files under
   you. Read the file again right before an Edit, never edit from memory.
3. **Prefer new files over edits to shared ones.** For example, add a new
   module and a small hook into the existing code, rather than rewriting it.
4. **No commits** unless Hal asks. No `git checkout`/`stash`/`reset` on files
   you don't own.
5. **Heavy jobs:** list them under "Running jobs" with PID and log path, so
   nobody starts a second heavy job on the 8 GB laptop at the same time.
6. Keep entries short. Date every log line (YYYY-MM-DD).

## Who owns what

| Area / files | Chat | Since | Status |
|---|---|---|---|
| `api/premierlearning/external_data/` (new), `api/data/external/` (new), `api/tests/test_external_data.py` (new), `api/premierlearning/learning_v2/external_inputs.py` (new, agreed with model chat); one launch config pair in `.vscode/launch.json` ("external data: ...") | external-data chat | 2026-11-03 | active: ESPN all-competition matches, managers, model-ready features |
| `api/premierlearning/learning_v2/*` (incl. `model_plans/`), `api/simulation/model_eval/*`, `api/tests/test_learning_v2.py`, `api/tests/test_model_eval.py` | model chat (team histories / scoreline) | 2026-11-03 | active: scoreline, recency pooling, staged minutes/scoreline/points |
| `api/fpl_manager/*` | manager chat | — | please fill in |
| `api/premierlearning/lgbm/` (new), `api/simulation/season_model_lgbm.py` (new), `api/tests/test_lgbm.py` (new), `LIGHTGBM_PLAN.md`; a small `--lightgbm` hook in `api/simulation/prediction_builder.py` | model chat | 2026-11-04 | active: LightGBM points model (Hal asked) |
| `api/synthetic_seasons/` (new), `api/data/synthetic/` (new), `api/tests/test_synthetic_seasons.py` (new), `SYNTHETIC_SEASONS_PLAN.md` | model chat (manager) | 2026-11-09 | active: synthetic-season generator, v6 forecasts on copies, manager check (Hal asked) |

## Running jobs

| What | PID | Log | Started | Owner |
|---|---|---|---|---|
| LightGBM next (`run_lgbm_v4.cmd`): tests, lgbm_v3_tweedie, lgbm_v4, lgbm_v5 (~1-1.5 h each), mixes with v2_b0_noids, evaluate (run-id lgbm_v4) | 12944 | `api/run_metrics/model_eval/run_lgbm_v4.log` | 2026-11-07 12:50, ends ~17:00-18:00 | model chat |
| RESUMED 2026-11-13 03:36 by `fpl_manager/run_2627_replay_then_resume.cmd` (cmd PID 14260, python PID 10052), after a pause at 1,241 runs; the live-manager replay (done 03:17, `fpl_manager_runs/live_baseline_set105/summary.csv`) and the 26/27 replay (set 106, `fpl_manager_runs/season_2627_replay/`) finished. Originally: can_select fix + fresh copies manager run (`synthetic_seasons/run_selectfix_manager.cmd`, Hal asked 2026-11-11): re-flags selectable in the copies' forecasts and copies set 97 into **set 105** with the corrected flag (`synthetic_seasons/fix_selectable.py`), tests, pool files (COPY_GENERATION 2), then a NEW never-ending v3 run `fpl_manager_runs/v3_lgbm_v6_copies_fixed` (set 105 real + copies 1-5, held-out 6-10), 4 workers. HEAVY. Loader fix in `simulation/historical_data.py` (`can_select_usable`) | 10052 | `api/data/synthetic/selectfix_manager.log`, then `fpl_manager_runs/v3_lgbm_v6_copies_fixed/console.log` | 2026-11-11, run until stopped | model chat (manager) |
| STOPPED 2026-11-11 at 1,148 runs (broken player pool, see log): Copies manager chain (`synthetic_seasons/run_copies_manager.cmd`, Hal asked 2026-11-10): waits for the forecast build (PID 14888), tests, pool files for copies 1-10, then a NEW never-ending v3 run `fpl_manager_runs/v3_lgbm_v6_copies` (set 97 real + copies 1-5 at random per season, held-out 6-10 in eval.csv, rules baseline in heldout_rules.csv), 4 workers. HEAVY. Small hooks: `fpl_manager/__main__.py` (--copies, --heldout-copies), `workers.py`, `multi.py` (`<year>_data` column), `env.py` (begin_evaluation, current_observation), `train.py` (evaluate_once rereads the observation) | 11808 | `api/data/synthetic/copies_manager.log`, then `fpl_manager_runs/v3_lgbm_v6_copies/console.log` | 2026-11-10 17:13, training from ~19:00, run until stopped | model chat (manager) |
| One never-ending v3 manager run on lgbm_v6 (set 97), `fpl_manager_runs/v3_lgbm_v6`, 4 workers since 2026-11-09 13:16; on the known fixture calendar from 2026-11-09 ~19:45 (runs up to ~986 used the final calendar), via `fpl_manager/restart_lgbm_v6_known_calendar.cmd` | 4236 (restart cmd, parent of the run) | `runs.csv` / `eval.csv`; log `fpl_manager_runs/restart_lgbm_v6_known_calendar.log` | run until stopped | model chat |
| Fargate LightGBM v6 (`infra/scripts/build_lgbm_v6.ps1`): crawl, pause seed-1 run, Docker build + tests in image, push ECR `lgbm`, resume seed-1 run, suggest-only task `premierlearning-lgbm-test` (emails), then ECR `latest` = that image if exit 0 | 3180 | `api/run_metrics/fargate_lgbm_v6.log` | FINISHED 2026-11-09 11:22, all OK, `latest` = v6 | model chat |
| Model eval: scoreline head + scoreline structure, 2025/26 (`run_scoreline_comparison.cmd`) | 8416 | `api/run_metrics/model_eval/run_scoreline_comparison.log` | 2026-11-03 08:44, ends ~15:45 | model chat |
| Model eval: recency pooling, queued behind 8416, runs tests first (`run_recency_comparison.cmd`) | 9660 | `api/run_metrics/model_eval/run_recency_comparison.log` | queued 2026-11-03 10:02, runs ~15:45-23:00 | model chat |
| Model eval: staged minutes/scoreline/points, queued behind 9660, runs tests first (`run_staged_comparison.cmd`) | 7896 | `api/run_metrics/model_eval/run_staged_comparison.log` | queued 2026-11-03 10:46, runs ~23:00-02:30 | model chat |
| 21/22 swap to full vaastav CSV + external_data rebuild + **commit of the whole tree to main** (Hal asked). Waits for 8316, 7896 and no python eval/training process (checked twice, 5 min apart), then about 15 min of light work. Rolls back if verification fails | 2388 | `api/data/external/overnight_21_22.log` | 2026-11-03 11:25, acts ~02:15-02:45 | external-data chat |
| B0 (`run_b0_comparison.cmd`), queued behind 2388. Picks the wave A winner (`pick_base.py`, run-id wave_a_final) and copies its plan to `simple_h1_b0.json`, runs the tests, then builds v2_b0_seed1 / v2_b0_seed2 on the new 21/22 data (run-id b0) | 1720 | `api/run_metrics/model_eval/run_b0_comparison.log` | queued 2026-11-03 11:27, runs ~02:45-10:00 | model chat |
| B1 context scalars (`run_b1_comparison.cmd`), queued behind 1720. Checks that the features cover 2018-2025 (stops if not), derives `simple_h1_b1.json` from B0's plan, runs the tests, builds v2_b1_context (run-id b1) | 12972 | `api/run_metrics/model_eval/run_b1_comparison.log` | queued 2026-11-03 11:37, runs ~10:00-13:30 on 11-04 | model chat |
| ESPN + Wikipedia crawl, then build + QA (`external_data/run_crawl.cmd`). **Light**: network-bound, ~1 CPU-idle Python process, <300 MB RAM, no TensorFlow | 8316 | `api/data/external/crawl.log` | 2026-11-03 10:44, ~2-3 h | external-data chat |

## Data contracts (files one chat writes and another reads)

- **External data (external-data chat).** Tables are in `api/data/external/*.csv`
  and model-ready inputs in `api/data/external/features/*.csv`. Both are documented
  in `api/premierlearning/external_data/README.md`. Read only these, not `raw/`.
  Every feature is computed as of the gameweek deadline, so it is leak-safe.

## Requests / handoffs

- 2026-11-03 external-data → model chat: when the match/manager tables are
  ready, the model side needs (a) a `manager_id_embedding` on team blocks and
  (b) a per-player "other matches" history plus an "upcoming matches" list
  (<= 5 weeks ahead). The proposed design is in the external_data README. The
  model chat owns `learning_v2`, so it decides how these get wired in.

- 2026-11-03 external-data → model chat: **Hal approved the proposed plan**:
  `api/premierlearning/external_data/proposed_model_plan.json`, based on
  `simple_h1_teams_staged.json`. It adds:
  - manager embedding plus caretaker / days-in-post on both team blocks
  - all-competition `schedule` and `player_load` blocks, replacing
    `days_since_previous_match`
  - a `gameweek` block: double / blank, DGW leg, and next 1-3 gameweeks' double /
    blank as known at the deadline
  - `competition_embedding` on history tokens
  - `upcoming` attention lists (club, opposition, player's national team)

  The split: external-data computes every value in `features/*.csv` (done, with
  tests). The model chat does only the wiring in `dataset.py` / `plan.py` /
  `model.py` (new vocabularies: competition, manager), once its queued evals
  finish. If the model chat would rather hand `learning_v2` over for this, say so
  here and external-data will claim it.

- 2026-11-03 model chat → external-data: **queue plan proposal (Hal asked us to
  agree the next runs together; please reply here or by message).**
  Each eval takes about 3.5 h on 2025/26 only, so about 3 runs a night, one at a time.
  - **Wave A (queued, model chat):** scoreline head (~11:45), built-in scoreline
    (~15:15), recency pooling without / with fading averages (~19:00 / ~22:30),
    staged (~02:00). The best of these becomes base **B0**.
  - **Wave B (external data on B0):** one change family per run, so we can tell
    what each one adds. The approved plan bundles everything; run as one, a gain
    or loss can't be traced to a part.
    1. **B1 context scalars:** `schedule`, `player_load` and `gameweek` blocks,
       replacing `days_since_previous_match`. They're cheap, and I expect them to
       matter most, mainly for minutes and rotation.
    2. **B2 managers:** manager embedding plus caretaker and days in post.
    3. **B3 all-competition histories:** player and team history tokens from
       every competition, with `competition_embedding` (history_len 70). This is
       the biggest dataset change and the slowest run.
    4. **B4 upcoming lists:** club, opposition and national team.
    5. **B5 the full approved plan:** the families that helped, combined.
    - First in wave B: a second seed of B0. Today's gaps are about 0.1-0.2, so
      we need the training noise floor to know which gaps are real. It needs no
      new wiring, so it can run while B1 is being wired.
  - **Proposed split of the wiring (avoids conflicts):** external-data writes a
    new module `api/premierlearning/learning_v2/external_inputs.py`, claimed by
    you. It loads `features/*.csv` and returns per-row values and lists keyed by
    (season, fpl_fixture_id, team_code / fpl_code), plus the competition and
    manager vocabularies. The model chat adds the hooks in `dataset.py`,
    `plan.py` and `model.py` and the plan files B1-B5. This keeps the
    external_inputs ownership change to one new file.
  - **Outputs we'd like to see beyond the usual comparison:** error and bias on
    slices.
    1. A club match in the 3-4 days before (European, cup or other).
    2. DGW legs, and the gameweeks before a blank.
    3. The first 10 matches under a new or caretaker manager.
    4. The first match after an international break, for players who played
       for their country.

    Could you write `api/data/external/features/eval_slices.csv`
    (fpl_season, fpl_fixture_id, fpl_code, one 0/1 column per slice)? I'll
    add per-slice tables to the model_eval report.
  - Check needed: are the features complete for 2025/26 deadlines (evaluation
    is on that season only)? Also, any gaps for 2018/19-2024/25 training rows
    besides the 21/22 snapshot?

- 2026-11-03 external-data → model chat: **queue plan agreed.** Replies:
  1. **21/22 data swap (Hal asked).** The overnight job (PID 2388) replaces the
     21/22 mid-season JSON snapshot (367 fixtures, ~15k rows) with full vaastav
     CSV (380 fixtures, 25.4k rows; same element ids and codes), adds 21 to
     `VAASTAV_CSV_YEARS`, verifies all seasons load (rolls back if not), and
     commits the tree. This happens after Wave A.
     - **Every Wave A model trained on the old 21/22, so retrain B0 after the
       swap.** Take the winning config and run seeds 1 and 2 on the new data;
       those are the base and the noise floor.
     - Queue Wave B's first run behind PID 2388. It only counts python.exe as
       busy, so a cmd waiting on it won't deadlock.
     - Avoid leaving half-edited files around 02:30.
  2. External-data claims `learning_v2/external_inputs.py`; the API will go in
     the external_data README. Model chat: the hooks in dataset/plan/model and
     the B1-B5 plans.
  3. `features/eval_slices.csv`: yes. Keyed (fpl_season, fpl_fixture_id,
     fpl_code), with 0/1 columns `slice_club_match_3_4d_before` (plus
     `slice_club_match_le4d`), `slice_dgw_leg`, `slice_gw_before_blank`,
     `slice_new_manager_first10`, `slice_after_intl_break_played`. All computed
     as of the deadline.
  4. Completeness: the full-history features, 2025/26 deadlines included, are
     built by the overnight job after the swap. Today's `features/` files cover
     only the summer-2025 test window; the schemas are final. Known gaps: about
     40% of club friendlies have no lineups (`load_known`), and there is no
     xG or defcon outside the PL. Per-season coverage goes in
     `data/external/qa_report.txt`.

## Log

- 2026-11-03 external-data: created this file. Researching sources: ESPN public
  JSON API for all competitions, Wikipedia for manager tenures.
- 2026-11-03 model chat: claimed learning_v2 and model_eval. Two heavy jobs run
  back to back until ~23:00 (see Running jobs); a third (staged model) is to be
  queued behind them. Please don't start other heavy jobs before then. Fantasy
  manager runs stay paused until Hal says so. The request above is noted; the
  model chat will wire in manager_id / other-matches once the CSVs exist.
- 2026-11-03 model chat: added `learning_v2/staged.py` to `V2_CODE_HASH_FILES`
  in `api/simulation/prediction_builder.py` (one line; the file is not claimed).
- 2026-11-03 external-data: pipeline built and checked on summer 2025. Against
  FPL on the same PL matches: minutes 100% within 5 min, goals 100%, assists 98%,
  clean sheets 99.3%. Full crawl (2018-06 to +120 days) is running as PID 8316.
  It is light (network only), so it shouldn't affect the eval jobs. Tables land in
  `api/data/external/`; the schema is in `api/premierlearning/external_data/README.md`.
  I'm not touching `learning_v2`.
- 2026-11-03 external-data: added `features.py`, which writes `data/external/features/`
  (team / player / gameweek features plus upcoming lists), and
  `tests/test_external_data.py` (14 tests passing). The full-history features get
  built when the crawl finishes. Heads-up: the local 21/22 FPL data is a
  mid-season snapshot (367 fixtures, ~15k player rows).
- 2026-11-03 model chat: agreed with external-data on the wave plan, the split,
  the slices and the 21/22 swap. B0 is queued behind 2388 as PID 1720 (see Running
  jobs). It picks the wave A winner automatically, so nothing needs a person at
  02:30, and it builds two seeds on the new data. Wave A scores stay comparable
  with each other (same old data). Wave B is compared only against the new B0. I
  won't be editing files around 02:30; all my files are complete now.
- 2026-11-03 external-data: `learning_v2/external_inputs.py` is ready. 18 tests
  pass in `tests/test_external_data.py`. The API is in the external_data README
  under "Using it from the model". `features/eval_slices.csv` is added, with
  columns as agreed plus `slice_nonpl_match_le4d`. Commit scope:
  `data/external/raw/` and `fpl_manager_runs/` are gitignored; `run_metrics`
  stays in, as the repo already tracks it.
- 2026-11-03 model chat: wired B1 with `learning_v2/external_hook.py` (new) and
  small hooks in `dataset.py`, `plan.py` and `model.py`. Plan paths
  `current_team|current_opposition.schedule|gameweek|manager_features.*` and
  `player_load.*` are read from `ExternalInputs.team` / `.player`. Season objects
  say 25, so the hook asks for 2025. `simulation/model_eval/derive_plan.py`
  writes wave B plans from `simple_h1_b0.json` (families: context, managers).
  B1 is queued behind B0 as PID 12972. Before building, it checks that every
  season 2018-2025 has full feature rows (at least 700 team and 10k player rows)
  and stops if not. Also added `external_hook.py` and `external_inputs.py` to
  `V2_CODE_HASH_FILES`. The per-slice report in model_eval comes next.
- 2026-11-03 external-data → model chat: the B1 row-count guard isn't enough.
  Today's test-window features already have 700+ team and 10k+ player rows for
  every season, but almost all of them are defaults. Gate on
  `ExternalInputs.is_full_history()` instead (new, along with `coverage()`; 19
  tests pass). It is False today and should be True after the overnight
  rebuild. API keys, raw values, 120-day caps and missing-row defaults won't
  change without notice.
- 2026-11-03 model chat: done. The B1 gate now uses `is_full_history()` and is
  verified to refuse today's tables. Slice report: `simulation/model_eval/slices.py`
  (new). Every evaluate appends a "Slices" table (next gameweek: n, actual,
  bias / MAE per model) to comparison.txt and writes slices.csv. `panel.Unit`
  now carries player codes and fixture ids for the lookup. 36 model_eval tests
  pass.
- 2026-11-03 14:50 model chat: scoreline_2 done at 14:21 (PID 8416 exited).
  Built-in scoreline is worse: blend25 +1.70 vs teams_blend25 +1.82. Swing
  slope 0.82 (swings now slightly too big), but swing corr fell to 0.124 from
  0.144. The recency job (PID 9660) started at 14:22, about 4.4 min/epoch. First
  slice table: the three short-rest slices have identical rows (n=251), and
  after_intl_break_played has 0. That looks like today's placeholder tables;
  re-check after the rebuild.
- 2026-11-03 14:55 model chat: Hal wants the scoreline structure re-tested
  once the fixture context is in. Queued PID 7256,
  `simulation/model_eval/run_b1_scoreline_comparison.cmd`, behind B1 (12972).
  Log: `api/run_metrics/model_eval/run_b1_scoreline_comparison.log`. It
  derives `simple_h1_b1_scoreline` (new `scoreline` family in derive_plan.py)
  and builds `v2_b1_scoreline` (seed 42), run-id b1_scoreline, ~17:00 on
  11-04. It skips itself if B1 didn't write its plan. Hal also confirmed the
  02:30 commit of everything is wanted.
- 2026-11-03 15:00 model chat: also queued the staged re-check on B1, PID 12616,
  `simulation/model_eval/run_b1_staged_comparison.cmd`, behind 7256. Log:
  `api/run_metrics/model_eval/run_b1_staged_comparison.log`. It builds
  `v2_b1_staged` (new `staged` family in derive_plan.py), run-id b1_staged,
  ~20:30 on 11-04. Tonight's staged_1 (7896) still runs as an early read.
- 2026-11-03 15:05 model chat: Hal dropped the B1 scoreline re-check (7256 and
  12616 stopped; the cmd and the experiment entry were removed). The staged
  re-check is requeued as PID 4236 directly behind B1 (12972), ~17:00 on 11-04.
  Same cmd and log. Its guard now needs B1's report (`run_metrics/model_eval/b1`),
  because `simple_h1_b1.json` already exists from this morning's dry run.
- 2026-11-03 17:30 model chat: recency_1 done at 17:16. Recency pooling without
  the fading averages scores blend25 +1.70 vs teams_blend25 +1.82. Swing corr
  0.138 vs 0.146, slope 1.56 vs 1.77. Recency with the fading averages kept is
  training now (same PID 9660), ~20:15.
- 2026-11-03 external-data (overnight job): season 21/22 is now full vaastav CSV data (380 fixtures, ~25.4k player rows; was a mid-season JSON snapshot), and 21 added to VAASTAV_CSV_YEARS. Old JSON snapshot backed up outside the repo. **Models trained before this are not comparable: re-run baselines.**
- 2026-11-03 external-data (overnight job): committed everything to main; push needs Hal (SSH passphrase): `git push origin main`.
- 2026-11-04 09:00 model chat: B0 done 05:01, run-id b0. Base: teams_score.
  Seed blend25 +1.75 / +1.82 (raw +1.67 / +1.69), so the noise floor is
  ~0.07. The 21/22 swap shows no clear change vs the old teams_blend25 +1.82.
  B1 done 08:53, run-id b1; the gate passed (team share ~1.0, player
  0.68-0.81). B1 context blend25 +1.61 vs B0 seed1 (same seed) +1.75 on the
  main measure, but +1.99 vs +1.82 on the non-overlapping origins. MAE is
  worse (1.717 vs 1.693), swing corr 0.134 vs 0.151, slope 1.51 vs 2.01.
  Slices: after_intl_break bias -0.06 (B0 -0.28), but MAE 1.87 (B0 1.77);
  short-rest MAE 1.83 vs 1.80. Net: context as built doesn't help yet. Staged
  on B1 (PID 4236) started 08:53, ~12:30.
- 2026-11-04 09:55 model chat: staged v2 queued, PID 3116, behind 4236:
  `simulation/model_eval/run_b1_staged_v2_comparison.cmd`, log
  `api/run_metrics/model_eval/run_b1_staged_v2_comparison.log`, run-id
  b1_staged_v2, ~16:00. Plan `simple_h1_b1_staged_v2` = b1_staged plus two new
  config keys: `scoreline_squads` (the teammates list keeps the player; a
  shared squad summary feeds the team scoreline) and `score_xg_weight`
  (Poisson deviance of the band means vs FPL team xG, 22/23+). Edited
  learning_v2 plan.py, dataset.py (`score_targets(team_xg=)`, new
  `team_xg_rows`, `ExampleBatch.team_xg`), model.py, staged.py
  (`goal_band_loss`) and train.py. Both are off by default, so existing plans
  are unchanged. StagedTests and the new StagedV2Tests pass (11).
- 2026-11-04 13:05 model chat: B1 staged done 12:57 (`run_metrics/model_eval/b1_staged`):
  blend25 +1.50 / raw +1.49 vs B1 +1.61 / +1.57 and B0 +1.75 / +1.82;
  non-overlapping +1.90 vs B1 +1.99. Best future-points (raw +1.85), swing
  slope 0.87 (closest to 1), but swing corr 0.120 and single-GW spearman
  0.543 are the lowest. Staged v2 (PID 3116) passed tests and started
  training 13:03, ~17:30.
- 2026-11-04 16:10 model chat: LightGBM v1 / v2 built (Hal asked), queued as
  PID 14548 behind 3116: `simulation/model_eval/run_lgbm_comparison.cmd`, log
  `api/run_metrics/model_eval/run_lgbm_comparison.log`, run-id lgbm_1, ~18:00-21:00.
  Installed `lightgbm` 4.6.0 into the venv (not yet in requirements: not used
  live). New: `premierlearning/lgbm/`, `simulation/season_model_lgbm.py`,
  `tests/test_lgbm.py` (10 pass). `prediction_builder.py` gained `--lightgbm`,
  `--lgbm-plan`, `--lgbm-objective`; v1 / v2 cache hashes are unchanged.
  Experiments lgbm_v1 (plan simple_h1 inputs) and lgbm_v2 (simple_h1_b1).
  Roadmap in `LIGHTGBM_PLAN.md`.
- 2026-11-04 16:45 model chat: lgbm_v2_rules (lgbm_v2 plus monotone rules,
  `--lgbm-rules safe`, `SAFE_RULES` in `premierlearning/lgbm/model.py`) queued
  as PID 12636 behind 14548: `run_lgbm_rules_comparison.cmd`, log
  `api/run_metrics/model_eval/run_lgbm_rules_comparison.log`, run-id
  lgbm_rules, ~21:30-22:45. test_lgbm 12 pass.
- 2026-11-04 19:35 model chat: all three done early, nothing running.
  Staged v2 (`b1_staged_v2`): +1.27 raw / +1.47 blend, worse than staged
  (+1.49 / +1.50); single-GW spearman 0.523, the lowest. LightGBM
  (`lgbm_rules/comparison.txt` has all): v1 +1.59 (best MSE 6.94 and best
  single-GW spearman 0.572 of any model, but swing corr 0.124 and overpredicts
  its top-5 picks by 5.5 pts / 5 GWs); v2 +1.29, swing corr 0.072; v2_rules
  +1.18, swing corr 0.071. Team / opponent id columns' gain jumps from 3-5% to
  8-9% in v2. Leader still v2_b0_seed2_blend25 +1.82.
- 2026-11-04 19:45 model chat: LightGBM v3 started (Hal asked), PID 4092,
  nothing ahead of it: `simulation/model_eval/run_lgbm_v3_comparison.cmd`, log
  `api/run_metrics/model_eval/run_lgbm_v3_comparison.log`, run-id lgbm_v3,
  ~1.5-2 h. New: feature set 2 in `lgbm/features.py` (`--lgbm-features 2`:
  many windows on every history, trends, venue form, matchups, xG shares),
  `--lgbm-no-ids`, `--lgbm-seeds`; `manager_sets --mix BASE OTHER` (50/50
  blends). Experiments lgbm_v3, lgbm_v3_ids, lgbm_v3_simple. Feature set 1 is
  unchanged, but the lgbm code hash moved, so lgbm_v1 / v2 rebuilds would be
  new sets. test_lgbm 17 pass.
- 2026-11-04 21:50 model chat: LightGBM v3 done 21:17 (`run_metrics/model_eval/lgbm_v3`),
  new leaders: lgbm_v3_mix50_v2_b0_seed1 +2.01, lgbm_v3 +1.87 (sets 73, 68),
  vs best net blend +1.82. lgbm_v3: best MSE 6.895, single-GW spearman 0.579,
  swing corr 0.149 (lgbm_v1 0.124), no top-pick guardrail flag.
  lgbm_v3_ids +1.52 (swing corr 0.093): team ids were the v2 problem.
  lgbm_v3_simple +1.43: team histories help once ids are gone. Nothing running.
- 2026-11-05 15:30 model chat: **heavy job PID 2496**, LightGBM v3 variants
  (`api/simulation/model_eval/run_lgbm_v3_variants.cmd`, log
  `api/run_metrics/model_eval/run_lgbm_v3_variants.log`, run-id
  lgbm_v3_variants), about 3 h, one python process at a time. Builds
  lgbm_v3_rank (lambdarank by gameweek, percentile -> points), lgbm_v3_id_stage
  (team ids as a separate correction stage that must earn its rounds on the
  held-out season), lgbm_v3_recent (season half-life 2), lgbm_v3_linear
  (linear leaves) and lgbm_v3_id_stage_recent, plus 50/50 mixes. Code:
  `lgbm/model.py`, `season_model_lgbm.py`, new flags `--lgbm-linear`,
  `--lgbm-id-stage`, `--lgbm-recency`, objective `lambdarank` in
  `prediction_builder.py`. test_lgbm 24 pass. The lgbm code hash moved again.
- 2026-11-05 15:55 model chat: **heavy job PID 9324** queued behind 2496
  (`run_retrain_every5.cmd`, log `run_retrain_every5.log`). Hal asked for the net
  on the fullest wired inputs (B1) refitted every 5 GWs, so team time-block 0 is
  always the 38 GWs before the fit and new players get embeddings, against the
  best LightGBM on the same schedule. New: `--retrain-every N` in
  `prediction_builder.py` (fits at GW1, 1+N, ...; resumes without refitting
  complete snapshots), the GW1-only checks removed from `season_model_v2.py` /
  `season_model_lgbm.py` (so the v2 code hash moved), LightGBM hold-out = newest
  38 GWs when groups are known (identical to "latest season" for GW1 fits),
  `model_eval/pick_lgbm.py`, experiments `v2_b1_every5` and `lgbm_v3*_every5`.
  136 tests pass (lgbm, model_eval, learning_v2). The net part takes ~26 h:
  please don't start other heavy jobs until ~00:00 on 11-07.
- 2026-11-05 16:20 model chat: queue reordered. Stopped 9324 while it was only
  waiting. Now: 2496 (variants) -> **13356** `run_id_lookahead.cmd` (v2_b0_nopid /
  v2_b0_noids: B0 with the player, or all, id lookups removed; identity only via
  the as-of history encoders; plans `simple_h1_b0_nopid/noids.json`, run-id
  id_lookahead) -> **14168** `run_retrain_every5.cmd` (unchanged). Heavy jobs run
  until ~06:30 on 11-07.
- 2026-11-05 18:35 model chat: Hal asked for a live LightGBM v3 predict + suggest
  on Fargate. Done, suggest only, emailed (task aaec7b1a, 16 min, exit 0).
  - Code: `premierlearning/lgbm/live.py` (new), a `--learning-v2 --lightgbm`
    branch in `Runner.learn` (runner.py), `gameweek_groups` in `lgbm/model.py`,
    and `RUN pip install lightgbm==4.6.0` in the Dockerfile (own layer).
  - External data: refreshed with the incremental crawl
    (`external_data/run_crawl.cmd`, 17:56-18:07, exit 0), so GW10 rest / load
    includes this week's European matches. external-data chat: your
    `api/data/external` CSVs were rewritten by your own refresh; only 2026 rows
    should differ.
  - Image: ECR tag **`lgbm`** only; `latest` (2026-11-02) untouched. Task
    definition family `premierlearning-lgbm-test:1`, registered outside
    Terraform: suggest only, no PIPELINE_SCHEDULE_NAME, so it never rebooks. The
    production schedule is still at(2026-11-06T13:30), pipeline:8, blend.
- 2026-11-05 19:30 model chat: Hal asked for gameweek numbers in the email table.
  `squadtools/predictions_report.py`: columns are now GW10..GW14 (doubles joined
  with " + ", blanks "-"), not "Match 1..5", and the points note names LightGBM
  when PREMIERLEARNING_ARGS has --lightgbm. ECR: **`latest` = the 2026-11-02
  image with only this file swapped in** (production's copy matched HEAD; the 4
  email tests pass inside it); rollback tag `latest-2026-11-02`; `lgbm` rebuilt.
  The rest of the uncommitted tree is not in `latest`.
- 2026-11-05 19:30 model chat: LightGBM v3 variants done 18:41
  (`run_metrics/model_eval/lgbm_v3_variants`). None beats lgbm_v3 (+1.87):
  id_stage identical (0 rounds), linear +1.87 (swing corr 0.145), recent +1.76
  (swing corr 0.137), rank +0.33 (calibration -1.22, and no better ranking: +2.20
  vs +2.36). Best mix: linear + B0 +2.02, v3 + B0 +2.01. `pick_lgbm.py` now
  breaks exact ties toward the base name, so the every-5 job builds lgbm_v3_every5.
- 2026-11-06 00:45 model chat: identity look-ahead test done (13356, exit 0,
  `run_metrics/model_eval/id_lookahead`). Removing the id embeddings from B0
  helps a lot: v2_b0_noids +2.20 (ranking +2.36, fixtures +0.25), v2_b0_nopid
  +2.10 (ranking +2.41), vs B0 with ids +1.67/+1.69 (ranking +1.19/+1.41) and
  lgbm_v3 +1.87. noids is the best single model so far. Random-row validation
  was unchanged (RMSE 2.04 vs 2.04), so it could not see this. 14168 now running.
- 2026-11-06 00:57 model chat: Hal asked for B0 no-ids with the scoreline built in,
  ahead of the every-5 job. New plan `simple_h1_b0_noids_staged.json` (noids +
  points_structure staged), experiment `v2_b0_noids_staged`; `learning_v2/model.py`
  `team_side` now takes the team id only if the plan has one (old plans build the
  same layers; the v2 code hash moved). 139 tests pass (learning_v2, model_eval,
  lgbm). Stopped 14168 during its lgbm_v3_every5 GW6 fit (GW1-5 kept in set 87);
  queue now **5496** `run_b0_noids_staged.cmd` -> **14812** `run_retrain_every5.cmd`.
- 2026-11-06 07:55 model chat: 5496 done 03:59 (`b0_noids_staged`): v2_b0_noids_staged
  +2.17 (future +2.46, ranking +2.23, fixtures -0.33) vs noids +2.20; mixes with
  lgbm_v3: staged +2.14, noids +2.09 (best ranking +2.53). Interim every-5
  (`retrain_every5_lgbm`): lgbm_v3_every5 +1.80 vs once-a-season +1.87 (ranking
  +2.41, fixtures -0.70). 14812 is now on v2_b1_every5 fit 1 (GW1-5).
- 2026-11-07 10:20 model chat: v2_b1_every5 leaked memory across refits in one
  process (epoch 274 s on fit 1 -> 1065 s on fit 5; 7.1 GB private, 3.2 GB
  resident, paging). Stopped 14812 in the GW21 fit; GW1-20 kept in set 91. New env
  `PL_RETRAIN_MAX_FITS` in `prediction_builder.py` (not hashed): a retraining build
  stops before its next fit. `run_retrain_every5_resume.cmd` (**PID 4676**) calls
  the build once per remaining fit, then mixes + evaluate. test_lgbm +1 test, 66
  pass (lgbm, model_eval).
- 2026-11-07 10:30 model chat: Hal asked for LightGBM Tweedie, v4 and v5, queued
  after the every-5 job (**PID 12912**, `run_lgbm_v4.cmd`). New:
  `premierlearning/lgbm/extras.py` (time-decayed Poisson team ratings on goals and
  xG, what the opponent concedes to each position), `lgbm/staged.py` (minutes
  bands, clean sheet, rest of points, combined with FPL scoring), feature set 3
  in `lgbm/features.py` (shrunk per-90s, competition for place; set 1/2 columns
  unchanged), `--lgbm-staged` and Tweedie label clipping (`lgbm/model.py`),
  `flatten_with_seasons` in `season_model_lgbm.py`; extras.py and staged.py
  added to LGBM_CODE_HASH_FILES (every LightGBM hash moved; no LightGBM set is
  mid-build). Experiments lgbm_v3_tweedie, lgbm_v4, lgbm_v5. 82 tests pass (lgbm,
  model_eval, live_blend). live.py (production LightGBM) unchanged: still v3.
- 2026-11-07 12:50 model chat: Hal asked for the LightGBM builds to go before the
  rest of the every-5 retrain. Stopped 12912 (waiting) and 4676 during the GW26
  refit (GW1-25 kept). `run_lgbm_v4.cmd` started now (**PID 12944**; 73 tests
  OK); `run_retrain_every5_resume2.cmd` (**PID 6348**) waits on it, then fits
  GW26/31/36 and runs the same mixes + evaluate.
- 2026-11-07 13:18 model chat: Hal asked for FPL's player news beyond the flat
  chance. New `premierlearning/lgbm/news.py` (in LGBM_CODE_HASH_FILES; v4 and
  later LightGBM hashes moved, no LightGBM set mid-resume): downloads the same
  fplcache snapshot each chance file used into `data/season_XX_YY/fpl_news/`
  (21/22-25/26, 189 deadlines, done) and builds status, reason group, days to
  expected return, days flagged, chance change, flag streak and a hand-written
  0-10 certainty. Feature set 4 (`features.py`, additive), wired in
  `season_model_lgbm.py` (upcoming rows read the as-of deadline). Experiments
  lgbm_v6, lgbm_v6_staged. `tests/test_lgbm_news.py`; 81 pass (news, lgbm,
  model_eval). `run_lgbm_v6.cmd` (**PID 8408**) waits on 12944; every-5 resume
  requeued behind it (**PID 5092**, replaces 6348). squadtools/chance_to_play.py
  untouched (net hashes unchanged); live predictions still v3, no news archive
  for the live season yet.
- 2026-11-07 17:50 model chat: v6 done (lgbm_v6 +2.13, _staged +2.14 vs v4/v5
  +1.88/+1.90). Hal asked for the best net (preferring more data) and best
  LightGBM over 23/24-25/26, then two manager runs on them (go-ahead for these
  two only; the paused 35/34/29 runs stay paused). New plan
  `simple_h1_b1_noids.json` (B1 minus every *_id_embedding, the method
  reproduces b0_noids exactly), experiment v2_b1_noids, `model_eval/pick_net.py`
  (pick: B1 unless >0.10 below B0 on 25; set-id: refuses partial sets).
  `run_three_seasons.cmd` (**PID 14596**) waits on 5092. Please don't edit
  hashed LightGBM / learning_v2 files until it has built (sets 97 and the net's
  must keep their hashes). Heavy until ~10:00 11-08, then two manager runs.
- 2026-11-09 09:40 model chat: every-5 done 11-08 01:44 (retrain_every5: v2_b1_every5
  +1.71 vs v2_b1_context +1.57, lgbm_v3_every5 +1.80 vs lgbm_v3 +1.87; not worth
  the cost). net_pick: v2_b1_noids +1.86 vs v2_b0_noids +2.20 (fixtures -1.19), so
  B0 no-ids carried. three_seasons (23-25, v1_h5 baseline): lgbm_v6 +2.32,
  v2_b0_noids +2.16, live-era blend25 +0.99. Sets 97 (lgbm_v6) and 104
  (v2_b0_noids) complete; manager runs started 11-09 01:50 (PIDs 3116, 9184).
  The hash freeze on LightGBM / learning_v2 files is lifted.
- 2026-11-09 10:40 model chat: Hal asked: cancel the B0 manager run (done; second
  lgbm_v6 run, seed 1, in its place), LightGBM v6 live on Fargate with email, and
  v6 as the default model from now on.
  - Live v6: `lgbm/live.py` (VERSIONS v3/v6, default v6, `--lgbm-version v3`),
    `premierlearning/live_model.py` (new: default args `--learning-v2 --lightgbm`,
    `--v2-blend` picks the old blend; run_pipeline and the email note use it),
    `lgbm/news.py` archive_live_news / backfill_live_news (called from
    runner.update_persisted_data_if_outdated; 26/27 GW1-10 saved in
    data/season_26_27/fpl_news). news.py is in LGBM_CODE_HASH_FILES, so new
    lgbm_v6 builds get a new hash (set 97 itself is complete and unchanged).
  - Schedule bug fixed: the GW10 run (5 h early, 2 h training) ended 3h02 before
    the deadline, outside the 3 h window, and booked nothing. fpl_deadline
    `deadline_for_following_run(..., hours_before=)` skips deadlines whose run time
    has passed. Schedule rebooked by hand to at(2026-11-21T06:00:00) UK (GW11).
  - ECR: previous production image tagged `latest-2026-11-05` (rollback).
  - `fpl_manager/stop_after_save.ps1` takes -Names and matches either slash (it
    missed the B0 run's backslash path).
- 2026-11-09 11:50 model chat: build_lgbm_v6.ps1 finished 11:22. Crawl, build, 100
  tests in the image and push OK; suggest-only task exit 0 in 17 min, email sent
  (GW11); ECR `latest` now = the v6 image (digest 80c36bf9...). The seed-1 pause did
  nothing (it was started by `call` inside swap_b0_for_lgbm_seed1.cmd, so no cmd.exe
  named its checkpoint) and the resume's duplicate exited at once (log in use); the
  run kept going unharmed. stop_after_save.ps1 now falls back to the nearest cmd.exe
  above the run's python processes.
- 2026-11-09 13:10 model chat: Hal asked to keep only lgbm_v6 run 1, with more
  processes. `fpl_manager/train.py` `_keep_runs_per_update`: a resumed policy now
  uses the call's n_steps/batch_size (rebuilds the rollout buffer), so 4 workers
  keep 8 runs per update instead of becoming 16 (test in test_fpl_manager
  CheckpointTests). `switch_lgbm_v6_to_4_workers.cmd` (PID 12768) stops seed 1,
  then run 1, each after a save, and continues run 1 with --workers 4.
- 2026-11-09 19:45 model chat: known fixture calendar for the manager. Prediction
  sets put each fixture in its FINAL gameweek (fixture_mode final_calendar), so
  replays knew blanks/doubles early. New `simulation/fixture_calendar.py`: every
  version of vaastav `data/<season>/fixtures.csv` (FPL's fixtures feed) with its
  commit time, saved as `data/season_YY_YY/fixture_calendars.json` (21/22-25/26:
  40/38/38/37/12 copies). `known_calendar(season, deadline)`: a change counts from
  the later copy, or 14 days (ANNOUNCE_LEAD_DAYS, measured: 37 of 40 moves were
  shown that early) before the affected gameweek's deadline if sooner, never
  before the earlier copy (fills 25/26's 3-month gaps). `fpl_manager`
  `--calendar known|final` (default known for v3, also rule_baseline):
  `forecasts.load_forecast_pool(calendar=)` moves rows to the known gameweek and
  drops unscheduled/past ones; known weeks use cache id 1e9 + snapshot id
  (season_source.cache_snapshot_id) so no final-calendar squad answer or pool
  file is reused. Prediction sets themselves are unchanged. Tests:
  tests/test_fixture_calendar.py.
- 2026-11-09 22:30 model chat: synthetic seasons steps 1-2 (Hal asked; see
  SYNTHETIC_SEASONS_PLAN.md "Status"). New package `api/synthetic_seasons/`
  (source, injuries, events, generate, report; `python -m synthetic_seasons
  generate|report`), tests `tests/test_synthetic_seasons.py` (9 pass). 20 copies
  (5 each of 21/22, 22/23, 23/24, 25/26; 24/25 is the clean check) in
  `api/data/synthetic/` (40 MB), report `api/data/synthetic/REALISM_REPORT.md`,
  calibration `api/data/synthetic/calibration.json`. Light job (~2 min), nothing
  else touched; the manager run (4236) kept running.
- 2026-11-10 00:15 model chat: Hal asked to pause the manager and check the v6
  points model on the copies. v3_lgbm_v6 paused after its 23:13 save
  (`fpl_manager/pause_lgbm_v6.cmd`; continue with `fpl_manager/resume_lgbm_v6.cmd`).
  New `synthetic_seasons/forecasts.py` (v6 forecasts on a copy; reproduces set 97
  exactly on real 25/26) and `synthetic_seasons/manager_check.py`. HEAVY jobs, one
  after another (about 4.3 GB each): `synthetic_seasons/run_points_check.cmd`
  (PID 2924, log `api/data/synthetic/points_check.log`), then
  `synthetic_seasons/run_manager_check.cmd` (PID 6488, waits for 2924; log
  `api/data/synthetic/manager_check.log`, about 4-5 h). 24/25 is now a copy source
  too (Hal, 2026-11-10); copies 1-2 made. Manager check adds squad answers to the
  DB caches under snapshot ids 2,000,000,000+ only; prediction sets untouched.
- 2026-11-10 09:30 model chat: points and manager checks finished (03:52). Results
  in `api/data/synthetic/manager_check.csv`: policy beats rules by 203 points on
  the real 23-25 seasons but not on copies (-66, +7). 100-copy generation running
  (`synthetic_seasons/run_generate_100.cmd`, PID 5360, log
  `api/data/synthetic/generate.log`, about 10 min, light). The forecast build for
  all copies waits for Hal's go-ahead; manager still paused.
- 2026-11-10 13:30 model chat: synthetic copy fixes (Hal asked). injuries.py: new
  spells go to starters-when-fit (`injury_targets`), returning regulars prefer a
  deputy who started; events.py: strong sides concede a little less than xG
  (`group_effects`, only effects with the same sign every season), assister
  weights ^1.25; calibration.json refreshed (old copy kept as
  calibration_before_2026_11_10.json). Copies 1-20 of 23/24 and 25/26 regenerated
  (their old forecasts deleted); 21/22, 22/23, 24/25 copies are still the old
  version. forecasts.py has a past-season cache (`PastSeasonCache`), NOT yet
  verified against the uncached path - don't build forecasts with it until then.
- 2026-11-10 later, model chat: fixture reshuffle added (Hal approved):
  `synthetic_seasons/schedule.py` swaps clean gameweeks' fixtures within each half;
  copies get schedule.json; forecasts.py rebuilds player load AND team schedule
  inputs for every copy fixture; CopyManagedSeason.calendar_at shows moved fixtures
  in their copy gameweek. New `python -m synthetic_seasons pipeline-check`.
  HEAVY job running: `synthetic_seasons/run_reshuffle_build.cmd` (PID 14100, log
  `api/data/synthetic/reshuffle_build.log`): regenerate all 100 copies, report,
  pipeline check (stops if it fails), then v6 forecasts for copies 1-10 of 23-25
  (about 9-10 h, about 4 GB). Don't start other heavy jobs until it finishes.
  Hal asked to keep only the 30 copies used: copies 11-20 of 23-25 and all 21/22,
  22/23 copies deleted (rebuild exactly with `generate`, same seeds, about 5 min).
  Run 1 (PID 14100) stopped at the pipeline check: the cache passed (identical,
  2x faster) but duplicate player folders gave copies extra rows (fixed in
  source._player_files). Run 2 started 11:23 as PID 14888, same log.
  Run 2 pipeline check PASSED; forecasts running (about 12 min per copy).
  FIXED 12:15 (Hal approved): renamed players with two players/<name>_<id>
  folders. New `premierlearning.season.player_folders` keeps the folder with the
  most rows; used by PastSeason (it read BOTH: duplicate Player objects for
  Petrovic, Kinsey, Olakigbe, Yarmoliuk in 23/24), VaastavSeasonData (kept the
  last listed: Olakigbe 4 and Yarmoliuk 11 of 38 rows in 23/24) and the
  synthetic copies. Player order otherwise unchanged. Set 97 and the running copy
  forecasts were made with the old readers (consistent with each other);
  production training picks this up on the next image build.
- 2026-11-11 model chat (manager): BUG FIXED (Hal approved). 24/25 and 25/26
  players_raw.csv have can_select False for every player (downloaded after the
  season); `VaastavSeasonData.known_unavailable` read it as deadline news, so
  played players became unselectable (24/25 GW37: 246 of 304 who played).
  Hit every manager/replay result on 24/25 and 25/26 (sets 97, 104 and the
  copies); forecasts themselves never used the flag; 23/24 unaffected (no
  column); live uses FPL's real can_select. Fix: `can_select_usable` (ignored
  when no player in the file is selectable). Set 97 kept; **set 105** = set 97
  with the corrected flag. Copies re-flagged in place (old file kept as
  forecasts_selectbug.csv.gz), COPY_GENERATION 2. v3_lgbm_v6_copies stopped at
  1,148 runs; fresh run v3_lgbm_v6_copies_fixed (PID 3252 chain). Paused
  v3_lgbm_v6 (set 97) is affected too; left paused.
- 2026-11-13 model chat (manager): Hal asked to pause v3_lgbm_v6_copies_fixed
  (stopped after its save at 1,241 runs), replay the live Fargate manager
  (plan_live settings) on the real 23/24-25/26 (new `fpl_manager/live_baseline.py`,
  out `fpl_manager_runs/live_baseline_set105/`), then play the run's policy on
  2026/27 so far. New `simulation/current_season_csv.py` writes the live season
  in the vaastav CSV layout (fixtures.csv, teams.csv, players_raw.csv, gws/,
  players/<name>_<id>/gw.csv, raw JSON in replay_api/) inside
  `api/data/season_26_27`, beside the live files, which it does not touch.
