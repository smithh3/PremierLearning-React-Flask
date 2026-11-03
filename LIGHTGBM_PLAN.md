# LightGBM points model: plan (not started)

Status: idea saved for later, 2026-11-03. Nothing here is built. If we pick it
up, claim the files in `AGENT_NOTES.md` first and queue its runs there (the
laptop can't run it alongside a v2 eval: one eval uses about 3.4 GB of 8 GB).

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
- Attack and defence ratings for each team as of each date, from a small Poisson
  model fitted on past scorelines only.
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

## Cost

- About 200k rows x 200 columns, about 160 MB in memory.
- Minutes to train on the CPU once the machine is free.
- `pip install lightgbm` into the venv (Python 3.8 wheels exist), or use
  sklearn's HistGradientBoosting.
