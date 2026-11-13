# Synthetic seasons for the fantasy manager (design, 2026-11-09)

## Status (2026-11-09)

Steps 1-2 of the build order are done: `api/synthetic_seasons/` (generator,
tests in `api/tests/test_synthetic_seasons.py`), 20 copies in
`api/data/synthetic/season_XX_YY/copy_001..005`, report in
`api/data/synthetic/REALISM_REPORT.md`. Run with
`python -m synthetic_seasons generate|report --seasons 21,22,23,25 --copies 5`
(from `api/`).

Hal's decision on sources: copies are made from 2021/22, 2022/23, 2023/24
and 2025/26 (25/26 adds a different pattern). 2024/25 is the one real season
never copied, so it is the clean check. 2020/21 is left out for now: its raw
files are in another format (API dumps) and have no FPL flags.

Changes from the design below, found while checking realism:
- Team goals use a shot-like draw (binomial, variance 0.74 x mean, fitted on
  22/23-25/26) instead of Poisson: Poisson on real match xG gives 29% of teams
  failing to score against 21% in reality.
- Assisters are weighted by xA x the player's FPL-assists-per-xA (shrunk to his
  position's ratio): FPL assists include penalties won and rebounds, and
  forwards get far more than their xA. Finishing is shrunk to the position's
  ratio too and scaled to the source season's goal total.
- New injuries go to regulars weighted by how injury-prone each player was over
  2021/22-2025/26, not uniformly, so durable stars stay mostly fit.
- 2021/22 has no FPL xG: each player's xG/xA come from his own threat and
  creativity ratios in the xG seasons, shrunk to the league ratio.
- When a removed injury hands minutes back, a regular starter gives way only if
  no squad player fits.
- Saves scale by a smoothed ratio (xG + 0.5 on both sides).

Known differences (report): the best players' season points are 5-12% below the
real season's, because real top players beat their xG that season and the
copies are centred on xG with shrunk finishing. Not done yet: the points-model
error check (needs step 3, forecasts on the copies).

Goal: train the v3 manager on ~200 made-up but realistic seasons (5 drawn at
random per run, one weight update from all of them), and keep scoring it every
few updates on the real 2023/24, 2024/25 and 2025/26 seasons with its best
choices, as eval.csv does now.

What exists already: `fpl_manager/scenarios.py` (`ScenarioSeason`) redraws each
match's score from the match's real xG, picks scorers/assisters by xG/xA, ties
clean sheets and goals conceded to the opponent's drawn score, and keeps real
minutes and bonus. It is tested but never wired into training. This plan
replaces it with the fuller generator below.

## 1. Which real seasons the synthetic ones are made from

A synthetic season is a copy of one real season (same teams, players, fixture
calendar as FPL showed it at each deadline, real prices) with new injuries, new
team form and new match events.

If synthetic copies of 2023/24-2025/26 were used for training, the real-season
check would no longer be a clean test (the manager would have practised on
those exact squads and calendars). So, recommended:

- Source seasons: 2020/21, 2021/22, 2022/23 (about 67 copies each). 2019/20 is
  left out (47 "gameweeks" after the Covid restart); 2018/19 can be added once
  2016/17-2017/18 are downloaded so its points model has earlier seasons to
  train on.
- Check seasons: real 2023/24, 2024/25, 2025/26, never used to make copies.
- Data limits, all FPL's own numbers: FPL xG/xA exist from 2022/23. Before that,
  each player's xG and xA rates are estimated from FPL's ICT "threat" and
  "creativity" (fitted on 2022/23-2025/26, where both exist). FPL news flags
  (fplcache) exist from 2021/22; 2020/21 has injuries from minutes only.

## 2. Making one synthetic season (source season S, seed k)

All steps use a seeded random generator, so a season can be rebuilt exactly.

### 2a. Team form
Each team gets an attack and a defence multiplier, 1 + N(0, 0.08), that drifts
a little each week (random walk, sd 0.01). So in some copies a team has a good
or bad season, and the points model sees that only through results, as in real
life.

### 2b. Injuries and availability (minutes)
From the real season:
- A player's "absence spells": runs of matches he missed while normally a
  starter, with FPL's flags around them (e.g. 25% the week before, 75% before
  his return) and his comeback minutes (often sub appearances first).
- A "deputy" list for each regular: the teammates whose minutes went up when he
  was out, with how much.
- Injury rates by position: spells per player-season and their lengths.

In the copy:
- Each real spell is kept with probability 0.5 and, if kept, moved to a random
  point in the season (same length, no overlap with another spell).
- New spells are added to other regulars, so the total injury rate matches the
  real one, using spell templates of the same position.
- When a player is out in a week he really played, a deputy (drawn from his
  deputy list) takes his minutes. When he plays in a week he was really out,
  the player who covered for him that week gives the minutes back.
- FPL's news flags move with the spells: removed with a removed spell, copied
  from the template for a new one. The points model and the squad optimiser
  both read these flags, so they have to match.
- Checks per team-match: one goalkeeper, 11 starters, at most 5 subs, about 990
  minutes in total.
- Real suspensions, red cards and January transfers stay as they happened.

### 2c. Match events
For each match:
1. Expected team xG = blend of the real match's team xG and the team-rating
   expectation for that pairing (weight 0.5 each), times the attack multiplier
   and the opponent's defence multiplier. The realised team xG is that times a
   lognormal(0, 0.25) draw.
2. The team xG is split across players on the pitch by their xG-per-90 rate x
   minutes on the pitch (starters are on from 0', subs until 90').
3. Goals per player ~ Poisson(player xG x finishing). Finishing = each player's
   goals/xG over every season we have, shrunk strongly toward 1 (a prior worth
   about 30 xG), so only proven over- or under-performers move away from 1.
   Own goals ~ Poisson(0.07) per team. Team goals = the sum.
4. Each goal gets a minute inside the scorer's time on the pitch. It is
   assisted with the real assisted share (about 75%), the assister drawn by
   xA-per-90 among teammates on the pitch at that minute.
5. Defence follows from the opponent's goals and their minutes: goals conceded
   while on the pitch, and a clean sheet only with 0 conceded and 60+ minutes.
   So every defender of a team gets a clean sheet together, and only when the
   other team failed to score.
6. Goalkeeper saves: the real saves x (opponent's drawn xG / real xG).
7. Bonus: BPS rebuilt from FPL's published BPS table. The real BPS minus the
   real goal/assist/clean-sheet/conceded parts, plus the same parts for the
   drawn events. Then 3/2/1 to the top three in each match, using FPL's tie rules.
8. Points use FPL's scoring for that season from the drawn events. Cards,
   penalties saved and missed, and appearance points stay as they happened.

### 2d. On the xG question
Players who score close to their xG are not more predictable in this world.
Everyone's goals are a Poisson draw around the same xG, so the noise is the
same for all.

The real risk is the opposite one. With finishing set to 1 for everyone, true
elite finishers would score less in the copies than the points model (trained
on real seasons) expects. The manager would then see the model as
over-optimistic about them. The shrunk finishing factor in 2c.3 removes most of
that. A small leftover bias only nudges how much the manager trusts big
predicted gains (hits, chips). It does not change what it learns about squads,
because squads are picked by the optimiser from the forecasts.

## 3. Forecasts without retraining

No retraining for each synthetic season. LightGBM v6 is already fitted once per
season, at GW1, on the seasons before it (`--train-once-per-season`). Later
gameweeks only rebuild the inputs from the results so far and predict.

So for a copy of season S:
- fit (once, and save) the v6 model for S on the real seasons before S;
- for each gameweek n, rebuild the model's inputs from the real earlier seasons
  plus the copy's GW1..n-1 (minutes, xG, goals, assists, clean sheets, bonus,
  scores, team ratings), the copy's news flags at deadline n, and the known
  fixture calendar at deadline n; then predict;
- store only what the manager reads, the next 8 gameweeks per player, as one
  compact file per season (about 1 MB), not database rows.

Retraining on the copies would be worse. The model should learn football from
real seasons only. Trained on copies it would learn the generator's quirks,
and it would cost an hour or more per season.

Check before use: the v6 model's error on the copies should be close to its
error on real seasons (RMSE about 1.9, MAE about 0.97). Much lower would mean
the copies are too predictable, and much higher would mean too random.

## 4. Percentiles for made-up seasons

The reward uses each season's finish percentile, and real managers never
played the copies. So each copy is calibrated with the rule-based manager
(`fpl_manager/rule_baseline.py`), which is run once on every copy and on the
real source season. The percentile of a score X on a copy is the real source
season's percentile of X - (baseline on copy - baseline on real). That puts
easy and hard copies on the same scale.

## 5. Manager training

- Library: the 200 copies. Each run draws 5 without replacement and plays each
  as its own episode, rewarded as now (a quarter of each week's points that
  week, the season's points and percentile at its end).
- 8 runs per weight update (2 per worker with 4 workers), so 40 seasons feed
  each update.
- Every 5 updates: one top-choice pass over the real 2023/24-2025/26 (eval.csv,
  same as now). It also scores 10 fixed copies the manager never trains on, so
  we can see whether it is learning the copies' world rather than football.
- Workers load a season's files when it is drawn and drop them after, instead
  of holding all seasons in memory.

## 6. Cost and time (estimates, to measure on the first season)

- Generating one copy: seconds for the events, about 10-15 minutes for 38 weeks
  of forecasts (the real v6 build took about 20 minutes per season including
  its one fit). 200 copies: about 35-50 hours on the laptop with the manager
  run paused (memory), or about 1-2 hours on Fargate with about 25 tasks at
  once (a few dollars; to be priced before running).
- Manager speed: 5 seasons per run instead of 3, and fewer squad answers reused
  from the cache, so roughly 3x slower per run than now. A check game every
  2-3 hours instead of about every 40 minutes.

## 7. Build order

1. Generator for one source season, with tests for the rules (team goals = sum
   of player goals, clean sheets iff the opponent scored 0 while on the pitch,
   about 990 minutes per team-match, one keeper, spells don't overlap, flags
   follow spells).
2. Realism report: compare 20 copies with the real season (goals per match,
   0-0s, clean-sheet rate, team goal totals, top scorers, points by position,
   hauls, injuries per team) and the v6 error check from section 3. Hal reviews
   the report before anything else is built.
3. Forecast builder (saved per-season v6 model, compact forecast files).
4. Percentile calibration (rule baseline per copy).
5. Manager: season library, 5 per run, held-out copies in the check.
6. Generate 200 copies, then start the run.

## Open decisions for Hal

1. Source seasons: decided 2026-11-09, 21/22, 22/23, 23/24 and 25/26; 24/25
   stays the clean check.
2. Start the synthetic-season manager from scratch (clean comparison), or from
   the current v6 run's weights (faster start).
3. Generate on the laptop (about 2 days, manager paused) or on Fargate (about
   2 hours, small cost).
