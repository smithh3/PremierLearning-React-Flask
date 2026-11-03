# external_data: all-competition matches and managers

Extra data the points model can learn from, beyond FPL's Premier League rows:

* every match a PL club plays: PL, FA Cup, League Cup, Community Shield, CL, EL,
  Conference League, Super Cup, Club World Cup, and **pre-season friendlies**
* every men's senior and youth international (World Cup, Euros, AFCON, Copa,
  Asian Cup, Gold Cup, qualifiers, Nations League, friendlies, U21s)
* lineups, minutes, goals, assists, cards, saves, shots and clean sheets for every
  FPL player in those matches
* scheduled future fixtures (up to 120 days ahead), for "upcoming matches" inputs
* Premier League manager spells (caretakers included) and the manager of each side
  in every FPL fixture

## Sources

| Source | What | Notes |
|---|---|---|
| ESPN public JSON API (`site.api.espn.com/apis/site/v2/sports/soccer/...`) | Fixtures, results, lineups, subs, goals, assists, cards | No key. One `all/scoreboard?dates=YYYYMMDD&limit=1000` request lists every competition for that day, friendlies included (checked against per-league feeds). Match `summary` has rosters and key events. Unofficial, so it could change. |
| Wikipedia "List of Premier League managers" | Every PL manager spell with exact dates, caretakers marked | Page says "up to date as of" a date. The "current managers" page patches newer appointments. |

Options considered and not used: FBref (Opta advanced data withdrawn in 2026;
no friendlies anyway); API-Football (free tier is 100 requests/day and only
seasons 2022-2024); FotMob / Sofascore (signed or Cloudflare-protected APIs);
BBC Sport (HTML only, no stable history); UEFA / other fantasy APIs (no
friendlies, cup or international coverage). ESPN covers all of it.

## Running

From `api/`:

```
..\venv\Scripts\python.exe -m premierlearning.external_data all     # crawl + build (re-run = incremental)
..\venv\Scripts\python.exe -m premierlearning.external_data.qa      # ESPN vs FPL agreement report
```

Or run detached: `premierlearning\external_data\run_crawl.cmd`, which logs to
`data\external\crawl.log`. The first full run takes about 2-3 h, almost all of it
waiting on the network. After that, re-runs fetch only the last 10 days, the
next 120 days and new finished matches.

The raw cache in `data/external/raw/` is git-ignored. The CSVs below are the
interface: read these, not the raw files.

## Output tables (`api/data/external/`)

**Join keys.** `fpl_code` is the stable FPL player code (players_raw `code`,
the same one `learning_v2` uses for players). `fpl_team_code` is the stable FPL
team code (teams.csv `code`, e.g. Arsenal = 3). `fpl_season` + `fpl_fixture_id`
point to the FPL fixture for Premier League rows. All times are UTC.

### `appearances.csv`: one row per FPL player per match-day squad (any competition)
Only ESPN athletes linked to an FPL code are included. A row means the player was
in the matchday squad; `minutes == 0` means an unused sub. If a match has
`has_lineups == 1` and a player isn't in it, he wasn't in the squad (injured,
rested, or not selected).

| column | meaning |
|---|---|
| event_id, kickoff_utc, league_slug, category | ESPN match id, time, competition |
| fpl_season, fpl_fixture_id | set only for Premier League matches |
| espn_athlete_id, fpl_code, name | player |
| team_id, fpl_team_code, home_away | the side he played for (ESPN id; FPL code if a PL club) |
| team_score, opponent_score | final score (before any shoot-out) |
| position | ESPN position abbreviation (G, CD-L, AM, F, SUB, ...) |
| starter, minutes | minutes from lineups + substitution times (extra time counts) |
| goals, assists, own_goals, yellow_cards, red_cards, saves, shots, shots_on_target | ESPN box score |
| goals_for_on_pitch, goals_against_on_pitch | team goals while he was on the pitch |
| clean_sheet | FPL rule: >= 60 min and nothing conceded while on |

### `matches.csv`: one row per relevant match (finished or scheduled)
event_id, kickoff_utc, season_year, league_id/slug, `category`, stage (ESPN
season slug, e.g. `league-phase`, `round-of-16`), status, completed, neutral,
has_lineups, fpl_season/fpl_fixture_id, home/away team ids, names and FPL codes,
scores, shoot-out scores, notes (e.g. aggregate score).

### `team_matches.csv`: the same matches, one row per side (easiest for schedules)
All `matches.csv` columns from one team's point of view (team / opponent /
goals_for / goals_against), plus:
* `known_from_utc`: a conservative estimate of the earliest time this match was
  surely on that team's schedule. **Use it to avoid leaking the future**: an
  "upcoming matches" list built at time T may only contain rows with
  `known_from_utc <= T`. Rules:
  * league / group / league-phase: 14 days before the team's first match of the stage
  * friendlies: 28 days before kickoff
  * knockout rounds: the team's previous match in that competition, because you
    only know you're in the next round once you've won the last one
* `manager_id`, `manager_caretaker` for PL clubs (blank otherwise).

### `manager_spells.csv`
manager_id (Wikipedia slug, e.g. `Mikel_Arteta`), manager_name, club,
fpl_team_code, from_date, until_date (blank = current), caretaker (0/1), source.
Spells overlapping 2018-06 onwards only.

### `fixture_managers.csv`: one row per local FPL fixture
fpl_season, fpl_fixture_id, kickoff_utc, then for home and away: team_code,
manager_id, caretaker, days_in_post (days since that manager's spell began).

### Lookups
`leagues.csv` (ESPN league id, slug, name, category), `espn_team_map.csv`
(ESPN team id -> FPL team per season), `espn_player_map.csv` (ESPN athlete ->
FPL code with match votes/score, for auditing), `qa_report.txt`.

### `category` values (suggested embedding vocabulary)
`premier_league`, `domestic_cup`, `domestic_super_cup`, `champions_league`,
`europa_league`, `conference_league`, `club_super_cup`, `club_world_cup`,
`club_friendly`, `intl_tournament`, `intl_qualifier`, `intl_nations_league`,
`intl_friendly`, `intl_youth`, `other_club`. `league_slug` is available too if
finer detail helps (e.g. FA Cup vs League Cup).

## Quality (ESPN vs official FPL on the same PL matches)

On the test window (summer 2025, 1,029 player-matches): minutes median error 0
and 100% within 5 min; goals 100% exact; assists 98%; clean sheets 99.3%; goals
conceded while on the pitch 99.7%; every FPL player who played was found. The
full-run numbers are in `data/external/qa_report.txt`.

Known gaps:
* About 60% of club friendlies have lineups on ESPN (higher for big clubs, lower
  for behind-closed-doors games). Without lineups you still get the scoreline and
  the fact that the match happened (`has_lineups = 0`).
* No xG / xA / defensive contributions / bonus outside the PL. Mask those
  features for non-PL tokens.
* Assists follow ESPN's definition, which is slightly stricter than FPL's.

## Model-ready features (`api/data/external/features/`, from `features.py`)

Built with `python -m premierlearning.external_data features` (also the last step of `all`).
Every value is computed **as of the deadline of the fixture's gameweek**, so it only uses
what was known when the prediction is made. Results must be before the deadline, and
future matches must have `known_from_utc` <= deadline. To get another horizon, call
`team_schedule`, `player_load` or `gw_flags` with a different `as_of`.

| file | key | columns |
|---|---|---|
| `team_fixture_features.csv` | fpl_season, fpl_fixture_id, team_code (one row per side; join it again on opponent_code for the opposition) | manager_id, manager_caretaker, manager_days_in_post_log; days_since_last_match, matches_last_14d, days_to_next_match, matches_next_7d (all competitions); gw_fixtures, gw_double, dgw_leg (1 or 2); next{1,2,3}_gw_double_known, next{1,2,3}_gw_blank_known |
| `team_upcoming.csv` | fpl_season, fpl_fixture_id, team_code, rank (1-14) | club's matches in the next 35 days, any competition including the PL: category, league_slug, opponent_fpl_team_code (blank if not a PL club), home_away, neutral, knockout, days_after_current |
| `player_fixture_features.csv` | fpl_season, fpl_fixture_id, fpl_code | days_since_last_played, days_since_last_started, minutes_last_{7,14,28}d (all competitions, internationals included), in_squad_last_club_match, load_known, national_team_id |
| `player_international_upcoming.csv` | fpl_season, fpl_fixture_id, fpl_code, rank | the player's senior national-team matches in the next 35 days (same columns as team_upcoming) |
| `gameweek_schedule.csv` | fpl_season, gw, team_code | fixtures (0/1/2), is_blank, is_double, cause (`normal`, `double`, `blank_cup`, `blank_other`), known_from_utc |

Day counts are capped at 120 (meaning "none known").

**Double and blank gameweeks.** The current gameweek's count is always known at
the deadline. For the next 3 gameweeks, the `_known` flags are 1 only once the
double or blank was public. Past FPL data only has the final fixture list, so
"public" is estimated:
* doubles: 21 days before that gameweek's deadline
* blanks caused by a cup tie: once the club reached that round (from the cup data)
* other blanks (postponements): only 3 days before, so late surprises (COVID,
  weather) are never revealed early

For the live season, anything in today's FPL data counts as known now.

**History tokens.** Non-PL tokens for `player_fixture_history` come straight
from `appearances.csv` (rows with blank `fpl_fixture_id`), filtered to kickoff
before the deadline. PL tokens stay as they are today, built from FPL data with
xG, points and defcon. Set `fpl_stats_known = 0` on ESPN tokens.

**Data caveat (not from this pipeline).** The local 21/22 FPL folder is a
mid-season JSON snapshot: 367 fixtures and about 15k player rows, against 24-29k
for other seasons.

## Using it from the model: `learning_v2/external_inputs.py`

The model reads these features only through this module. The external-data
chat owns it; the model chat owns the hooks that call it.

```python
from premierlearning.learning_v2.external_inputs import ExternalInputs, competition_index
ext = ExternalInputs.load()                          # ~5 s
ext.team(season, fixture_id, team_code)              # {'schedule.days_since_last_match': .., 'gameweek.dgw_leg': .., 'manager_features.caretaker': ..}
ext.manager_id(season, fixture_id, team_code)        # 'Mikel_Arteta'; map with ext.manager_vocab(train_seasons, min_matches=10)
ext.player(season, fixture_id, fpl_code)             # {'player_load.minutes_last_7d': .., 'player_load.load_known': ..}
ext.team_upcoming(season, fixture_id, team_code)     # [{'competition_embedding', 'opposition_team_code' (None = unknown), 'home_away', 'neutral', 'knockout', 'days_after_current', 'national_team'}]
ext.player_upcoming(season, fixture_id, fpl_code, team_code)  # club + national team, ordered by days ahead
ext.player_history(fpl_code, before_datetime, n)     # non-PL appearance tokens; merge with PL tokens by 'kickoff'
ext.slices(season, fixture_id, fpl_code)             # eval_slices.csv flags (reports only)
```

The dict keys equal the proposed plan's paths (`schedule.*`, `gameweek.*`,
`manager_features.*`, `player_load.*`), so plan leaves map one to one. Missing
rows return "nothing known" defaults (days 120, counts 0). Competition index 0
is pad / unknown; the order in `COMPETITIONS` is append-only.

## Suggested model design (for the model chat; `learning_v2` is theirs)

The full proposed plan is in `proposed_model_plan.json`, based on
`simple_h1_teams_staged.json`. Hal approved it on 2026-11-03. Each new plan field
maps to a column in the feature files above. `days_since_previous_match` is
replaced by the all-competition `schedule` and `player_load` blocks, and a
`gameweek` block is added for doubles and blanks.

1. **Manager embedding.** Add `manager_id_embedding` to `current_team`,
   `current_opposition` and the team-history tokens, from `fixture_managers.csv`.
   Give managers with very few matches (mostly caretakers) the unknown index, and
   add `caretaker` and `log1p(days_in_post)` as scalars (new-manager bounce,
   rotation habits). Managers move clubs (e.g. Iraola from Bournemouth to
   Liverpool), so the embedding can carry a manager's style across clubs.
2. **One mixed, chronological match list with a competition embedding**,
   rather than a separate list per competition. This answers "should PL fixtures
   go in the same list?": yes, because the interleaving is the signal (a 120-minute
   CL tie three days before a PL game). Each token gets `category` (and maybe
   `league_slug`) embeddings, home/away/neutral, `days_from_target`, the
   opponent's team embedding when it's a PL club (else unknown), and a
   `has_lineups` / stat mask.
   * **History tokens** come from `appearances.csv`, strictly before kickoff: the
     player's minutes, starter, goals, assists, clean_sheet, plus team score.
     FPL-only stats are masked on non-PL tokens.
   * **Upcoming tokens** come from `team_matches.csv` for the player's club, plus
     his national team if he's been capped in the last ~12 months. Window:
     (target kickoff, target + 35 days], filtered by `known_from_utc <=` the
     prediction time. The attention can learn that, for example, two big CL games
     next week means a rest this weekend.
3. **Cheap schedule scalars**, which may be worth more than the attention lists:
   days since the club's last match in any competition (today's
   `days_since_previous_match` only sees PL games), player minutes in the last
   7 / 14 days across all competitions, number of club matches in the next 7 days,
   and whether he was in the squad for the last cup / European game.
