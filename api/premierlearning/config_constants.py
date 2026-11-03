import os

SEASON_YR = 26
PLAYER_JSON_FILE_LOCATION = 'data/season_%i_%i/players/player_'

# Past seasons that use vaastav-style CSV (players_raw.csv, fixtures.csv, players/*/gw.csv)
VAASTAV_CSV_YEARS = (18, 19, 21, 22, 23, 24, 25)

# Points total window for the multi-GW flat model (summed team xg/xc + redistribute)
POINTS_HORIZON = 5

# Early-season prior interaction: last_pp90 * exp(-matches_played / tau)
# Raw last_season / two_season_prior rates stay unscaled.
PRIOR_EARLY_TAU = 6.0
# Appearance window behind past_stats['avg'] (short window is last 3).
# Horizon-sweep can override with PL_SHORT_FORM_WINDOW / PL_FORM_AVG_WINDOW.
FORM_AVG_WINDOW = int(os.environ.get('PL_FORM_AVG_WINDOW', '7'))
SHORT_FORM_WINDOW = int(os.environ.get('PL_SHORT_FORM_WINDOW', '3'))
# Appearances below this are dropped from last-3 / last-7 per-90 form windows.
FORM_MIN_MINUTES = 20
# Last-season second half for recency fill into last-1/2/5 minutes (not a prior).
HALF_SEASON_FIXTURES = 19
LAST_SEASON_FIXTURES = 38
# Predict-only: drop DNP runs of this length from minutes features, but only if
# the player appeared in the last RETURN_CONSECUTIVE_APPS fixtures (back in side).
INJURY_DNP_STREAK = 3
RETURN_CONSECUTIVE_APPS = 2

# Summer squad-quality prior: top-N FPL costs, end of last season vs start of this.
# Centered z nudges last-season team scoring rates; weight is 1 at GW1 and 0 at GW5.
SQUAD_COST_TOP_N = 15
SQUAD_COST_FADE_GAMES = 4
SQUAD_COST_Z_CLIP = 2.0
SQUAD_COST_RATE_SCALE = 0.06
SQUAD_COST_MIN_RATE = 0.2
