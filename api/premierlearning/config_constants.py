SEASON_YR = 26
PLAYER_JSON_FILE_LOCATION = 'data/season_%i_%i/players/player_'

# Past seasons that use vaastav-style CSV (players_raw.csv, fixtures.csv, players/*/gw.csv)
VAASTAV_CSV_YEARS = (18, 19, 23, 24, 25)

# Points total window for the multi-GW flat model (summed team xg/xc + redistribute)
POINTS_HORIZON = 5

# Early-season prior: (last_pp90 + 0.5 * two_season_pp90) * exp(-matches_played / tau)
# At matches_played=0 → weight 1; fades toward 0 as this-season apps grow.
PRIOR_EARLY_TAU = 6.0
TWO_SEASON_PRIOR_SCALE = 0.5
# Last-season second half for minutes priors (DNP-inclusive fixture count)
HALF_SEASON_FIXTURES = 19
# Predict-only: drop DNP runs of this length from minutes features, but only if
# the player appeared in the last RETURN_CONSECUTIVE_APPS fixtures (back in side).
INJURY_DNP_STREAK = 3
RETURN_CONSECUTIVE_APPS = 2
