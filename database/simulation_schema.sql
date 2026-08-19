create table if not exists prediction_sets (
    id integer primary key autoincrement not null,
    name varchar(255) not null,
    model_config_hash varchar(64) not null,
    horizon_gameweeks integer not null,
    training_policy varchar(64) not null,
    availability_mode varchar(64) not null,
    fixture_mode varchar(64) not null,
    random_seed integer not null,
    config_json text not null,
    created_at varchar(64) not null default current_timestamp,
    unique (model_config_hash, horizon_gameweeks, training_policy, availability_mode, fixture_mode)
);

create table if not exists prediction_snapshots (
    id integer primary key autoincrement not null,
    prediction_set_id integer not null,
    season_start_year integer not null,
    as_of_gameweek integer not null,
    status varchar(32) not null,
    row_count integer not null default 0,
    min_training_season integer not null default 18,
    max_training_season integer not null,
    max_training_gameweek integer not null,
    warnings_json text not null default '[]',
    started_at varchar(64) not null default current_timestamp,
    completed_at varchar(64),
    unique (prediction_set_id, season_start_year, as_of_gameweek),
    foreign key (prediction_set_id) references prediction_sets(id)
);

create table if not exists historical_player_predictions (
    id integer primary key autoincrement not null,
    prediction_snapshot_id integer not null,
    season_start_year integer not null,
    as_of_gameweek integer not null,
    target_gameweek integer not null,
    fixture_id integer not null,
    fixture_order integer not null,
    player_id integer not null,
    player_code integer not null,
    player_name varchar(255) not null,
    position integer not null,
    team_id integer not null,
    opponent_team_id integer not null,
    opponent varchar(255) not null,
    is_home integer not null,
    price integer not null,
    selectable integer not null,
    availability_mode varchar(64) not null default 'minutes_proxy',
    fixture_mode varchar(64) not null default 'final_calendar',
    predicted_minutes real not null,
    predicted_points real not null,
    points_error real not null default 0,
    foreign key (prediction_snapshot_id) references prediction_snapshots(id),
    unique (prediction_snapshot_id, player_id, fixture_id)
);

create index if not exists idx_historical_predictions_lookup
    on historical_player_predictions (
        season_start_year, as_of_gameweek, target_gameweek, player_id
    );

create index if not exists idx_prediction_snapshots_lookup
    on prediction_snapshots (prediction_set_id, season_start_year, as_of_gameweek);

create table if not exists simulation_runs (
    id integer primary key autoincrement not null,
    prediction_set_id integer not null,
    season_start_year integer not null,
    config_hash varchar(64) not null,
    config_json text not null,
    status varchar(32) not null,
    total_points real,
    transfers_made integer not null default 0,
    final_bank integer,
    final_squad_value integer,
    availability_warning_count integer not null default 0,
    autosub_recoveries integer not null default 0,
    availability_affected_gameweeks integer not null default 0,
    zero_minute_captains integer not null default 0,
    started_at varchar(64) not null default current_timestamp,
    completed_at varchar(64),
    foreign key (prediction_set_id) references prediction_sets(id)
);

create table if not exists simulation_gameweeks (
    id integer primary key autoincrement not null,
    simulation_run_id integer not null,
    gameweek integer not null,
    free_transfers_before integer not null,
    free_transfers_after integer not null,
    bank integer not null,
    expected_hold_current real not null,
    expected_selected_current real not null,
    expected_hold_horizon real not null,
    expected_selected_horizon real not null,
    actual_points real not null,
    cumulative_points real not null,
    captain_id integer,
    vice_captain_id integer,
    captain_scored_id integer,
    starters_json text not null,
    bench_json text not null,
    autosubs_json text not null,
    unexpected_zero_minute_starters integer not null default 0,
    unique (simulation_run_id, gameweek),
    foreign key (simulation_run_id) references simulation_runs(id)
);

create table if not exists simulation_transfers (
    id integer primary key autoincrement not null,
    simulation_run_id integer not null,
    gameweek integer not null,
    player_out_id integer not null,
    player_in_id integer not null,
    sell_price integer not null,
    buy_price integer not null,
    foreign key (simulation_run_id) references simulation_runs(id)
);

create index if not exists idx_simulation_runs_lookup
    on simulation_runs (prediction_set_id, season_start_year, config_hash);

create index if not exists idx_simulation_transfers_lookup
    on simulation_transfers (simulation_run_id, gameweek);
