import os
import json
import sqlite3
from pathlib import Path
from urllib.parse import urlparse, unquote

from flask import g

from premierlearning import RawMatch
from premierlearning import RawPlayer

_DB_DIR = Path(__file__).resolve().parents[2] / 'database'
DATABASE = str(_DB_DIR / 'fantasy.db')
DROP = str(_DB_DIR / 'drop_tables.sql')
SCHEMA = str(_DB_DIR / 'schema.sql')
POSTGRES_DROP = str(_DB_DIR / 'drop_tables.postgres.sql')
POSTGRES_SCHEMA = str(_DB_DIR / 'schema.postgres.sql')
SIMULATION_SCHEMA = str(_DB_DIR / 'simulation_schema.sql')
POSTGRES_SIMULATION_SCHEMA = str(_DB_DIR / 'simulation_schema.postgres.sql')


def database_config():
    url = os.environ.get('DATABASE_URL', '').strip()
    path = os.environ.get('DATABASE_PATH', '').strip()
    if url.startswith('postgres://') or url.startswith('postgresql://'):
        if url.startswith('postgres://'):
            url = 'postgresql://' + url[len('postgres://'):]
        return 'postgres', url
    if url.startswith('sqlite:///'):
        raw = url[len('sqlite:///'):]
        return 'sqlite', raw
    if path:
        return 'sqlite', path
    return 'sqlite', DATABASE


def _connect_postgres(url):
    import psycopg2
    parsed = urlparse(url)
    connect_kwargs = {
        'host': parsed.hostname,
        'port': parsed.port or 5432,
        'user': unquote(parsed.username or ''),
        'password': unquote(parsed.password or ''),
        'dbname': (parsed.path or '/postgres').lstrip('/') or 'postgres',
    }
    sslmode = None
    if parsed.query:
        for part in parsed.query.split('&'):
            if part.startswith('sslmode='):
                sslmode = part.split('=', 1)[1]
    if sslmode:
        connect_kwargs['sslmode'] = sslmode
    return psycopg2.connect(**connect_kwargs)


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        kind, target = database_config()
        g._database_kind = kind
        if kind == 'postgres':
            db = g._database = _connect_postgres(target)
        else:
            parent = os.path.dirname(os.path.abspath(target))
            if parent:
                os.makedirs(parent, exist_ok=True)
            db = g._database = sqlite3.connect(target, timeout=60)
            db.execute('pragma journal_mode=WAL')
            db.execute('pragma busy_timeout=60000')
    return db


def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


class DB_Handler:

    def __init__(self):
        self.kind = getattr(g, '_database_kind', None) or database_config()[0]
        self.connection = get_db()
        self.kind = getattr(g, '_database_kind', self.kind)
        self.cursor = self.connection.cursor()
        self._simulation_schema_ready = False

    def close_connection(self):
        self.cursor.close()
        self.connection.close()

    def _sql(self, sql):
        if self.kind == 'postgres':
            return sql.replace('?', '%s')
        return sql

    def run_sql_script(self, path):
        with open(path) as f:
            script = f.read()
        if self.kind == 'postgres':
            statements = [part.strip() for part in script.split(';') if part.strip()]
            for statement in statements:
                self.cursor.execute(statement)
            return
        self.cursor.executescript(script)

    def init_db(self):
        if self.kind == 'postgres':
            self.run_sql_script(POSTGRES_DROP)
            self.run_sql_script(POSTGRES_SCHEMA)
        else:
            self.run_sql_script(DROP)
            self.run_sql_script(SCHEMA)
        self.ensure_future_match_columns()
        # Historical predictions are intentionally outside the destructive live
        # snapshot lifecycle.
        self.ensure_simulation_schema()

    def ensure_simulation_schema(self):
        if self._simulation_schema_ready:
            return
        self.run_sql_script(
            POSTGRES_SIMULATION_SCHEMA if self.kind == 'postgres' else SIMULATION_SCHEMA
        )
        if self.kind == 'postgres':
            self.cursor.execute(
                """alter table historical_player_predictions
                   add column if not exists availability_mode varchar(64)
                   not null default 'minutes_proxy'"""
            )
            self.cursor.execute(
                """alter table historical_player_predictions
                   add column if not exists fixture_mode varchar(64)
                   not null default 'final_calendar'"""
            )
            self.cursor.execute(
                """alter table prediction_snapshots
                   add column if not exists min_training_season integer
                   not null default 18"""
            )
            self.cursor.execute(
                """alter table simulation_gameweeks
                   add column if not exists captain_scored_id integer"""
            )
            self.cursor.execute(
                """alter table historical_player_predictions
                   add column if not exists predicted_pp90 double precision"""
            )
            self.cursor.execute(
                """alter table historical_player_predictions
                   add column if not exists chance_of_playing integer"""
            )
            for column in (
                'autosub_recoveries',
                'availability_affected_gameweeks',
                'zero_minute_captains',
            ):
                self.cursor.execute(
                    f"""alter table simulation_runs
                        add column if not exists {column} integer
                        not null default 0"""
                )
        else:
            self.cursor.execute('pragma table_info(historical_player_predictions)')
            columns = {row[1] for row in self.cursor.fetchall()}
            if 'availability_mode' not in columns:
                self.cursor.execute(
                    """alter table historical_player_predictions
                       add column availability_mode varchar(64)
                       not null default 'minutes_proxy'"""
                )
            if 'fixture_mode' not in columns:
                self.cursor.execute(
                    """alter table historical_player_predictions
                       add column fixture_mode varchar(64)
                       not null default 'final_calendar'"""
                )
            self.cursor.execute('pragma table_info(prediction_snapshots)')
            snapshot_columns = {row[1] for row in self.cursor.fetchall()}
            if 'min_training_season' not in snapshot_columns:
                self.cursor.execute(
                    """alter table prediction_snapshots
                       add column min_training_season integer
                       not null default 18"""
                )
            self.cursor.execute('pragma table_info(simulation_gameweeks)')
            gameweek_columns = {row[1] for row in self.cursor.fetchall()}
            if 'captain_scored_id' not in gameweek_columns:
                self.cursor.execute(
                    """alter table simulation_gameweeks
                       add column captain_scored_id integer"""
                )
            self.cursor.execute('pragma table_info(historical_player_predictions)')
            prediction_columns = {row[1] for row in self.cursor.fetchall()}
            if 'predicted_pp90' not in prediction_columns:
                self.cursor.execute(
                    """alter table historical_player_predictions
                       add column predicted_pp90 real"""
                )
            if 'chance_of_playing' not in prediction_columns:
                self.cursor.execute(
                    """alter table historical_player_predictions
                       add column chance_of_playing integer"""
                )
            self.cursor.execute('pragma table_info(simulation_runs)')
            run_columns = {row[1] for row in self.cursor.fetchall()}
            for column in (
                'autosub_recoveries',
                'availability_affected_gameweeks',
                'zero_minute_captains',
            ):
                if column not in run_columns:
                    self.cursor.execute(
                        f"""alter table simulation_runs
                            add column {column} integer not null default 0"""
                    )
        self.connection.commit()
        self._simulation_schema_ready = True

    def ensure_future_match_columns(self):
        """Keep chance and pp90 on live future_matches so squadtools can scale XI scores.

        h1_points and horizon_scale are the pre-redistribution prediction and the
        factor applied to it, kept so the report can show both sides of the split.
        points_basis 'raw' marks expected points that already carry minutes risk
        (learning_v2): the optimiser uses them as they are. minutes_band_* are
        the bands head's chance of 0, 1-59 and 60+ minutes, when it ran.
        """
        added = (
            ('predicted_pp90', 'double precision', 'real'),
            ('chance_of_playing', 'integer', 'integer'),
            ('h1_points', 'double precision', 'real'),
            ('horizon_scale', 'double precision', 'real'),
            ('points_basis', 'varchar(16)', 'varchar(16)'),
            ('minutes_band_0', 'double precision', 'real'),
            ('minutes_band_1_59', 'double precision', 'real'),
            ('minutes_band_60', 'double precision', 'real'),
        )
        if self.kind == 'postgres':
            for column, postgres_type, _sqlite_type in added:
                self.cursor.execute(
                    'alter table future_matches add column if not exists %s %s'
                    % (column, postgres_type)
                )
        else:
            self.cursor.execute('pragma table_info(future_matches)')
            columns = {row[1] for row in self.cursor.fetchall()}
            for column, _postgres_type, sqlite_type in added:
                if column not in columns:
                    self.cursor.execute(
                        'alter table future_matches add column %s %s' % (column, sqlite_type)
                    )
        self.connection.commit()

    def _insert_id(self, sql, params):
        if self.kind == 'postgres':
            self.cursor.execute(self._sql(sql) + ' returning id', params)
            return int(self.cursor.fetchone()[0])
        self.cursor.execute(self._sql(sql), params)
        return int(self.cursor.lastrowid)

    def get_or_create_prediction_set(self, metadata):
        self.ensure_simulation_schema()
        key = (
            metadata['model_config_hash'],
            int(metadata['horizon_gameweeks']),
            metadata['training_policy'],
            metadata['availability_mode'],
            metadata['fixture_mode'],
        )
        self.cursor.execute(self._sql(
            '''select id from prediction_sets
               where model_config_hash = ? and horizon_gameweeks = ?
                 and training_policy = ? and availability_mode = ?
                 and fixture_mode = ?'''
        ), key)
        existing = self.cursor.fetchone()
        if existing:
            return int(existing[0])
        prediction_set_id = self._insert_id(
            '''insert into prediction_sets
               (name, model_config_hash, horizon_gameweeks, training_policy,
                availability_mode, fixture_mode, random_seed, config_json)
               values (?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                metadata['name'],
                metadata['model_config_hash'],
                int(metadata['horizon_gameweeks']),
                metadata['training_policy'],
                metadata['availability_mode'],
                metadata['fixture_mode'],
                int(metadata['random_seed']),
                json.dumps(metadata.get('config') or {}, sort_keys=True),
            ),
        )
        self.connection.commit()
        return prediction_set_id

    def replace_prediction_cache(self, metadata):
        """Drop every historical prediction set and insert one fresh cache."""
        self.ensure_simulation_schema()
        self.cursor.execute('delete from historical_player_predictions')
        self.cursor.execute('delete from prediction_snapshots')
        self.cursor.execute('delete from simulation_transfers')
        self.cursor.execute('delete from simulation_gameweeks')
        self.cursor.execute('delete from simulation_runs')
        self.cursor.execute('delete from prediction_sets')
        self.connection.commit()
        return self.get_or_create_prediction_set(metadata)

    def get_prediction_set(self, prediction_set_id):
        self.ensure_simulation_schema()
        self.cursor.execute(self._sql(
            '''select id, name, model_config_hash, horizon_gameweeks,
                      training_policy, availability_mode, fixture_mode,
                      random_seed, config_json, created_at
               from prediction_sets where id = ?'''
        ), (int(prediction_set_id),))
        row = self.cursor.fetchone()
        if not row:
            return None
        return {
            'id': int(row[0]),
            'name': row[1],
            'model_config_hash': row[2],
            'horizon_gameweeks': int(row[3]),
            'training_policy': row[4],
            'availability_mode': row[5],
            'fixture_mode': row[6],
            'random_seed': int(row[7]),
            'config': json.loads(row[8] or '{}'),
            'created_at': str(row[9]),
        }

    def update_prediction_set_metadata(self, prediction_set_id, metadata):
        self.ensure_simulation_schema()
        self.cursor.execute(self._sql(
            '''update prediction_sets
               set name = ?, model_config_hash = ?, horizon_gameweeks = ?,
                   training_policy = ?, availability_mode = ?,
                   fixture_mode = ?, random_seed = ?, config_json = ?
               where id = ?'''
        ), (
            metadata['name'],
            metadata['model_config_hash'],
            int(metadata['horizon_gameweeks']),
            metadata['training_policy'],
            metadata['availability_mode'],
            metadata['fixture_mode'],
            int(metadata['random_seed']),
            json.dumps(metadata.get('config') or {}, sort_keys=True),
            int(prediction_set_id),
        ))
        if self.cursor.rowcount != 1:
            self.connection.rollback()
            raise ValueError(f'Unknown prediction set {prediction_set_id}')
        self.connection.commit()

    def latest_prediction_set_id(self, season_start_year=None):
        """Newest set, ignoring model-evaluation sets (named eval-*): those
        are experiments and must never become a replay's or manager's default.
        ('%%' because psycopg2 formats queries that have parameters; to SQLite
        it is two wildcards.)"""
        self.ensure_simulation_schema()
        if season_start_year is None:
            self.cursor.execute(
                "select id from prediction_sets where name not like 'eval-%' "
                'order by id desc limit 1'
            )
        else:
            self.cursor.execute(self._sql(
                '''select distinct p.id
                   from prediction_sets p
                   join prediction_snapshots s on s.prediction_set_id = p.id
                   where s.season_start_year = ? and s.status = 'complete'
                     and p.name not like 'eval-%%'
                   order by p.id desc limit 1'''
            ), (int(season_start_year),))
        row = self.cursor.fetchone()
        return int(row[0]) if row else None

    def get_prediction_snapshot(self, prediction_set_id, season_start_year, as_of_gameweek):
        self.ensure_simulation_schema()
        self.cursor.execute(self._sql(
            '''select id, status, row_count, min_training_season,
                      max_training_season, max_training_gameweek, warnings_json
               from prediction_snapshots
               where prediction_set_id = ? and season_start_year = ?
                 and as_of_gameweek = ?'''
        ), (
            int(prediction_set_id),
            int(season_start_year),
            int(as_of_gameweek),
        ))
        row = self.cursor.fetchone()
        if not row:
            return None
        return {
            'id': int(row[0]),
            'status': row[1],
            'row_count': int(row[2]),
            'min_training_season': int(row[3]),
            'max_training_season': int(row[4]),
            'max_training_gameweek': int(row[5]),
            'warnings': json.loads(row[6] or '[]'),
        }

    def begin_prediction_snapshot(
        self,
        prediction_set_id,
        season_start_year,
        as_of_gameweek,
        force=False,
        expected_max_target_gameweek=None,
    ):
        existing = self.get_prediction_snapshot(
            prediction_set_id, season_start_year, as_of_gameweek
        )
        if existing and existing['status'] == 'complete' and not force:
            self.cursor.execute(self._sql(
                '''select count(*), min(target_gameweek), max(target_gameweek)
                   from historical_player_predictions
                   where prediction_snapshot_id = ?'''
            ), (existing['id'],))
            row = self.cursor.fetchone()
            stored_rows = int(row[0])
            target_bounds_valid = (
                row[1] is not None
                and row[2] is not None
                and int(row[1]) == int(as_of_gameweek)
                and (
                    expected_max_target_gameweek is None
                    or int(row[2]) == int(expected_max_target_gameweek)
                )
            )
            if (
                existing['row_count'] > 0
                and stored_rows == existing['row_count']
                and target_bounds_valid
            ):
                return existing['id'], False
        max_training_season = int(season_start_year)
        max_training_gameweek = max(0, int(as_of_gameweek) - 1)
        if existing:
            snapshot_id = existing['id']
            self.cursor.execute(self._sql(
                'delete from historical_player_predictions where prediction_snapshot_id = ?'
            ), (snapshot_id,))
            self.cursor.execute(self._sql(
                '''update prediction_snapshots
                   set status = ?, row_count = 0, min_training_season = 18,
                       max_training_season = ?,
                       max_training_gameweek = ?, warnings_json = ?,
                       started_at = current_timestamp, completed_at = null
                   where id = ?'''
            ), ('building', max_training_season, max_training_gameweek, '[]', snapshot_id))
        else:
            snapshot_id = self._insert_id(
                '''insert into prediction_snapshots
                   (prediction_set_id, season_start_year, as_of_gameweek, status,
                    min_training_season, max_training_season,
                    max_training_gameweek)
                   values (?, ?, ?, ?, ?, ?, ?)''',
                (
                    int(prediction_set_id),
                    int(season_start_year),
                    int(as_of_gameweek),
                    'building',
                    18,
                    max_training_season,
                    max_training_gameweek,
                ),
            )
        self.connection.commit()
        return snapshot_id, True

    def persist_historical_predictions(self, prediction_snapshot_id, rows):
        sql = self._sql(
            '''insert into historical_player_predictions
               (prediction_snapshot_id, season_start_year, as_of_gameweek,
                target_gameweek, fixture_id, fixture_order, player_id,
                player_code, player_name, position, team_id, opponent_team_id,
                opponent, is_home, price, selectable, predicted_minutes,
                predicted_pp90, predicted_points, chance_of_playing,
                points_error, availability_mode, fixture_mode)
               values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        )
        values = [
            (
                int(prediction_snapshot_id),
                int(row['season_start_year']),
                int(row['as_of_gameweek']),
                int(row['target_gameweek']),
                int(row['fixture_id']),
                int(row['fixture_order']),
                int(row['player_id']),
                int(row['player_code']),
                row['player_name'],
                int(row['position']),
                int(row['team_id']),
                int(row['opponent_team_id']),
                row['opponent'],
                bool(row['is_home']),
                int(row['price']),
                bool(row['selectable']),
                float(row['predicted_minutes']),
                None if row.get('predicted_pp90') is None else float(row['predicted_pp90']),
                float(row['predicted_points']),
                None if row.get('chance_of_playing') is None else int(row['chance_of_playing']),
                float(row.get('points_error') or 0.0),
                row.get('availability_mode') or 'minutes_proxy',
                row.get('fixture_mode') or 'final_calendar',
            )
            for row in rows
        ]
        self.cursor.executemany(sql, values)

    def complete_prediction_snapshot(self, prediction_snapshot_id, row_count, warnings=None):
        self.cursor.execute(self._sql(
            '''update prediction_snapshots
               set status = ?, row_count = ?, warnings_json = ?,
                   completed_at = current_timestamp
               where id = ?'''
        ), (
            'complete',
            int(row_count),
            json.dumps(warnings or []),
            int(prediction_snapshot_id),
        ))
        self.connection.commit()

    def fail_prediction_snapshot(self, prediction_snapshot_id, warning):
        self.connection.rollback()
        self.cursor.execute(self._sql(
            '''update prediction_snapshots
               set status = ?, warnings_json = ?, completed_at = current_timestamp
               where id = ?'''
        ), ('failed', json.dumps([str(warning)]), int(prediction_snapshot_id)))
        self.connection.commit()

    def get_historical_predictions(
        self, prediction_set_id, season_start_year, as_of_gameweek,
        max_target_gameweek=None,
    ):
        self.ensure_simulation_schema()
        params = [
            int(prediction_set_id),
            int(season_start_year),
            int(as_of_gameweek),
        ]
        target_filter = ''
        if max_target_gameweek is not None:
            target_filter = 'and p.target_gameweek <= ?'
            params.append(int(max_target_gameweek))
        sql = '''select p.target_gameweek, p.fixture_id, p.fixture_order,
                      p.player_id, p.player_code, p.player_name, p.position,
                      p.team_id, p.opponent_team_id, p.opponent, p.is_home,
                      p.price, p.selectable, p.predicted_minutes,
                      p.predicted_pp90, p.predicted_points, p.chance_of_playing,
                      p.points_error, p.availability_mode, p.fixture_mode
               from historical_player_predictions p
               join prediction_snapshots s on s.id = p.prediction_snapshot_id
               where s.prediction_set_id = ? and s.season_start_year = ?
                 and s.as_of_gameweek = ? and s.status = 'complete'
                 %s
               order by p.player_id, p.target_gameweek, p.fixture_order,
                        p.fixture_id''' % target_filter
        self.cursor.execute(self._sql(sql), tuple(params))
        columns = (
            'target_gameweek', 'fixture_id', 'fixture_order', 'player_id',
            'player_code', 'player_name', 'position', 'team_id',
            'opponent_team_id', 'opponent', 'is_home', 'price', 'selectable',
            'predicted_minutes', 'predicted_pp90', 'predicted_points',
            'chance_of_playing', 'points_error', 'availability_mode',
            'fixture_mode',
        )
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]

    def prediction_coverage(self, prediction_set_id):
        self.ensure_simulation_schema()
        self.cursor.execute(self._sql(
            '''select s.season_start_year, s.as_of_gameweek, s.status,
                      s.row_count, min(p.target_gameweek),
                      max(p.target_gameweek), count(p.id)
               from prediction_snapshots s
               left join historical_player_predictions p
                 on p.prediction_snapshot_id = s.id
               where s.prediction_set_id = ?
               group by s.id, s.season_start_year, s.as_of_gameweek,
                        s.status, s.row_count
               order by s.season_start_year, s.as_of_gameweek'''
        ), (int(prediction_set_id),))
        return [
            {
                'season_start_year': int(row[0]),
                'as_of_gameweek': int(row[1]),
                'status': row[2],
                'row_count': int(row[3]),
                'min_target_gameweek': None if row[4] is None else int(row[4]),
                'max_target_gameweek': None if row[5] is None else int(row[5]),
                'stored_row_count': int(row[6]),
            }
            for row in self.cursor.fetchall()
        ]

    def create_simulation_run(
        self, prediction_set_id, season_start_year, config_hash, config
    ):
        self.ensure_simulation_schema()
        run_id = self._insert_id(
            '''insert into simulation_runs
               (prediction_set_id, season_start_year, config_hash, config_json, status)
               values (?, ?, ?, ?, ?)''',
            (
                int(prediction_set_id),
                int(season_start_year),
                config_hash,
                json.dumps(config, sort_keys=True),
                'running',
            ),
        )
        self.connection.commit()
        return run_id

    def get_simulation_run(self, simulation_run_id):
        self.ensure_simulation_schema()
        self.cursor.execute(self._sql(
            '''select id, prediction_set_id, season_start_year, config_hash,
                      config_json, status, total_points, transfers_made,
                      final_bank, final_squad_value,
                      availability_warning_count, autosub_recoveries,
                      availability_affected_gameweeks, zero_minute_captains,
                      started_at, completed_at
               from simulation_runs where id = ?'''
        ), (int(simulation_run_id),))
        row = self.cursor.fetchone()
        if not row:
            return None
        return {
            'id': int(row[0]),
            'prediction_set_id': int(row[1]),
            'season_start_year': int(row[2]),
            'config_hash': row[3],
            'config': json.loads(row[4] or '{}'),
            'status': row[5],
            'total_points': None if row[6] is None else float(row[6]),
            'transfers_made': int(row[7]),
            'final_bank': row[8],
            'final_squad_value': row[9],
            'availability_warning_count': int(row[10]),
            'autosub_recoveries': int(row[11]),
            'availability_affected_gameweeks': int(row[12]),
            'zero_minute_captains': int(row[13]),
            'started_at': str(row[14]),
            'completed_at': None if row[15] is None else str(row[15]),
        }

    def get_simulation_gameweeks(self, simulation_run_id):
        self.ensure_simulation_schema()
        self.cursor.execute(self._sql(
            '''select gameweek, free_transfers_before, free_transfers_after,
                      bank, expected_hold_current, expected_selected_current,
                      expected_hold_horizon, expected_selected_horizon,
                      actual_points, cumulative_points, captain_id,
                      vice_captain_id, captain_scored_id, starters_json,
                      bench_json, autosubs_json,
                      unexpected_zero_minute_starters
               from simulation_gameweeks where simulation_run_id = ?
               order by gameweek'''
        ), (int(simulation_run_id),))
        result = []
        for row in self.cursor.fetchall():
            result.append({
                'gameweek': int(row[0]),
                'free_transfers_before': int(row[1]),
                'free_transfers_after': int(row[2]),
                'bank': int(row[3]),
                'expected_hold_current': float(row[4]),
                'expected_selected_current': float(row[5]),
                'expected_hold_horizon': float(row[6]),
                'expected_selected_horizon': float(row[7]),
                'actual_points': float(row[8]),
                'cumulative_points': float(row[9]),
                'captain_id': row[10],
                'vice_captain_id': row[11],
                'captain_scored_id': row[12],
                'starters': json.loads(row[13] or '[]'),
                'bench': json.loads(row[14] or '[]'),
                'autosubs': json.loads(row[15] or '[]'),
                'unexpected_zero_minute_starters': int(row[16]),
            })
        return result

    def persist_simulation_gameweek(self, simulation_run_id, row):
        self.cursor.execute(self._sql(
            '''insert into simulation_gameweeks
               (simulation_run_id, gameweek, free_transfers_before,
                free_transfers_after, bank, expected_hold_current,
                expected_selected_current, expected_hold_horizon,
                expected_selected_horizon, actual_points, cumulative_points,
                captain_id, vice_captain_id, captain_scored_id, starters_json,
                bench_json, autosubs_json, unexpected_zero_minute_starters)
               values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        ), (
            int(simulation_run_id),
            int(row['gameweek']),
            int(row['free_transfers_before']),
            int(row['free_transfers_after']),
            int(row['bank']),
            float(row['expected_hold_current']),
            float(row['expected_selected_current']),
            float(row['expected_hold_horizon']),
            float(row['expected_selected_horizon']),
            float(row['actual_points']),
            float(row['cumulative_points']),
            row.get('captain_id'),
            row.get('vice_captain_id'),
            row.get('captain_scored_id'),
            json.dumps(row.get('starters') or []),
            json.dumps(row.get('bench') or []),
            json.dumps(row.get('autosubs') or []),
            int(row.get('unexpected_zero_minute_starters') or 0),
        ))

    def persist_simulation_transfers(self, simulation_run_id, gameweek, transfers):
        if not transfers:
            return
        self.cursor.executemany(self._sql(
            '''insert into simulation_transfers
               (simulation_run_id, gameweek, player_out_id, player_in_id,
                sell_price, buy_price)
               values (?, ?, ?, ?, ?, ?)'''
        ), [
            (
                int(simulation_run_id),
                int(gameweek),
                int(transfer['player_out_id']),
                int(transfer['player_in_id']),
                int(transfer['sell_price']),
                int(transfer['buy_price']),
            )
            for transfer in transfers
        ])

    def complete_simulation_run(
        self,
        simulation_run_id,
        total_points,
        transfers_made,
        final_bank,
        final_squad_value,
        availability_warning_count,
        autosub_recoveries=0,
        availability_affected_gameweeks=0,
        zero_minute_captains=0,
    ):
        self.cursor.execute(self._sql(
            '''update simulation_runs
               set status = ?, total_points = ?, transfers_made = ?,
                   final_bank = ?, final_squad_value = ?,
                   availability_warning_count = ?, autosub_recoveries = ?,
                   availability_affected_gameweeks = ?,
                   zero_minute_captains = ?,
                   completed_at = current_timestamp
               where id = ?'''
        ), (
            'complete',
            float(total_points),
            int(transfers_made),
            int(final_bank),
            int(final_squad_value),
            int(availability_warning_count),
            int(autosub_recoveries),
            int(availability_affected_gameweeks),
            int(zero_minute_captains),
            int(simulation_run_id),
        ))
        self.connection.commit()

    def fail_simulation_run(self, simulation_run_id):
        self.connection.rollback()
        self.cursor.execute(self._sql(
            '''update simulation_runs
               set status = ?, completed_at = current_timestamp where id = ?'''
        ), ('failed', int(simulation_run_id)))
        self.connection.commit()

    def persist_players(self, players):
        for player in players:
            self.persist_player(player)

    def persist_player(self, player):
        player_id = int(player.id)
        self.cursor.execute(self._sql(
            '''insert into players (id, first_name, surname, team_id, team, current_cost, position, code)
            values (?, ?, ?, ?, ?, ?, ?, ?)'''
        ), player.format_as_db_insert())
        for match in (player.matches or []):
            self.persist_match(match, player_id=player_id)
        for future_match in (player.future_matches or []):
            self.persist_future_match(future_match, player_id=player_id)

    def persist_future_match(self, match, player_id=None):
        self.ensure_future_match_columns()
        row = list(match.format_as_future_db_insert())
        if player_id is not None:
            row[0] = int(player_id)
        self.cursor.execute(self._sql(
            '''insert into future_matches (
                   player_id, minutes, points, gameweek, opponent, points_error,
                   predicted_pp90, chance_of_playing, h1_points, horizon_scale,
                   points_basis, minutes_band_0, minutes_band_1_59, minutes_band_60
               ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        ), tuple(row))

    def persist_match(self, match, player_id=None):
        row = list(match.format_as_db_insert())
        if player_id is not None:
            row[0] = int(player_id)
        self.cursor.execute(self._sql(
            '''insert into matches (player_id, minutes, points, gameweek, opponent)
            values (?, ?, ?, ?, ?)'''
        ), tuple(row))

    def get_future_match_jsons(self):
        """player_id -> list of future match dicts in persist order."""
        self.ensure_future_match_columns()
        future_matches = {}
        for db_future_match in self.cursor.execute(
            'select player_id, minutes, points, gameweek, opponent, points_error, '
            'predicted_pp90, chance_of_playing, h1_points, horizon_scale, '
            'points_basis, minutes_band_0, minutes_band_1_59, minutes_band_60 '
            'from future_matches order by id'
        ).fetchall():
            player_id = db_future_match[0]
            future_match = {
                'minutes': db_future_match[1],
                'points': db_future_match[2],
                'gameweek': db_future_match[3],
                'opponent': db_future_match[4],
                'points_error': db_future_match[5] if len(db_future_match) > 5 else 0.0,
                'predicted_pp90': db_future_match[6] if len(db_future_match) > 6 else None,
                'chance_of_playing': db_future_match[7] if len(db_future_match) > 7 else None,
                'h1_points': db_future_match[8] if len(db_future_match) > 8 else None,
                'horizon_scale': db_future_match[9] if len(db_future_match) > 9 else None,
                'points_basis': db_future_match[10] if len(db_future_match) > 10 else None,
                'minutes_bands': (
                    tuple(db_future_match[11:14])
                    if len(db_future_match) > 13 and db_future_match[11] is not None
                    else None
                ),
            }
            future_matches.setdefault(player_id, []).append(future_match)
        return future_matches

    @staticmethod
    def convert_match_from_json(player_id, match_json):
        match = RawMatch()
        match.player_id = player_id
        match.minutes = match_json['minutes']
        match.points = match_json['points']
        match.gameweek = match_json['gameweek']
        match.opponent = match_json['opponent']
        match.points_error = float(match_json.get('points_error') or 0.0)
        pp90 = match_json.get('predicted_pp90')
        match.predicted_pp90 = None if pp90 is None or pp90 == '' else float(pp90)
        chance = match_json.get('chance_of_playing')
        match.chance_of_playing = None if chance is None or chance == '' else int(chance)
        h1_points = match_json.get('h1_points')
        match.h1_points = None if h1_points is None or h1_points == '' else float(h1_points)
        scale = match_json.get('horizon_scale')
        match.horizon_scale = None if scale is None or scale == '' else float(scale)
        match.points_basis = match_json.get('points_basis') or None
        bands = match_json.get('minutes_bands')
        match.minutes_bands = (
            None if not bands or bands[0] is None
            else tuple(float(value) for value in bands)
        )
        return match

    def get_match_jsons(self):
        """player_id -> list of past match dicts in persist order (carry then this season)."""
        matches = {}
        for db_match in self.cursor.execute(
            'select player_id, minutes, points, gameweek, opponent from matches order by id'
        ).fetchall():
            player_id = db_match[0]
            match = {
                'minutes': db_match[1],
                'points': db_match[2],
                'gameweek': db_match[3],
                'opponent': db_match[4]
            }
            matches.setdefault(player_id, []).append(match)
        return matches

    def get_player_jsons(self):
        players = {}
        matches = self.get_match_jsons()
        future_matches = self.get_future_match_jsons()
        for db_player in self.cursor.execute(
            'select id, first_name, surname, team_id, team, current_cost, position, code from players'
        ).fetchall():
            player_id = db_player[0]
            player = {
                'id': player_id,
                'first_name': db_player[1],
                'last_name': db_player[2],
                'team_id': db_player[3],
                'team': db_player[4],
                'current_cost': db_player[5],
                'position': db_player[6],
                'code': db_player[7],
                'matches': matches.get(player_id, []),
                'future_matches': future_matches.get(player_id, [])
            }
            players[player_id] = player
        return players

    def get_player_objects(self):
        players = {}
        player_jsons = self.get_player_jsons()
        for player_id, player_json in player_jsons.items():
            player = RawPlayer()
            player.id = player_id
            player.first_name = player_json['first_name']
            player.last_name = player_json['last_name']
            player.team_id = player_json['team_id']
            player.team_name = player_json['team']
            player.current_cost = player_json['current_cost']
            player.position = player_json['position']
            player.code = player_json['code']
            player.matches = [
                self.convert_match_from_json(player_id, match_json)
                for match_json in player_json['matches']
            ]
            player.future_matches = [
                self.convert_match_from_json(player_id, future_match_json)
                for future_match_json in player_json['future_matches']
            ]
            players[player_id] = player
        return players
