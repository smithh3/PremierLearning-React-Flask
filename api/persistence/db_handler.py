import os
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
            db = g._database = sqlite3.connect(target)
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

    def persist_players(self, players):
        for player in players:
            self.persist_player(player)

    def persist_player(self, player):
        self.cursor.execute(self._sql(
            '''insert into players (id, first_name, surname, team_id, team, current_cost, position, code)
            values (?, ?, ?, ?, ?, ?, ?, ?)'''
        ), player.format_as_db_insert())
        for match in player.matches:
            self.persist_match(match)
        for future_match in player.future_matches:
            self.persist_future_match(future_match)

    def persist_future_match(self, match):
        self.cursor.execute(self._sql(
            '''insert into future_matches (player_id, minutes, points, gameweek, opponent, points_error)
            values (?, ?, ?, ?, ?, ?)'''
        ), match.format_as_future_db_insert())

    def persist_match(self, match):
        self.cursor.execute(self._sql(
            '''insert into matches (player_id, minutes, points, gameweek, opponent)
            values (?, ?, ?, ?, ?)'''
        ), match.format_as_db_insert())

    def get_future_match_jsons(self):
        """player_id -> list of future match dicts in persist order."""
        future_matches = {}
        for db_future_match in self.cursor.execute(
            'select player_id, minutes, points, gameweek, opponent, points_error '
            'from future_matches order by id'
        ).fetchall():
            player_id = db_future_match[0]
            future_match = {
                'minutes': db_future_match[1],
                'points': db_future_match[2],
                'gameweek': db_future_match[3],
                'opponent': db_future_match[4],
                'points_error': db_future_match[5] if len(db_future_match) > 5 else 0.0,
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
