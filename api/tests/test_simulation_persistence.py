import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from flask import Flask

from persistence import DB_Handler
from simulation.prediction_builder import HistoricalPredictionBuilder


class SimulationPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_path = os.environ.get('DATABASE_PATH')
        os.environ['DATABASE_PATH'] = str(Path(self.temp_dir.name) / 'test.db')
        self.app = Flask(__name__)
        self.context = self.app.app_context()
        self.context.push()
        self.db = DB_Handler()
        self.db.init_db()

    def tearDown(self):
        self.db.close_connection()
        self.context.pop()
        if self.previous_path is None:
            os.environ.pop('DATABASE_PATH', None)
        else:
            os.environ['DATABASE_PATH'] = self.previous_path
        self.temp_dir.cleanup()

    @staticmethod
    def metadata():
        return {
            'name': 'test',
            'model_config_hash': 'a' * 64,
            'horizon_gameweeks': 5,
            'training_policy': 'strict_point_in_time',
            'availability_mode': 'minutes_proxy',
            'fixture_mode': 'final_calendar',
            'random_seed': 42,
            'config': {'test': True},
        }

    @staticmethod
    def prediction_row():
        return {
            'season_start_year': 23,
            'as_of_gameweek': 1,
            'target_gameweek': 1,
            'fixture_id': 1,
            'fixture_order': 1,
            'player_id': 10,
            'player_code': 100,
            'player_name': 'Test Player',
            'position': 3,
            'team_id': 1,
            'opponent_team_id': 2,
            'opponent': 'TST',
            'is_home': True,
            'price': 50,
            'selectable': True,
            'predicted_minutes': 80.0,
            'predicted_points': 5.0,
            'points_error': 1.0,
        }

    def test_prediction_cache_survives_live_database_reset(self):
        set_id = self.db.get_or_create_prediction_set(self.metadata())
        snapshot_id, should_build = self.db.begin_prediction_snapshot(set_id, 23, 1)
        self.assertTrue(should_build)
        self.db.persist_historical_predictions(
            snapshot_id, [self.prediction_row()]
        )
        self.db.complete_prediction_snapshot(snapshot_id, 1)

        self.db.init_db()

        snapshot = self.db.get_prediction_snapshot(set_id, 23, 1)
        self.assertEqual(18, snapshot['min_training_season'])
        self.assertEqual(0, snapshot['max_training_gameweek'])
        rows = self.db.get_historical_predictions(set_id, 23, 1)
        self.assertEqual(1, len(rows))
        self.assertEqual(5.0, rows[0]['predicted_points'])
        self.assertEqual('minutes_proxy', rows[0]['availability_mode'])
        self.assertEqual('final_calendar', rows[0]['fixture_mode'])
        coverage = self.db.prediction_coverage(set_id)[0]
        self.assertEqual(1, coverage['min_target_gameweek'])
        self.assertEqual(1, coverage['max_target_gameweek'])
        self.assertEqual(1, coverage['stored_row_count'])
        _, should_build = self.db.begin_prediction_snapshot(set_id, 23, 1)
        self.assertFalse(should_build)

    def test_force_rebuild_is_idempotent(self):
        set_id = self.db.get_or_create_prediction_set(self.metadata())
        snapshot_id, _ = self.db.begin_prediction_snapshot(set_id, 23, 1)
        self.db.persist_historical_predictions(
            snapshot_id, [self.prediction_row()]
        )
        self.db.complete_prediction_snapshot(snapshot_id, 1)
        rebuilt_id, should_build = self.db.begin_prediction_snapshot(
            set_id, 23, 1, force=True
        )
        self.assertTrue(should_build)
        self.assertEqual(snapshot_id, rebuilt_id)
        self.assertEqual([], self.db.get_historical_predictions(set_id, 23, 1))

    def test_complete_snapshot_with_wrong_row_count_is_rebuilt(self):
        set_id = self.db.get_or_create_prediction_set(self.metadata())
        snapshot_id, _ = self.db.begin_prediction_snapshot(set_id, 23, 1)
        self.db.persist_historical_predictions(
            snapshot_id, [self.prediction_row()]
        )
        self.db.complete_prediction_snapshot(snapshot_id, 2)
        rebuilt_id, should_build = self.db.begin_prediction_snapshot(
            set_id, 23, 1
        )
        self.assertTrue(should_build)
        self.assertEqual(snapshot_id, rebuilt_id)

    def test_snapshot_that_does_not_reach_gw38_is_rebuilt(self):
        set_id = self.db.get_or_create_prediction_set(self.metadata())
        snapshot_id, _ = self.db.begin_prediction_snapshot(set_id, 23, 1)
        self.db.persist_historical_predictions(
            snapshot_id, [self.prediction_row()]
        )
        self.db.complete_prediction_snapshot(snapshot_id, 1)
        rebuilt_id, should_build = self.db.begin_prediction_snapshot(
            set_id,
            23,
            1,
            expected_max_target_gameweek=38,
        )
        self.assertTrue(should_build)
        self.assertEqual(snapshot_id, rebuilt_id)

    def test_duplicate_player_fixture_prediction_is_rejected(self):
        set_id = self.db.get_or_create_prediction_set(self.metadata())
        snapshot_id, _ = self.db.begin_prediction_snapshot(set_id, 23, 1)
        row = self.prediction_row()
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.persist_historical_predictions(snapshot_id, [row, row])
        self.db.connection.rollback()

    def test_explicit_cache_rejects_a_different_model_hash(self):
        set_id = self.db.get_or_create_prediction_set(self.metadata())
        with self.assertRaisesRegex(ValueError, 'different model/config hash'):
            HistoricalPredictionBuilder(
                self.db,
                seasons=(23,),
                prediction_set_id=set_id,
            )

    def test_simulation_results_are_persisted(self):
        set_id = self.db.get_or_create_prediction_set(self.metadata())
        run_id = self.db.create_simulation_run(set_id, 23, 'b' * 64, {'x': 1})
        row = {
            'gameweek': 1,
            'free_transfers_before': 15,
            'free_transfers_after': 1,
            'bank': 5,
            'expected_hold_current': 0,
            'expected_selected_current': 50,
            'expected_hold_horizon': 0,
            'expected_selected_horizon': 250,
            'actual_points': 60,
            'cumulative_points': 60,
            'captain_id': 10,
            'vice_captain_id': 11,
            'captain_scored_id': 10,
            'starters': list(range(1, 12)),
            'bench': list(range(12, 16)),
            'autosubs': [],
            'unexpected_zero_minute_starters': 0,
        }
        self.db.persist_simulation_gameweek(run_id, row)
        self.db.persist_simulation_transfers(run_id, 1, [])
        self.db.complete_simulation_run(run_id, 60, 0, 5, 1005, 0)
        self.db.cursor.execute(
            'select status, total_points from simulation_runs where id = ?',
            (run_id,),
        )
        self.assertEqual(('complete', 60.0), self.db.cursor.fetchone())
        self.assertEqual('complete', self.db.get_simulation_run(run_id)['status'])
        gameweeks = self.db.get_simulation_gameweeks(run_id)
        self.assertEqual(1, len(gameweeks))
        self.assertEqual(10, gameweeks[0]['captain_scored_id'])


if __name__ == '__main__':
    unittest.main()
