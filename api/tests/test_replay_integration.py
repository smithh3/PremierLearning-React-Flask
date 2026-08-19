import os
import tempfile
import unittest
from pathlib import Path

from flask import Flask

from persistence import DB_Handler
from simulation.historical_data import POSITION_IDS, VaastavSeasonData
from simulation.sim_runner import SimRunner, SimulationConfig


class ReplayIntegrationTests(unittest.TestCase):
    def test_report_path_does_not_change_strategy_hash(self):
        first = SimulationConfig(
            prediction_set_id=1,
            season_start_year=23,
            output_dir='one',
        )
        second = SimulationConfig(
            prediction_set_id=1,
            season_start_year=23,
            output_dir='two',
        )
        self.assertEqual(first.hash(), second.hash())

    def test_one_gameweek_replay_uses_cached_predictions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_path = os.environ.get('DATABASE_PATH')
            os.environ['DATABASE_PATH'] = str(Path(temp_dir) / 'replay.db')
            app = Flask(__name__)
            context = app.app_context()
            context.push()
            db = DB_Handler()
            try:
                db.init_db()
                metadata = {
                    'name': 'integration',
                    'model_config_hash': 'c' * 64,
                    'horizon_gameweeks': 5,
                    'training_policy': 'strict_point_in_time',
                    'availability_mode': 'minutes_proxy',
                    'fixture_mode': 'final_calendar',
                    'random_seed': 42,
                    'config': {'prediction_scope': 'rest_of_season'},
                }
                set_id = db.get_or_create_prediction_set(metadata)
                snapshot_id, _ = db.begin_prediction_snapshot(set_id, 23, 1)
                source = VaastavSeasonData(23)
                state = source.state_at_gameweek(1)
                rows = []
                for player_id, snapshot in state.items():
                    raw = source.players_raw.get(player_id)
                    if raw is None:
                        continue
                    team_id = source.team_id_from_snapshot(snapshot, raw)
                    position = POSITION_IDS.get(
                        str(snapshot.get('position') or '').upper()
                    )
                    fixtures = source.fixtures_for_team(team_id, 1, 2)
                    if position is None or not fixtures:
                        continue
                    for order, fixture in enumerate(fixtures, start=1):
                        opponent_id = (
                            fixture['team_a']
                            if fixture['team_h'] == team_id
                            else fixture['team_h']
                        )
                        rows.append({
                            'season_start_year': 23,
                            'as_of_gameweek': 1,
                            'target_gameweek': 1,
                            'fixture_id': fixture['id'],
                            'fixture_order': order,
                            'player_id': player_id,
                            'player_code': int(float(raw['code'])),
                            'player_name': snapshot.get('name') or str(player_id),
                            'position': position,
                            'team_id': team_id,
                            'opponent_team_id': opponent_id,
                            'opponent': str(opponent_id),
                            'is_home': fixture['team_h'] == team_id,
                            'price': int(snapshot['value']),
                            'selectable': True,
                            'predicted_minutes': 75.0,
                            'predicted_points': (
                                2.0 + position + (player_id % 7) / 10.0
                            ),
                            'points_error': 1.0,
                        })
                db.persist_historical_predictions(snapshot_id, rows)
                db.complete_prediction_snapshot(snapshot_id, len(rows))
                output_dir = Path(temp_dir) / 'reports'
                summary = SimRunner(
                    SimulationConfig(
                        prediction_set_id=set_id,
                        season_start_year=23,
                        gameweek_from=1,
                        gameweek_to=1,
                        horizon_gameweeks=10,
                        output_dir=str(output_dir),
                    ),
                    db_handler=db,
                ).run()
                self.assertGreaterEqual(summary['total_points'], 0)
                self.assertTrue((output_dir / 'summary.json').is_file())
                self.assertEqual(0, summary['transfers_made'])
                self.assertIn('autosub_recoveries', summary)
                self.assertIn('zero_minute_captains', summary)
                repeat_dir = Path(temp_dir) / 'reports-repeat'
                repeat = SimRunner(
                    SimulationConfig(
                        prediction_set_id=set_id,
                        season_start_year=23,
                        gameweek_from=1,
                        gameweek_to=1,
                        horizon_gameweeks=10,
                        output_dir=str(repeat_dir),
                    ),
                    db_handler=db,
                ).run()
                self.assertEqual(
                    db.get_simulation_gameweeks(summary['simulation_run_id']),
                    db.get_simulation_gameweeks(repeat['simulation_run_id']),
                )
            finally:
                db.close_connection()
                context.pop()
                if previous_path is None:
                    os.environ.pop('DATABASE_PATH', None)
                else:
                    os.environ['DATABASE_PATH'] = previous_path


if __name__ == '__main__':
    unittest.main()
