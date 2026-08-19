import unittest
from collections import Counter, defaultdict
from types import SimpleNamespace
from unittest.mock import patch

from premierlearning.runner import Runner
from simulation.historical_data import VaastavSeasonData
from simulation.prediction_builder import _prediction_rows


class HistoricalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = VaastavSeasonData(23)

    def test_snapshot_history_stops_before_decision_gameweek(self):
        season = self.source.build_current_season(10, 5)
        gameweeks = [
            match.gameweek
            for player in season.players
            for match in player.matches
            if match.gameweek is not None
        ]
        self.assertTrue(gameweeks)
        self.assertLess(max(gameweeks), 10)
        runner = Runner(
            current_season=season,
            target_season_year=23,
            as_of_gameweek=10,
            horizon_gameweeks=5,
            update_data=False,
        )
        runner.validate_temporal_cutoff()
        future_fixtures = [
            fixture
            for fixture in season.fixtures_json
            if fixture.get('event') is not None and fixture['event'] >= 10
        ]
        self.assertTrue(future_fixtures)
        self.assertTrue(all(
            not fixture['finished']
            and fixture.get('team_h_score') is None
            and fixture.get('team_a_score') is None
            for fixture in future_fixtures
        ))

    def test_runner_loads_prior_seasons_only(self):
        current = SimpleNamespace(
            years=(23, 24),
            as_of_gameweek=1,
            players=[],
            player_code_dict={},
        )

        def fake_season(year):
            return SimpleNamespace(
                years=(year, year + 1),
                players=[],
                player_code_dict={},
            )

        runner = Runner(
            current_season=current,
            target_season_year=23,
            as_of_gameweek=1,
            update_data=False,
        )
        with patch('premierlearning.runner.PastSeason', side_effect=fake_season):
            runner.populate_past_seasons()
        self.assertEqual([22, 21, 20, 19, 18], [
            season.years[0] for season in runner.past_seasons
        ])
        self.assertTrue(all(
            season.years[0] < 23 for season in runner.past_seasons
        ))

    def test_runner_rejects_a_cutoff_leak(self):
        current = SimpleNamespace(
            years=(23, 24),
            as_of_gameweek=10,
            players=[
                SimpleNamespace(
                    id=1,
                    matches=[
                        SimpleNamespace(
                            gameweek=10,
                            is_last_season_carry=False,
                        )
                    ],
                )
            ],
        )
        runner = Runner(
            current_season=current,
            target_season_year=23,
            as_of_gameweek=10,
            update_data=False,
        )
        with self.assertRaisesRegex(ValueError, 'leaked training matches'):
            runner.validate_temporal_cutoff()

    def test_snapshot_predicts_every_remaining_fixture(self):
        season = self.source.build_current_season(10, 5)
        player = next(
            player
            for player in season.players
            if self.source.fixtures_for_team(player.team_id, 10, 39)
        )
        player.process_matches()
        self.assertTrue(player.future_matches)
        self.assertTrue(all(
            10 <= match.gameweek <= 38 for match in player.future_matches
        ))
        self.assertTrue(any(
            match.gameweek >= 15 for match in player.future_matches
        ))
        runner = Runner(
            current_season=season,
            target_season_year=23,
            as_of_gameweek=10,
            horizon_gameweeks=5,
            update_data=False,
        )
        horizon_rows = runner.build_horizon_predict_data(5)
        self.assertTrue(all(
            match.gameweek < 15
            for window in horizon_rows['fixture_windows']
            for match in window
        ))

    def test_source_contains_double_and_blank_gameweeks(self):
        counts = Counter()
        for fixture in self.source.fixtures:
            event = fixture.get('event')
            if event in (None, ''):
                continue
            event = int(float(event))
            counts[(int(float(fixture['team_h'])), event)] += 1
            counts[(int(float(fixture['team_a'])), event)] += 1
        self.assertTrue(any(count > 1 for count in counts.values()))
        team_ids = {int(float(team['id'])) for team in self.source.teams_rows}
        self.assertTrue(any(
            counts.get((team_id, gameweek), 0) == 0
            for team_id in team_ids
            for gameweek in range(1, 39)
        ))

    def test_prediction_rows_number_double_gameweek_fixtures(self):
        def fixture(fixture_id, kickoff):
            return SimpleNamespace(
                gameweek=34,
                fixture_id=fixture_id,
                fixture_date=kickoff,
                opponent_team=SimpleNamespace(id=2),
                opponent='OPP',
                home=True,
                minutes=80,
                points=4,
                points_error=1,
            )

        player = SimpleNamespace(
            id=1,
            code=100,
            get_name=lambda: 'Double Player',
            position=3,
            team_id=1,
            current_cost=75,
            historical_selectable=True,
            future_matches=[
                fixture(200, '2024-04-20'),
                fixture(100, '2024-04-17'),
            ],
        )
        rows = _prediction_rows(
            SimpleNamespace(players=[player]),
            season_start_year=23,
            as_of_gameweek=34,
        )
        self.assertEqual([100, 200], [row['fixture_id'] for row in rows])
        self.assertEqual([1, 2], [row['fixture_order'] for row in rows])

    def test_actuals_sum_every_fixture_in_a_double(self):
        duplicate = None
        for player_id, rows in self.source.player_rows.items():
            by_gameweek = defaultdict(list)
            for row in rows:
                if row.get('round') not in (None, ''):
                    by_gameweek[int(float(row['round']))].append(row)
            duplicate = next(
                (
                    (player_id, gameweek, gameweek_rows)
                    for gameweek, gameweek_rows in by_gameweek.items()
                    if len(gameweek_rows) > 1
                ),
                None,
            )
            if duplicate:
                break
        self.assertIsNotNone(duplicate)
        player_id, gameweek, rows = duplicate
        actual = self.source.actuals_for_gameweek(gameweek)[player_id]
        expected = sum(float(row['total_points']) for row in rows)
        self.assertEqual(expected, actual.points)
        self.assertEqual(len(rows), len(actual.fixture_ids))

    def test_current_registry_uses_gameweek_price_not_final_price(self):
        state = self.source.state_at_gameweek(1)
        player_id = next(iter(state))
        season = self.source.build_current_season(1, 5)
        player = next(player for player in season.players if player.id == player_id)
        self.assertEqual(int(state[player_id]['value']), int(player.current_cost))

    def test_only_dated_known_unavailability_is_excluded(self):
        unavailable_ids = {
            player_id
            for player_id, raw in self.source.players_raw.items()
            if self.source.known_unavailable(raw, 38)
        }
        self.assertTrue(unavailable_ids)
        season = self.source.build_current_season(38, 1)
        flagged = {
            player.id
            for player in season.players
            if not player.historical_selectable
        }
        self.assertTrue(unavailable_ids & flagged)


if __name__ == '__main__':
    unittest.main()
