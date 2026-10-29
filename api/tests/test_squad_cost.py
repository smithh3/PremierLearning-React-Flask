import csv
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from premierlearning import squad_cost as squad_mod
from premierlearning.squad_cost import (
    centered_z,
    nudge_rate,
    squad_cost_early_weight,
    squad_cost_z_by_code,
    top_n_sum,
    use_squad_cost,
)
from premierlearning.team_expectancy import seed_form_from_last_season


def _write_teams_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['code', 'id', 'name', 'short_name'])
        writer.writeheader()
        writer.writerows(rows)


def _write_gw_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['name', 'team', 'value'])
        writer.writeheader()
        writer.writerows(rows)


class SquadCostTests(unittest.TestCase):
    def test_use_squad_cost_flag(self):
        with mock.patch.dict(os.environ, {'PL_SQUAD_COST': '0'}):
            self.assertFalse(use_squad_cost())
        with mock.patch.dict(os.environ, {'PL_SQUAD_COST': '1'}):
            self.assertTrue(use_squad_cost())
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('PL_SQUAD_COST', None)
            with mock.patch('sys.argv', ['premierlearning', '--no-squad-cost']):
                self.assertFalse(use_squad_cost())

    def test_top_n_ignores_cheap_tail(self):
        values = [100, 90, 80] + [40] * 20
        self.assertEqual(270.0, top_n_sum(values, 3))

    def test_centered_z_is_relative(self):
        z = centered_z({1: 10.0, 2: 0.0, 3: -10.0})
        self.assertAlmostEqual(0.0, z[2])
        self.assertGreater(z[1], 0.0)
        self.assertLess(z[3], 0.0)
        self.assertAlmostEqual(z[1], -z[3])

    def test_fade_is_gone_by_gameweek_5(self):
        self.assertAlmostEqual(1.0, squad_cost_early_weight(1))
        self.assertAlmostEqual(0.75, squad_cost_early_weight(2))
        self.assertAlmostEqual(0.5, squad_cost_early_weight(3))
        self.assertAlmostEqual(0.25, squad_cost_early_weight(4))
        self.assertAlmostEqual(0.0, squad_cost_early_weight(5))
        self.assertAlmostEqual(0.0, squad_cost_early_weight(10))

    def test_nudge_raises_scored_and_lowers_conceded(self):
        scored = nudge_rate(2.0, z=2.0, weight=1.0, sign=1.0)
        conceded = nudge_rate(1.0, z=2.0, weight=1.0, sign=-1.0)
        self.assertGreater(scored, 2.0)
        self.assertLess(conceded, 1.0)
        self.assertAlmostEqual(2.0, nudge_rate(2.0, z=2.0, weight=0.0, sign=1.0))

    def test_file_snapshots_use_top_n_and_center(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prev_teams = [
                {'code': 1, 'id': 1, 'name': 'Man Utd', 'short_name': 'MUN'},
                {'code': 43, 'id': 2, 'name': 'Man City', 'short_name': 'MCI'},
            ]
            curr_teams = list(prev_teams)
            _write_teams_csv(root / 'season_24_25' / 'teams.csv', prev_teams)
            _write_teams_csv(root / 'season_25_26' / 'teams.csv', curr_teams)
            # End of last season: United cheap XI, City expensive; lots of junk.
            _write_gw_csv(
                root / 'season_24_25' / 'gws' / 'gw38.csv',
                [{'name': 'a', 'team': 'Man Utd', 'value': v} for v in [50] * 15 + [40] * 20]
                + [{'name': 'b', 'team': 'Man City', 'value': v} for v in [90] * 15 + [40] * 20],
            )
            _write_gw_csv(
                root / 'season_25_26' / 'gws' / 'gw1.csv',
                [{'name': 'a', 'team': 'Man Utd', 'value': v} for v in [80] * 15]
                + [{'name': 'b', 'team': 'Man City', 'value': v} for v in [85] * 15],
            )
            squad_mod._CLUB_MAPS_CACHE.clear()
            z = squad_cost_z_by_code(24, 25, root=root, n=15)
            self.assertGreater(z[1], 0.0)
            self.assertLess(z[43], 0.0)

    def test_promoted_club_is_absent_from_z(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_teams_csv(
                root / 'season_24_25' / 'teams.csv',
                [
                    {'code': 1, 'id': 1, 'name': 'Man Utd', 'short_name': 'MUN'},
                    {'code': 8, 'id': 2, 'name': 'Chelsea', 'short_name': 'CHE'},
                ],
            )
            _write_teams_csv(
                root / 'season_25_26' / 'teams.csv',
                [
                    {'code': 1, 'id': 1, 'name': 'Man Utd', 'short_name': 'MUN'},
                    {'code': 2, 'id': 2, 'name': 'Leeds', 'short_name': 'LEE'},
                ],
            )
            _write_gw_csv(
                root / 'season_24_25' / 'gws' / 'gw38.csv',
                [{'name': 'a', 'team': 'Man Utd', 'value': 70}] * 15
                + [{'name': 'b', 'team': 'Chelsea', 'value': 70}] * 15,
            )
            _write_gw_csv(
                root / 'season_25_26' / 'gws' / 'gw1.csv',
                [{'name': 'a', 'team': 'Man Utd', 'value': 70}] * 15
                + [{'name': 'c', 'team': 'Leeds', 'value': 70}] * 15,
            )
            squad_mod._CLUB_MAPS_CACHE.clear()
            z = squad_cost_z_by_code(24, 25, root=root, n=15)
            self.assertIn(1, z)
            self.assertNotIn(2, z)
            self.assertNotIn(8, z)

    def test_seed_nudges_gw1_and_leaves_gw5(self):
        prior = SimpleNamespace(
            name='MUN',
            code=1,
            average_scored_so_far=2.0,
            average_conceded_so_far=1.0,
            _final_scored_last_5=2.0,
            _final_conceded_last_5=1.0,
            team_match_dict={},
        )
        prev_season = SimpleNamespace(
            years=(24, 25),
            teams={1: prior},
            players=[],
        )

        def make_match(event, scored_avg=0.0, conc_avg=0.0):
            return SimpleNamespace(
                event=event,
                scored_season_avg=scored_avg,
                conceded_season_avg=conc_avg,
                scored_last_5_avg=scored_avg,
                conceded_last_5_avg=conc_avg,
                average_scored_in_previous_matches=scored_avg,
                average_conceded_in_previous_matches=conc_avg,
            )

        gw1 = make_match(1)
        gw5 = make_match(5)
        team = SimpleNamespace(
            name='MUN',
            code=1,
            team_match_dict={},
            future_team_match_dict={1: gw1, 5: gw5},
            average_scored_so_far=0.0,
            average_conceded_so_far=0.0,
        )
        current = SimpleNamespace(
            years=(25, 26),
            teams={1: team},
            players=[],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_teams_csv(
                root / 'season_24_25' / 'teams.csv',
                [
                    {'code': 1, 'id': 1, 'name': 'Man Utd', 'short_name': 'MUN'},
                    {'code': 43, 'id': 2, 'name': 'Man City', 'short_name': 'MCI'},
                ],
            )
            _write_teams_csv(
                root / 'season_25_26' / 'teams.csv',
                [
                    {'code': 1, 'id': 1, 'name': 'Man Utd', 'short_name': 'MUN'},
                    {'code': 43, 'id': 2, 'name': 'Man City', 'short_name': 'MCI'},
                ],
            )
            _write_gw_csv(
                root / 'season_24_25' / 'gws' / 'gw38.csv',
                [{'name': 'a', 'team': 'Man Utd', 'value': 50}] * 15
                + [{'name': 'b', 'team': 'Man City', 'value': 90}] * 15,
            )
            _write_gw_csv(
                root / 'season_25_26' / 'gws' / 'gw1.csv',
                [{'name': 'a', 'team': 'Man Utd', 'value': 80}] * 15
                + [{'name': 'b', 'team': 'Man City', 'value': 85}] * 15,
            )
            squad_mod._CLUB_MAPS_CACHE.clear()
            with mock.patch.object(squad_mod, 'data_root', return_value=root):
                seed_form_from_last_season(current, [prev_season])
                seed_form_from_last_season(current, [prev_season])
        self.assertGreater(gw1.scored_season_avg, 2.0)
        self.assertLess(gw1.conceded_season_avg, 1.0)
        self.assertAlmostEqual(2.0, gw5.scored_season_avg)
        self.assertAlmostEqual(1.0, gw5.conceded_season_avg)
        self.assertGreater(team.squad_cost_yoy_z, 0.0)


class LiveSquadCostDataTests(unittest.TestCase):
    def test_24_25_to_25_26_united_up_city_down(self):
        z = squad_cost_z_by_code(24, 25)
        if not z:
            self.skipTest('season gw snapshots not present')
        self.assertGreater(z[1], 1.0)  # Man Utd
        self.assertLess(z[43], 0.0)  # Man City


if __name__ == '__main__':
    unittest.main()
