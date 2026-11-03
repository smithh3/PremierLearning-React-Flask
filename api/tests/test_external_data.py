"""Unit tests for premierlearning.external_data (no network, no data files needed)."""
import datetime as dt
import unittest

from premierlearning.external_data import competitions, espn, features, fpl_map, tables


def _player(starter=False, subbed_in=False, subbed_out=False, plays=(), athlete_id='1'):
    return {'athlete_id': athlete_id, 'starter': starter, 'subbed_in': subbed_in, 'subbed_out': subbed_out,
            'plays': [dict({'minute': None, 'sub': False, 'red': False}, **p) for p in plays]}


SUMMARY = {'status': 'STATUS_FULL_TIME', 'key_events': [], 'competitors': [{'team_id': 'A'}, {'team_id': 'B'}]}


class ClockTests(unittest.TestCase):
    def test_stoppage_time_sorts_inside_its_half(self):
        self.assertEqual(espn._minute({'displayValue': "45'+3'"}), 45.03)
        self.assertEqual(espn._minute({'displayValue': "90'"}), 90)
        self.assertLess(espn._minute({'displayValue': "45'+3'"}), espn._minute({'displayValue': "46'"}))
        self.assertIsNone(espn._minute({}))


class MinutesTests(unittest.TestCase):
    def test_full_match_starter(self):
        self.assertEqual(tables.minutes_played(_player(starter=True), SUMMARY), 90)

    def test_starter_subbed_off(self):
        p = _player(starter=True, subbed_out=True, plays=[{'minute': 64, 'sub': True}])
        self.assertEqual(tables.minutes_played(p, SUMMARY), 64)

    def test_sub_on_then_off(self):
        p = _player(subbed_in=True, subbed_out=True, plays=[{'minute': 46, 'sub': True}, {'minute': 80, 'sub': True}])
        self.assertEqual(tables.minutes_played(p, SUMMARY), 34)

    def test_unused_sub_and_red_card(self):
        self.assertEqual(tables.minutes_played(_player(), SUMMARY), 0)
        p = _player(starter=True, plays=[{'minute': 30, 'red': True}])
        self.assertEqual(tables.minutes_played(p, SUMMARY), 30)

    def test_extra_time(self):
        self.assertEqual(tables.minutes_played(_player(starter=True), dict(SUMMARY, status='STATUS_FINAL_AET')), 120)

    def test_stoppage_goal_counts_for_full_time_player(self):
        summary = dict(SUMMARY, key_events=[{'scoring': True, 'minute': 90.04, 'team_id': 'B', 'text': 'Goal'}])
        start, stop = tables.on_pitch(_player(starter=True), summary)
        timeline = tables._goal_timeline(summary)
        self.assertEqual(sum(1 for m, t in timeline if t == 'B' and start < m <= stop), 1)


class CategoryTests(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(competitions.category('fifa.world'), 'intl_tournament')
        self.assertEqual(competitions.category('fifa.wwc'), 'women')
        self.assertEqual(competitions.category('fifa.worldq.uefa'), 'intl_qualifier')
        self.assertEqual(competitions.category('club.friendly'), 'club_friendly')
        self.assertEqual(competitions.category('eng.w.1'), 'women')
        self.assertEqual(competitions.category('ger.1'), 'other_club')


class NameMatchTests(unittest.TestCase):
    def test_accents_and_short_names(self):
        fpl = {'first_name': 'Hugo', 'second_name': 'Ekitiké', 'web_name': 'Ekitiké'}
        self.assertGreater(fpl_map.name_score({'name': 'Hugo Ekitike', 'last_name': 'Ekitike'}, fpl), 0.9)
        other = {'first_name': 'Cody', 'second_name': 'Gakpo', 'web_name': 'Gakpo'}
        self.assertLess(fpl_map.name_score({'name': 'Hugo Ekitike', 'last_name': 'Ekitike'}, other), 0.5)

    def test_team_names(self):
        self.assertEqual(fpl_map.team_name_score('Wolverhampton Wanderers', 'Wolves'), 1.0)
        self.assertEqual(fpl_map.team_name_score('Brighton & Hove Albion', 'Brighton'), 1.0)


class _Ctx:
    """Minimal stand-in for features.Context."""

    def __init__(self, schedule, gws):
        self.schedule = schedule
        self.gw_order = {2025: gws}


class GameweekFlagTests(unittest.TestCase):
    def setUp(self):
        t = lambda d: f'2025-{d}T10:00Z'
        self.ctx = _Ctx({
            (2025, 10, 1): {'fixtures': 2, 'is_double': 1, 'is_blank': 0, 'cause': 'double',
                            'known_from_utc': t('10-01'), 'fixture_ids': '95 101'},
            (2025, 11, 1): {'fixtures': 0, 'is_double': 0, 'is_blank': 1, 'cause': 'blank_other',
                            'known_from_utc': t('11-05'), 'fixture_ids': ''},
            (2025, 12, 1): {'fixtures': 2, 'is_double': 1, 'is_blank': 0, 'cause': 'double',
                            'known_from_utc': t('10-20'), 'fixture_ids': '120 125'},
        }, [10, 11, 12])

    def test_this_gameweek_double_and_leg(self):
        flags = features.gw_flags(self.ctx, 2025, 10, 1, 101, dt.datetime(2025, 10, 25))
        self.assertEqual((flags['gw_fixtures'], flags['gw_double'], flags['dgw_leg']), (2, 1, 2))

    def test_future_irregular_gameweeks_only_once_known(self):
        flags = features.gw_flags(self.ctx, 2025, 10, 1, 95, dt.datetime(2025, 10, 25))
        self.assertEqual(flags['next1_gw_blank_known'], 0)   # blank only known from 5 Nov
        self.assertEqual(flags['next2_gw_double_known'], 1)  # double known from 20 Oct
        later = features.gw_flags(self.ctx, 2025, 10, 1, 95, dt.datetime(2025, 11, 6))
        self.assertEqual(later['next1_gw_blank_known'], 1)


class VisibilityTests(unittest.TestCase):
    def _match(self, kickoff, known_from, completed=True, status='STATUS_FULL_TIME'):
        return features.TeamMatch({
            'event_id': '1', 'kickoff_utc': kickoff, 'known_from_utc': known_from, 'completed': '1' if completed else '0',
            'has_lineups': '1', 'category': 'domestic_cup', 'league_slug': 'eng.fa', 'stage': 'quarterfinals',
            'opponent_fpl_team_code': '', 'home_away': 'home', 'neutral': '0', 'status': status})

    def test_future_knockout_hidden_until_known(self):
        as_of = dt.datetime(2025, 3, 1)
        m = self._match('2025-03-15T15:00Z', '2025-03-05T20:00Z', completed=False, status='STATUS_SCHEDULED')
        self.assertFalse(features.Context.visible(m, as_of))
        self.assertTrue(features.Context.visible(m, dt.datetime(2025, 3, 6)))
        self.assertEqual(m.knockout, 1)

    def test_postponed_never_counts(self):
        m = self._match('2025-02-15T15:00Z', '2025-01-01T00:00Z', status='STATUS_POSTPONED')
        self.assertFalse(features.Context.visible(m, dt.datetime(2025, 3, 1)))


class ExternalInputsTests(unittest.TestCase):
    def setUp(self):
        import os
        import tempfile
        self.dir = tempfile.mkdtemp()
        def write(name, header, rows):
            with open(os.path.join(self.dir, name), 'w', encoding='utf-8', newline='') as handle:
                handle.write(','.join(header) + '\n')
                for row in rows:
                    handle.write(','.join(str(v) for v in row) + '\n')
        from premierlearning.learning_v2 import external_inputs as ei
        self.ei = ei
        team_cols = ['fpl_season', 'fpl_fixture_id', 'team_code', 'manager_id'] + [c for c, _ in ei.TEAM_SCALARS.values()]
        team_vals = {c: 0 for c, _ in ei.TEAM_SCALARS.values()}
        team_vals.update(days_since_last_match=3.5, gw_double=1, dgw_leg=2, manager_caretaker=0)
        rows = [[2024, f, 14, 'Arne_Slot'] + [team_vals[c] for c, _ in ei.TEAM_SCALARS.values()] for f in range(1, 13)]
        rows.append([2024, 13, 14, 'Caretaker_X'] + [team_vals[c] for c, _ in ei.TEAM_SCALARS.values()])
        write('team_fixture_features.csv', team_cols, rows)
        player_cols = ['fpl_season', 'fpl_fixture_id', 'fpl_code'] + [c for c, _ in ei.PLAYER_SCALARS.values()]
        write('player_fixture_features.csv', player_cols, [[2024, 1, 999, 4.0, 4.0, 90, 180, 300, 1, 1]])
        up = ['fpl_season', 'fpl_fixture_id', 'team_code', 'rank', 'event_id', 'kickoff_utc', 'category', 'league_slug',
              'opponent_fpl_team_code', 'home_away', 'neutral', 'knockout', 'days_after_current']
        write('team_upcoming.csv', up, [[2024, 1, 14, 1, 'e1', '2024-09-17T19:00Z', 'champions_league', 'uefa.champions', '', 0, 0, 0, 3.2],
                                        [2024, 1, 14, 2, 'e2', '2024-09-21T14:00Z', 'premier_league', 'eng.1', 3, 1, 0, 0, 7.0]])
        iu = ['fpl_season', 'fpl_fixture_id', 'fpl_code', 'rank', 'national_team_id', 'event_id', 'kickoff_utc', 'category',
              'league_slug', 'home_away', 'neutral', 'knockout', 'days_after_current']
        write('player_international_upcoming.csv', iu, [[2024, 1, 999, 1, '448', 'e3', '2024-09-10T19:00Z', 'intl_nations_league',
                                                         'uefa.nations', 1, 0, 0, 1.5]])
        write('eval_slices.csv', ['fpl_season', 'fpl_fixture_id', 'fpl_code', 'slice_dgw_leg'], [[2024, 1, 999, 1]])
        write('appearances.csv', ['fpl_fixture_id', 'fpl_code', 'kickoff_utc', 'category', 'home_away', 'minutes', 'starter',
                                  'goals', 'assists', 'clean_sheet', 'team_score', 'opponent_score'],
              [['', 999, '2024-08-01T18:00Z', 'club_friendly', 'home', 45, 0, 1, 0, '', 2, 1],
               ['', 999, '2024-09-10T19:00Z', 'intl_nations_league', 'away', 90, 1, 0, 1, 1, 2, 0],
               [5, 999, '2024-08-17T14:00Z', 'premier_league', 'home', 90, 1, 0, 0, 1, 1, 0]])
        self.ext = ei.ExternalInputs.load(self.dir, self.dir)

    def test_scalars_and_defaults(self):
        team = self.ext.team(2024, 1, 14)
        self.assertEqual(team['schedule.days_since_last_match'], 3.5)
        self.assertEqual(team['gameweek.dgw_leg'], 2.0)
        self.assertEqual(self.ext.team(2030, 1, 14)['schedule.days_since_last_match'], self.ei.CAP_DAYS)
        self.assertEqual(self.ext.player(2024, 1, 999)['player_load.minutes_last_7d'], 90.0)
        self.assertEqual(self.ext.player(2024, 2, 999)['player_load.load_known'], 0.0)
        self.assertEqual(self.ext.slices(2024, 1, 999), {'slice_dgw_leg': 1})

    def test_upcoming_merges_club_and_country(self):
        tokens = self.ext.player_upcoming(2024, 1, 999, 14)
        self.assertEqual([t['days_after_current'] for t in tokens], [1.5, 3.2, 7.0])
        self.assertEqual(tokens[0]['national_team'], 1)
        self.assertIsNone(tokens[1]['opposition_team_code'])
        self.assertEqual(tokens[2]['opposition_team_code'], 3)
        self.assertEqual(tokens[1]['competition_embedding'], self.ei.competition_index('champions_league'))

    def test_history_excludes_pl_and_future(self):
        hist = self.ext.player_history(999, dt.datetime(2024, 9, 10, 19, 0))
        self.assertEqual(len(hist), 1)  # friendly only: PL row skipped, international not before cutoff
        self.assertEqual(hist[0]['minutes_played'], 45.0)
        self.assertEqual(len(self.ext.player_history(999, dt.datetime(2024, 9, 11))), 2)

    def test_manager_vocab_threshold(self):
        vocab = self.ext.manager_vocab([2024], min_matches=10)
        self.assertEqual(vocab, {'Arne_Slot': 1})
        self.assertEqual(self.ext.manager_index(vocab, 'Caretaker_X'), 0)
        self.assertEqual(self.ei.competition_index('unknown_cup'), 0)

    def test_coverage_counts_real_values_not_rows(self):
        cov = self.ext.coverage()[2024]
        self.assertEqual(cov['team_rows'], 13)
        self.assertEqual(cov['team_real_share'], 1.0)   # all fixture rows have days_since_last_match 3.5
        self.assertEqual(cov['player_real_share'], 1.0)
        self.assertFalse(self.ext.is_full_history(seasons=[2024]))  # only 13 team rows, under 700
        self.assertFalse(self.ext.is_full_history(seasons=[2023]))  # season missing entirely


if __name__ == '__main__':
    unittest.main()
