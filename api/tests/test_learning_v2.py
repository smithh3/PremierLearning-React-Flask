"""Plan parsing, pre-match dataset rows, and masked encoder outputs."""
from datetime import datetime, timedelta
import unittest

import numpy

from premierlearning.learning_v2.dataset import build_dataset, unsupported_leaves
from premierlearning.learning_v2.score import (
    recent_gameweek_rates,
    score_by_recent_form,
    score_by_tier,
)
from premierlearning.learning_v2.kickoff import parse_kickoff
from premierlearning.learning_v2.plan import (
    contains_nested_attention,
    input_specs,
    load_plan,
    plan_from_dict,
    tensor_key,
)


def _stats(**kwargs):
    base = {
        'total_points': 0.0,
        'minutes': 0.0,
        'expected_goals': 0.0,
        'expected_assists': 0.0,
        'expected_goals_conceded': 0.0,
        'goals_scored': 0.0,
        'assists': 0.0,
        'defensive_contribution': 0.0,
    }
    base.update(kwargs)
    return base


class _TeamMatch:
    def __init__(self, fixture_id, event, scored, conceded, xg, xc):
        self.id = fixture_id
        self.event = event
        self.scored = scored
        self.conceded = conceded
        self.predicted_xg = xg
        self.predicted_xc = xc
        self.fixture_json = {}


class _Team:
    def __init__(self, code):
        self.code = code
        self.team_match_dict = {}
        self.future_team_match_dict = {}


class _Player:
    def __init__(self, code, position, team):
        self.code = code
        self.position = position
        self.team = team
        self.matches = []
        self.future_matches = []
        self.skip_historical_predict = False

    def changed_club_feature(self, _apps):
        return 0


class _Match:
    def __init__(self, player, fixture_id, kickoff, team, opponent, team_match, opp_match, stats, home=True):
        self.player = player
        self.fixture_id = fixture_id
        self.gameweek = fixture_id
        self.kickoff_time = kickoff
        self.home = home
        self.team = team
        self.opponent_team = opponent
        self.team_match = team_match
        self.opponent_team_match = opp_match
        self.game_stats = stats
        self.minutes = stats['minutes']
        self.points = stats['total_points']
        self.matches_played = 3
        self.new_stats_included = 1
        self.is_last_season_carry = False
        self.chance_of_playing = 25


class _Season:
    def __init__(self, teams, players):
        self.years = (25, 26)
        self.teams = teams
        self.players = players


def _fixture_world():
    day0 = datetime(2025, 8, 1)
    kicks = [day0, day0 + timedelta(days=10), day0 + timedelta(days=24), day0 + timedelta(days=34)]
    arsenal = _Team(1)
    chelsea = _Team(2)
    arsenal_matches = [
        _TeamMatch(1, 1, scored=2, conceded=0, xg=1.0, xc=0.5),
        _TeamMatch(2, 2, scored=1, conceded=1, xg=3.0, xc=1.0),
        _TeamMatch(3, 3, scored=5, conceded=2, xg=9.0, xc=4.0),
    ]
    arsenal.team_match_dict = {match.id: match for match in arsenal_matches}
    arsenal_future = _TeamMatch(4, 4, scored=0, conceded=0, xg=0.1, xc=0.2)
    arsenal.future_team_match_dict = {4: arsenal_future}
    chelsea_matches = [
        _TeamMatch(1, 1, scored=0, conceded=2, xg=0.4, xc=1.1),
        _TeamMatch(2, 2, scored=1, conceded=1, xg=0.7, xc=1.2),
        _TeamMatch(3, 3, scored=2, conceded=5, xg=8.0, xc=3.0),
    ]
    chelsea.team_match_dict = {match.id: match for match in chelsea_matches}
    chelsea_future = _TeamMatch(4, 4, scored=0, conceded=0, xg=0.3, xc=0.4)
    chelsea.future_team_match_dict = {4: chelsea_future}

    saka = _Player(101, 3, arsenal)
    mate = _Player(102, 4, arsenal)
    opp = _Player(201, 2, chelsea)

    def add(player, index, stats, future=False):
        if player is opp:
            team, opponent = chelsea, arsenal
            team_match = chelsea_future if future else chelsea_matches[index]
            opp_match = arsenal_future if future else arsenal_matches[index]
        else:
            team, opponent = arsenal, chelsea
            team_match = arsenal_future if future else arsenal_matches[index]
            opp_match = chelsea_future if future else chelsea_matches[index]
        match = _Match(
            player, index + 1, kicks[index], team, opponent, team_match, opp_match, stats,
            home=player is not opp,
        )
        if future:
            player.future_matches.append(match)
        else:
            player.matches.append(match)
        return match

    add(saka, 0, _stats(minutes=10, total_points=1, expected_goals=0.2))
    add(saka, 1, _stats(minutes=0, total_points=0, expected_goals=0.0))
    add(saka, 2, _stats(minutes=90, total_points=8, expected_goals=0.5))
    add(saka, 3, _stats(minutes=0, total_points=0), future=True)
    add(mate, 0, _stats(minutes=90, total_points=2, expected_goals=0.0))
    add(mate, 1, _stats(minutes=90, total_points=2, expected_goals=0.0))
    add(mate, 2, _stats(minutes=90, total_points=2, expected_goals=9.0))
    add(opp, 1, _stats(minutes=90, total_points=2, expected_goals=4.0))
    add(opp, 2, _stats(minutes=90, total_points=2, expected_goals=8.0))
    saka.future_matches[0].chance_of_playing = 50
    return _Season({1: arsenal, 2: chelsea}, [saka, mate, opp]), saka


class PlanTests(unittest.TestCase):
    def test_simple_and_full_parse(self):
        simple = load_plan('simple_h1')
        full = load_plan('full_h1')
        self.assertFalse(contains_nested_attention(simple))
        self.assertTrue(contains_nested_attention(full))
        self.assertEqual([], unsupported_leaves(simple))
        self.assertTrue(unsupported_leaves(full))
        for plan in (simple, full):
            tables = {leaf.path: leaf.table for leaf in plan.leaves() if leaf.dtype == 'embedding'}
            self.assertNotIn('team_fixture_history', tables)
            player_tables = {table for path, table in tables.items() if path.endswith('player_id_embedding')}
            self.assertEqual({'player'}, player_tables)
            team_tables = {
                table for path, table in tables.items()
                if path.endswith('team_id_embedding') or path.endswith('opposition_team_id_embedding')
            }
            self.assertEqual({'team'}, team_tables)

    def test_unknown_leaf_type_fails(self):
        with self.assertRaises(ValueError):
            plan_from_dict({'player_id_embedding': 'string'})

    def test_kickoff_parser(self):
        parsed = parse_kickoff('2025-08-16T14:00:00Z')
        self.assertEqual(datetime(2025, 8, 16, 14, 0, 0), parsed)
        self.assertIsNone(parse_kickoff(None))


class DatasetTests(unittest.TestCase):
    def test_history_includes_dnp_and_excludes_the_target_fixture(self):
        season, saka = _fixture_world()
        saka.played_last_season = True
        saka.points_per_min_last_season = 0.05
        saka.xgi_per_min_last_season = 0.01
        saka.minutes_last_season = 2000
        saka.points_per_min_two_seasons_ago = 0.04
        plan = load_plan('simple_h1')
        plan.config.history_len = 4
        plan.config.max_squad = 3
        plan.config.team_form_n = 5
        train, predict, vocab = build_dataset([season], plan, predict_season=season)
        row = next(
            index for index, match in enumerate(train.matches)
            if match.player is saka and match.fixture_id == 3
        )
        arrays = train.arrays
        minutes = arrays[tensor_key('player_fixture_history.player_performance.minutes_played')][row]
        days = arrays[tensor_key('player_fixture_history.match_facts.days_before_current')][row]
        history_mask = arrays[tensor_key('player_fixture_history.mask')][row]
        self.assertTrue(numpy.array_equal(history_mask, numpy.array([0, 0, 1, 1], dtype=numpy.float32)))
        self.assertEqual(0.0, float(minutes[0]))
        self.assertEqual(10.0, float(minutes[2]))
        self.assertEqual(0.0, float(minutes[3]))
        self.assertGreater(float(days[2]), float(days[3]))
        self.assertAlmostEqual(24.0, float(days[2]))
        self.assertAlmostEqual(14.0, float(days[3]))

        goals = arrays[tensor_key('current_team.team_features.goals_last_n')][row, 0]
        self.assertAlmostEqual(1.5, float(goals))
        self.assertNotIn(tensor_key('match_facts.team_xg'), arrays)
        self.assertNotIn(tensor_key('current_team.team_features.xG_last_n'), arrays)

        hist_goals = arrays[tensor_key('player_fixture_history.team_features_pre_match.goals')][row]
        self.assertAlmostEqual(2.0, float(hist_goals[3]))

        mate_xg = arrays[tensor_key('current_team.teammates.xG_last_n')][row]
        mate_ids = arrays[tensor_key('current_team.teammates.player_id_embedding')][row]
        mate_mask = arrays[tensor_key('current_team.teammates.mask')][row]
        self.assertEqual(1.0, float(mate_mask[0]))
        self.assertEqual(0.0, float(mate_mask[1]))
        self.assertAlmostEqual(0.0, float(mate_xg[0]))
        self.assertEqual(vocab.player_index(102), int(mate_ids[0]))
        self.assertNotEqual(vocab.player_index(101), int(mate_ids[0]))
        self.assertNotIn(vocab.player_index(101), set(int(value) for value in mate_ids))

        opp_xg = arrays[tensor_key('current_opposition.players.xG_last_n')][row]
        self.assertAlmostEqual(4.0, float(opp_xg[0]))
        self.assertAlmostEqual(1.0, float(arrays[tensor_key('context.chance_of_playing')][row, 0]))
        self.assertEqual(0.0, float(arrays[tensor_key('last_season.mask')][row].sum()))
        self.assertNotIn(tensor_key('current_player_features.minutes_last_n'), arrays)

        dnp_rows = [
            match for match in train.matches
            if match.player is saka and match.fixture_id == 2
        ]
        self.assertEqual(1, len(dnp_rows))
        self.assertEqual(0.0, float(dnp_rows[0].game_stats['total_points']))

        future = predict.matches[0]
        self.assertEqual(4, future.fixture_id)
        future_row = 0
        self.assertAlmostEqual(
            0.5, float(predict.arrays[tensor_key('context.chance_of_playing')][future_row, 0]),
        )
        self.assertAlmostEqual(
            8.0 / 3.0,
            float(predict.arrays[tensor_key('current_team.team_features.goals_last_n')][future_row, 0]),
            places=5,
        )
        future_minutes = predict.arrays[
            tensor_key('player_fixture_history.player_performance.minutes_played')
        ][future_row]
        future_mask = predict.arrays[tensor_key('player_fixture_history.mask')][future_row]
        self.assertEqual(90.0, float(future_minutes[int(numpy.where(future_mask > 0)[0][-1])]))

    def test_later_future_uses_played_matches_only(self):
        season, saka = _fixture_world()
        first = saka.future_matches[0]
        later_kick = first.kickoff_time + timedelta(days=10)
        saka.future_matches.append(_Match(
            saka, 5, later_kick, first.team, first.opponent_team,
            first.team_match, first.opponent_team_match,
            _stats(minutes=0, total_points=0),
        ))
        mate = next(player for player in season.players if player.code == 102)
        for fixture_id, kickoff in ((4, first.kickoff_time), (5, later_kick)):
            mate.future_matches.append(_Match(
                mate, fixture_id, kickoff, first.team, first.opponent_team,
                first.team_match, first.opponent_team_match,
                _stats(minutes=0, total_points=0, expected_goals=50.0),
            ))
        plan = load_plan('simple_h1')
        plan.config.history_len = 4
        plan.config.team_form_n = 5
        _train, predict, vocab = build_dataset([season], plan, predict_season=season)
        by_fixture = {
            match.fixture_id: index
            for index, match in enumerate(predict.matches)
            if match.player is saka
        }
        minutes_key = tensor_key('player_fixture_history.player_performance.minutes_played')
        mask_key = tensor_key('player_fixture_history.mask')
        for fixture_id in (4, 5):
            row = by_fixture[fixture_id]
            minutes = predict.arrays[minutes_key][row]
            mask = predict.arrays[mask_key][row]
            played = [float(value) for value in minutes[mask > 0]]
            self.assertEqual([10.0, 0.0, 90.0], played)
        later = by_fixture[5]
        mate_xg = predict.arrays[tensor_key('current_team.teammates.xG_last_n')][later]
        mate_ids = predict.arrays[tensor_key('current_team.teammates.player_id_embedding')][later]
        mate_mask = predict.arrays[tensor_key('current_team.teammates.mask')][later]
        slot = next(
            index for index, flag in enumerate(mate_mask)
            if flag > 0 and int(mate_ids[index]) == vocab.player_index(102)
        )
        self.assertAlmostEqual(3.0, float(mate_xg[slot]))

    def test_history_keeps_earlier_seasons_and_skips_the_carry_copy(self):
        day0 = datetime(2024, 8, 1)
        arsenal = _Team(1)
        chelsea = _Team(2)
        past_player = _Player(101, 3, arsenal)
        current = _Player(101, 3, arsenal)
        made = []
        for index, stat in enumerate((
            _stats(minutes=10, total_points=1),
            _stats(minutes=0, total_points=0),
            _stats(minutes=90, total_points=8),
        )):
            kick = day0 + timedelta(days=10 * index)
            team_match = _TeamMatch(index + 1, index + 1, scored=1, conceded=0, xg=1.0, xc=1.0)
            match = _Match(
                past_player, index + 1, kick, arsenal, chelsea, team_match, team_match, stat,
            )
            past_player.matches.append(match)
            made.append(match)
        last = made[-1]
        carry = _Match(
            current, last.fixture_id, last.kickoff_time, arsenal, chelsea,
            last.team_match, last.opponent_team_match,
            _stats(minutes=90, total_points=8),
        )
        carry.is_last_season_carry = True
        current.matches.append(carry)
        future_team = _TeamMatch(9, 9, scored=0, conceded=0, xg=0.0, xc=0.0)
        future = _Match(
            current, 9, day0 + timedelta(days=400), arsenal, chelsea,
            future_team, future_team, _stats(),
        )
        current.future_matches.append(future)
        past = _Season({1: arsenal, 2: chelsea}, [past_player])
        past.years = (24, 25)
        now = _Season({1: arsenal, 2: chelsea}, [current])
        now.years = (25, 26)
        plan = load_plan('simple_h1')
        plan.config.history_len = 50
        _train, predict, _vocab = build_dataset([past, now], plan, predict_season=now)
        row = next(index for index, match in enumerate(predict.matches) if match.player is current)
        minutes = predict.arrays[
            tensor_key('player_fixture_history.player_performance.minutes_played')
        ][row]
        mask = predict.arrays[tensor_key('player_fixture_history.mask')][row]
        played = [float(value) for value in minutes[mask > 0]]
        self.assertEqual([10.0, 0.0, 90.0], played)
        season_mask = predict.arrays[tensor_key('last_season.mask')][row]
        self.assertEqual(1.0, float(season_mask.sum()))
        slot = int(numpy.where(season_mask > 0)[0][-1])
        self.assertAlmostEqual(
            1.0, float(predict.arrays[tensor_key('last_season.seasons_before')][row, slot]),
        )
        self.assertAlmostEqual(
            1.0, float(predict.arrays[tensor_key('last_season.played')][row, slot]),
        )
        self.assertAlmostEqual(
            100.0, float(predict.arrays[tensor_key('last_season.minutes')][row, slot]),
        )
        self.assertAlmostEqual(
            8.1, float(predict.arrays[tensor_key('last_season.points_per_90')][row, slot]),
            places=5,
        )
        past_row = next(
            index for index, match in enumerate(_train.matches) if match.player is past_player
        )
        self.assertEqual(0.0, float(_train.arrays[tensor_key('last_season.mask')][past_row].sum()))


class ScoreTests(unittest.TestCase):
    def test_high_actual_scores_count_more_than_the_prediction(self):
        actual = [0, 2, 4, 8, 12]
        predicted = [0, 1, 4, 5, 6]
        report = score_by_tier(actual, predicted)
        self.assertAlmostEqual(2.0, report['mae'])
        self.assertAlmostEqual(61.0 / 16.0, report['weighted_mae'])
        by_name = {tier['name']: tier for tier in report['tiers']}
        self.assertEqual(2, by_name['low']['count'])
        self.assertAlmostEqual(0.5, by_name['low']['mae'])
        self.assertAlmostEqual(0.0, by_name['medium']['mae'])
        self.assertAlmostEqual(3.0, by_name['high']['mae'])
        self.assertAlmostEqual(6.0, by_name['very_high']['mae'])
        self.assertAlmostEqual(-6.0, by_name['very_high']['bias'])

    def test_tier_weights_match_the_report_bands(self):
        from premierlearning.learning_v2.score import normalized_tier_weights, tier_weights

        points = [-1, 0, 2.9, 3, 5.9, 6, 9.9, 10, 15]
        weights = tier_weights(points)
        self.assertEqual([1, 1, 1, 2, 2, 4, 4, 8, 8], [float(value) for value in weights])
        scaled = normalized_tier_weights([0, 10])
        self.assertAlmostEqual(1.0, float(scaled.mean()))
        self.assertAlmostEqual(8.0, float(scaled[1] / scaled[0]))

    def test_training_weights_are_a_fifth_of_the_way_to_the_bands(self):
        from premierlearning.learning_v2.score import blended_tier_weights

        weights = blended_tier_weights([0, 3, 6, 10])
        self.assertAlmostEqual(1.0, float(weights.mean()), places=5)
        # Equal weight is 1. Bands are 1, 2, 4, 8. Mix 0.2 is 1, 1.2, 1.6, 2.4.
        self.assertAlmostEqual(1.2, float(weights[1] / weights[0]), places=5)
        self.assertAlmostEqual(1.6, float(weights[2] / weights[0]), places=5)
        self.assertAlmostEqual(2.4, float(weights[3] / weights[0]), places=5)

    def test_recent_level_ignores_the_fixture_being_scored(self):
        class _Player:
            def __init__(self):
                self.matches = []

        class _Match:
            def __init__(self, player, gameweek, points, day, carry=False):
                self.player = player
                self.gameweek = gameweek
                self.points = points
                self.game_stats = {'total_points': points}
                self.kickoff_time = datetime(2025, 8, 1) + timedelta(days=day)
                self.is_last_season_carry = carry
                player.matches.append(self)

        player = _Player()
        for gameweek, points in ((1, 10), (2, 10), (3, 10), (4, 0), (5, 2)):
            _Match(player, gameweek, points, gameweek * 7)
        _Match(player, 4, 8, 4 * 7 + 1)
        target = _Match(player, 6, 1, 50)
        rates = recent_gameweek_rates([target])
        self.assertAlmostEqual((10 + 10 + 10 + 8 + 2) / 5.0, float(rates[0]))
        report = score_by_recent_form([1.0], [2.0], rates)
        by_name = {tier['name']: tier for tier in report['tiers']}
        self.assertEqual(1, by_name['high']['count'])
        self.assertEqual(0, by_name['low']['count'])
        self.assertAlmostEqual(1.0, by_name['high']['mean_actual'])
        self.assertAlmostEqual(2.0, by_name['high']['mean_predicted'])


class ReportTests(unittest.TestCase):
    def test_opening_five_sums_double_gameweeks_and_ranks_players(self):
        from premierlearning.learning_v2.learn import format_opening_predictions

        class _Named:
            def __init__(self, name, team):
                self.team_name = team
                self._name = name

            def get_name(self):
                return self._name

        class _Fixture:
            def __init__(self, player, gameweek, points):
                self.player = player
                self.gameweek = gameweek
                self.points = points

        haaland = _Named('Erling Haaland', 'MCI')
        saka = _Named('Bukayo Saka', 'ARS')
        bench = _Named('Bench Player', 'SUN')
        text = format_opening_predictions([
            _Fixture(haaland, 1, 3.0),
            _Fixture(haaland, 1, 1.0),
            _Fixture(haaland, 2, 2.0),
            _Fixture(saka, 1, 5.0),
            _Fixture(bench, 3, 0.5),
            _Fixture(haaland, 8, 9.0),
        ], gameweeks=(1, 2, 3), limit=2)
        lines = text.splitlines()
        self.assertIn('Haaland', lines[2])
        self.assertIn('4.00', lines[2])
        self.assertIn('2.00', lines[2])
        self.assertIn('Saka', lines[3])
        self.assertNotIn('Bench', text)
        self.assertNotIn('9.00', text)

    def test_default_loss_is_raw_huber(self):
        from premierlearning.learning_v2.train import resolve_loss_name

        self.assertEqual('huber_raw', resolve_loss_name())
        self.assertEqual('huber', resolve_loss_name('huber'))

    def test_plateau_patience_is_a_fifth_of_epochs_run_and_at_most_40(self):
        from premierlearning.learning_v2.train import stop_after_plateau

        self.assertFalse(stop_after_plateau(5, 1))
        self.assertTrue(stop_after_plateau(6, 1))
        self.assertFalse(stop_after_plateau(23, 19))
        self.assertFalse(stop_after_plateau(28, 23))
        self.assertTrue(stop_after_plateau(29, 23))
        self.assertFalse(stop_after_plateau(230, 200))
        self.assertTrue(stop_after_plateau(240, 200))

    def test_next_five_starts_at_the_upcoming_gameweek(self):
        from premierlearning.learning_v2.learn import upcoming_gameweeks

        class _Season:
            next_gameweek = 9
            current_gameweek = 8

        self.assertEqual((9, 10, 11, 12, 13), upcoming_gameweeks(_Season()))

        class _Stale:
            next_gameweek = 1
            current_gameweek = 0
            fantasy_json = {'events': [
                {'id': 1, 'deadline_time': '2026-08-21T17:30:00Z', 'is_next': True},
                {'id': 8, 'deadline_time': '2026-10-24T12:30:00Z'},
                {'id': 9, 'deadline_time': '2026-10-31T13:30:00Z'},
            ]}

        self.assertEqual(
            (9, 10, 11, 12, 13),
            upcoming_gameweeks(_Stale(), now=datetime(2026, 10, 25, 21, 0, 0)),
        )


class ModelTests(unittest.TestCase):
    def test_padded_slots_do_not_change_the_output(self):
        import tensorflow as tf

        from premierlearning.learning import PlayerNetHParams
        from premierlearning.learning_v2.model import build_model

        tf.random.set_seed(1)
        plan = load_plan('simple_h1')
        plan.config.history_len = 4
        plan.config.max_squad = 3
        plan.config.d_model = 8
        plan.config.n_heads = 2
        plan.config.n_layers = 1
        plan.config.player_emb_dim = 4
        plan.config.team_emb_dim = 4
        plan.config.position_emb_dim = 2

        class _Vocab:
            player_size = 16
            team_size = 16
            position_size = 5

        hparams = PlayerNetHParams()
        hparams.dropout_1 = 0.0
        hparams.dropout_2 = 0.0
        hparams.n_hidden = 1
        hparams.width_mult = 1.0
        model = build_model(plan, _Vocab(), hparams, loss_name='huber')
        specs = input_specs(plan)
        self.assertEqual({spec.key for spec in specs}, set(model.input_names))

        feed = {}
        for spec in specs:
            if spec.kind == 'embedding':
                feed[spec.key] = numpy.zeros((2, *spec.shape), dtype=numpy.int32)
            else:
                feed[spec.key] = numpy.zeros((2, *spec.shape), dtype=numpy.float32)
        history_mask = tensor_key('player_fixture_history.mask')
        squad_mask = tensor_key('current_team.teammates.mask')
        opp_mask = tensor_key('current_opposition.players.mask')
        feed[history_mask][:, 2:] = 1.0
        feed[squad_mask][:, 0] = 1.0
        feed[opp_mask][:, 0] = 1.0
        feed[tensor_key('player_id_embedding')][:, 0] = 1
        feed[tensor_key('player_fixture_history.player_performance.points_scored')][:, 2] = 3.0

        def predict(batch):
            return model(batch, training=False).numpy()

        baseline = predict(feed)
        changed = {key: numpy.array(value, copy=True) for key, value in feed.items()}
        for spec in specs:
            if spec.path.startswith('player_fixture_history.') and not spec.path.endswith('mask'):
                if spec.shape == (4,):
                    changed[spec.key][:, 0] = 4 if spec.path.endswith('position') else 7
            if spec.path.startswith('current_team.teammates.') and not spec.path.endswith('mask'):
                changed[spec.key][:, 1] = 4 if spec.path.endswith('position') else 7
            if spec.path.startswith('current_opposition.players.') and not spec.path.endswith('mask'):
                changed[spec.key][:, 2] = 4 if spec.path.endswith('position') else 7
        numpy.testing.assert_allclose(baseline, predict(changed), atol=1e-5)


if __name__ == '__main__':
    unittest.main()
