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

    def test_frozen_vocab_keeps_indexes_and_new_players_read_as_unknown(self):
        season, saka = _fixture_world()
        plan = load_plan('simple_h1')
        plan.config.history_len = 4
        plan.config.max_squad = 3
        _train, _predict, vocab = build_dataset([season], plan, predict_season=season)
        vocab.freeze()
        sizes = (vocab.player_size, vocab.team_size)
        saka_index = vocab.player_index(101)
        arsenal = season.teams[1]
        newcomer = _Player(999, 3, arsenal)
        first = saka.future_matches[0]
        newcomer.future_matches.append(_Match(
            newcomer, 4, first.kickoff_time, arsenal, first.opponent_team,
            first.team_match, first.opponent_team_match, _stats(),
        ))
        season.players.append(newcomer)
        train, predict, same = build_dataset(
            [season], plan, predict_season=season, vocab=vocab, include_train=False,
        )
        self.assertIs(vocab, same)
        self.assertEqual(sizes, (vocab.player_size, vocab.team_size))
        self.assertEqual(0, len(train.matches))
        ids = predict.arrays[tensor_key('player_id_embedding')]
        rows = {match.player.code: index for index, match in enumerate(predict.matches)}
        self.assertEqual(saka_index, int(ids[rows[101], 0]))
        self.assertEqual(0, int(ids[rows[999], 0]))

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


class ChanceInputTests(unittest.TestCase):
    def setUp(self):
        from premierlearning.learning_v2 import dataset
        self.dataset = dataset
        self.saved = dict(dataset._DEADLINE_CHANCES)
        dataset._DEADLINE_CHANCES.clear()
        # 2021/22 GW10 has a flag file; 2019/20 GW10 has none.
        dataset._DEADLINE_CHANCES[(21, 10)] = {7: 0, 8: None, 9: 75}
        dataset._DEADLINE_CHANCES[(19, 10)] = {}

    def tearDown(self):
        self.dataset._DEADLINE_CHANCES.clear()
        self.dataset._DEADLINE_CHANCES.update(self.saved)

    def _row(self, player_id, year, gameweek, is_future=False, flag=None):
        from types import SimpleNamespace
        return SimpleNamespace(
            player=SimpleNamespace(id=player_id),
            match=SimpleNamespace(gameweek=gameweek, chance_of_playing=flag),
            year=year, is_future=is_future,
        )

    def test_finished_fixtures_use_the_flag_known_at_their_deadline(self):
        known = self.dataset._chance_and_known
        self.assertEqual((0.0, 1.0), known(self._row(7, 21, 10)))
        self.assertEqual((0.75, 1.0), known(self._row(9, 21, 10)))
        # No flag for a listed player, or not listed at all: no doubt.
        self.assertEqual((1.0, 1.0), known(self._row(8, 21, 10)))
        self.assertEqual((1.0, 1.0), known(self._row(99, 21, 10)))
        # A season without flag history reads as unknown, not as fit.
        self.assertEqual((1.0, 0.0), known(self._row(7, 19, 10)))

    def test_upcoming_fixtures_use_the_flag_on_the_match(self):
        known = self.dataset._chance_and_known
        self.assertEqual((0.25, 1.0), known(self._row(7, 25, 3, is_future=True, flag=25)))
        self.assertEqual((1.0, 1.0), known(self._row(7, 25, 3, is_future=True, flag=None)))


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

    def test_smooth_weights_track_the_bands_and_keep_rising(self):
        from premierlearning.learning_v2.score import smooth_weights

        weights = smooth_weights([-2, 0, 3, 6, 10, 20])
        self.assertAlmostEqual(1.0, float(weights.mean()), places=5)
        ratio = [float(value / weights[1]) for value in weights]
        self.assertAlmostEqual(1.0, ratio[0], places=5)
        self.assertAlmostEqual(1.23, ratio[2], places=2)
        self.assertAlmostEqual(1.65, ratio[3], places=2)
        self.assertAlmostEqual(2.4, ratio[4], places=4)
        self.assertGreater(ratio[5], 4.9)

    def test_capped_smooth_plateau_and_hump_weights(self):
        from premierlearning.learning_v2.score import LossWeighting, hump_weights

        points = [0, 3, 6, 10, 15, 20]

        def ratios(weights):
            return [float(value / weights[0]) for value in weights]

        capped = ratios(LossWeighting('smooth', cap=2.4).weights(points))
        self.assertAlmostEqual(2.4, capped[3], places=4)
        self.assertAlmostEqual(2.4, capped[5], places=4)
        plateau = ratios(LossWeighting('plateau').weights(points))
        self.assertAlmostEqual(1.2, plateau[1], places=5)
        self.assertAlmostEqual(1.6, plateau[2], places=5)
        self.assertAlmostEqual(1.6, plateau[5], places=5)
        hump = ratios(hump_weights([0, 3, 6, 8, 10, 15, 20]))
        self.assertAlmostEqual(1.7, hump[3], places=4)
        self.assertGreater(hump[3], hump[4])
        self.assertGreater(hump[4], hump[5])
        self.assertGreater(hump[5], hump[6])
        self.assertAlmostEqual(1.0, float(hump_weights(points).mean()), places=5)
        fast = ratios(LossWeighting(
            'hump', hump_peak=6, hump_height=2, hump_power=1, hump_width=15,
        ).weights([0, 3, 6, 15, 20]))
        self.assertAlmostEqual(2.0, fast[1], places=4)
        self.assertAlmostEqual(3.0, fast[2], places=4)
        self.assertAlmostEqual(2.40, fast[3], places=2)
        self.assertAlmostEqual(1.84, fast[4], places=2)
        configs = [LossWeighting(kind).config() for kind in ('tier', 'plateau', 'smooth', 'hump')]
        configs.append(LossWeighting('smooth', cap=2.4).config())
        configs.append(LossWeighting('hump', hump_peak=6).config())
        self.assertEqual(len(configs), len({repr(sorted(c.items())) for c in configs}))
        # Uncapped smooth hashes exactly as before the cap existed.
        self.assertNotIn('cap', LossWeighting('smooth').config())

    def test_loss_weighting_kinds(self):
        from premierlearning.learning_v2.score import LossWeighting

        points = [0, 3, 6, 10]
        self.assertEqual([1.0] * 4, [float(v) for v in LossWeighting('none').weights(points)])
        tier = LossWeighting('tier').weights(points)
        self.assertAlmostEqual(2.4, float(tier[3] / tier[0]), places=5)
        self.assertEqual({'kind': 'none'}, LossWeighting('none').config())
        self.assertNotEqual(LossWeighting('tier').config(), LossWeighting('smooth').config())
        with self.assertRaises(ValueError):
            LossWeighting('predicted')

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

    def test_fitted_points_scale_inputs_average_fits_and_floor_at_zero(self):
        from collections import namedtuple

        from premierlearning.learning_v2.train import FittedPoints

        Spec = namedtuple('Spec', 'kind key path')

        class _Model:
            def __init__(self, offset):
                self.offset = offset
                self.seen = None

            def predict(self, arrays, verbose=0):
                self.seen = float(arrays['x'][0, 0])
                return arrays['x'][:, :1] + self.offset

        class _Batch:
            arrays = {'x': numpy.array([[3.0], [-9.0]], dtype=numpy.float32)}
            matches = ['a', 'b']

        models = [_Model(0.0), _Model(2.0)]
        fitted = FittedPoints(models, [Spec('float', 'x', 'context.x')], {'x': (1.0, 2.0)}, False)
        numpy.testing.assert_allclose([2.0, 0.0], fitted.predict(_Batch()))
        self.assertEqual(1.0, models[0].seen)
        self.assertEqual(3.0, float(_Batch.arrays['x'][0, 0]))
        logged = FittedPoints([_Model(0.0)], [Spec('float', 'x', 'context.x')], {'x': (0.0, 1.0)}, True)
        self.assertAlmostEqual(numpy.expm1(3.0), float(logged.predict(_Batch())[0]), places=4)
        smeared = FittedPoints(
            [_Model(0.0)], [Spec('float', 'x', 'context.x')], {'x': (0.0, 1.0)}, True,
            smearing={'S': 1.5}, smear_mode='global',
        )
        self.assertAlmostEqual(numpy.exp(3.0) * 1.5 - 1.0, float(smeared.predict(_Batch())[0]), places=3)

        class _Empty:
            arrays = {}
            matches = []

        self.assertEqual(0, fitted.predict(_Empty()).size)

    def test_weighting_comes_from_flags_unless_given(self):
        import sys

        from premierlearning.learning_v2.score import LossWeighting
        from premierlearning.learning_v2.train import resolve_weighting

        given = LossWeighting('none')
        self.assertIs(given, resolve_weighting(given))
        self.assertEqual('tier', resolve_weighting().kind)
        previous = list(sys.argv)
        sys.argv[:] = previous + ['--v2-weighting', 'smooth', '--v2-smooth-power=2']
        try:
            weighting = resolve_weighting()
        finally:
            sys.argv[:] = previous
        self.assertEqual(('smooth', 2.0), (weighting.kind, weighting.power))

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


class _TinyVocab:
    player_size = 16
    team_size = 16
    position_size = 5


def _tiny_plan_and_hparams():
    from premierlearning.learning import PlayerNetHParams

    plan = load_plan('simple_h1')
    plan.config.history_len = 4
    plan.config.max_squad = 3
    plan.config.d_model = 8
    plan.config.n_heads = 2
    plan.config.n_layers = 1
    plan.config.player_emb_dim = 4
    plan.config.team_emb_dim = 4
    plan.config.position_emb_dim = 2
    hparams = PlayerNetHParams()
    hparams.dropout_1 = 0.0
    hparams.dropout_2 = 0.0
    hparams.n_hidden = 1
    hparams.width_mult = 1.0
    return plan, hparams


def _zero_feed(specs, rows):
    feed = {}
    for spec in specs:
        if spec.kind == 'embedding':
            feed[spec.key] = numpy.zeros((rows, *spec.shape), dtype=numpy.int32)
        else:
            feed[spec.key] = numpy.zeros((rows, *spec.shape), dtype=numpy.float32)
    return feed


class WindowRateTests(unittest.TestCase):
    def test_rates_count_doubles_and_did_not_plays_and_need_complete_windows(self):
        from types import SimpleNamespace as NS

        from premierlearning.learning_v2.dataset import ExampleBatch, window_rate_batch

        player, other = object(), object()
        weeks = [1, 2, 2, 3, 4, 1, 2, 3, 4]
        points = [2.0, 6.0, 1.0, 0.0, 3.0, 4.0, 4.0, 4.0, 4.0]
        owners = [player] * 5 + [other] * 4
        batch = ExampleBatch(
            {'x': numpy.arange(9.0).reshape(-1, 1)}, numpy.array(points, dtype=numpy.float32).reshape(-1, 1),
            [NS(gameweek=week) for week in weeks], owners, [25] * 9,
        )
        windowed, groups = window_rate_batch(batch, 3)
        # GW4 has no complete window; GW2's double keeps one row and counts twice.
        self.assertEqual([0.0, 1.0, 5.0, 6.0], windowed.arrays['x'].reshape(-1).tolist())
        numpy.testing.assert_allclose([9.0 / 4, 10.0 / 4, 4.0, 4.0], windowed.labels.reshape(-1))
        self.assertEqual([0, 0, 1, 1], groups.tolist())

    def test_window_total_splits_by_h1_shares_and_leaves_later_fixtures(self):
        from types import SimpleNamespace as NS

        from premierlearning.learning_v2.learn import spread_window_rates

        player = object()
        matches = [NS(gameweek=5, points=1.0), NS(gameweek=6, points=3.0), NS(gameweek=7, points=9.0)]
        batch = NS(owners=[player] * 3, matches=matches)
        spread_window_rates(batch, [2.0, 4.0, 50.0], 5, 2)
        self.assertEqual([1.5, 4.5, 9.0], [match.points for match in matches])
        self.assertEqual(3.0, matches[1].h1_points)

    def test_group_split_keeps_each_group_on_one_side(self):
        from premierlearning.learning_v2.train import _group_split

        groups = numpy.repeat(numpy.arange(10), 4)
        kept, held = _group_split(numpy.arange(40), groups, 0.2)
        self.assertFalse(set(groups[kept]) & set(groups[held]))


class ModelTests(unittest.TestCase):
    def test_grouped_fit_uses_a_held_out_validation_set(self):
        from types import SimpleNamespace
        from unittest import mock

        from premierlearning.learning_v2 import train
        from premierlearning.learning_v2.score import LossWeighting

        plan, hparams = _tiny_plan_and_hparams()
        specs = input_specs(plan)
        labels = numpy.arange(24, dtype=numpy.float32) % 5
        matches = [SimpleNamespace(gameweek=1)] * 24
        batch = SimpleNamespace(arrays=_zero_feed(specs, 24), labels=labels, matches=matches)
        with mock.patch.object(train, 'EPOCHS', 1), \
                mock.patch.object(train.sys, 'argv', ['test', '--quick']), \
                mock.patch.object(train, 'recent_gameweek_rates', return_value=[0.0] * 6), \
                mock.patch.object(train, 'score_by_recent_form', return_value=[]), \
                mock.patch.object(train, 'format_form_report', return_value=''):
            fitted = train.fit_points(
                plan, _TinyVocab(), batch, hparams=hparams, loss_name='huber', reps=1,
                weighting=LossWeighting('none'), smear_mode='global',
                groups=numpy.repeat(numpy.arange(6), 4),
            )
        self.assertEqual('global', fitted.smear_mode)
        self.assertTrue(numpy.isfinite(numpy.asarray(fitted.predict(batch))).all())

    def test_oracle_minutes_read_real_results_for_upcoming_fixtures(self):
        from types import SimpleNamespace
        from unittest import mock

        from premierlearning.learning_v2 import dataset

        player = SimpleNamespace(id=7)
        finished = SimpleNamespace(row=SimpleNamespace(), minutes=45)
        played = SimpleNamespace(is_future=False, match=finished, player=player, year=25)
        upcoming = SimpleNamespace(
            is_future=True, match=SimpleNamespace(fixture_id=3), player=player, year=25,
        )
        benched = SimpleNamespace(
            is_future=True, match=SimpleNamespace(fixture_id=4), player=player, year=25,
        )
        with mock.patch.dict(dataset._ACTUAL_MINUTES, {25: {(7, 3): 90.0}}):
            self.assertEqual(45.0, dataset._oracle_minutes(played))
            self.assertEqual(90.0, dataset._oracle_minutes(upcoming))
            self.assertEqual(0.0, dataset._oracle_minutes(benched))

    def test_a_manager_refuses_a_set_that_sees_future_results(self):
        from unittest import mock

        from fpl_manager.season_source import load_managed_season

        db = mock.Mock()
        db.get_prediction_set.return_value = {'config': {'uses_future_information': True}}
        with self.assertRaises(ValueError):
            load_managed_season(db, 33, 25, 1, 38)

    def test_a_manager_takes_a_cheating_set_only_when_told(self):
        from unittest import mock

        from fpl_manager import season_source

        db = mock.Mock()
        db.get_prediction_set.return_value = {'config': {'uses_future_information': True}}
        with mock.patch.object(season_source, 'ManagedSeason') as managed:
            season_source.load_managed_season(db, 33, 25, 1, 38, allow_oracle=True)
        managed.assert_called_once()

    def test_an_oracle_run_must_say_so_in_its_checkpoint_name(self):
        from unittest import mock

        import fpl_manager.__main__ as cli

        db = mock.Mock()
        db.get_prediction_set.return_value = {'config': {'uses_future_information': True}}
        with mock.patch.object(cli, 'DB_Handler', return_value=db), \
                mock.patch.object(cli, '_observation_version', return_value='v2'), \
                mock.patch.object(cli, '_write_features') as write_features:
            with self.assertRaises(ValueError):
                cli.main(['--prediction-set-id', '34', '--allow-oracle', '--checkpoint', 'runs/plain'])
        write_features.assert_not_called()

    def test_minutes_bands_split_at_one_and_sixty_minutes(self):
        from types import SimpleNamespace

        from premierlearning.learning_v2.dataset import minutes_targets

        matches = [SimpleNamespace(minutes=value) for value in (0, 1, 59, 60, 90)]
        self.assertEqual([0, 1, 1, 2, 2], minutes_targets(matches, 'bands').reshape(-1).tolist())

    def test_log_minutes_run_from_0_to_1_and_stretch_the_first_minutes(self):
        from types import SimpleNamespace

        from premierlearning.learning_v2.dataset import minutes_targets

        matches = [SimpleNamespace(minutes=value) for value in (0, 1, 60, 90, 96)]
        scaled = minutes_targets(matches, 'log').reshape(-1)
        numpy.testing.assert_allclose([0.0, 1.0, 1.0], [scaled[0], scaled[3], scaled[4]])
        self.assertGreater(scaled[1] - scaled[0], scaled[3] - scaled[2])

    def test_minutes_head_trains_but_is_dropped_before_predicting(self):
        for target, weight in (('bands', 0.1), ('log', 1.0)):
            with self.subTest(target=target):
                self._check_minutes_head(target, weight)

    def _check_minutes_head(self, target, weight):
        from types import SimpleNamespace
        from unittest import mock

        from premierlearning.learning_v2 import train
        from premierlearning.learning_v2.score import LossWeighting

        plan, hparams = _tiny_plan_and_hparams()
        specs = input_specs(plan)
        labels = numpy.arange(24, dtype=numpy.float32) % 5
        matches = [SimpleNamespace(gameweek=1, minutes=value * 4) for value in range(24)]
        batch = SimpleNamespace(arrays=_zero_feed(specs, 24), labels=labels, matches=matches)
        monitored = []
        real_stopping = train.PlateauStopping

        class SpyStopping(real_stopping):
            def on_epoch_end(self, epoch, logs=None):
                monitored.append((logs or {}).get(self.monitor))
                super().on_epoch_end(epoch, logs)

        with mock.patch.object(train, 'EPOCHS', 1), \
                mock.patch.object(train, 'PlateauStopping', SpyStopping), \
                mock.patch.object(train.sys, 'argv', ['test', '--quick']), \
                mock.patch.object(train, 'recent_gameweek_rates', return_value=[0.0] * 6), \
                mock.patch.object(train, 'score_by_recent_form', return_value=[]), \
                mock.patch.object(train, 'format_form_report', return_value=''):
            fitted = train.fit_points(
                plan, _TinyVocab(), batch, hparams=hparams, loss_name='huber', reps=1,
                weighting=LossWeighting('none'), smear_mode='global',
                groups=numpy.repeat(numpy.arange(6), 4), minutes_weight=weight, minutes_target=target,
            )

        # A wrong metric name would leave the fit with no early stopping at all.
        self.assertTrue(monitored and all(value is not None for value in monitored))
        self.assertEqual(['points'], fitted.models[0].output_names)
        predicted = numpy.asarray(fitted.predict(batch))
        self.assertEqual((24,), predicted.shape)
        self.assertTrue(numpy.isfinite(predicted).all())
        # The bands head is kept beside the points model so its output can be read.
        bands = fitted.predict_minutes_bands(batch)
        if target == 'bands':
            self.assertEqual((24, 3), bands.shape)
            numpy.testing.assert_allclose(numpy.ones(24), bands.sum(axis=1), rtol=1e-5)
        else:
            self.assertIsNone(bands)

    def test_no_minutes_head_keeps_the_single_points_output(self):
        from premierlearning.learning_v2.model import build_model

        plan, hparams = _tiny_plan_and_hparams()
        model = build_model(plan, _TinyVocab(), hparams, loss_name='huber')
        self.assertEqual(['points'], model.output_names)

    def test_poisson_fit_floors_negative_scores_and_stays_positive(self):
        from types import SimpleNamespace
        from unittest import mock

        import tensorflow as tf

        from premierlearning.learning_v2 import train
        from premierlearning.learning_v2.model import build_model
        from premierlearning.learning_v2.score import LossWeighting

        tf.random.set_seed(1)
        plan, hparams = _tiny_plan_and_hparams()
        specs = input_specs(plan)
        labels =numpy.array([-2.0, 0.0, 1.0, 2.0, 6.0, 12.0], dtype=numpy.float32)
        batch = SimpleNamespace(arrays=_zero_feed(specs, 6), labels=labels, matches=[])
        seen = []

        def spy_build(*args, **kwargs):
            model = build_model(*args, **kwargs)
            real_fit = model.fit

            def fit(x, y, **options):
                seen.append(numpy.array(y))
                return real_fit(x, y, **options)

            model.fit = fit
            return model

        with mock.patch.object(train, 'build_model', spy_build), \
                mock.patch.object(train, 'EPOCHS', 1), \
                mock.patch.object(train.sys, 'argv', ['test', '--quick']):
            fitted = train.fit_points(
                plan, _TinyVocab(), batch, hparams=hparams, loss_name='poisson', reps=1,
                weighting=LossWeighting('none'),
            )

        self.assertGreaterEqual(float(seen[0].min()), 0.0)
        self.assertEqual(12.0, float(seen[0].max()))
        predicted = numpy.asarray(fitted.predict(batch))
        self.assertTrue(numpy.isfinite(predicted).all())
        self.assertTrue((predicted > 0).all())

    def test_padded_slots_do_not_change_the_output(self):
        import tensorflow as tf

        from premierlearning.learning_v2.model import build_model

        tf.random.set_seed(1)
        plan, hparams = _tiny_plan_and_hparams()
        model = build_model(plan, _TinyVocab(), hparams, loss_name='huber')
        specs = input_specs(plan)
        self.assertEqual({spec.key for spec in specs}, set(model.input_names))

        feed = _zero_feed(specs, 2)
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


def _teams_plan(team_blocks=1, player_blocks=1):
    plan = load_plan('simple_h1_teams')
    plan.config.history_len = 4
    plan.config.max_squad = 3
    plan.config.team_history_len = 4
    plan.config.team_blocks = team_blocks
    plan.config.player_blocks = player_blocks
    return plan


def _row(batch, player, fixture_id):
    return next(
        index for index, match in enumerate(batch.matches)
        if match.player is player and match.fixture_id == fixture_id
    )


class TeamHistoryTests(unittest.TestCase):
    """Team and opponent match histories, their fading averages, and time blocks."""

    def test_new_plans_parse_and_the_dataset_fills_them(self):
        for name in ('simple_h1_teams', 'simple_h1_teams_pblocks', 'simple_h1_teams_attn'):
            plan = load_plan(name)
            self.assertEqual([], unsupported_leaves(plan))
            self.assertEqual(114, plan.config.team_history_len)
            self.assertEqual(5, plan.config.team_blocks)
            history = plan.root.child('current_opposition').child('fixture_history')
            self.assertEqual('attention' if name.endswith('_attn') else 'transformer', history.kind)
            masks = {
                spec.path: spec.shape for spec in input_specs(plan)
                if spec.path in ('current_team.fixture_history.mask', 'current_opposition.fixture_history.mask')
            }
            self.assertEqual(2, len(masks))
            self.assertEqual({(114,)}, set(masks.values()))
        self.assertEqual(1, load_plan('simple_h1_teams').config.player_blocks)
        self.assertEqual(5, load_plan('simple_h1_teams_pblocks').config.player_blocks)

    def test_attention_variant_trains_from_the_shared_table(self):
        from unittest import mock

        from premierlearning.learning import PlayerNetHParams
        from premierlearning.learning_v2 import train as train_module
        from premierlearning.learning_v2.score import LossWeighting

        season, _saka = _fixture_world()
        plan = load_plan('simple_h1_teams_attn')
        plan.config.history_len = 4
        plan.config.max_squad = 3
        plan.config.team_history_len = 4
        plan.config.d_model = 8
        plan.config.n_heads = 2
        train, predict, vocab = build_dataset([season], plan, predict_season=season)
        hparams = PlayerNetHParams()
        hparams.n_hidden = 1
        hparams.batch_size = 4
        with mock.patch.object(train_module, 'EPOCHS', 1), \
                mock.patch.object(train_module.sys, 'argv', ['test', '--quick']):
            fitted = train_module.fit_points(
                plan, vocab, train, hparams=hparams, loss_name='huber', reps=1,
                weighting=LossWeighting('none'), smear_mode='global',
            )
        self.assertTrue(numpy.isfinite(fitted.predict(predict)).all())

    def test_b0_plans_without_identity_embeddings_train(self):
        """The id-free B0 plans: identity only from the as-of history encoders."""
        from unittest import mock

        from premierlearning.learning import PlayerNetHParams
        from premierlearning.learning_v2 import train as train_module
        from premierlearning.learning_v2.plan import input_specs
        from premierlearning.learning_v2.score import LossWeighting

        for name, gone in (('simple_h1_b0_nopid', ('player_id_embedding',)),
                           ('simple_h1_b0_noids', ('player_id_embedding', 'team_id_embedding')),
                           # Staged scoreline with no team ids: each side's goal bands
                           # come from its fading averages and match history only.
                           ('simple_h1_b0_noids_staged', ('player_id_embedding', 'team_id_embedding')),
                           # B1 inputs (all-competition rest, load, DGW flags) with no ids.
                           ('simple_h1_b1_noids', ('player_id_embedding', 'team_id_embedding'))):
            season, _saka = _fixture_world()
            plan = load_plan(name)
            paths = [spec.path for spec in input_specs(plan)]
            for suffix in gone:
                self.assertFalse([path for path in paths if path.endswith(suffix)], name)
            plan.config.history_len = 4
            plan.config.max_squad = 3
            plan.config.team_history_len = 4
            plan.config.d_model = 8
            plan.config.n_heads = 2
            train, predict, vocab = build_dataset([season], plan, predict_season=season)
            hparams = PlayerNetHParams()
            hparams.n_hidden = 1
            hparams.batch_size = 4
            with mock.patch.object(train_module, 'EPOCHS', 1), \
                    mock.patch.object(train_module.sys, 'argv', ['test', '--quick']):
                fitted = train_module.fit_points(
                    plan, vocab, train, hparams=hparams, loss_name='huber', reps=1,
                    weighting=LossWeighting('none'), smear_mode='global',
                    minutes_weight=0.1, minutes_target='bands',
                )
            self.assertTrue(numpy.isfinite(fitted.predict(predict)).all(), name)

    def test_old_plan_has_no_shared_tables(self):
        season, _saka = _fixture_world()
        plan = load_plan('simple_h1')
        plan.config.history_len = 4
        plan.config.max_squad = 3
        train, predict, vocab = build_dataset([season], plan, predict_season=season)
        self.assertEqual({}, train.tables)
        self.assertEqual({}, predict.table_inputs)
        self.assertFalse([key for key in train.arrays if key.startswith('__ref__')])
        self.assertIsNone(vocab.block_reference)

    def test_team_history_holds_only_earlier_matches_and_is_shared(self):
        from premierlearning.learning_v2.teams import ref_key

        season, saka = _fixture_world()
        mate = next(player for player in season.players if player.code == 102)
        train, predict, vocab = build_dataset([season], _teams_plan(), predict_season=season)
        table = train.tables['team_history']
        team_ref = train.arrays[ref_key('current_team.fixture_history')]
        opp_ref = train.arrays[ref_key('current_opposition.fixture_history')]

        first = _row(train, saka, 1)
        self.assertEqual(0.0, float(table['mask'][team_ref[first]].sum()))

        row = _row(train, saka, 3)
        entry = team_ref[row]
        self.assertEqual([0, 0, 1, 1], table['mask'][entry].astype(int).tolist())
        self.assertEqual([2.0, 1.0], table['goals_for'][entry][2:].tolist())
        self.assertEqual([0.0, 1.0], table['goals_against'][entry][2:].tolist())
        numpy.testing.assert_allclose([24.0, 14.0], table['days_before_current'][entry][2:])
        # Fixture 1 has no Chelsea player row, so its xG is unknown on both sides.
        numpy.testing.assert_allclose([0.2, 0.0], table['xg_for'][entry][2:], atol=1e-6)
        self.assertEqual([0.0, 1.0], table['xg_known'][entry][2:].tolist())
        self.assertAlmostEqual(4.0, float(table['xg_against'][entry][3]))
        self.assertEqual(
            [vocab.team_index(2)] * 2, table['opposition_team_id_embedding'][entry][2:].tolist(),
        )
        # Chelsea's side of the same match: their own earlier results.
        self.assertEqual([0.0, 1.0], table['goals_for'][opp_ref[row]][2:].tolist())
        # Every Arsenal player in fixture 3 points at the same stored history.
        self.assertEqual(entry, team_ref[_row(train, mate, 3)])

        upcoming = predict.arrays[ref_key('current_team.fixture_history')][_row(predict, saka, 4)]
        self.assertEqual([2.0, 1.0, 5.0], table['goals_for'][upcoming][1:].tolist())
        self.assertIs(train.tables, predict.tables)
        self.assertEqual(
            ('team_history', 'goals_for', ref_key('current_opposition.fixture_history')),
            train.table_inputs[tensor_key('current_opposition.fixture_history.goals_for')],
        )

    def test_fading_averages_shrink_toward_the_promoted_prior(self):
        import math

        from premierlearning.learning_v2.teams import PRIOR_WEIGHT, PROMOTED_PRIOR

        season, saka = _fixture_world()
        plan = _teams_plan()
        train, _predict, _vocab = build_dataset([season], plan, predict_season=season)
        key = tensor_key('current_team.team_features.goals_for_decay')
        self.assertAlmostEqual(PROMOTED_PRIOR['goals_for'], float(train.arrays[key][_row(train, saka, 1), 0]), places=5)
        lam = math.log(2.0) / plan.config.team_decay_half_life
        weights = [math.exp(-lam * 24.0), math.exp(-lam * 14.0)]
        expected = (
            (weights[0] * 2.0 + weights[1] * 1.0 + PRIOR_WEIGHT * PROMOTED_PRIOR['goals_for'])
            / (sum(weights) + PRIOR_WEIGHT)
        )
        self.assertAlmostEqual(expected, float(train.arrays[key][_row(train, saka, 3), 0]), places=5)
        known = tensor_key('current_team.team_features.xg_weight')
        self.assertAlmostEqual(
            weights[1] / sum(weights), float(train.arrays[known][_row(train, saka, 3), 0]), places=5,
        )

    def test_blocks_count_back_from_the_cutoff(self):
        from premierlearning.learning_v2.teams import block_of, block_reference, encode_block

        self.assertEqual(0, block_of(-5, 5, 38, 76))
        self.assertEqual(0, block_of(38, 5, 38, 76))
        self.assertEqual(1, block_of(39, 5, 38, 76))
        self.assertEqual(1, block_of(114, 5, 38, 76))
        self.assertEqual(2, block_of(115, 5, 38, 76))
        self.assertEqual(4, block_of(1000, 5, 38, 76))
        self.assertEqual(0, block_of(1000, 1, 38, 76))
        self.assertEqual(7, encode_block(7, 3, 1))
        self.assertEqual(38, encode_block(7, 3, 5))
        self.assertEqual(25 * 38 + 1, block_reference(25, 0))

    def test_block_ids_and_a_frozen_vocab_keep_the_fit_reference(self):
        past_season, _saka = _fixture_world()
        past_season.years = (23, 24)
        season, saka = _fixture_world()
        plan = _teams_plan(team_blocks=5, player_blocks=5)
        train, predict, vocab = build_dataset([past_season, season], plan, predict_season=season, cutoff_gameweek=3)
        self.assertEqual(25 * 38 + 4, vocab.block_reference)
        team_key = tensor_key('current_team.team_id_embedding')
        player_key = tensor_key('player_id_embedding')
        old = next(index for index, match in enumerate(train.matches) if match.player not in season.players)
        # Two seasons back is 76 gameweeks: past the newest 38, inside block 1.
        self.assertEqual(vocab.team_index(1) * 5 + 1, int(train.arrays[team_key][old, 0]))
        new = _row(train, saka, 3)
        self.assertEqual(vocab.team_index(1) * 5, int(train.arrays[team_key][new, 0]))
        self.assertEqual(vocab.player_index(101) * 5, int(train.arrays[player_key][new, 0]))
        upcoming = _row(predict, saka, 4)
        self.assertEqual(vocab.player_index(101) * 5, int(predict.arrays[player_key][upcoming, 0]))

        vocab.freeze()
        _train, later, same = build_dataset(
            [past_season, season], plan, predict_season=season, cutoff_gameweek=10,
            vocab=vocab, include_train=False,
        )
        self.assertEqual(25 * 38 + 4, same.block_reference)

    def test_window_rows_keep_the_shared_table(self):
        from premierlearning.learning_v2.dataset import window_rate_batch

        season, _saka = _fixture_world()
        train, _predict, _vocab = build_dataset([season], _teams_plan(), predict_season=season)
        windowed, _groups = window_rate_batch(train, 1)
        self.assertIs(train.tables, windowed.tables)
        self.assertEqual(train.table_inputs, windowed.table_inputs)

    def test_block_embedding_adds_a_shared_base_and_a_block_part(self):
        import tensorflow as tf

        from premierlearning.learning_v2.model import BlockEmbedding

        layer = BlockEmbedding(4, 3, 2, 1e-4)
        same_team = layer(tf.constant([[6], [7], [8]]))[:, 0, :].numpy()
        # Block parts start at zero, so every block of team 2 starts on its base.
        numpy.testing.assert_allclose(same_team[0], same_team[1])
        numpy.testing.assert_allclose(same_team[0], same_team[2])
        self.assertEqual((4, 2), layer.base.embeddings.shape)
        self.assertEqual((12, 2), layer.block.embeddings.shape)

    def test_teams_model_trains_and_predicts_from_the_shared_table(self):
        from unittest import mock

        from premierlearning.learning import PlayerNetHParams
        from premierlearning.learning_v2 import train as train_module
        from premierlearning.learning_v2.model import build_model
        from premierlearning.learning_v2.score import LossWeighting

        season, saka = _fixture_world()
        plan = _teams_plan(team_blocks=3, player_blocks=2)
        plan.config.d_model = 8
        plan.config.n_heads = 2
        plan.config.player_emb_dim = 4
        plan.config.team_emb_dim = 4
        plan.config.position_emb_dim = 2
        train, predict, vocab = build_dataset([season], plan, predict_season=season)
        model = build_model(plan, vocab, PlayerNetHParams(), loss_name='huber')
        self.assertEqual({spec.key for spec in input_specs(plan)}, set(model.input_names))
        hparams = PlayerNetHParams()
        hparams.dropout_1 = 0.0
        hparams.dropout_2 = 0.0
        hparams.n_hidden = 1
        hparams.batch_size = 4
        with mock.patch.object(train_module, 'EPOCHS', 1), \
                mock.patch.object(train_module.sys, 'argv', ['test', '--quick']):
            fitted = train_module.fit_points(
                plan, vocab, train, hparams=hparams, loss_name='huber', reps=1,
                weighting=LossWeighting('none'), smear_mode='global',
                minutes_weight=0.1, minutes_target='bands',
            )
        before = {key: numpy.array(value, copy=True) for key, value in train.tables['team_history'].items()}
        points = fitted.predict(predict)
        bands = fitted.predict_minutes_bands(predict)
        self.assertEqual((len(predict.matches),), points.shape)
        self.assertTrue(numpy.isfinite(points).all())
        numpy.testing.assert_allclose(numpy.ones(len(predict.matches)), bands.sum(axis=1), rtol=1e-5)
        # Predicting scales copies: the shared table is left as built.
        for key, value in train.tables['team_history'].items():
            numpy.testing.assert_array_equal(before[key], value)
        self.assertIn(tensor_key('current_team.fixture_history.goals_for'), fitted.stats)

    def test_table_fit_holds_out_test_and_validation_rows(self):
        from types import SimpleNamespace
        from unittest import mock

        from premierlearning.learning import PlayerNetHParams
        from premierlearning.learning_v2 import train as train_module
        from premierlearning.learning_v2.score import LossWeighting
        from premierlearning.learning_v2.teams import is_team_history_path, ref_key, team_history_side

        plan = _teams_plan()
        plan.config.d_model = 8
        plan.config.n_heads = 2
        specs = input_specs(plan)
        rows = 30
        arrays = _zero_feed([spec for spec in specs if not is_team_history_path(spec.path)], rows)
        table = {'mask': numpy.zeros((3, 4), dtype=numpy.float32)}
        table['mask'][1:, 2:] = 1.0
        table_inputs = {}
        for spec in specs:
            if not is_team_history_path(spec.path):
                continue
            side, name = team_history_side(spec.path)
            if name != 'mask':
                dtype = numpy.int32 if spec.kind == 'embedding' else numpy.float32
                table[name] = (numpy.arange(12).reshape(3, 4) % 3).astype(dtype)
            table_inputs[spec.key] = ('team_history', name, ref_key(side))
            arrays[ref_key(side)] = numpy.arange(rows) % 3
        batch = SimpleNamespace(
            arrays=arrays, labels=numpy.arange(rows, dtype=numpy.float32) % 4,
            matches=[SimpleNamespace(gameweek=1, minutes=90)] * rows,
            tables={'team_history': table}, table_inputs=table_inputs,
        )
        hparams = PlayerNetHParams()
        hparams.n_hidden = 1
        hparams.batch_size = 8
        for groups in (None, numpy.repeat(numpy.arange(10), 3)):
            with self.subTest(grouped=groups is not None), \
                    mock.patch.object(train_module, 'EPOCHS', 2), \
                    mock.patch.object(train_module.sys, 'argv', ['test', '--quick']), \
                    mock.patch.object(train_module, 'recent_gameweek_rates', return_value=[0.0] * 6), \
                    mock.patch.object(train_module, 'score_by_recent_form', return_value=[]), \
                    mock.patch.object(train_module, 'format_form_report', return_value=''):
                fitted = train_module.fit_points(
                    plan, _TinyVocab(), batch, hparams=hparams, loss_name='huber', reps=1,
                    weighting=LossWeighting('none'), smear_mode='global', groups=groups,
                )
                predicted = fitted.predict(batch)
                self.assertEqual((rows,), predicted.shape)
                self.assertTrue(numpy.isfinite(predicted).all())
                self.assertIsNotNone(fitted.smearing)

    def test_table_feed_matches_the_row_copy(self):
        from premierlearning.learning_v2.train import TableFeed

        arrays = {'x': numpy.arange(5.0).reshape(-1, 1), '__ref__t': numpy.array([2, 0, 1, 2, 1])}
        table = numpy.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32)
        feed = TableFeed(arrays, ['x'], {'t_in': (table, '__ref__t')}, y=numpy.arange(5.0), batch_size=2)
        self.assertEqual(3, len(feed))
        x, y = feed[0]
        numpy.testing.assert_array_equal([[3.0, 4.0], [0.0, 0.0]], x['t_in'])
        numpy.testing.assert_array_equal([0.0, 1.0], y)
        self.assertEqual((1, 2), feed[2][0]['t_in'].shape)


class ScorelineTests(unittest.TestCase):
    """The team-goals head and points built from minutes bands and the scoreline."""

    def test_plans_parse_with_the_scoreline_settings(self):
        from premierlearning.learning_v2.plan import plan_from_dict

        score = load_plan('simple_h1_teams_score')
        scoreline = load_plan('simple_h1_teams_scoreline')
        self.assertEqual([], unsupported_leaves(scoreline))
        self.assertEqual((0.1, 'direct'), (score.config.score_head_weight, score.config.points_structure))
        self.assertEqual((0.1, 'scoreline'), (scoreline.config.score_head_weight, scoreline.config.points_structure))
        plain = load_plan('simple_h1_teams')
        self.assertEqual((0.0, 'direct'), (plain.config.score_head_weight, plain.config.points_structure))
        with self.assertRaises(ValueError):
            plan_from_dict({'_config': {'points_structure': 'other'}})

    def test_score_targets_are_the_team_side_and_skip_unknown_scores(self):
        from types import SimpleNamespace

        from premierlearning.learning_v2.dataset import score_targets

        matches = [
            SimpleNamespace(team_match=SimpleNamespace(scored=2, conceded=1)),
            SimpleNamespace(team_match=None),
            SimpleNamespace(team_match=SimpleNamespace(scored=None, conceded=0)),
        ]
        numpy.testing.assert_array_equal([[2.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]], score_targets(matches))

    def test_masked_poisson_skips_unknown_scores(self):
        import tensorflow as tf

        from premierlearning.learning_v2.model import masked_poisson

        pred = tf.constant([[1.5, 1.0], [1.5, 1.0]])
        loss = masked_poisson(tf.constant([[2.0, 1.0], [-1.0, -1.0]]), pred).numpy()
        expected = ((1.5 - 2.0 * numpy.log(1.5)) + 1.0) / 2.0
        self.assertAlmostEqual(expected, float(loss[0]), places=5)
        self.assertEqual(0.0, float(loss[1]))

    def test_scoreline_points_follow_the_fpl_parts(self):
        import tensorflow as tf

        from premierlearning.learning_v2.model import ScorelinePoints

        def inverse_softplus(value):
            return numpy.log(numpy.expm1(value))

        bands = tf.constant([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
        goals = tf.constant([[1.5, 0.8], [1.5, 0.8]])
        # attack 0.4, sub share 0.5, other 0.6 (full) and 0.3 (sub)
        part_row = [inverse_softplus(0.4), 0.0, inverse_softplus(0.6), inverse_softplus(0.3)]
        parts = tf.constant([part_row, part_row], dtype=tf.float32)
        position = tf.constant([[2], [4]])
        raw = ScorelinePoints(log_output=False)([bands, goals, parts, position]).numpy().reshape(-1)
        sub = 1.0 + 0.5 * 0.4 * 1.5 + 0.3
        defender = 2.0 + 4.0 * numpy.exp(-0.8) - 0.5 * 0.8 + 0.4 * 1.5 + 0.6
        forward = 2.0 + 0.4 * 1.5 + 0.6
        numpy.testing.assert_allclose([0.2 * sub + 0.7 * defender, 0.2 * sub + 0.7 * forward], raw, rtol=1e-5)
        logged = ScorelinePoints(log_output=True)([bands, goals, parts, position]).numpy().reshape(-1)
        numpy.testing.assert_allclose(numpy.log1p(raw), logged, rtol=1e-5)

    def _fit(self, name, minutes_weight):
        from unittest import mock

        from premierlearning.learning import PlayerNetHParams
        from premierlearning.learning_v2 import train as train_module
        from premierlearning.learning_v2.score import LossWeighting

        season, _saka = _fixture_world()
        plan = load_plan(name)
        plan.config.history_len = 4
        plan.config.max_squad = 3
        plan.config.team_history_len = 4
        plan.config.d_model = 8
        plan.config.n_heads = 2
        train, predict, vocab = build_dataset([season], plan, predict_season=season)
        hparams = PlayerNetHParams()
        hparams.n_hidden = 1
        hparams.batch_size = 4
        with mock.patch.object(train_module, 'EPOCHS', 1), \
                mock.patch.object(train_module.sys, 'argv', ['test', '--quick']):
            fitted = train_module.fit_points(
                plan, vocab, train, hparams=hparams, loss_name='huber', reps=1,
                weighting=LossWeighting('none'), smear_mode='global',
                minutes_weight=minutes_weight, minutes_target='bands',
            )
        return fitted, predict

    def test_both_variants_train_and_predict_points_only(self):
        for name, minutes_weight in (
            ('simple_h1_teams_score', 0.1),
            ('simple_h1_teams_scoreline', 0.1),
            ('simple_h1_teams_scoreline', 0.0),
        ):
            with self.subTest(name=name, minutes_weight=minutes_weight):
                fitted, predict = self._fit(name, minutes_weight)
                self.assertEqual(['points'], fitted.models[0].output_names)
                points = fitted.predict(predict)
                self.assertEqual((len(predict.matches),), points.shape)
                self.assertTrue(numpy.isfinite(points).all())
                bands = fitted.predict_minutes_bands(predict)
                if minutes_weight:
                    numpy.testing.assert_allclose(numpy.ones(len(predict.matches)), bands.sum(axis=1), rtol=1e-5)
                else:
                    self.assertIsNone(bands)


class RecencyPoolTests(unittest.TestCase):
    """Attention pooling whose weights fade with age, in place of the flat average."""

    def test_plans_parse_and_find_the_age_leaves(self):
        from premierlearning.learning_v2.plan import age_leaf, plan_from_dict

        kept = load_plan('simple_h1_teams_recency')
        dropped = load_plan('simple_h1_teams_recency_nodecay')
        self.assertEqual('recency', kept.config.history_pool)
        self.assertEqual('mean', load_plan('simple_h1_teams').config.history_pool)
        self.assertEqual([], unsupported_leaves(dropped))
        self.assertIn('xg_against_decay', [c.name for c in kept.root.child('current_opposition').child('team_features').children])
        self.assertEqual(
            ['goals_last_n', 'goals_conceded_last_n'],
            [c.name for c in dropped.root.child('current_opposition').child('team_features').children],
        )
        ages = {
            name: age_leaf(kept.root.child(name) if '.' not in name else
                           kept.root.child(name.split('.')[0]).child(name.split('.')[1])).path
            for name in ('player_fixture_history', 'last_season', 'current_team.fixture_history')
        }
        self.assertEqual('player_fixture_history.match_facts.days_before_current', ages['player_fixture_history'])
        self.assertEqual('last_season.seasons_before', ages['last_season'])
        self.assertEqual('current_team.fixture_history.days_before_current', ages['current_team.fixture_history'])
        with self.assertRaises(ValueError):
            plan_from_dict({'_config': {'history_pool': 'max'}})

    def test_a_fast_fade_reads_the_newest_item_and_padding_is_ignored(self):
        import tensorflow as tf

        from premierlearning.learning_v2.model import RecencyPool

        layer = RecencyPool(4, 1)
        x = tf.constant([[[9.0, 9.0, 9.0, 9.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
        mask = tf.constant([[0.0, 1.0, 1.0, 1.0]])
        age = tf.constant([[0.0, 3.0, 2.0, 1.0]])
        layer(x, mask, age)
        layer.rate.assign([20.0])
        layer.query.assign(tf.zeros_like(layer.query))
        pooled = layer(x, mask, age).numpy()
        newest = layer(x[:, 3:, :], tf.constant([[1.0]]), tf.constant([[1.0]])).numpy()
        numpy.testing.assert_allclose(newest[0, :4], pooled[0, :4], atol=1e-5)
        self.assertAlmostEqual(numpy.log1p(3.0) / numpy.log1p(4.0), float(pooled[0, 4]), places=5)
        empty = layer(x, tf.zeros((1, 4)), age).numpy()
        numpy.testing.assert_allclose(numpy.zeros(5), empty[0], atol=1e-6)

    def test_both_recency_plans_train_and_predict(self):
        from unittest import mock

        from premierlearning.learning import PlayerNetHParams
        from premierlearning.learning_v2 import train as train_module
        from premierlearning.learning_v2.score import LossWeighting

        for name in ('simple_h1_teams_recency', 'simple_h1_teams_recency_nodecay'):
            with self.subTest(name=name):
                season, _saka = _fixture_world()
                plan = load_plan(name)
                plan.config.history_len = 4
                plan.config.max_squad = 3
                plan.config.team_history_len = 4
                plan.config.d_model = 8
                plan.config.n_heads = 2
                train, predict, vocab = build_dataset([season], plan, predict_season=season)
                hparams = PlayerNetHParams()
                hparams.n_hidden = 1
                hparams.batch_size = 4
                with mock.patch.object(train_module, 'EPOCHS', 1), \
                        mock.patch.object(train_module.sys, 'argv', ['test', '--quick']):
                    fitted = train_module.fit_points(
                        plan, vocab, train, hparams=hparams, loss_name='huber', reps=1,
                        weighting=LossWeighting('none'), smear_mode='global',
                        minutes_weight=0.1, minutes_target='bands',
                    )
                points = fitted.predict(predict)
                self.assertEqual((len(predict.matches),), points.shape)
                self.assertTrue(numpy.isfinite(points).all())


def _small_staged_plan():
    plan = load_plan('simple_h1_teams_staged')
    plan.config.history_len = 4
    plan.config.max_squad = 3
    plan.config.team_history_len = 4
    plan.config.d_model = 8
    plan.config.n_heads = 2
    return plan


class _FakeExternal:
    """Stands in for ExternalInputs: values encode which side and club were asked for."""

    def __init__(self):
        self.calls = []

    def team(self, season, fixture_id, team_code):
        from premierlearning.learning_v2.external_inputs import TEAM_DEFAULTS

        self.calls.append(('team', season, fixture_id, team_code))
        values = dict(TEAM_DEFAULTS)
        values['schedule.days_since_last_match'] = float(team_code * 10 + fixture_id)
        values['gameweek.gw_double'] = 1.0 if fixture_id == 3 else 0.0
        return values

    def player(self, season, fixture_id, fpl_code):
        from premierlearning.learning_v2.external_inputs import PLAYER_DEFAULTS

        self.calls.append(('player', season, fixture_id, fpl_code))
        values = dict(PLAYER_DEFAULTS)
        values['player_load.minutes_last_14d'] = float(fpl_code + fixture_id)
        return values


class ExternalHookTests(unittest.TestCase):
    """Plan leaves filled from the external-data features (B1: schedule, gameweek, player_load)."""

    def _context_plan(self):
        import json

        from simulation.model_eval.derive_plan import add_family
        from premierlearning.learning_v2.plan import PLANS_DIR, plan_from_dict

        with (PLANS_DIR / 'simple_h1_teams.json').open(encoding='utf-8') as handle:
            raw = add_family(json.load(handle), 'context')
        plan = plan_from_dict(json.loads(json.dumps(raw)), 'b1_test')
        plan.config.history_len = 4
        plan.config.max_squad = 3
        plan.config.team_history_len = 4
        return plan

    def test_context_family_adds_the_blocks_and_drops_pl_only_rest_days(self):
        plan = self._context_plan()
        self.assertEqual([], unsupported_leaves(plan))
        names = [child.name for child in plan.root.child('current_opposition').children]
        self.assertEqual(['schedule', 'gameweek'], names[names.index('team_features') + 1:names.index('team_features') + 3])
        self.assertEqual(['home_away'], [child.name for child in plan.root.child('match_facts').children])
        self.assertIn('player_load', [child.name for child in plan.root.children])

    def test_scoreline_family_switches_the_points_structure(self):
        import json

        from simulation.model_eval.derive_plan import add_family
        from premierlearning.learning_v2.plan import PLANS_DIR, POINTS_SCORELINE, plan_from_dict

        with (PLANS_DIR / 'simple_h1_teams.json').open(encoding='utf-8') as handle:
            raw = add_family(add_family(json.load(handle), 'context'), 'scoreline')
        plan = plan_from_dict(json.loads(json.dumps(raw)), 'b1_scoreline_test')
        self.assertEqual(POINTS_SCORELINE, plan.config.points_structure)
        self.assertEqual(0.1, plan.config.score_head_weight)
        self.assertEqual([], unsupported_leaves(plan))

    def test_staged_family_switches_the_points_structure_once(self):
        import json

        from simulation.model_eval.derive_plan import add_family
        from premierlearning.learning_v2.plan import PLANS_DIR, POINTS_STAGED, plan_from_dict

        with (PLANS_DIR / 'simple_h1_teams.json').open(encoding='utf-8') as handle:
            raw = add_family(add_family(json.load(handle), 'context'), 'staged')
        plan = plan_from_dict(json.loads(json.dumps(raw)), 'b1_staged_test')
        self.assertEqual(POINTS_STAGED, plan.config.points_structure)
        self.assertEqual(0.1, plan.config.score_head_weight)
        with self.assertRaises(SystemExit):
            add_family(raw, 'staged')

    def test_rows_read_their_own_side_club_and_player(self):
        from unittest import mock

        from premierlearning.learning_v2 import external_hook

        season, saka = _fixture_world()
        fake = _FakeExternal()
        with mock.patch.object(external_hook, '_LOADED', fake):
            train, predict, _vocab = build_dataset([season], self._context_plan(), predict_season=season)
        row = _row(train, saka, 3)
        match = train.matches[row]
        team = int(match.team.code)
        opp = int(match.opponent_team.code)
        days = train.arrays[tensor_key('current_team.schedule.days_since_last_match')]
        self.assertEqual(float(team * 10 + 3), float(days[row, 0]))
        opp_days = train.arrays[tensor_key('current_opposition.schedule.days_since_last_match')]
        self.assertEqual(float(opp * 10 + 3), float(opp_days[row, 0]))
        self.assertEqual(1.0, float(train.arrays[tensor_key('current_team.gameweek.gw_double')][row, 0]))
        load = train.arrays[tensor_key('player_load.minutes_last_14d')]
        self.assertEqual(float(saka.code + 3), float(load[row, 0]))
        # Season objects say 25; the feature tables are keyed by 2025.
        self.assertTrue(all(call[1] == 2025 for call in fake.calls))
        upcoming = _row(predict, saka, 4)
        self.assertEqual(float(team * 10 + 4), float(predict.arrays[tensor_key('current_team.schedule.days_since_last_match')][upcoming, 0]))

    def test_old_plans_never_load_the_feature_tables(self):
        from unittest import mock

        from premierlearning.learning_v2 import external_hook

        season, _saka = _fixture_world()
        with mock.patch.object(external_hook, 'load_external', side_effect=AssertionError('loaded')):
            build_dataset([season], _teams_plan(), predict_season=season)

    def test_context_plan_trains_and_predicts(self):
        from unittest import mock

        from premierlearning.learning import PlayerNetHParams
        from premierlearning.learning_v2 import external_hook
        from premierlearning.learning_v2 import train as train_module
        from premierlearning.learning_v2.score import LossWeighting

        season, _saka = _fixture_world()
        plan = self._context_plan()
        plan.config.d_model = 8
        plan.config.n_heads = 2
        with mock.patch.object(external_hook, '_LOADED', _FakeExternal()):
            train, predict, vocab = build_dataset([season], plan, predict_season=season)
        hparams = PlayerNetHParams()
        hparams.n_hidden = 1
        hparams.batch_size = 4
        with mock.patch.object(train_module, 'EPOCHS', 1), \
                mock.patch.object(train_module.sys, 'argv', ['test', '--quick']):
            fitted = train_module.fit_points(
                plan, vocab, train, hparams=hparams, loss_name='huber', reps=1,
                weighting=LossWeighting('none'), smear_mode='global',
                minutes_weight=0.1, minutes_target='bands',
            )
        self.assertIn(tensor_key('player_load.minutes_last_14d'), fitted.stats)
        self.assertTrue(numpy.isfinite(fitted.predict(predict)).all())


class StagedTests(unittest.TestCase):
    """Minutes bands, a team-level scoreline in goal bands, and points from both, in one network."""

    def test_weighted_score_targets_count_each_team_fixture_once(self):
        from types import SimpleNamespace

        from premierlearning.learning_v2.dataset import score_targets

        arsenal = SimpleNamespace(scored=2, conceded=0)
        chelsea = SimpleNamespace(scored=0, conceded=2)
        matches = [SimpleNamespace(team_match=arsenal)] * 3 + [SimpleNamespace(team_match=chelsea),
                                                               SimpleNamespace(team_match=None)]
        targets = score_targets(matches, weighted=True)
        self.assertEqual((5, 3), targets.shape)
        # 4 known rows in 2 team fixtures: each fixture's rows share 2 of weight.
        numpy.testing.assert_allclose([2 / 3.0] * 3 + [2.0, 0.0], targets[:, 2], rtol=1e-6)
        numpy.testing.assert_array_equal([2.0, 0.0], targets[0, :2])

    def test_goal_band_loss_picks_the_real_bands_and_skips_unknown(self):
        import tensorflow as tf

        from premierlearning.learning_v2.staged import masked_goal_band_ce

        bands = numpy.array([[0.1, 0.2, 0.3, 0.2, 0.2, 0.5, 0.2, 0.1, 0.1, 0.1]] * 2, dtype=numpy.float32)
        y_true = tf.constant([[6.0, 0.0, 2.0], [-1.0, -1.0, 1.0]])
        loss = masked_goal_band_ce(y_true, tf.constant(bands)).numpy()
        # 6 goals sits in the 4+ band.
        self.assertAlmostEqual(-0.5 * (numpy.log(0.2) + numpy.log(0.5)) * 2.0, float(loss[0]), places=5)
        self.assertEqual(0.0, float(loss[1]))

    def test_band_points_follow_the_fpl_parts(self):
        import tensorflow as tf

        from premierlearning.learning_v2.staged import BandScorelinePoints

        minutes = tf.constant([[0.1, 0.2, 0.7]] * 2)
        for_bands = [0.2, 0.3, 0.3, 0.1, 0.1]
        against_bands = [0.4, 0.3, 0.2, 0.05, 0.05]
        goals = tf.constant([for_bands + against_bands] * 2)
        part_row = [numpy.log(numpy.expm1(0.4)), 0.0, numpy.log(numpy.expm1(0.6)), numpy.log(numpy.expm1(0.3))]
        parts = tf.constant([part_row] * 2, dtype=tf.float32)
        position = tf.constant([[1], [3]])
        raw = BandScorelinePoints(log_output=False)([minutes, goals, parts, position]).numpy().reshape(-1)
        goals_for = 0.3 + 0.6 + 0.3 + 0.1 * 4.4
        steps = 0.2 + 0.05 + 2 * 0.05
        sub = 1.0 + 0.5 * 0.4 * goals_for + 0.3
        keeper = 2.0 + 4.0 * 0.4 - steps + 0.4 * goals_for + 0.6
        midfielder = 2.0 + 1.0 * 0.4 + 0.4 * goals_for + 0.6
        numpy.testing.assert_allclose([0.2 * sub + 0.7 * keeper, 0.2 * sub + 0.7 * midfielder], raw, rtol=1e-5)

    def test_scoreline_is_team_level_and_mirrored(self):
        from premierlearning.learning import PlayerNetHParams
        from premierlearning.learning_v2.model import build_model
        from premierlearning.learning_v2.teams import ref_key

        season, _saka = _fixture_world()
        plan = _small_staged_plan()
        train, _predict, vocab = build_dataset([season], plan, predict_season=season)
        model = build_model(plan, vocab, PlayerNetHParams(), loss_name='huber', minutes_head='bands')
        self.assertEqual(['points', 'minutes', 'score'], model.output_names)
        names = [layer.name for layer in model.layers]
        self.assertIn('hist_team_shared', names)
        self.assertFalse([name for name in names if name.startswith('hist_current_')])
        x = {}
        for spec in input_specs(plan):
            if spec.key in train.table_inputs:
                table, field, ref = train.table_inputs[spec.key]
                x[spec.key] = train.tables[table][field][train.arrays[ref]]
            else:
                x[spec.key] = train.arrays[spec.key]
        _points, _minutes, score = model.predict(x, verbose=0)
        team_ref = train.arrays[ref_key('current_team.fixture_history')]
        rows_by_fixture = {}
        for index, match in enumerate(train.matches):
            rows_by_fixture.setdefault((match.fixture_id, match.team.code), []).append(index)
        shared = [rows for rows in rows_by_fixture.values() if len(rows) > 1]
        self.assertTrue(shared)
        for rows in shared:
            self.assertEqual(1, len({int(team_ref[row]) for row in rows}))
            numpy.testing.assert_allclose(score[rows[0]], score[rows[1]], atol=1e-5)
        numpy.testing.assert_allclose(numpy.ones((len(score), 2)), score.reshape(-1, 2, 5).sum(axis=2), rtol=1e-5)

    def test_staged_model_trains_and_predicts_points_only(self):
        from unittest import mock

        from premierlearning.learning import PlayerNetHParams
        from premierlearning.learning_v2 import train as train_module
        from premierlearning.learning_v2.score import LossWeighting

        season, _saka = _fixture_world()
        plan = _small_staged_plan()
        train, predict, vocab = build_dataset([season], plan, predict_season=season)
        hparams = PlayerNetHParams()
        hparams.n_hidden = 1
        hparams.batch_size = 4
        with mock.patch.object(train_module, 'EPOCHS', 1), \
                mock.patch.object(train_module.sys, 'argv', ['test', '--quick']):
            fitted = train_module.fit_points(
                plan, vocab, train, hparams=hparams, loss_name='huber', reps=1,
                weighting=LossWeighting('none'), smear_mode='global',
                minutes_weight=0.1, minutes_target='bands',
            )
        self.assertEqual(['points'], fitted.models[0].output_names)
        points = fitted.predict(predict)
        self.assertEqual((len(predict.matches),), points.shape)
        self.assertTrue(numpy.isfinite(points).all())


def _small_staged_v2_plan():
    plan = _small_staged_plan()
    plan.config.scoreline_squads = True
    plan.config.score_xg_weight = 1.0
    return plan


class StagedV2Tests(unittest.TestCase):
    """Staged with a squad summary in the scoreline and xG as a second scoreline target."""

    def test_settings_parse_and_need_the_staged_structure(self):
        from premierlearning.learning_v2.plan import PlanConfig

        config = PlanConfig.from_dict({'points_structure': 'staged', 'scoreline_squads': True, 'score_xg_weight': 1})
        self.assertTrue(config.scoreline_squads)
        self.assertEqual(1.0, config.score_xg_weight)
        with self.assertRaises(ValueError):
            PlanConfig.from_dict({'points_structure': 'direct', 'scoreline_squads': True})
        with self.assertRaises(ValueError):
            PlanConfig.from_dict({'points_structure': 'staged', 'score_xg_weight': -1})

    def test_score_targets_carry_the_team_xg_columns(self):
        from types import SimpleNamespace

        from premierlearning.learning_v2.dataset import score_targets

        team = SimpleNamespace(scored=2, conceded=1)
        matches = [SimpleNamespace(team_match=team)] * 2
        xg = numpy.array([[1.7, 0.4, 1.0], [1.7, 0.4, 1.0]], dtype=numpy.float32)
        targets = score_targets(matches, weighted=True, team_xg=xg)
        self.assertEqual((2, 6), targets.shape)
        numpy.testing.assert_allclose([2.0, 1.0, 1.0, 1.7, 0.4, 1.0], targets[0], rtol=1e-6)
        with self.assertRaises(ValueError):
            score_targets(matches, weighted=False, team_xg=xg)

    def test_xg_term_is_zero_at_the_xg_and_skipped_without_it(self):
        import tensorflow as tf

        from premierlearning.learning_v2.staged import goal_band_loss, masked_goal_band_ce

        self.assertIs(masked_goal_band_ce, goal_band_loss(0.0))
        loss = goal_band_loss(1.0)
        # Band means: for = 0.5 + 2 * 0.5 = 1.5; against = 1.0.
        bands = numpy.array([[0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 3, dtype=numpy.float32)
        y_true = tf.constant([
            [1.0, 1.0, 1.0, 1.5, 1.0, 1.0],    # xG equals the means: only the band term
            [1.0, 1.0, 1.0, 0.5, 2.0, 1.0],    # xG away from the means: adds deviance
            [1.0, 1.0, 1.0, 0.5, 2.0, 0.0],    # xG unknown: only the band term
        ])
        values = loss(y_true, tf.constant(bands)).numpy()
        base = masked_goal_band_ce(y_true[:, :3], tf.constant(bands)).numpy()
        self.assertAlmostEqual(float(base[0]), float(values[0]), places=4)
        deviance = (1.5 - 0.5 + 0.5 * numpy.log(0.5 / 1.5)) + (1.0 - 2.0 + 2.0 * numpy.log(2.0 / 1.0))
        self.assertAlmostEqual(float(base[1]) + 0.5 * deviance, float(values[1]), places=4)
        self.assertAlmostEqual(float(base[2]), float(values[2]), places=4)

    def test_teammates_list_keeps_the_player_only_with_scoreline_squads(self):
        season, saka = _fixture_world()
        for flag in (False, True):
            plan = _small_staged_plan()
            plan.config.max_squad = 30
            plan.config.scoreline_squads = flag
            from premierlearning.learning_v2.dataset import _Catalog, Vocab

            catalog = _Catalog([season], plan, Vocab())
            match = saka.matches[-1]
            row = catalog.row(saka, match, int(season.years[0]), False)
            self.assertEqual(flag, any(player is saka for player in row.teammates))

    def test_squad_scoreline_is_team_level_and_mirrored(self):
        from premierlearning.learning import PlayerNetHParams
        from premierlearning.learning_v2.model import build_model

        season, _saka = _fixture_world()
        plan = _small_staged_v2_plan()
        train, _predict, vocab = build_dataset([season], plan, predict_season=season)
        model = build_model(plan, vocab, PlayerNetHParams(), loss_name='huber', minutes_head='bands')
        self.assertIn('squad_proj_team_shared', [layer.name for layer in model.layers])
        x = {}
        for spec in input_specs(plan):
            if spec.key in train.table_inputs:
                table, field, ref = train.table_inputs[spec.key]
                x[spec.key] = train.tables[table][field][train.arrays[ref]]
            else:
                x[spec.key] = train.arrays[spec.key]
        _points, _minutes, score = model.predict(x, verbose=0)
        rows_by_side = {}
        for index, match in enumerate(train.matches):
            rows_by_side.setdefault((match.fixture_id, match.team.code), []).append(index)
        checked = 0
        for (fixture_id, code), rows in rows_by_side.items():
            for row in rows[1:]:
                numpy.testing.assert_allclose(score[rows[0]], score[row], atol=1e-5)
                checked += 1
            for (other_fixture, other_code), other_rows in rows_by_side.items():
                if other_fixture == fixture_id and other_code != code:
                    mine = score[rows[0]].reshape(2, 5)
                    theirs = score[other_rows[0]].reshape(2, 5)
                    numpy.testing.assert_allclose(mine[0], theirs[1], atol=1e-5)
                    numpy.testing.assert_allclose(mine[1], theirs[0], atol=1e-5)
                    checked += 1
        self.assertTrue(checked)

    def test_staged_v2_trains_with_xg_and_predicts_points_only(self):
        from unittest import mock

        from premierlearning.learning import PlayerNetHParams
        from premierlearning.learning_v2 import train as train_module
        from premierlearning.learning_v2.score import LossWeighting

        season, _saka = _fixture_world()
        plan = _small_staged_v2_plan()
        train, predict, vocab = build_dataset([season], plan, predict_season=season)
        self.assertEqual((len(train.matches), 3), train.team_xg.shape)
        hparams = PlayerNetHParams()
        hparams.n_hidden = 1
        hparams.batch_size = 4
        with mock.patch.object(train_module, 'EPOCHS', 1), \
                mock.patch.object(train_module.sys, 'argv', ['test', '--quick']):
            fitted = train_module.fit_points(
                plan, vocab, train, hparams=hparams, loss_name='huber', reps=1,
                weighting=LossWeighting('none'), smear_mode='global',
                minutes_weight=0.1, minutes_target='bands',
            )
        points = fitted.predict(predict)
        self.assertEqual((len(predict.matches),), points.shape)
        self.assertTrue(numpy.isfinite(points).all())


if __name__ == '__main__':
    unittest.main()
