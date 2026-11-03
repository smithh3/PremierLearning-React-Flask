"""One evaluation unit per (season, origin gameweek), shared by every model.

A unit holds the players eligible at that deadline and, for each of the next
`horizon` gameweeks, whether they had a fixture, what they actually scored, and
what each model predicted at the deadline. Eligibility depends only on what was
known at the deadline, never on a model or on the target fixture, so every model
is scored on exactly the same rows. A player a model did not predict counts as a
prediction of 0.

Double gameweeks are summed per player per gameweek on both sides. A blank
gameweek is 0 predicted and 0 actual: it counts in 5-GW totals and is left out
of single-GW metrics.
"""

from collections import defaultdict

import numpy

from simulation.historical_data import VaastavSeasonData, _float

# How each kind of prediction set is turned into the number the optimiser uses.
# v1 stores raw pp90 and minutes and leaves chance-to-play to squadtools. v2
# predicts raw points with chance-to-play as an input. Sets built before
# v2_chance_input trained on a constant 1.0 flag, so the model never learned it;
# those get chance applied here as for v1.
SCORE_OPTIMISER = 'optimiser'
SCORE_RAW_POINTS = 'raw_points'
SCORE_RAW_POINTS_CHANCE = 'raw_points_x_chance'


def scorer_for_set(prediction_set):
    config = (prediction_set or {}).get('config') or {}
    if config.get('model') == 'learning_v2':
        if config.get('v2_chance_input'):
            return SCORE_RAW_POINTS
        return SCORE_RAW_POINTS_CHANCE
    return SCORE_OPTIMISER


def row_points(row, scorer):
    if scorer == SCORE_RAW_POINTS:
        return float(row.get('predicted_points') or 0.0)
    from squadtools.availability import chance_multiplier, row_optimiser_points
    if scorer == SCORE_RAW_POINTS_CHANCE:
        return float(row.get('predicted_points') or 0.0) * chance_multiplier(row.get('chance_of_playing'))
    return float(row_optimiser_points(row))


class Unit:
    """minutes and fixture_count (player x lead, realised) pick the rows the
    fixture-swing metric uses; units built without them simply skip it."""

    def __init__(self, season, origin, player_ids, has_fixture, actual, valid_leads,
                 minutes=None, fixture_count=None, codes=None, fixtures=None):
        self.season = int(season)
        self.origin = int(origin)
        self.player_ids = numpy.asarray(player_ids, dtype=int)
        self.has_fixture = numpy.asarray(has_fixture, dtype=bool)
        self.actual = numpy.asarray(actual, dtype=float)
        self.valid_leads = int(valid_leads)
        self.minutes = None if minutes is None else numpy.asarray(minutes, dtype=float)
        self.fixture_count = None if fixture_count is None else numpy.asarray(fixture_count, dtype=int)
        # Stable FPL code per player (0 unknown) and the fixture ids each player
        # played per lead: slices.py looks the eval slices up by them.
        self.codes = None if codes is None else numpy.asarray(codes, dtype=int)
        self.fixtures = fixtures
        self.preds = {}
        self.missing = {}

    @property
    def key(self):
        return (self.season, self.origin)

    @property
    def horizon(self):
        return int(self.actual.shape[1])

    @property
    def full_window(self):
        return self.valid_leads == self.horizon

    def target(self, lead):
        return self.origin + int(lead)


class SeasonFacts:
    """Deadline-known eligibility, fixtures and realised points for one season."""

    def __init__(self, source, previous=None):
        self.source = source
        self.events_by_team = defaultdict(set)
        for fixture in source.fixtures:
            event = fixture.get('event')
            if event in (None, ''):
                continue
            event = int(float(event))
            self.events_by_team[int(float(fixture['team_h']))].add(event)
            self.events_by_team[int(float(fixture['team_a']))].add(event)
        self.last_season_minutes = {}
        if previous is not None:
            for player_id, rows in previous.player_rows.items():
                raw = previous.players_raw.get(player_id)
                if raw is None or raw.get('code') in (None, ''):
                    continue
                code = int(float(raw['code']))
                minutes = sum(_float(row.get('minutes')) for row in rows)
                self.last_season_minutes[code] = self.last_season_minutes.get(code, 0.0) + minutes
        self._actuals = {}

    def actuals(self, gameweek):
        if gameweek not in self._actuals:
            self._actuals[gameweek] = self.source.actuals_for_gameweek(gameweek)
        return self._actuals[gameweek]

    def eligible(self, origin, recent_gws, last_season_min_minutes, exclude_builder_skipped=True):
        """{player_id: team_id at the deadline} for players eligible at `origin`.

        exclude_builder_skipped drops players the prediction builder does not
        predict at this deadline (VaastavSeasonData.predict_skips_at). That rule
        is deadline-known and the same for every model, so no model gets them as
        free zeros.
        """
        origin = int(origin)
        first_recent = origin - int(recent_gws)
        early = origin - 1 < int(recent_gws)
        registry = self.source.state_at_gameweek(origin)
        skipped = set()
        if exclude_builder_skipped and hasattr(self.source, 'predict_skips_at'):
            skipped = self.source.predict_skips_at(origin)
        chosen = {}
        for player_id in sorted(registry):
            raw = self.source.players_raw.get(player_id)
            if raw is None or player_id in skipped:
                continue
            team_id = self.source.team_id_from_snapshot(registry[player_id], raw)
            if team_id is None:
                continue
            played_recently = any(
                first_recent <= int(float(row.get('round') or 0)) < origin
                and _float(row.get('minutes')) > 0
                for row in self.source.player_rows.get(player_id, ())
            )
            qualifies = played_recently
            if not qualifies and early and raw.get('code') not in (None, ''):
                code = int(float(raw['code']))
                qualifies = self.last_season_minutes.get(code, 0.0) >= float(last_season_min_minutes)
            if qualifies:
                chosen[player_id] = int(team_id)
        return chosen

    def build_unit(self, origin, horizon, eligibility):
        eligible = self.eligible(
            origin, eligibility['recent_gws'], eligibility['last_season_min_minutes'],
            eligibility.get('exclude_builder_skipped', True),
        )
        player_ids = sorted(eligible)
        valid_leads = max(0, min(int(horizon), 38 - int(origin) + 1))
        has_fixture = numpy.zeros((len(player_ids), int(horizon)), dtype=bool)
        actual = numpy.zeros((len(player_ids), int(horizon)), dtype=float)
        minutes = numpy.zeros((len(player_ids), int(horizon)), dtype=float)
        fixture_count = numpy.zeros((len(player_ids), int(horizon)), dtype=int)
        fixtures = [[() for _ in range(int(horizon))] for _ in player_ids]
        codes = []
        for player_id in player_ids:
            raw = self.source.players_raw.get(player_id) or {}
            try:
                codes.append(int(float(raw.get('code'))))
            except (TypeError, ValueError):
                codes.append(0)
        for lead in range(valid_leads):
            gameweek = int(origin) + lead
            realised = self.actuals(gameweek)
            for row, player_id in enumerate(player_ids):
                result = realised.get(player_id)
                scheduled = gameweek in self.events_by_team.get(eligible[player_id], ())
                has_fixture[row, lead] = scheduled or bool(result and result.fixture_ids)
                actual[row, lead] = float(result.points) if result is not None else 0.0
                if result is not None:
                    minutes[row, lead] = float(result.minutes or 0.0)
                    fixture_count[row, lead] = len(result.fixture_ids or ())
                    fixtures[row][lead] = tuple(int(f) for f in (result.fixture_ids or ()))
        return Unit(
            self.source.season_start_year, origin, player_ids, has_fixture, actual, valid_leads,
            minutes=minutes, fixture_count=fixture_count, codes=codes, fixtures=fixtures,
        )


MAX_FIXTURES = 3


class FixtureParts:
    """Per-fixture pieces of a model's prediction, for calibrate.py.

    base: points before chance-to-play, one slot per fixture in the gameweek.
    multiplier: the flag applied after calibration (1 for v2 sets that learned it).
    flagged: the deadline flag was below 100.
    keep_flagged: flagged rows keep their raw prediction instead of being calibrated
    (v2 sets whose points already carry the flag).
    """

    def __init__(self, shape, keep_flagged):
        self.base = numpy.zeros(shape + (MAX_FIXTURES,), dtype=float)
        self.count = numpy.zeros(shape, dtype=int)
        self.multiplier = numpy.ones(shape, dtype=float)
        self.flagged = numpy.zeros(shape, dtype=bool)
        self.keep_flagged = bool(keep_flagged)


def _row_parts(row, scorer):
    """(points before chance, multiplier applied after) for one stored fixture row."""
    from squadtools.availability import chance_multiplier, unadjusted_points
    flag = chance_multiplier(row.get('chance_of_playing'))
    if scorer == SCORE_RAW_POINTS:
        return float(row.get('predicted_points') or 0.0), 1.0
    if scorer == SCORE_RAW_POINTS_CHANCE:
        return float(row.get('predicted_points') or 0.0), flag
    pp90 = row.get('predicted_pp90')
    if pp90 is None or pp90 == '':
        return float(row.get('predicted_points') or 0.0), 1.0
    return unadjusted_points(pp90, row.get('predicted_minutes')), flag


def attach_predictions(unit, name, rows, scorer):
    """Sum a model's stored fixture rows into the unit's player x lead grid."""
    index = {player_id: row for row, player_id in enumerate(unit.player_ids.tolist())}
    grid = numpy.zeros(unit.actual.shape, dtype=float)
    seen = numpy.zeros(unit.actual.shape, dtype=bool)
    parts = FixtureParts(unit.actual.shape, keep_flagged=scorer == SCORE_RAW_POINTS)
    for row in rows:
        lead = int(row['target_gameweek']) - unit.origin
        position = index.get(int(row['player_id']))
        if position is None or not (0 <= lead < unit.valid_leads):
            continue
        grid[position, lead] += row_points(row, scorer)
        seen[position, lead] = True
        base, multiplier = _row_parts(row, scorer)
        slot = parts.count[position, lead]
        if slot < MAX_FIXTURES:
            parts.base[position, lead, slot] = base
        parts.count[position, lead] += 1
        parts.multiplier[position, lead] = multiplier
        flag = row.get('chance_of_playing')
        parts.flagged[position, lead] = flag not in (None, '') and float(flag) < 100.0
    unit.preds[name] = grid
    if not hasattr(unit, 'parts'):
        unit.parts = {}
    unit.parts[name] = parts
    window = numpy.zeros(unit.actual.shape, dtype=bool)
    window[:, :unit.valid_leads] = True
    unit.missing[name] = int((unit.has_fixture & window & ~seen).sum())


def complete_origins(db, prediction_set_id, season):
    return {
        int(row['as_of_gameweek'])
        for row in db.prediction_coverage(prediction_set_id)
        if (
            int(row['season_start_year']) == int(season)
            and row['status'] == 'complete'
            and row['row_count'] > 0
            and row['stored_row_count'] == row['row_count']
        )
    }


def load_panel(db, sets, seasons, config, data_root=None, log=print):
    """Units for every (season, origin) that all prediction sets have complete.

    `sets` maps model name -> prediction set id. Origins missing from any set are
    dropped for all models, and reported, so every model sees the same rows.
    """
    horizon = int(config['horizon'])
    origins_range = range(int(config['origin_from']), int(config['origin_to']) + 1)
    scorers = {}
    for name, set_id in sets.items():
        prediction_set = db.get_prediction_set(set_id)
        if prediction_set is None:
            raise ValueError('Unknown prediction set %s for %s' % (set_id, name))
        scorers[name] = scorer_for_set(prediction_set)
        log('  %s: prediction set %s, scored as %s' % (name, set_id, scorers[name]))
    units = []
    dropped = []
    for season in seasons:
        try:
            previous = VaastavSeasonData(int(season) - 1, data_root)
        except (FileNotFoundError, ValueError):
            previous = None
            log('  no %s/%s data: early-season eligibility uses this season only'
                % (int(season) - 1, int(season)))
        facts = SeasonFacts(VaastavSeasonData(int(season), data_root), previous)
        available = set(origins_range) & set(facts.source.gameweeks)
        for name, set_id in sets.items():
            have = complete_origins(db, set_id, season)
            missing = sorted(available - have)
            if missing:
                dropped.append({'model': name, 'season': int(season), 'origins': missing})
                log('  %s is missing complete snapshots for %s GW %s; those origins are dropped for every model'
                    % (name, season, ','.join(str(gw) for gw in missing)))
            available &= have
        empty = []
        for origin in sorted(available):
            unit = facts.build_unit(origin, horizon, config['eligibility'])
            if not len(unit.player_ids):
                # No last-season data and no recent minutes yet (2023/24 GW1).
                empty.append(origin)
                continue
            for name, set_id in sets.items():
                rows = db.get_historical_predictions(
                    set_id, season, origin, max_target_gameweek=origin + horizon - 1,
                )
                attach_predictions(unit, name, rows, scorers[name])
            units.append(unit)
        if empty:
            log('  season %s GW %s: no eligible players, skipped'
                % (season, ','.join(str(gw) for gw in empty)))
            available -= set(empty)
        log('  season %s: %s origins, %s eligible players on average'
            % (season, len(available),
               round(numpy.mean([len(u.player_ids) for u in units if u.season == int(season)]), 1)
               if available else 0))
    units.sort(key=lambda unit: unit.key)
    return units, scorers, dropped
