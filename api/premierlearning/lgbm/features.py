"""Flatten a learning_v2 ExampleBatch into one table of columns for LightGBM.

The net reads lists: the player's last matches, his past seasons, both squads
and both teams' match histories. Trees need fixed columns, so each list becomes
summaries such as means over the last few items, faded means, per-90 rates and
counts. Every column comes from the batch the net would see, so both models get
the same information as of each deadline, with no second pipeline to leak.

Lists are right-aligned (newest last, padded in front), except the squads,
which are ranked by minutes from slot 0. NaN marks "unknown" (no matches yet),
which LightGBM handles natively.

Feature sets: 1 is the original (lgbm_v1 / v2). 2 adds more windows on every
history (short to long, so the trees can weigh recent form against long-run
level), trends (short mean minus long mean), same-venue form, fixture matchup
columns (his team's attack against the opponent's defence and the reverse) and
the player's share of his team's xG / xA.

3 (lgbm_v4) adds per-90 rates shrunk toward his position's average when he has
few minutes, competition for his place (same-position teammates' recent
minutes against his own and the usual number of starters), and, via
lgbm/extras.py (they need the seasons, not just the batch), team ratings and
what the opponent concedes to each position.

4 (lgbm_v6) adds FPL's own player news beyond the flat chance, via
lgbm/news.py: status, injury type, days until the expected return, how long
he has been flagged, the change since the last deadline and a 0-10 certainty.
"""
from dataclasses import dataclass, replace

import numpy

from premierlearning.learning_v2.plan import input_specs
from premierlearning.learning_v2.teams import TEAM_HISTORY_PATHS, TEAM_HISTORY_TABLE, ref_key

HISTORY = 'player_fixture_history'
SEASONS = 'last_season'
SQUADS = ('current_team.teammates', 'current_opposition.players')
SQUAD_STARTERS = 11
PERFORMANCE = 'player_performance.'
MINUTES = PERFORMANCE + 'minutes_played'
TEAM_STATS = ('goals_for', 'goals_against', 'xg_for', 'xg_against')
LAST_ONLY = ('match_facts.days_before_current', 'match_facts.days_since_previous_match', 'days_before_current')
HOME = 'match_facts.home_away'
VENUE_FIELDS = ('points_scored', 'minutes_played', 'xG', 'xA')
TREND_FIELDS = ('points_scored', 'minutes_played', 'xG', 'xA')
TREND_PAIRS = ((3, 10), (5, 50))
TEAM_TREND = (5, 38)
MATCHUP_WINDOWS = ('mean5', 'mean10', 'mean38', 'faded10')
# Shrunk per-90s: his rate counts as if he also played SHRINK_MINUTES at his
# position's average rate, so 200 hot minutes no longer read as an elite rate.
SHRINK_MINUTES = 270.0
SHRINK_FIELDS = ('points_scored', 'xG', 'xA', 'goals', 'assists', 'defcon', 'xGC')
PRIOR_WINDOW = 50
SQUAD_MINUTES = 'minutes_last_n'
SQUAD_POSITION = 'position'
# Usual starters per position (GK, DEF, MID, FWD); unknown positions use 2.
STARTER_SLOTS = {1: 1, 2: 4, 3: 4, 4: 2}
POSITION_COLUMN = 'position'


@dataclass(frozen=True)
class FeatureSettings:
    history_windows: tuple      # means over the player's last n matches
    history_half_lives: tuple   # faded means, in matches
    rate_windows: tuple         # per-90 rates
    share_windows: tuple        # appearance and 60+ shares
    other_windows: tuple        # non-performance history fields (pre-match team form etc.)
    team_windows: tuple         # team / opponent match histories
    team_half_lives: tuple
    venue_window: int = 0       # same-venue means over this many matches; 0 = none
    extras: bool = False        # trends, fixture matchups, xG shares
    shrink_windows: tuple = ()  # shrunk per-90 rates over these windows
    place: bool = False         # competition for his place
    season_columns: bool = False  # team ratings and conceded-by-position (lgbm/extras.py)
    news: bool = False          # FPL's player news: status, reason, return date, certainty (lgbm/news.py)


FEATURE_SETS = {
    1: FeatureSettings(
        history_windows=(1, 3, 5, 10, 50), history_half_lives=(3.0, 10.0), rate_windows=(5, 10, 50),
        share_windows=(5, 10), other_windows=(5,), team_windows=(5, 10, 38), team_half_lives=(10.0,),
    ),
    2: FeatureSettings(
        history_windows=(1, 2, 3, 5, 10, 20, 50), history_half_lives=(1.5, 3.0, 10.0, 25.0),
        rate_windows=(3, 5, 10, 20, 50), share_windows=(3, 5, 10, 20, 50), other_windows=(1, 3, 5, 10),
        team_windows=(1, 3, 5, 10, 19, 38, 76, 114), team_half_lives=(3.0, 10.0, 38.0),
        venue_window=10, extras=True,
    ),
}
FEATURE_SETS[3] = replace(FEATURE_SETS[2], shrink_windows=(5, 10, 20, 50), place=True, season_columns=True)
FEATURE_SETS[4] = replace(FEATURE_SETS[3], news=True)
DEFAULT_FEATURES = 1


@dataclass
class FeatureTable:
    names: list
    values: numpy.ndarray       # float32 [rows, columns]; NaN is unknown
    categorical: list           # names of the categorical columns
    priors: dict = None         # per-minute position rates the shrunk per-90s used

    def column(self, name):
        return self.values[:, self.names.index(name)]

    def with_columns(self, extra):
        """A copy with {name: values} appended, in the given order."""
        if not extra:
            return self
        clash = [name for name in extra if name in self.names]
        if clash:
            raise ValueError('duplicate columns %s' % clash)
        added = numpy.column_stack([numpy.asarray(values, dtype=numpy.float32) for values in extra.values()])
        return FeatureTable(list(self.names) + list(extra), numpy.hstack([self.values, added]),
                            list(self.categorical), self.priors)


def _split(path):
    for prefix in (HISTORY, SEASONS) + SQUADS + TEAM_HISTORY_PATHS:
        if path.startswith(prefix + '.'):
            return prefix, path[len(prefix) + 1:]
    return None, path


def _divide(total, count):
    out = numpy.full(total.shape, numpy.nan, dtype=numpy.float64)
    has = count > 0
    out[has] = total[has] / count[has]
    return out


def _mean(values, mask, weights=None):
    mask = mask if weights is None else mask * weights
    return _divide((values * mask).sum(axis=1), mask.sum(axis=1))


def _faded(values, mask, half_life):
    """Newest item weight 1, halving every half_life items back."""
    ages = numpy.arange(values.shape[1] - 1, -1, -1, dtype=numpy.float64)
    return _mean(values, mask, 0.5 ** (ages / half_life))


def _last(values, mask, back=0):
    """The newest real item (back=1: the one before), NaN when there is none."""
    index = values.shape[1] - 1 - back
    if index < 0:
        return numpy.full(values.shape[0], numpy.nan)
    out = values[:, index].astype(numpy.float64)
    out[mask[:, index] <= 0] = numpy.nan
    return out


def _since(condition, mask):
    """Real items since the newest one meeting the condition (0 = the newest); NaN if none."""
    hit = (condition & (mask > 0))[:, ::-1]
    out = hit.argmax(axis=1).astype(numpy.float64)
    out[~hit.any(axis=1)] = numpy.nan
    return out


def _newest(selected, count):
    """Keep only the newest `count` selected items of each row (lists are newest last)."""
    flipped = selected[:, ::-1]
    keep = flipped * (numpy.cumsum(flipped, axis=1) <= count)
    return keep[:, ::-1]


def _ratio(top, bottom):
    out = numpy.full(top.shape, numpy.nan, dtype=numpy.float64)
    ok = numpy.isfinite(top) & numpy.isfinite(bottom) & (bottom > 1e-6)
    out[ok] = top[ok] / bottom[ok]
    return out


class _Columns:
    def __init__(self, rows):
        self.rows = rows
        self.names = []
        self.values = []
        self.categorical = []
        self.index = {}

    def add(self, name, values, categorical=False):
        values = numpy.asarray(values, dtype=numpy.float32).reshape(-1)
        if values.shape[0] != self.rows:
            raise ValueError('%s has %s rows, expected %s' % (name, values.shape[0], self.rows))
        if name in self.index:
            raise ValueError('duplicate column %s' % name)
        self.index[name] = len(self.names)
        self.names.append(name)
        self.values.append(values)
        if categorical:
            self.categorical.append(name)

    def get(self, name):
        """A column as float64, or None when the plan does not give it."""
        position = self.index.get(name)
        return None if position is None else self.values[position].astype(numpy.float64)

    def table(self, priors=None):
        values = numpy.empty((self.rows, len(self.values)), dtype=numpy.float32)
        for index, column in enumerate(self.values):
            values[:, index] = column
        return FeatureTable(list(self.names), values, list(self.categorical), priors)


def _row_positions(columns):
    positions = columns.get(POSITION_COLUMN)
    if positions is None:
        return None
    positions = numpy.nan_to_num(positions).astype(numpy.int64)
    positions[(positions < 0) | (positions > 4)] = 0
    return positions


def per90_priors(fields, mask, positions):
    """{stat: per-minute rate for positions 0 (everyone) to 4} over the rows' last PRIOR_WINDOW matches."""
    minutes = fields[MINUTES][:, -PRIOR_WINDOW:] * mask[:, -PRIOR_WINDOW:]
    priors = {}
    for short in SHRINK_FIELDS:
        values = fields.get(PERFORMANCE + short)
        if values is None:
            continue
        done = (values[:, -PRIOR_WINDOW:] * mask[:, -PRIOR_WINDOW:]).sum(axis=1)
        spent = minutes.sum(axis=1)
        overall = float(done.sum() / max(spent.sum(), 1.0))
        rates = [overall]
        for position in range(1, 5):
            rows = positions == position
            total = spent[rows].sum()
            rates.append(float(done[rows].sum() / total) if total > 0 else overall)
        priors[short] = rates
    return priors


def _shrunk_rates(columns, fields, mask, settings, priors):
    """Per-90 rates shrunk toward the position's rate; returns the priors used."""
    positions = _row_positions(columns)
    minutes = fields.get(MINUTES)
    if positions is None or minutes is None:
        return priors
    if priors is None:
        priors = per90_priors(fields, mask, positions)
    for short in SHRINK_FIELDS:
        values = fields.get(PERFORMANCE + short)
        if values is None or short not in priors:
            continue
        prior = numpy.asarray(priors[short], dtype=numpy.float64)[positions]
        for window in settings.shrink_windows:
            spent = (minutes[:, -window:] * mask[:, -window:]).sum(axis=1)
            done = (values[:, -window:] * mask[:, -window:]).sum(axis=1)
            columns.add('hist.%s.shrunk90_%s' % (short, window),
                        90.0 * (done + SHRINK_MINUTES * prior) / (spent + SHRINK_MINUTES))
    return priors


def _place_columns(columns, squad, squad_ints, mask):
    """Competition for his place: same-position teammates' recent minutes against his own.

    The teammates list leaves the player out. Minutes are means over the last
    few matches (team_form_n), compared with his own mean over his last 5.
    """
    positions = _row_positions(columns)
    own = columns.get('hist.minutes_played.mean5')
    rival_minutes = squad.get(SQUAD_MINUTES)
    rival_positions = squad_ints.get(SQUAD_POSITION)
    if positions is None or own is None or rival_minutes is None or rival_positions is None:
        return
    same = (rival_positions == positions[:, None]) & (mask > 0) & (positions[:, None] > 0)
    minutes = numpy.where(same, rival_minutes, -numpy.inf)
    ordered = -numpy.sort(-minutes, axis=1)
    slots = numpy.array([STARTER_SLOTS.get(int(position), 2) for position in positions])
    rivals = same.sum(axis=1).astype(numpy.float64)
    slot_minutes = ordered[numpy.arange(len(slots)), numpy.minimum(slots, ordered.shape[1]) - 1]
    slot_minutes = numpy.where(rivals >= slots, slot_minutes, 0.0)
    top = numpy.where(rivals > 0, ordered[:, 0], 0.0)
    ahead = (same & (rival_minutes > numpy.nan_to_num(own, nan=-1.0)[:, None])).sum(axis=1).astype(numpy.float64)
    ahead[~numpy.isfinite(own)] = numpy.nan
    columns.add('place.rivals', rivals)
    columns.add('place.rivals_ahead', ahead)
    columns.add('place.ahead_minus_slots', ahead - slots)
    columns.add('place.top_rival_minutes', top)
    columns.add('place.slot_rival_minutes', slot_minutes)
    columns.add('place.own_minus_slot', own - slot_minutes)
    columns.add('place.rival_minutes_sum', numpy.where(same, rival_minutes, 0.0).sum(axis=1))


def _player_history(columns, fields, mask, settings, home=None, venues=None):
    columns.add('hist.count', mask.sum(axis=1))
    for name, values in fields.items():
        if name in LAST_ONLY:
            columns.add('hist.%s.last' % name, _last(values, mask))
            continue
        if not name.startswith(PERFORMANCE):
            for window in settings.other_windows:
                columns.add('hist.%s.mean%s' % (name, window), _mean(values[:, -window:], mask[:, -window:]))
            continue
        short = name[len(PERFORMANCE):]
        for window in settings.history_windows:
            columns.add('hist.%s.mean%s' % (short, window), _mean(values[:, -window:], mask[:, -window:]))
        for half_life in settings.history_half_lives:
            columns.add('hist.%s.faded%g' % (short, half_life), _faded(values, mask, half_life))
        if settings.extras and short in TREND_FIELDS:
            for near, far in TREND_PAIRS:
                columns.add('hist.%s.trend%s_%s' % (short, near, far),
                            _mean(values[:, -near:], mask[:, -near:]) - _mean(values[:, -far:], mask[:, -far:]))
        if settings.venue_window and home is not None and venues is not None and short in VENUE_FIELDS:
            same = _newest(mask * (venues == home[:, None]), settings.venue_window)
            columns.add('hist.%s.venue%s' % (short, settings.venue_window), _mean(values, same))
    minutes = fields.get(MINUTES)
    if minutes is None:
        return
    for name, values in fields.items():
        if not name.startswith(PERFORMANCE) or name == MINUTES:
            continue
        short = name[len(PERFORMANCE):]
        for window in settings.rate_windows:
            spent = (minutes[:, -window:] * mask[:, -window:]).sum(axis=1)
            done = (values[:, -window:] * mask[:, -window:]).sum(axis=1)
            rate = _divide(done * 90.0, numpy.where(spent >= 1.0, spent, 0.0))
            columns.add('hist.%s.per90_%s' % (short, window), rate)
    for window in settings.share_windows:
        recent = mask[:, -window:]
        played = (minutes[:, -window:] > 0).astype(numpy.float64)
        started = (minutes[:, -window:] >= 60).astype(numpy.float64)
        columns.add('hist.played_share%s' % window, _mean(played, recent))
        columns.add('hist.sixty_share%s' % window, _mean(started, recent))
    columns.add('hist.since_played', _since(minutes > 0, mask))
    columns.add('hist.since_sixty', _since(minutes >= 60, mask))


def _seasons(columns, fields, mask):
    columns.add('seasons.count', mask.sum(axis=1))
    for name, values in fields.items():
        columns.add('seasons.%s.last' % name, _last(values, mask))
        columns.add('seasons.%s.before' % name, _last(values, mask, back=1))


def _squad(columns, side, fields, mask):
    label = 'team' if side == SQUADS[0] else 'opp'
    columns.add('%s_squad.count' % label, mask.sum(axis=1))
    starters = numpy.zeros_like(mask)
    starters[:, :SQUAD_STARTERS] = 1.0
    for name, values in fields.items():
        real = values * mask
        columns.add('%s_squad.%s.sum' % (label, name), real.sum(axis=1))
        columns.add('%s_squad.%s.mean' % (label, name), _mean(values, mask))
        columns.add('%s_squad.%s.top11' % (label, name), (real * starters).sum(axis=1))
        peak = numpy.where(mask > 0, values, -numpy.inf).max(axis=1).astype(numpy.float64)
        peak[~numpy.isfinite(peak)] = numpy.nan
        columns.add('%s_squad.%s.max' % (label, name), peak)


def _team_history(columns, side, fields, batch, settings, home=None):
    """Summaries per stored history, then looked up by each row's reference."""
    label = 'team' if side == TEAM_HISTORY_PATHS[0] else 'opp'
    table = (getattr(batch, 'tables', None) or {}).get(TEAM_HISTORY_TABLE)
    refs = batch.arrays.get(ref_key(side))
    if table is None or refs is None:
        raise ValueError('batch has no team-history table for %s' % side)
    refs = numpy.asarray(refs, dtype=numpy.int64).reshape(-1)
    mask = table['mask'].astype(numpy.float64)
    xg_mask = mask * table['xg_known'] if 'xg_known' in table else mask
    venue_rows = None
    if settings.venue_window and home is not None and 'home_away' in table:
        # This side's venue in the row's fixture: the opponent's is the reverse.
        venue_rows = home if label == 'team' else 1.0 - home
        venues = table['home_away'].astype(numpy.float64)
    columns.add('%s_hist.count' % label, mask.sum(axis=1)[refs])
    for name in fields:
        if name not in table:
            continue
        values = table[name].astype(numpy.float64)
        if name in LAST_ONLY:
            columns.add('%s_hist.%s.last' % (label, name), _last(values, mask)[refs])
            continue
        if name not in TEAM_STATS:
            continue
        use = xg_mask if name.startswith('xg_') else mask
        for window in settings.team_windows:
            columns.add('%s_hist.%s.mean%s' % (label, name, window), _mean(values[:, -window:], use[:, -window:])[refs])
        for half_life in settings.team_half_lives:
            columns.add('%s_hist.%s.faded%g' % (label, name, half_life), _faded(values, use, half_life)[refs])
        if settings.extras:
            near, far = TEAM_TREND
            trend = _mean(values[:, -near:], use[:, -near:]) - _mean(values[:, -far:], use[:, -far:])
            columns.add('%s_hist.%s.trend%s_%s' % (label, name, near, far), trend[refs])
        if venue_rows is not None:
            at_home = _mean(values, _newest(use * (venues == 1.0), settings.venue_window))[refs]
            away = _mean(values, _newest(use * (venues == 0.0), settings.venue_window))[refs]
            columns.add('%s_hist.%s.venue%s' % (label, name, settings.venue_window),
                        numpy.where(venue_rows >= 0.5, at_home, away))


def _fixture_columns(columns):
    """His team's attack against the opponent's defence, and the reverse, in one column each.

    Trees add two columns poorly, so the matchup that sets a fixture's
    difficulty is spelled out, as is how this fixture compares to the team's
    usual level (ratio above 1: an easier fixture than usual for the attack).
    """
    for stat in ('goals', 'xg'):
        for window in MATCHUP_WINDOWS:
            parts = {key: columns.get('%s_hist.%s_%s.%s' % (side, stat, kind, window))
                     for key, (side, kind) in {'tf': ('team', 'for'), 'ta': ('team', 'against'),
                                               'of': ('opp', 'for'), 'oa': ('opp', 'against')}.items()}
            if any(part is None for part in parts.values()):
                continue
            columns.add('fixture.attack_%s.%s' % (stat, window), (parts['tf'] + parts['oa']) / 2.0)
            columns.add('fixture.defence_%s.%s' % (stat, window), (parts['ta'] + parts['of']) / 2.0)
        attack = columns.get('fixture.attack_%s.mean10' % stat)
        defence = columns.get('fixture.defence_%s.mean10' % stat)
        if attack is None or defence is None:
            continue
        attack_ratio = _ratio(attack, columns.get('team_hist.%s_for.mean10' % stat))
        defence_ratio = _ratio(defence, columns.get('team_hist.%s_against.mean10' % stat))
        columns.add('fixture.attack_%s.ratio10' % stat, attack_ratio)
        columns.add('fixture.defence_%s.ratio10' % stat, defence_ratio)
        xg, xa = columns.get('hist.xG.per90_10'), columns.get('hist.xA.per90_10')
        if xg is not None and xa is not None:
            columns.add('fixture.player_xgi90_x_attack_%s' % stat, (xg + xa) * attack_ratio)
        points = columns.get('hist.points_scored.mean10')
        if points is not None:
            columns.add('fixture.player_points_x_attack_%s' % stat, points * attack_ratio)
            columns.add('fixture.player_points_x_defence_%s' % stat, points * defence_ratio)
    team_xg = columns.get('team_hist.xg_for.mean10')
    for short in ('xG', 'xA'):
        mine = columns.get('hist.%s.mean10' % short)
        if mine is not None and team_xg is not None:
            columns.add('hist.%s.share10' % short, _ratio(mine, team_xg))
    # Every plan also carries team form scalars (last n, and in some decayed).
    for label, scored, conceded in (('goals.last_n', 'goals_last_n', 'goals_conceded_last_n'),
                                    ('goals.decay', 'goals_for_decay', 'goals_against_decay'),
                                    ('xg.decay', 'xg_for_decay', 'xg_against_decay')):
        parts = {key: columns.get('%s.team_features.%s' % (side, name)) for key, side, name in (
            ('tf', 'current_team', scored), ('ta', 'current_team', conceded),
            ('of', 'current_opposition', scored), ('oa', 'current_opposition', conceded))}
        if any(part is None for part in parts.values()):
            continue
        stat, window = label.split('.')
        columns.add('fixture.attack_%s.%s' % (stat, window), (parts['tf'] + parts['oa']) / 2.0)
        columns.add('fixture.defence_%s.%s' % (stat, window), (parts['ta'] + parts['of']) / 2.0)


def flatten(batch, plan, features=DEFAULT_FEATURES, ids=True, priors=None) -> FeatureTable:
    """One float32 row per batch row. Column order depends only on the plan and options.

    features: a FEATURE_SETS key. ids: False leaves out the team and opponent
    id columns (position stays), so the trees must judge teams by their form.
    priors: the position rates for shrunk per-90s (set 3): None works them out
    from this batch (training), and a fitted model passes its training ones back
    for predictions. Season columns (set 3) are added by lgbm/extras.py, not here.
    """
    if features not in FEATURE_SETS:
        raise ValueError('features must be one of %s' % sorted(FEATURE_SETS))
    settings = FEATURE_SETS[features]
    rows = len(batch.matches)
    columns = _Columns(rows)
    lists = {}
    int_lists = {}
    masks = {}
    team_sides = {}
    home = None
    for spec in input_specs(plan):
        group, rest = _split(spec.path)
        if group in TEAM_HISTORY_PATHS:
            if rest != 'mask' and spec.dtype != 'int32':
                team_sides.setdefault(group, []).append(rest)
            continue
        if spec.key not in batch.arrays:
            raise KeyError('batch is missing %s' % spec.path)
        values = batch.arrays[spec.key]
        if group is None:
            if spec.path.endswith('player_id_embedding'):
                continue
            if spec.path == HOME:
                home = values.reshape(rows, -1)[:, 0].astype(numpy.float64)
            is_team_id = spec.path.endswith('team_id_embedding')
            if is_team_id and not ids:
                continue
            categorical = is_team_id or spec.path == 'position'
            columns.add(spec.path, values.reshape(rows, -1)[:, 0], categorical=categorical)
            continue
        if rest == 'mask':
            masks[group] = values.astype(numpy.float64)
        elif spec.dtype != 'int32':
            lists.setdefault(group, {})[rest] = values.astype(numpy.float64)
        else:
            int_lists.setdefault(group, {})[rest] = values.astype(numpy.float64)
    if HISTORY in masks:
        venues = lists.get(HISTORY, {}).get(HOME)
        if venues is None:
            venues = int_lists.get(HISTORY, {}).get(HOME)
        _player_history(columns, lists.get(HISTORY, {}), masks[HISTORY], settings, home, venues)
        if settings.shrink_windows:
            priors = _shrunk_rates(columns, lists.get(HISTORY, {}), masks[HISTORY], settings, priors)
    if SEASONS in masks:
        _seasons(columns, lists.get(SEASONS, {}), masks[SEASONS])
    for side in SQUADS:
        if side in masks:
            _squad(columns, side, lists.get(side, {}), masks[side])
    if settings.place and SQUADS[0] in masks:
        _place_columns(columns, lists.get(SQUADS[0], {}), int_lists.get(SQUADS[0], {}), masks[SQUADS[0]])
    for side in TEAM_HISTORY_PATHS:
        if side in team_sides:
            _team_history(columns, side, team_sides[side], batch, settings, home)
    if settings.extras:
        _fixture_columns(columns)
    return columns.table(priors if settings.shrink_windows else None)
