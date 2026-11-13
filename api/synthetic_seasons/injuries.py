"""Move, remove and add injury spells, and give the minutes to the right teammates.

A spell is a run of team matches a player missed while normally a starter,
with an FPL injury or doubt flag ('i' or 'd') at one of those deadlines.
Suspensions, loans and 'not in squad' runs stay as they happened.

In a copy each real spell is kept with probability KEEP and moved to a random
point in the player's time at the club; otherwise it is removed. For every
removed spell a new one is added to a random regular of the same position
(a starter when fit: injury_targets),
with the length and FPL flag pattern of a random real spell of that position,
so the number of injuries per position stays the same on average.

Minutes move by swapping a player's whole match line with a teammate's:
- out in a match he really played: an unused teammate (a "deputy": someone
  whose minutes went up when he was really out, else a same-position backup)
  takes his minutes, cards, defensive numbers and the rest of his BPS;
- playing in a match he really missed: the teammate who covered for him that
  week hands the line back.
So every team-match keeps its real minutes, one goalkeeper and its subs.
Goals, assists, clean sheets, saves, bonus and points are redrawn afterwards.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np

KEEP = 0.5
INJURY_STATUSES = ('i', 'd')
UNFIT = ('i', 's', 'u', 'n')
NEIGHBOURS = 4
REGULAR_START_SHARE = 0.5
REGULAR_MIN_STARTS = 10
TARGET_MIN_STARTS = 5        # starts when fit, to be given a new injury spell
PLACEMENT_TRIES = 60
FLAG_LOOKBACK = 3
FLAG_LOOKAHEAD = 2
# Fields that belong to a match line rather than to the player, so they move
# with the minutes in a swap. Goals and the rest are redrawn later.
LINE_FIELDS = (
    'minutes', 'starts', '_started', 'yellow_cards', 'red_cards', 'penalties_saved',
    'penalties_missed', 'defensive_contribution', 'clearances_blocks_interceptions',
    'tackles', 'recoveries', 'saves', 'influence', 'creativity', 'threat', 'ict_index',
    '_bps_rest', '_points_rest', '_saves_real',
)
AVAILABLE = {
    'status': 'a', 'news': '', 'news_added': None,
    'chance_of_playing_next_round': None, 'chance_of_playing_this_round': None,
}
DATE_IN_NEWS = re.compile(r'((?:expected back|suspended until)\s+)(\d{1,2})\s+([A-Za-z]{3})', re.IGNORECASE)
MONTHS = ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec')


@dataclass
class Spell:
    element: int
    team: int
    position: int
    start: int                      # index into the player's match list at the club
    length: int
    gameweeks: tuple                # gameweeks of the missed matches
    flags: Dict[int, dict] = field(default_factory=dict)   # gw -> FPL news entry
    match_gws: tuple = ()           # gameweek of each missed match, in order

    @property
    def first_gw(self) -> int:
        return min(self.gameweeks)


def is_flagged(entry: dict) -> bool:
    if not entry:
        return False
    if (entry.get('status') or 'a') != 'a':
        return True
    chance = entry.get('chance_of_playing_next_round')
    return chance not in (None, '') and float(chance) < 100


class Squads:
    """Index of a season's match lines by player-at-club and by team-match."""

    def __init__(self, rows: List[dict], source):
        self.source = source
        self.by_key = {}                       # (fixture, element) -> row
        self.lines = defaultdict(list)         # (element, team) -> rows in kickoff order
        self.team_match = defaultdict(list)    # (fixture, team) -> rows
        for row in rows:
            self.by_key[(row['fixture'], row['element'])] = row
            self.lines[(row['element'], row['_team'])].append(row)
            self.team_match[(row['fixture'], row['_team'])].append(row)
        self.season_minutes = defaultdict(int)
        self.starts = defaultdict(int)
        for (element, team), line in self.lines.items():
            self.season_minutes[(element, team)] = sum(r['minutes'] for r in line)
            self.starts[(element, team)] = sum(1 for r in line if r['_started'])

    def is_regular(self, element, team) -> bool:
        line = self.lines[(element, team)]
        starts = self.starts[(element, team)]
        return starts >= REGULAR_MIN_STARTS and starts >= REGULAR_START_SHARE * len(line)


def find_spells(squads: Squads) -> List[Spell]:
    source = squads.source
    spells = []
    for (element, team), line in squads.lines.items():
        if len(line) < 10:
            continue
        index = 0
        while index < len(line):
            if line[index]['minutes'] > 0:
                index += 1
                continue
            end = index
            while end + 1 < len(line) and line[end + 1]['minutes'] == 0:
                end += 1
            spell = _spell_from_run(source, element, team, line, index, end)
            if spell is not None:
                spells.append(spell)
            index = end + 1
    return spells


def _spell_from_run(source, element, team, line, first, last) -> Optional[Spell]:
    gameweeks = tuple(sorted({line[i]['_gw'] for i in range(first, last + 1)}))
    statuses = [(source.flag(gw, element).get('status') or 'a') for gw in gameweeks]
    if any(status in ('s', 'u', 'n') for status in statuses):
        return None
    if not any(status in INJURY_STATUSES for status in statuses):
        return None
    around = line[max(0, first - NEIGHBOURS):first] + line[last + 1:last + 1 + NEIGHBOURS]
    if not around or sum(1 for row in around if row['_started']) < REGULAR_START_SHARE * len(around):
        return None
    flag_gws = set(gameweeks)
    gw = min(gameweeks) - 1
    while gw >= max(1, min(gameweeks) - FLAG_LOOKBACK) and is_flagged(source.flag(gw, element)):
        flag_gws.add(gw)
        gw -= 1
    gw = max(gameweeks) + 1
    while gw <= min(38, max(gameweeks) + FLAG_LOOKAHEAD) and is_flagged(source.flag(gw, element)):
        flag_gws.add(gw)
        gw += 1
    flags = {gw: source.flag(gw, element) for gw in sorted(flag_gws) if source.flag(gw, element)}
    match_gws = tuple(line[i]['_gw'] for i in range(first, last + 1))
    return Spell(element, team, line[first]['_position'], first, last - first + 1, gameweeks, flags, match_gws)


def deputy_weights(squads: Squads, element: int, team: int) -> Dict[int, float]:
    """Teammates whose minutes went up when this player was really out."""
    line = squads.lines[(element, team)]
    absent = {row['fixture'] for row in line if row['minutes'] == 0}
    present = {row['fixture'] for row in line if row['_started']}
    if not absent or len(present) < 3:
        return {}
    position = line[0]['_position']
    weights = {}
    for (other, other_team), other_line in squads.lines.items():
        if other_team != team or other == element:
            continue
        other_position = other_line[0]['_position']
        if (position == 1) != (other_position == 1):
            continue
        out_minutes = [r['minutes'] for r in other_line if r['fixture'] in absent]
        in_minutes = [r['minutes'] for r in other_line if r['fixture'] in present]
        if not out_minutes or len(in_minutes) < 3:
            continue
        delta = sum(out_minutes) / len(out_minutes) - sum(in_minutes) / len(in_minutes)
        if delta > 10:
            weights[other] = delta * (1.0 if other_position == position else 0.3)
    return weights


@dataclass
class InjuryPlan:
    removed: List[Spell]
    moved: List[tuple]          # (real spell, new start index)
    added: List[tuple]          # (template spell, target element, team, start index)
    out: set                    # (fixture, element) out in the copy
    back: set                   # (fixture, element) freed from a real spell


def plan_injuries(squads: Squads, spells: List[Spell], rng, proneness: Dict[int, float] = None) -> InjuryPlan:
    """proneness: element -> weight for receiving a new spell (default equal)."""
    occupied = defaultdict(set)     # (element, team) -> indices not free for a new spell
    for (element, team), line in squads.lines.items():
        for index, row in enumerate(line):
            status = squads.source.flag(row['_gw'], element).get('status') or 'a'
            if row['minutes'] == 0 and status in ('s', 'u', 'n'):
                occupied[(element, team)].add(index)
    keep = [spell for spell in spells if rng.random() < KEEP]
    kept_ids = {id(spell) for spell in keep}
    removed = [spell for spell in spells if id(spell) not in kept_ids]
    moved, added = [], []
    for spell in keep:
        key = (spell.element, spell.team)
        start = _place(rng, len(squads.lines[key]), spell.length, occupied[key])
        if start is None:
            start = spell.start
        occupied[key].update(range(start - 1, start + spell.length + 1))
        moved.append((spell, start))
    by_position = defaultdict(list)
    for spell in spells:
        by_position[spell.position].append(spell)
    regulars = injury_targets(squads, spells)
    for spell in removed:
        templates, targets = by_position[spell.position], regulars[spell.position]
        if not templates or not targets:
            continue
        template = templates[int(rng.integers(len(templates)))]
        weights = np.array([(proneness or {}).get(element, 1.0) for element, _team in targets], dtype=float)
        weights /= weights.sum()
        for _try in range(10):
            key = targets[int(rng.choice(len(targets), p=weights))]
            start = _place(rng, len(squads.lines[key]), template.length, occupied[key])
            if start is not None:
                occupied[key].update(range(start - 1, start + template.length + 1))
                added.append((template, key[0], key[1], start))
                break
    out, back = set(), set()
    for spell in spells:
        for row in squads.lines[(spell.element, spell.team)][spell.start:spell.start + spell.length]:
            back.add((row['fixture'], spell.element))
    for spell, start in moved:
        for row in squads.lines[(spell.element, spell.team)][start:start + spell.length]:
            out.add((row['fixture'], spell.element))
    for template, element, team, start in added:
        for row in squads.lines[(element, team)][start:start + template.length]:
            out.add((row['fixture'], element))
    return InjuryPlan(removed, moved, added, out, back - out)


def injury_targets(squads: Squads, spells: List[Spell]) -> Dict[int, List[tuple]]:
    """position -> players at a club who can get a new spell: regular starters when fit.

    Judged on the matches outside the player's real injury spells. On the
    whole season (Squads.is_regular) most real injury victims are not regulars,
    because the injury cut their starts, so new spells all went to the players
    who stayed fit and regulars lost about twice their real injured matches.
    """
    injured = defaultdict(set)
    for spell in spells:
        injured[(spell.element, spell.team)].update(range(spell.start, spell.start + spell.length))
    targets = defaultdict(list)
    for (element, team), line in squads.lines.items():
        fit = [row for index, row in enumerate(line) if index not in injured[(element, team)]]
        starts = sum(1 for row in fit if row['_started'])
        if starts >= TARGET_MIN_STARTS and starts >= REGULAR_START_SHARE * len(fit):
            targets[line[0]['_position']].append((element, team))
    return targets


def _place(rng, line_length, length, occupied) -> Optional[int]:
    if length >= line_length:
        return None
    for _try in range(PLACEMENT_TRIES):
        start = int(rng.integers(0, line_length - length + 1))
        if not any(index in occupied for index in range(start, start + length)):
            return start
    return None


def apply_minutes(squads: Squads, plan: InjuryPlan, rng) -> dict:
    """Swap match lines so the copy's injuries hold. Returns counts for the log."""
    counts = {'out_swaps': 0, 'back_swaps': 0, 'out_no_deputy': 0, 'back_no_cover': 0}
    deputies = {}
    touched = set()
    changes = [(key, 'out') for key in plan.out] + [(key, 'back') for key in plan.back]
    changes.sort(key=lambda item: (item[0][0], item[1], item[0][1]))
    for (fixture, element), kind in changes:
        row = squads.by_key[(fixture, element)]
        if kind == 'out' and row['minutes'] == 0:
            continue
        if kind == 'back' and row['minutes'] > 0:
            continue
        team = row['_team']
        if (element, team) not in deputies:
            deputies[(element, team)] = deputy_weights(squads, element, team)
        weights = deputies[(element, team)]
        partner = _partner(squads, row, kind, weights, plan, touched, rng)
        if partner is None:
            counts['out_no_deputy' if kind == 'out' else 'back_no_cover'] += 1
            continue
        _swap(row, partner)
        touched.update({(fixture, element), (fixture, partner['element'])})
        counts['out_swaps' if kind == 'out' else 'back_swaps'] += 1
    return counts


def _partner(squads, row, kind, weights, plan, touched, rng):
    fixture, team, position = row['fixture'], row['_team'], row['_position']
    options, values = [], []
    for other in squads.team_match[(fixture, team)]:
        key = (fixture, other['element'])
        if other is row or key in touched or key in plan.out or key in plan.back:
            continue
        other_position = other['_position']
        if (position == 1) != (other_position == 1):
            continue
        if kind == 'out':
            if other['minutes'] > 0:
                continue
            status = squads.source.flag(other['_gw'], other['element']).get('status') or 'a'
            if status in UNFIT:
                continue
            base = weights.get(other['element'])
            if base is None:
                # Squad players by how much they play; someone who never
                # played (often the backup keeper) only when nobody else fits.
                minutes = squads.season_minutes[(other['element'], team)]
                base = 0.05 * (minutes / 90.0 + 0.5) if minutes > 0 else 1e-4
        else:
            if other['minutes'] <= 0:
                continue
            base = weights.get(other['element'])
            if base is None:
                share = squads.starts[(other['element'], team)] / max(1, len(squads.lines[(other['element'], team)]))
                base = 0.05 * other['minutes'] * (1.05 - share)
            else:
                # Prefer the deputy who started: a returning regular takes a
                # full line, not a substitute's half hour.
                base *= other['minutes'] / 90.0
            # The minutes come back from whoever covered, normally a squad
            # player; a regular starter only gives way when nobody else fits.
            if squads.is_regular(other['element'], team):
                base *= 0.02
        if other_position != position:
            base *= 0.15
        if base > 0:
            options.append(other)
            values.append(base)
    if not options:
        return None
    total = sum(values)
    return options[int(rng.choice(len(options), p=[value / total for value in values]))]


def _swap(row, partner):
    for name in LINE_FIELDS:
        if name in row or name in partner:
            row[name], partner[name] = partner.get(name, 0), row.get(name, 0)


def flag_overrides(squads: Squads, plan: InjuryPlan) -> Dict[int, Dict[int, dict]]:
    """{gw: {element: FPL news entry}} replacing the real flags in the copy."""
    source = squads.source
    overrides = defaultdict(dict)
    for spell in plan.removed + [spell for spell, _start in plan.moved]:
        for gw in spell.flags:
            overrides[gw][spell.element] = dict(AVAILABLE)
    placements = [(spell, spell.element, spell.team, start) for spell, start in plan.moved] + list(plan.added)
    for template, element, team, start in placements:
        new_gws = [row['_gw'] for row in squads.lines[(element, team)][start:start + template.length]]
        for gw, entry in template.flags.items():
            target = _matching_gw(gw, template.match_gws, new_gws)
            if not 1 <= target <= 38:
                continue
            old_deadline, new_deadline = source.deadline(gw), source.deadline(target)
            shift = (new_deadline - old_deadline) if (old_deadline and new_deadline) else timedelta(days=7 * (target - gw))
            existing = overrides[target].get(element)
            # Two spells of one player can sit a match apart: an unflagged
            # deadline from one must not wipe the other's flag.
            if existing and is_flagged(existing) and not is_flagged(entry):
                continue
            overrides[target][element] = shift_entry(entry, shift)
    return {gw: dict(players) for gw, players in overrides.items()}


def _matching_gw(gw: int, old_gws, new_gws) -> int:
    """Where a template's flag lands: on the same missed match, or the same
    number of deadlines before the first / after the last missed match."""
    if gw in old_gws:
        return new_gws[list(old_gws).index(gw)]
    if gw < old_gws[0]:
        return new_gws[0] - (old_gws[0] - gw)
    return new_gws[-1] + (gw - old_gws[-1])


def shift_entry(entry: dict, shift: timedelta) -> dict:
    """The same flag moved in time: news_added and any "Expected back" date shift too."""
    from .source import parse_time
    entry = dict(entry)
    added = parse_time(entry.get('news_added'))
    if added is not None:
        entry['news_added'] = (added + shift).isoformat().replace('+00:00', 'Z')
    text = entry.get('news') or ''
    if text and added is not None:
        def _move(match):
            month = match.group(3).lower()
            if month not in MONTHS:
                return match.group(0)
            for year in (added.year - 1, added.year, added.year + 1):
                try:
                    when = added.replace(year=year, month=MONTHS.index(month) + 1, day=int(match.group(2)))
                except ValueError:
                    continue
                if when >= added - timedelta(days=14):
                    moved = when + shift
                    return '%s%d %s' % (match.group(1), moved.day, MONTHS[moved.month - 1].capitalize())
            return match.group(0)
        entry['news'] = DATE_IN_NEWS.sub(_move, text)
    return entry
