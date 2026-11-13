"""Reshuffle which gameweek each round of fixtures is played in.

Every team keeps its real match dates, and with them everything that belongs
to a date rather than an opponent: who played and for how long, injuries, FPL
flags and news, prices, cards. What moves is the opponent: the ten fixtures
of one real gameweek are played in another gameweek's slot.

Only "clean" gameweeks swap: ten fixtures, every team once, and every fixture
in that gameweek in every saved FPL calendar (simulation/fixture_calendar.py),
so nothing in it was ever postponed or moved. Blanks, doubles and rearranged
matches stay where they were. Gameweeks swap within their half of the season
(1-19, 20-38), so each team still meets everyone once in each half.

A search over swaps keeps two rules the real fixture lists keep: no team plays
the same opponent twice within MIN_REMATCH_GAP gameweeks, and no team has a
longer run of home or away matches than the real season's longest.

Kick-off times: the ten fixtures moved into a gameweek take that gameweek's
ten real kick-off times, each fixture the time of one of its two teams' real
matches (a perfect matching always exists). So every deadline stays where it
was, and each team plays at its own real time or its opponent's.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

MIN_REMATCH_GAP = 3        # gameweeks between a fixture and its reverse
SEARCH_STEPS = 4000
HALVES = ((1, 19), (20, 38))
SCHEDULE_FILE = 'schedule.json'


def clean_gameweeks(source) -> List[int]:
    """Gameweeks with ten fixtures, every team once, never moved in FPL's calendar."""
    moved = set()
    path = source.directory / 'fixture_calendars.json'
    if path.is_file():
        for snapshot in json.loads(path.read_text(encoding='utf-8')).get('snapshots') or []:
            for fixture_id, event in (snapshot.get('events') or {}).items():
                fixture = source.fixtures.get(int(fixture_id))
                if fixture is not None and (event is None or int(event) != fixture.gameweek):
                    moved.add(fixture.id)
    else:
        return []
    by_gw = defaultdict(list)
    for fixture in source.fixtures.values():
        if fixture.gameweek is not None:
            by_gw[fixture.gameweek].append(fixture)
    clean = []
    for gw, fixtures in sorted(by_gw.items()):
        teams = Counter([f.home for f in fixtures] + [f.away for f in fixtures])
        if len(fixtures) == 10 and len(teams) == 20 and not any(f.id in moved for f in fixtures):
            clean.append(gw)
    return clean


@dataclass
class Reshuffle:
    slots: Dict[int, int]                         # real gameweek -> gameweek its fixtures are played in
    fixtures: Dict[int, object]                   # fixture id -> Fixture as played in the copy
    slot_fixture: Dict[Tuple[int, int], int]      # (real fixture, team) -> copy fixture on that date
    kickoff_text: Dict[int, str] = field(default_factory=dict)   # moved fixture -> kickoff_time string
    cost: int = 0                                 # rule breaks left after the search (0 normally)

    def to_json(self) -> dict:
        return {
            'slots': {str(gw): slot for gw, slot in sorted(self.slots.items())},
            'rule_breaks': self.cost,
            'moved': {str(fid): [self.fixtures[fid].gameweek, text] for fid, text in sorted(self.kickoff_text.items())},
            'slot_fixture': [[real, team, new] for (real, team), new in sorted(self.slot_fixture.items())],
        }


def draw_reshuffle(source, rng) -> Reshuffle:
    clean = clean_gameweeks(source)
    by_gw = defaultdict(list)
    for fixture in source.fixtures.values():
        if fixture.gameweek is not None:
            by_gw[fixture.gameweek].append(fixture)
    limit = _longest_run(source, by_gw, {gw: gw for gw in by_gw})
    halves = [[gw for gw in clean if lo <= gw <= hi] for lo, hi in HALVES]
    slots = {gw: gw for gw in by_gw}
    for half in halves:
        for gw, slot in zip(half, rng.permutation(half)):
            slots[gw] = int(slot)
    cost = _cost(source, by_gw, slots, limit)
    for _step in range(SEARCH_STEPS):
        if cost == 0:
            break
        half = halves[int(rng.integers(len(halves)))]
        if len(half) < 2:
            continue
        a, b = rng.choice(half, size=2, replace=False)
        trial = dict(slots)
        trial[int(a)], trial[int(b)] = slots[int(b)], slots[int(a)]
        trial_cost = _cost(source, by_gw, trial, limit)
        if trial_cost <= cost:
            slots, cost = trial, trial_cost
    real_in_slot = {slot: gw for gw, slot in slots.items()}
    fixtures, slot_fixture, kickoff_text = {}, {}, {}
    for fixture in source.fixtures.values():
        if fixture.gameweek is None or slots[fixture.gameweek] == fixture.gameweek:
            fixtures[fixture.id] = fixture
    for slot, real_gw in real_in_slot.items():
        if slot == real_gw:
            continue
        old = by_gw[slot]           # the real fixtures on these dates
        new = by_gw[real_gw]        # the fixtures now played on them
        times = _match_kickoffs(old, new, rng)
        own = {team: f for f in old for team in (f.home, f.away)}
        for fixture in new:
            timed = times[fixture.id]
            raw = dict(fixture.raw, event=str(slot), kickoff_time=timed.raw.get('kickoff_time'))
            fixtures[fixture.id] = replace(fixture, gameweek=slot, kickoff=timed.kickoff, raw=raw)
            kickoff_text[fixture.id] = timed.raw.get('kickoff_time')
            for team in (fixture.home, fixture.away):
                slot_fixture[(own[team].id, team)] = fixture.id
    return Reshuffle(slots, fixtures, slot_fixture, kickoff_text, cost)


def _match_kickoffs(old, new, rng) -> Dict[int, object]:
    """new fixture id -> the real fixture (in `old`) whose kick-off it takes; every old time used once."""
    by_team = {team: f for f in old for team in (f.home, f.away)}
    options = {f.id: list({by_team[f.home].id: by_team[f.home], by_team[f.away].id: by_team[f.away]}.values())
               for f in new}
    owner = {}     # old fixture id -> new fixture id

    def augment(new_id, seen):
        choices = list(options[new_id])
        rng.shuffle(choices)
        for old_fixture in choices:
            if old_fixture.id in seen:
                continue
            seen.add(old_fixture.id)
            if old_fixture.id not in owner or augment(owner[old_fixture.id], seen):
                owner[old_fixture.id] = new_id
                return True
        return False

    order = [f.id for f in new]
    rng.shuffle(order)
    for new_id in order:
        if not augment(new_id, set()):
            raise ValueError('no kick-off matching for gameweek fixtures %s' % order)
    old_by_id = {f.id: f for f in old}
    return {new_id: old_by_id[old_id] for old_id, new_id in owner.items()}


def _team_sequences(by_gw, slots):
    """team -> [(gameweek, home?, opponent)] in gameweek order, with the slots applied."""
    seq = defaultdict(list)
    for gw, fixtures in by_gw.items():
        slot = slots[gw]
        for f in fixtures:
            order = (slot, f.kickoff.timestamp() if f.kickoff else 0)
            seq[f.home].append((order, True, f.away))
            seq[f.away].append((order, False, f.home))
    return {team: sorted(items) for team, items in seq.items()}


def _longest_run(source, by_gw, slots) -> int:
    longest = 1
    for items in _team_sequences(by_gw, slots).values():
        run = 1
        for (_a, home_a, _o), (_b, home_b, _p) in zip(items, items[1:]):
            run = run + 1 if home_a == home_b else 1
            longest = max(longest, run)
    return longest


def _cost(source, by_gw, slots, limit) -> int:
    cost = 0
    for items in _team_sequences(by_gw, slots).values():
        run = 1
        last_met = {}
        for index, ((gw_order, home, opponent)) in enumerate(items):
            if index and items[index - 1][1] == home:
                run += 1
                cost += int(run > limit)
            else:
                run = 1
            gw = gw_order[0]
            if opponent in last_met and gw - last_met[opponent] < MIN_REMATCH_GAP:
                cost += 1
            last_met[opponent] = gw
    return cost


def apply_reshuffle(rows, reshuffle: Reshuffle, match_xg) -> None:
    """Move each team's rows on a moved date to the fixture now played then (in place).

    Keeps the real opponent's xG on the row (_opp_xg_real): the keeper's real
    saves were made against that opponent.
    """
    for row in rows:
        row['_opp_xg_real'] = match_xg.get((row['fixture'], row['_opponent']), 0.0)
        new_id = reshuffle.slot_fixture.get((row['fixture'], row['_team']))
        if new_id is None:
            continue
        fixture = reshuffle.fixtures[new_id]
        home = fixture.home == row['_team']
        row['fixture'] = new_id
        row['_opponent'] = fixture.away if home else fixture.home
        row['_home'] = home
        row['_kickoff'] = fixture.kickoff
        row['opponent_team'] = row['_opponent']
        row['was_home'] = 'True' if home else 'False'
        row['kickoff_time'] = reshuffle.kickoff_text[new_id]


def load_reshuffle(directory) -> dict:
    """The copy's schedule.json: {'slots', 'moved': {fixture: [gw, kickoff]}, 'slot_fixture'}; {} if none."""
    path = directory / SCHEDULE_FILE
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['moved'] = {int(fid): (int(gw), text) for fid, (gw, text) in payload.get('moved', {}).items()}
    payload['slot_fixture'] = {(int(real), int(team)): int(new) for real, team, new in payload.get('slot_fixture', [])}
    return payload
