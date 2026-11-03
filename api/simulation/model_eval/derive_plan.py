"""Write a wave B plan by adding external-data families to the B0 plan.

Run from api/:
  python -m simulation.model_eval.derive_plan --base simple_h1_b0 --out simple_h1_b1 --family context

B0's plan is only chosen overnight (pick_base), so the wave B plans are derived
from it when their run starts, not written by hand in advance. Families:
  context  : schedule (all-competition rest and congestion) and gameweek
             (doubles / blanks) on both team blocks, plus the player_load root;
             drops match_facts.days_since_previous_match, which only saw PL games.
  managers : manager_features (caretaker, log days in post) on both team blocks.
             The manager embedding itself is a later step.
  scoreline: points built from minutes-band chances and expected goals for and
             against (points_structure scoreline, score head weight 0.1). Re-run
             on a wave B plan, once the fixture context is in.
  staged   : one team-level scoreline per fixture in goal bands 0-4+, shared by
             both sides, and points built from minutes bands and goal bands
             (points_structure staged, score head weight 0.1).
Fields come from learning_v2/external_inputs.py (TEAM_SCALARS, PLAYER_SCALARS).
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

PLANS_DIR = Path(__file__).resolve().parents[2] / 'premierlearning' / 'learning_v2' / 'model_plans'

# Leaf types: day and minute counts are floats (scaled); flags and small counts are ints.
SCHEDULE = OrderedDict([
    ('days_since_last_match', 'float'),
    ('matches_last_14d', 'float'),
    ('days_to_next_match', 'float'),
    ('matches_next_7d', 'float'),
])
GAMEWEEK = OrderedDict([
    ('gw_fixtures', 'int'),
    ('gw_double', 'int'),
    ('dgw_leg', 'int'),
    ('next1_gw_double_known', 'int'),
    ('next1_gw_blank_known', 'int'),
    ('next2_gw_double_known', 'int'),
    ('next2_gw_blank_known', 'int'),
    ('next3_gw_double_known', 'int'),
    ('next3_gw_blank_known', 'int'),
])
PLAYER_LOAD = OrderedDict([
    ('days_since_last_played', 'float'),
    ('days_since_last_started', 'float'),
    ('minutes_last_7d', 'float'),
    ('minutes_last_14d', 'float'),
    ('minutes_last_28d', 'float'),
    ('in_squad_last_club_match', 'int'),
    ('load_known', 'int'),
])
MANAGER = OrderedDict([
    ('caretaker', 'int'),
    ('days_in_post_log', 'float'),
])
SIDES = ('current_team', 'current_opposition')
FAMILIES = ('context', 'managers', 'scoreline', 'staged')


def _insert_after(block, key, name, value):
    """Put name: value after key in an ordered dict (at the end if key is missing)."""
    items = list(block.items())
    out = OrderedDict()
    placed = False
    for existing, existing_value in items:
        if existing == name:
            continue
        out[existing] = existing_value
        if existing == key:
            out[name] = value
            placed = True
    if not placed:
        out[name] = value
    return out


def add_family(plan, family):
    plan = json.loads(json.dumps(plan), object_pairs_hook=OrderedDict)
    if family == 'context':
        for side in SIDES:
            plan[side] = _insert_after(plan[side], 'team_features', 'schedule', OrderedDict(SCHEDULE))
            plan[side] = _insert_after(plan[side], 'schedule', 'gameweek', OrderedDict(GAMEWEEK))
        facts = plan.get('match_facts', OrderedDict())
        facts.pop('days_since_previous_match', None)
        plan['match_facts'] = facts
        plan = _insert_after(plan, 'context', 'player_load', OrderedDict(PLAYER_LOAD))
    elif family == 'managers':
        for side in SIDES:
            id_name = 'team_id_embedding' if side == 'current_team' else 'opposition_team_id_embedding'
            plan[side] = _insert_after(plan[side], id_name, 'manager_features', OrderedDict(MANAGER))
    elif family in ('scoreline', 'staged'):
        config = plan.setdefault('_config', OrderedDict())
        if config.get('points_structure') == family:
            raise SystemExit('The base plan already builds points the %s way' % family)
        config['score_head_weight'] = config.get('score_head_weight') or 0.1
        config['points_structure'] = family
    else:
        raise ValueError('Unknown family %s (choose from %s)' % (family, ', '.join(FAMILIES)))
    return plan


def check_coverage(seasons):
    """Stop unless every season has real feature values, not just rows of defaults.

    Every fixture gets a row even before the full build (missing data reads as
    the "nothing known" defaults), so row counts prove nothing. The external-data
    chat's is_full_history() checks the share of real (uncapped) values instead.
    """
    from premierlearning.learning_v2.external_inputs import ExternalInputs

    first, _, last = seasons.partition('-')
    wanted = range(int(first), int(last or first) + 1)
    ext = ExternalInputs.load()
    coverage = ext.coverage()
    for season in wanted:
        print('  %s: %s' % (season, coverage.get(season, 'no rows')))
    if not ext.is_full_history(seasons=wanted):
        raise SystemExit('Feature tables do not hold real values for every season %s yet' % seasons)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--family', action='append', required=True, choices=FAMILIES)
    parser.add_argument('--require-seasons', default='',
                        help='e.g. 2018-2025: stop unless the feature tables cover each of these seasons')
    args = parser.parse_args(argv)
    if args.require_seasons:
        check_coverage(args.require_seasons)
    with (PLANS_DIR / ('%s.json' % args.base)).open(encoding='utf-8') as handle:
        plan = json.load(handle, object_pairs_hook=OrderedDict)
    for family in args.family:
        plan = add_family(plan, family)
    from premierlearning.learning_v2.dataset import unsupported_leaves
    from premierlearning.learning_v2.plan import plan_from_dict

    parsed = plan_from_dict(json.loads(json.dumps(plan)), args.out)
    missing = unsupported_leaves(parsed)
    if missing:
        raise SystemExit('The dataset cannot fill: %s' % ', '.join(missing))
    with (PLANS_DIR / ('%s.json' % args.out)).open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(plan, handle, indent=2)
        handle.write('\n')
    print('Wrote %s from %s with %s' % (args.out, args.base, ', '.join(args.family)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
