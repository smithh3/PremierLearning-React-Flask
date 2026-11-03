"""Map ESPN league slugs to a small set of match categories for an embedding.

`category` is coarse (about 15 values) so rare competitions still share a
learned row. Keep the raw `league_slug` too if the model wants finer detail.
"""
from __future__ import annotations

import re

CATEGORY_BY_SLUG = {
    'eng.1': 'premier_league',
    'eng.fa': 'domestic_cup',
    'eng.league_cup': 'domestic_cup',
    'eng.charity': 'domestic_super_cup',
    'uefa.champions': 'champions_league',
    'uefa.champions_qual': 'champions_league',
    'uefa.europa': 'europa_league',
    'uefa.europa_qual': 'europa_league',
    'uefa.europa.conf': 'conference_league',
    'uefa.europa.conf_qual': 'conference_league',
    'uefa.super_cup': 'club_super_cup',
    'fifa.cwc': 'club_world_cup',
    'fifa.intercontinental_cup': 'club_world_cup',
    'club.friendly': 'club_friendly',
    'nonfifa': 'club_friendly',
    'friendly.emirates_cup': 'club_friendly',
    'esp.joan_gamper': 'club_friendly',
    'jpn.world_challenge': 'club_friendly',
    'global.club_challenge': 'club_friendly',
    'eng.trophy': 'domestic_cup',  # PL academy sides play in the EFL Trophy
    'fifa.world': 'intl_tournament',
    'uefa.euro': 'intl_tournament',
    'conmebol.america': 'intl_tournament',
    'caf.nations': 'intl_tournament',
    'afc.asian.cup': 'intl_tournament',
    'concacaf.gold': 'intl_tournament',
    'global.finalissima': 'intl_tournament',
    'uefa.nations': 'intl_nations_league',
    'concacaf.nations.league': 'intl_nations_league',
    'fifa.friendly': 'intl_friendly',
    'fifa.olympics': 'intl_youth',
    'uefa.euro_u21': 'intl_youth',
    'uefa.euro_u21_qual': 'intl_youth',
    'fifa.world.u20': 'intl_youth',
    'fifa.friendly_u21': 'intl_youth',
    'uefa.euro.u19': 'intl_youth',
    'fifa.world.u17': 'intl_youth',
}

QUALIFIER = re.compile(r'^(fifa\.worldq\.|fifa\.wcq\.ply$|uefa\.euroq$|caf\.nations_qual$|afc\.cupq$|concacaf\.gold_qual$)')
WOMEN = re.compile(r"(\.w\.|\.w$|^fifa\.ww|wworld|weuro|wchampions|femenina|nwsl|women|reina|shebelieves|pinatar|arnold\.clark)", re.I)

INTERNATIONAL = {'intl_tournament', 'intl_qualifier', 'intl_nations_league', 'intl_friendly', 'intl_youth'}

# Competitions with a schedule fixed well before the match (league tables,
# group / league phases). Knockout ties only become known as a team progresses.
FIXED_STAGE = re.compile(r'(regular|league-phase|group|league stage|^season$)', re.I)


def is_women(slug: str, name: str = '') -> bool:
    return bool(WOMEN.search(slug or '')) or 'women' in (name or '').lower()


def category(slug: str, name: str = '') -> str:
    slug = slug or ''
    if is_women(slug, name):
        return 'women'
    if slug in CATEGORY_BY_SLUG:
        return CATEGORY_BY_SLUG[slug]
    if QUALIFIER.search(slug):
        return 'intl_qualifier'
    if slug.startswith('usa.ncaa'):
        return 'college'
    return 'other_club'


def is_international(cat: str) -> bool:
    return cat in INTERNATIONAL
