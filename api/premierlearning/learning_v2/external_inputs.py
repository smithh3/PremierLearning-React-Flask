"""Read-only access to the external-data features for the v2 dataset builder.

Owned by the external-data chat (see AGENT_NOTES.md); the model chat wires it into
dataset.py / plan.py / model.py. Everything here is already computed as of each
fixture's gameweek deadline by premierlearning.external_data.features, so callers
need no extra leakage handling for the h1 setup.

    ext = ExternalInputs.load()                    # once per dataset build
    ext.team(2025, 17, 14)                         # dict of team scalars (own team; call again for the opposition)
    ext.player(2025, 17, 118748)                   # dict of player scalars
    ext.team_upcoming(2025, 17, 14)                # list of token dicts, ordered by kickoff
    ext.player_upcoming(2025, 17, 118748, 14)      # club tokens + national-team tokens, ordered by kickoff
    ext.player_history(118748, before_utc, n)      # non-PL appearance tokens (cups, Europe, friendlies, internationals)
    ext.competition_index('champions_league')      # embedding index, 0 = pad / unknown
    ext.manager_vocab(train_seasons=[2018, ..., 2024], min_matches=10)  # manager_id -> index, 0 = unknown
    ext.slices(2025, 17, 118748)                   # eval slice flags (for reports, not inputs)

Missing keys give the defaults in TEAM_DEFAULTS / PLAYER_DEFAULTS, meaning "nothing known".
Values are raw (days, minutes, counts); scale them in the model as for other scalars.
"""
from __future__ import annotations

import bisect
import csv
import datetime as dt
import math
import os
from collections import Counter, defaultdict

API_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTERNAL_DIR = os.path.join(API_ROOT, 'data', 'external')
FEATURE_DIR = os.path.join(EXTERNAL_DIR, 'features')

# Fixed order: index = position + 1 (0 is pad / unknown). Append only, so trained embeddings keep their rows.
COMPETITIONS = (
    'premier_league', 'domestic_cup', 'domestic_super_cup', 'champions_league', 'europa_league',
    'conference_league', 'club_super_cup', 'club_world_cup', 'club_friendly', 'intl_tournament',
    'intl_qualifier', 'intl_nations_league', 'intl_friendly', 'intl_youth', 'other_club',
)
CAP_DAYS = 120.0

TEAM_SCALARS = {
    # plan path suffix -> (csv column, default)
    'schedule.days_since_last_match': ('days_since_last_match', CAP_DAYS),
    'schedule.matches_last_14d': ('matches_last_14d', 0.0),
    'schedule.days_to_next_match': ('days_to_next_match', CAP_DAYS),
    'schedule.matches_next_7d': ('matches_next_7d', 0.0),
    'gameweek.gw_fixtures': ('gw_fixtures', 1.0),
    'gameweek.gw_double': ('gw_double', 0.0),
    'gameweek.dgw_leg': ('dgw_leg', 1.0),
    'gameweek.next1_gw_double_known': ('next1_gw_double_known', 0.0),
    'gameweek.next1_gw_blank_known': ('next1_gw_blank_known', 0.0),
    'gameweek.next2_gw_double_known': ('next2_gw_double_known', 0.0),
    'gameweek.next2_gw_blank_known': ('next2_gw_blank_known', 0.0),
    'gameweek.next3_gw_double_known': ('next3_gw_double_known', 0.0),
    'gameweek.next3_gw_blank_known': ('next3_gw_blank_known', 0.0),
    'manager_features.caretaker': ('manager_caretaker', 0.0),
    'manager_features.days_in_post_log': ('manager_days_in_post_log', 0.0),
}
PLAYER_SCALARS = {
    'player_load.days_since_last_played': ('days_since_last_played', CAP_DAYS),
    'player_load.days_since_last_started': ('days_since_last_started', CAP_DAYS),
    'player_load.minutes_last_7d': ('minutes_last_7d', 0.0),
    'player_load.minutes_last_14d': ('minutes_last_14d', 0.0),
    'player_load.minutes_last_28d': ('minutes_last_28d', 0.0),
    'player_load.in_squad_last_club_match': ('in_squad_last_club_match', 0.0),
    'player_load.load_known': ('load_known', 0.0),
}
TEAM_DEFAULTS = {path: default for path, (_, default) in TEAM_SCALARS.items()}
PLAYER_DEFAULTS = {path: default for path, (_, default) in PLAYER_SCALARS.items()}
UPCOMING_FIELDS = ('competition_embedding', 'opposition_team_code', 'home_away', 'neutral', 'knockout',
                   'days_after_current', 'national_team')
HISTORY_FIELDS = ('kickoff', 'competition_embedding', 'national_team', 'home_away', 'minutes_played', 'started',
                  'goals', 'assists', 'clean_sheet', 'team_goals', 'opponent_goals', 'fpl_stats_known')
INTERNATIONAL = {'intl_tournament', 'intl_qualifier', 'intl_nations_league', 'intl_friendly', 'intl_youth'}


def competition_index(category: str) -> int:
    try:
        return COMPETITIONS.index(category) + 1
    except ValueError:
        return 0


def _t(text):
    return dt.datetime.strptime(text[:16], '%Y-%m-%dT%H:%M') if text else None


def _f(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


class ExternalInputs:
    def __init__(self, feature_dir=FEATURE_DIR, external_dir=EXTERNAL_DIR):
        self.feature_dir = feature_dir
        self.external_dir = external_dir
        self._team = {}
        self._manager = {}
        self._player = {}
        self._team_upcoming = defaultdict(list)
        self._intl_upcoming = defaultdict(list)
        self._slices = {}
        self._history = None
        self._slice_columns = []

    @classmethod
    def load(cls, feature_dir=FEATURE_DIR, external_dir=EXTERNAL_DIR):
        ext = cls(feature_dir, external_dir)
        for r in _read(os.path.join(feature_dir, 'team_fixture_features.csv')):
            key = (int(r['fpl_season']), int(r['fpl_fixture_id']), int(r['team_code']))
            ext._team[key] = tuple(_f(r[col], default) for col, default in TEAM_SCALARS.values())
            ext._manager[key] = r['manager_id']
        for r in _read(os.path.join(feature_dir, 'player_fixture_features.csv')):
            key = (int(r['fpl_season']), int(r['fpl_fixture_id']), int(r['fpl_code']))
            ext._player[key] = tuple(_f(r[col], default) for col, default in PLAYER_SCALARS.values())
        for r in _read(os.path.join(feature_dir, 'team_upcoming.csv')):
            key = (int(r['fpl_season']), int(r['fpl_fixture_id']), int(r['team_code']))
            ext._team_upcoming[key].append(ext._token(r, national=0))
        for r in _read(os.path.join(feature_dir, 'player_international_upcoming.csv')):
            key = (int(r['fpl_season']), int(r['fpl_fixture_id']), int(r['fpl_code']))
            ext._intl_upcoming[key].append(ext._token(r, national=1))
        rows = _read(os.path.join(feature_dir, 'eval_slices.csv'))
        if rows:
            ext._slice_columns = [c for c in rows[0] if c.startswith('slice_')]
        for r in rows:
            key = (int(r['fpl_season']), int(r['fpl_fixture_id']), int(r['fpl_code']))
            ext._slices[key] = tuple(int(r[c]) for c in ext._slice_columns)
        return ext

    @staticmethod
    def _token(r, national):
        opp = r.get('opponent_fpl_team_code') or ''
        return {
            'competition_embedding': competition_index(r['category']),
            'opposition_team_code': int(float(opp)) if opp else None,  # None -> unknown team embedding
            'home_away': int(r['home_away']),
            'neutral': int(r['neutral']),
            'knockout': int(r['knockout']),
            'days_after_current': float(r['days_after_current']),
            'national_team': national,
        }

    # ------------------------------------------------------------ scalars

    def team(self, season, fixture_id, team_code) -> dict:
        values = self._team.get((int(season), int(fixture_id), int(team_code)))
        if values is None:
            return dict(TEAM_DEFAULTS)
        return dict(zip(TEAM_SCALARS, values))

    def manager_id(self, season, fixture_id, team_code) -> str:
        return self._manager.get((int(season), int(fixture_id), int(team_code)), '')

    def player(self, season, fixture_id, fpl_code) -> dict:
        values = self._player.get((int(season), int(fixture_id), int(fpl_code)))
        if values is None:
            return dict(PLAYER_DEFAULTS)
        return dict(zip(PLAYER_SCALARS, values))

    def slices(self, season, fixture_id, fpl_code) -> dict:
        values = self._slices.get((int(season), int(fixture_id), int(fpl_code)))
        return dict(zip(self._slice_columns, values or (0,) * len(self._slice_columns)))

    @property
    def slice_columns(self):
        return list(self._slice_columns)

    def coverage(self) -> dict:
        """Per season: row counts, and the share of rows holding real values rather than defaults.

        Row counts alone can't tell a full build from a partial one: every fixture gets a row
        and missing data shows up as defaults. Use the `*_real_share` values as the readiness gate.
        A full build should give about 0.95+ for team rows (every PL club has a recent match).
        Player rows are lower, because squad players who never feature stay at the 120-day cap.
        Today's test-window build gives 0.0 for every season except 2025.
        """
        out = defaultdict(lambda: {'team_rows': 0, 'team_real': 0, 'player_rows': 0, 'player_real': 0})
        for (season, _, _), values in self._team.items():
            out[season]['team_rows'] += 1
            out[season]['team_real'] += values[0] < CAP_DAYS  # schedule.days_since_last_match
        for (season, _, _), values in self._player.items():
            out[season]['player_rows'] += 1
            out[season]['player_real'] += values[0] < CAP_DAYS  # player_load.days_since_last_played
        for row in out.values():
            row['team_real_share'] = round(row['team_real'] / max(row['team_rows'], 1), 3)
            row['player_real_share'] = round(row['player_real'] / max(row['player_rows'], 1), 3)
        return dict(sorted(out.items()))

    def is_full_history(self, seasons=range(2018, 2026), min_team_share=0.9, min_player_share=0.3) -> bool:
        """True when every season in `seasons` has real (non-default) values, i.e. the full crawl was built."""
        cov = self.coverage()
        return all(s in cov and cov[s]['team_rows'] >= 700 and cov[s]['team_real_share'] >= min_team_share
                   and cov[s]['player_real_share'] >= min_player_share for s in seasons)

    # ------------------------------------------------------------ lists

    def team_upcoming(self, season, fixture_id, team_code, limit=14) -> list:
        return self._team_upcoming.get((int(season), int(fixture_id), int(team_code)), [])[:limit]

    def player_upcoming(self, season, fixture_id, fpl_code, team_code, limit=14) -> list:
        """The player's club schedule (all competitions) merged with his national team's, by days ahead."""
        merged = self.team_upcoming(season, fixture_id, team_code, limit) + \
            self._intl_upcoming.get((int(season), int(fixture_id), int(fpl_code)), [])
        return sorted(merged, key=lambda t: t['days_after_current'])[:limit]

    def _load_history(self):
        history = defaultdict(list)
        for r in _read(os.path.join(self.external_dir, 'appearances.csv')):
            if r['fpl_fixture_id']:
                continue  # PL rows come from FPL data (with xG / points) in dataset.py
            cat = r['category']
            history[int(r['fpl_code'])].append({
                'kickoff': _t(r['kickoff_utc']),
                'competition_embedding': competition_index(cat),
                'national_team': int(cat in INTERNATIONAL),
                'home_away': int(r['home_away'] == 'home'),
                'minutes_played': _f(r['minutes']),
                'started': int(r['starter'] or 0),
                'goals': _f(r['goals']),
                'assists': _f(r['assists']),
                'clean_sheet': int(_f(r['clean_sheet'])),
                'team_goals': _f(r['team_score']),
                'opponent_goals': _f(r['opponent_score']),
                'fpl_stats_known': 0,
            })
        self._history = {}
        for code, items in history.items():
            items.sort(key=lambda t: t['kickoff'])
            self._history[code] = (items, [t['kickoff'] for t in items])

    def player_history(self, fpl_code, before, n=20) -> list:
        """Last n non-PL appearances (squad rows, minutes may be 0) with kickoff strictly before `before`.

        Use the gameweek deadline (or kickoff, matching how PL history is cut) as `before`.
        Merge with the PL tokens by `kickoff` to get one chronological history.
        """
        if self._history is None:
            self._load_history()
        items, kicks = self._history.get(int(fpl_code), ([], []))
        end = bisect.bisect_left(kicks, before)
        return items[max(0, end - n):end]

    # ------------------------------------------------------------ vocabularies

    def manager_vocab(self, train_seasons, min_matches=10) -> dict:
        """manager_id -> embedding index (1..), for managers with >= min_matches PL matches in train_seasons.

        Everyone else (mostly caretakers and brand-new managers) maps to 0 = unknown.
        Ordered by first appearance so indices are stable when seasons are added.
        """
        seasons = {int(s) for s in train_seasons}
        counts, first = Counter(), {}
        for (season, fixture, _), manager in sorted(self._manager.items()):
            if season in seasons and manager:
                counts[manager] += 1
                first.setdefault(manager, (season, fixture))
        keep = sorted((m for m, c in counts.items() if c >= min_matches), key=lambda m: first[m])
        return {m: i + 1 for i, m in enumerate(keep)}

    @staticmethod
    def manager_index(vocab: dict, manager_id: str) -> int:
        return vocab.get(manager_id, 0)

    @staticmethod
    def days_in_post_from_log(value: float) -> float:
        return math.expm1(value)
