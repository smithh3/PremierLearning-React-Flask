"""Explain stored predictions for named players (no player-model fitting).

Loads the last persisted future_matches (Fargate / premierlearning persist)
and prints them in the same columns as the results email. Rebuilds processed
match rows (all players, no learning) so horizon and minutes model inputs can
be printed the same way as the Haaland / Fernandes sample on --horizon-only
and --minutes-only runs. Also dumps every played current-season and last-season
fixture (points + minutes).

Run from api/:
  python -m premierlearning.explain_player haaland watkins fernandes
  python -m premierlearning.explain_player "igor jesus" "bruno g" "gibbs white"

Players are printed in descending predicted points over the next POINTS_HORIZON
gameweeks (from the database). Team xG is refit locally, so those feature
values can differ slightly from the run that wrote the DB scores.

Stdout after setup is appended to explain_player.txt in the cwd (api/).
"""
import argparse
import os
import re
import sys
import unicodedata
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask

from premierlearning.clean_match import use_team_expectancies
from premierlearning.config_constants import POINTS_HORIZON, SEASON_YR
from premierlearning.player import Player
from premierlearning.runner import Runner
from premierlearning.season import CurrentSeason
from premierlearning.team_expectancy import run_team_expectancies
from persistence import DB_Handler
from squadtools.predictions_report import csv_headers, player_row

OUTPUT_PATH = 'explain_player.txt'


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _open_output(path, queries):
    out = open(path, 'a', encoding='utf-8')
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    names = ' '.join(queries)
    out.write('\n\n')
    out.write('=' * 72 + '\n')
    out.write(f'explain_player {stamp}  args: {names}\n')
    out.write('=' * 72 + '\n')
    out.flush()
    return out


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize('NFKD', text or '')
    return ''.join(ch for ch in normalized if not unicodedata.combining(ch))


def _name_tokens(text: str):
    cleaned = _strip_accents(text).lower()
    cleaned = re.sub(r'[^a-z0-9]+', ' ', cleaned)
    return [tok for tok in cleaned.split() if tok]


def _collapsed(tokens) -> str:
    return ''.join(tokens)


def _initials(tokens) -> str:
    return ''.join(tok[0] for tok in tokens if tok)


def _web_name(player) -> str:
    element = getattr(player, 'element_dict', None) or {}
    return str(element.get('web_name') or '')


def _player_name_parts(player):
    web_tokens = _name_tokens(_web_name(player))
    full_tokens = _name_tokens(f'{player.first_name} {player.last_name}')
    return web_tokens, full_tokens


def _is_subsequence(needle: str, haystack: str) -> bool:
    """True if needle chars appear in order in haystack (not necessarily contiguous)."""
    i = 0
    for ch in haystack:
        if i < len(needle) and ch == needle[i]:
            i += 1
            if i == len(needle):
                return True
    return i == len(needle)


def _token_matches(query: str, source: str) -> bool:
    if source.startswith(query):
        return True
    if len(query) >= 3 and len(source) >= 3 and query.startswith(source):
        return True
    if len(query) >= 3 and len(source) >= 4 and _is_subsequence(query, source):
        return True
    return False


def _prefix_seq_match(query_tokens, source_tokens) -> bool:
    """True if each query token matches some later source token, in order."""
    i = 0
    for query in query_tokens:
        found = False
        while i < len(source_tokens):
            if _token_matches(query, source_tokens[i]):
                found = True
                i += 1
                break
            i += 1
        if not found:
            return False
    return True


def _match_score(player, query: str) -> int:
    """Higher is better; 0 means no match."""
    query_tokens = _name_tokens(query)
    if not query_tokens:
        return 0
    query_join = ' '.join(query_tokens)
    query_flat = _collapsed(query_tokens)
    web_tokens, full_tokens = _player_name_parts(player)
    web_join = ' '.join(web_tokens)
    full_join = ' '.join(full_tokens)
    web_flat = _collapsed(web_tokens)
    full_flat = _collapsed(full_tokens)
    score = 0

    if query_join == web_join or query_flat == web_flat:
        score = max(score, 1000)
    if query_join == full_join or query_flat == full_flat:
        score = max(score, 950)

    if len(query_tokens) == 1 and len(query_flat) >= 2:
        if query_flat in {_initials(full_tokens), _initials(web_tokens)}:
            score = max(score, 900)

    if web_tokens and _prefix_seq_match(query_tokens, web_tokens):
        score = max(score, 800)
    if full_tokens and _prefix_seq_match(query_tokens, full_tokens):
        score = max(score, 700)

    if query_flat and query_flat in web_flat:
        score = max(score, 650)
    if query_flat and query_flat in full_flat:
        score = max(score, 600)

    if len(query_tokens) == 1:
        token = query_tokens[0]
        if token in web_tokens:
            score = max(score, 750)
        elif token in full_tokens:
            score = max(score, 550)

    return score


def resolve_player(query: str, players) -> object:
    """Pick one player for a name query; web_name / initials beat namesakes."""
    scored = [(p, _match_score(p, query)) for p in players]
    scored = [(p, score) for p, score in scored if score > 0]
    if not scored:
        print(f'Could not find a player matching {query!r}')
        return None
    scored.sort(
        key=lambda item: (item[1], float(getattr(item[0], 'current_cost', 0) or 0)),
        reverse=True,
    )
    chosen, best = scored[0]
    extras = [p for p, score in scored[1:] if score >= best - 50]
    extra = ''
    if extras:
        extra = '  [also matched: %s]' % ', '.join(
            f'{p.get_name()} ({_web_name(p) or "no web_name"})' for p in extras[:8]
        )
    print(
        f'  {query!r} -> {chosen.get_name()} '
        f'({_web_name(chosen) or chosen.team_name}){extra}'
    )
    return chosen


def _horizon_points(player, next_gameweek, n: int) -> float:
    if hasattr(player, 'get_points_within_n_gameweeks') and next_gameweek is not None:
        try:
            return float(player.get_points_within_n_gameweeks(next_gameweek, n) or 0)
        except Exception:
            pass
    return sum(float(getattr(m, 'points', 0) or 0) for m in (player.future_matches or [])[:n])


def _load_db_players():
    db_handler = None
    try:
        db_handler = DB_Handler()
        return db_handler.get_player_objects()
    except Exception as exc:
        print(f'Could not read players from the database: {exc}')
        return {}
    finally:
        if db_handler is not None:
            try:
                db_handler.close_connection()
            except Exception:
                pass


def _db_player_for(processed, db_by_id, db_by_code):
    found = db_by_id.get(processed.id)
    if found is not None:
        return found
    code = getattr(processed, 'code', None)
    if code is not None:
        return db_by_code.get(code)
    return None


def _print_email_table(db_players, n: int):
    headers = csv_headers(n)
    rows = [player_row(player, n) for player in db_players]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(cells):
        return '  '.join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells))

    print(fmt(headers))
    print('  '.join('-' * w for w in widths))
    for row in rows:
        print(fmt(row))


def _fmt_played_match(match) -> str:
    mins = Runner._match_minutes(match)
    pts = float(getattr(match, 'points', 0) or 0)
    gw = getattr(match, 'gameweek', '?')
    opp = str(getattr(match, 'opponent', '') or '?')
    home = getattr(match, 'home', None)
    if home is None:
        loc = '?'
    else:
        loc = 'H' if home else 'A'
    return f'  GW{str(gw):>2} {loc} vs {opp:<18} {pts:>6.1f} pts  {mins:>5.0f} mins'


def _print_match_block(title: str, matches):
    print(f'\n{title}')
    if not matches:
        print('  (none)')
        return
    for match in matches:
        print(_fmt_played_match(match))
    total_pts = sum(float(getattr(m, 'points', 0) or 0) for m in matches)
    total_mins = sum(Runner._match_minutes(m) for m in matches)
    apps = sum(1 for m in matches if Runner._match_minutes(m) > 0)
    print(
        f'  totals: {total_pts:.1f} pts, {total_mins:.0f} mins, '
        f'{apps}/{len(matches)} appearances'
    )


def print_played_history(processed, runner: Runner):
    current_played = Player.non_carry_matches(processed)
    _print_match_block(
        f'Played matches — current season {runner.current_season.years[0]}/'
        f'{runner.current_season.years[1]}',
        current_played,
    )

    last_years = (SEASON_YR - 1, SEASON_YR)
    last_season = next(
        (s for s in runner.past_seasons if s.years[0] == last_years[0]),
        None,
    )
    last_player = None
    if last_season is not None:
        last_player = last_season.player_code_dict.get(processed.code)
    last_matches = Player.non_carry_matches(last_player) if last_player else []
    _print_match_block(
        f'Played matches — last season {last_years[0]}/{last_years[1]}',
        last_matches,
    )


def _setup_runner() -> Runner:
    """Same as Runner.run through process_player_matches, without learning or persist."""
    runner = Runner()
    print('Setting up players and teams (no live FPL update, no player learning)...')
    runner.current_season = CurrentSeason()
    runner.populate_past_seasons()
    if use_team_expectancies():
        print('Running team expectancies (needed for team_xg / team_xc inputs)...')
        run_team_expectancies(runner.past_seasons, runner.current_season)
    else:
        print('Skipping team expectancies (--no-team-expectancies); xG features may be 0')
    runner.process_player_matches()
    return runner


def main(queries, runner=None):
    if runner is None:
        runner = _setup_runner()

    print('\nResolving players:')
    processed_list = []
    seen_ids = set()
    for query in queries:
        player = resolve_player(query, runner.current_season.players)
        if player is None:
            continue
        if player.id in seen_ids:
            print(f'  (skipping duplicate {player.get_name()})')
            continue
        seen_ids.add(player.id)
        processed_list.append(player)

    if not processed_list:
        raise SystemExit('No matching players')

    print('\nLoading stored predictions from the database...')
    db_players = _load_db_players()
    db_by_id = {p.id: p for p in db_players.values()}
    db_by_code = {
        p.code: p for p in db_players.values() if getattr(p, 'code', None) is not None
    }
    if not db_players:
        print(
            'Database has no players. Predictions table will be empty until '
            'premierlearning persists a run.'
        )

    next_gw = runner.current_season.next_gameweek
    n = POINTS_HORIZON
    paired = []
    for processed in processed_list:
        db_player = _db_player_for(processed, db_by_id, db_by_code)
        pts = _horizon_points(db_player, next_gw, n) if db_player else 0.0
        paired.append((pts, processed, db_player))
    paired.sort(key=lambda item: item[0], reverse=True)

    print(f'\n======== Ranked by predicted points over next {n} GWs (from DB) ========')
    print(f'Next gameweek: {next_gw}')
    print(f'{"#":>3}  {"Player":<36}  {"Pts":>6}  query/web')
    for i, (pts, processed, db_player) in enumerate(paired, start=1):
        src = 'DB' if db_player is not None else 'no DB row'
        print(
            f'{i:>3}  {processed.get_name():<36}  {pts:>6.1f}  '
            f'({_web_name(processed) or processed.team_name}, {src})'
        )

    db_ordered = [db for _, _, db in paired if db is not None]
    print(f'\n======== Predictions (same columns as the Fargate email / top_players.csv) ========')
    if db_ordered:
        _print_email_table(db_ordered, n)
        print(
            '\nPred and Pred mins are per-GW averages over the horizon. '
            'Adj is the minutes-scaled score used by the optimiser. '
            'Prev pp90 is pooled points/90 over the last appearances in the DB.'
        )
    else:
        print('No stored future_matches for these players.')

    print('\nTeam xG is refit locally (player nets are not). Feature values can')
    print('differ slightly from the Fargate run that wrote the DB scores.')
    for _, processed, _ in paired:
        print('\n' + '=' * 72)
        print(f'{processed.get_name()} ({processed.team_name})')
        print('=' * 72)
        print_played_history(processed, runner)
        runner.print_sample_horizon_feature_columns([processed])
        runner.print_sample_minutes_feature_columns([processed])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Print DB predictions, model inputs, and match history for named players.',
    )
    parser.add_argument(
        'players',
        nargs='+',
        help='Player names or FPL web names (e.g. haaland, "bruno g", "igor jesus")',
    )
    parser.add_argument(
        '--out',
        default=OUTPUT_PATH,
        help='Append the player dump to this text file (default: explain_player.txt)',
    )
    args, _unknown = parser.parse_known_args()

    app = Flask(__name__)
    with app.app_context():
        runner = _setup_runner()
        out_file = _open_output(args.out, args.players)
        original = sys.stdout
        sys.stdout = _Tee(original, out_file)
        try:
            main(args.players, runner)
        finally:
            sys.stdout = original
            out_file.close()
            print(f'Wrote player dump to {os.path.abspath(args.out)}')
