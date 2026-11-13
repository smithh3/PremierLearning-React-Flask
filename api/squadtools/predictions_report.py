"""Player-prediction table for CSV and the Fargate results email."""
import csv
import html
import os

from premierlearning.config_constants import POINTS_HORIZON
from premierlearning.helper_functions import implied_pp90
from premierlearning.player import LAST_5_WINDOW, is_form_appearance

POS_SHORT = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
CSV_PATH = 'top_players.csv'


def _match_minutes(match) -> float:
    """Same minutes source as the model: game_stats when present, else match.minutes."""
    stats = getattr(match, 'game_stats', None)
    if stats:
        try:
            return float(stats.get('minutes', 0) or 0)
        except (TypeError, ValueError):
            return 0.0
    return float(getattr(match, 'minutes', 0) or 0)


def _match_points(match) -> float:
    """FPL total_points from game_stats when present (model input), else match.points."""
    stats = getattr(match, 'game_stats', None)
    if stats and 'total_points' in stats:
        try:
            return float(stats.get('total_points', 0) or 0)
        except (TypeError, ValueError):
            return 0.0
    return float(getattr(match, 'points', 0) or 0)


def _match_adj(match) -> float:
    from squadtools.availability import match_optimiser_points
    return match_optimiser_points(match)


BASIS_FINAL = 'final'
BASIS_RAW = 'raw 1GW'


def _adj_minutes(match) -> float:
    """Minutes the optimiser scores this fixture on (same order as match_optimiser_points)."""
    stats = getattr(match, 'game_stats', None) or {}
    try:
        minutes = float(stats.get('minutes', 0) or 0)
    except (TypeError, ValueError):
        minutes = 0.0
    if minutes <= 0:
        minutes = float(getattr(match, 'minutes', 0) or 0)
    return minutes


def _horizon_scale(player, n: int):
    """Factor the 5GW model applied to this player's single-week points.

    One factor covers the whole window, so the first fixture carrying one speaks
    for the player. None when nothing was rescaled.
    """
    for match in _future_matches(player, n):
        scale = getattr(match, 'horizon_scale', None)
        if scale is not None:
            return float(scale)
    return None


def _match_scores(match, raw: bool):
    """(pp90, adj) for one fixture, either as scored now or from the 1GW model alone.

    The raw pair goes through the same minutes mapping and chance-to-play as the
    final one, so the two differ only by the horizon scale factor.
    """
    if not raw:
        adj = _match_adj(match)
        pp90 = getattr(match, 'predicted_pp90', None)
        if pp90 is None:
            pp90 = implied_pp90(adj, _adj_minutes(match))
        return float(pp90 or 0.0), adj
    from squadtools.availability import LIVE_MINUTES_SCALE, optimiser_points
    points = getattr(match, 'h1_points', None)
    if points is None:
        return None, None
    minutes = _adj_minutes(match)
    pp90 = implied_pp90(float(points), minutes)
    return pp90, optimiser_points(
        pp90,
        minutes,
        getattr(match, 'chance_of_playing', None),
        # match_optimiser_points defaults to the live mapping; without it the raw
        # row would miss the starter uplift and the two rows would not divide out.
        minutes_scale=LIVE_MINUTES_SCALE,
    )


def _player_name(player) -> str:
    if hasattr(player, 'get_name'):
        return player.get_name()
    first = getattr(player, 'first_name', '') or ''
    last = getattr(player, 'last_name', '') or ''
    return ('%s %s' % (first, last)).strip()


def _player_team(player) -> str:
    team = getattr(player, 'team_name', None)
    if team:
        return str(team)
    team_obj = getattr(player, 'team', None)
    if team_obj is not None:
        return str(getattr(team_obj, 'name', '') or '')
    return ''


def _player_pos(player) -> str:
    return POS_SHORT.get(int(getattr(player, 'position', 0) or 0), '')


def _player_price(player) -> float:
    return float(getattr(player, 'current_cost', 0) or 0)


def _future_matches(player, n: int):
    return list(getattr(player, 'future_matches', None) or [])[:n]


def _last_n_appearance_pp90(player, n: int = LAST_5_WINDOW) -> float:
    """Pooled points per 90 over the last n form appearances.

    Same construction as model ``points_last_3_per_90_excl_null``: appearances
    with minutes >= FORM_MIN_MINUTES, including last-season carry when those
    rows are on ``player.matches``. Prefer the already-built snapshot when
    present (in-memory after learning).
    """
    past = getattr(player, 'past_stats', None) or {}
    window = past.get('total_points') if isinstance(past, dict) else None
    if isinstance(window, dict) and n == LAST_5_WINDOW:
        try:
            stored = float(window.get('last_5_avg') or 0)
        except (TypeError, ValueError):
            stored = 0.0
        if stored:
            return stored
    played = [
        m for m in (getattr(player, 'matches', None) or [])
        if is_form_appearance(_match_minutes(m))
    ][-n:]
    total_mins = sum(_match_minutes(m) for m in played)
    if total_mins <= 0:
        return 0.0
    total_pts = sum(_match_points(m) for m in played)
    return total_pts * 90.0 / total_mins


def _avg_error(matches) -> float:
    if not matches:
        return 0.0
    errs = [float(getattr(m, 'points_error', 0.0) or 0.0) for m in matches]
    return sum(errs) / len(errs) if errs else 0.0


def csv_headers(n: int):
    headers = [
        'Player', 'Team', 'Pos', 'Price',
        'Basis', 'Pred', 'Pred mins', 'Prev pp90', 'Err', 'Scale',
    ]
    for i in range(1, n + 1):
        prefix = 'M%d' % i
        headers.extend([
            '%s opp' % prefix,
            '%s pp90' % prefix,
            '%s mins' % prefix,
            '%s adj' % prefix,
        ])
    return headers


def rank_players(players, n: int):
    ranked = list(players or [])
    ranked.sort(
        key=lambda p: (
            float(getattr(p, 'points_over_next_5_gameweeks', 0) or 0)
            or sum(_match_adj(m) for m in _future_matches(p, n))
        ),
        reverse=True,
    )
    return ranked


def format_predictions_table(players, n: int, limit=None) -> str:
    """Plain-text Fargate email / top_players.csv columns."""
    ranked = rank_players(players, n)
    if limit is not None:
        ranked = ranked[: int(limit)]
    headers = csv_headers(n)
    rows = [row for player in ranked for row in player_rows(player, n)]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(cells):
        return '  '.join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [fmt(headers), '  '.join('-' * w for w in widths)]
    lines.extend(fmt(row) for row in rows)
    return '\n'.join(lines)


def player_rows(player, n: int):
    """Two rows: the scores the optimiser uses, then the 1GW model on its own."""
    return [player_row(player, n), player_row(player, n, raw=True)]


def player_row(player, n: int, raw: bool = False):
    futs = _future_matches(player, n)
    scored = [_match_scores(match, raw) for match in futs]
    pred_pts = sum(adj for _pp90, adj in scored if adj is not None)
    scored_any = any(adj is not None for _pp90, adj in scored)
    pred_mins = sum(_match_minutes(m) for m in futs)
    scale = _horizon_scale(player, n)
    prev_pp90 = _last_n_appearance_pp90(player, LAST_5_WINDOW)
    row = [
        _player_name(player),
        _player_team(player),
        _player_pos(player),
        '%.1f' % _player_price(player),
        BASIS_RAW if raw else BASIS_FINAL,
        '%.1f' % (pred_pts / n if n else 0.0) if scored_any else '',
        '%.0f' % (pred_mins / n if n else 0.0),
        '%.1f' % prev_pp90,
        '%.2f' % _avg_error(futs),
        '' if scale is None else '%.2f' % scale,
    ]
    for i in range(n):
        if i >= len(futs):
            row.extend(['', '', '', ''])
            continue
        match = futs[i]
        pp90, adj = scored[i]
        row.extend([
            str(getattr(match, 'opponent', '') or ''),
            '' if pp90 is None else '%.1f' % pp90,
            '%.0f' % _match_minutes(match),
            '' if adj is None else '%.1f' % adj,
        ])
    return row


def sorted_players(player_dict, next_gameweek, n: int):
    players = list((player_dict or {}).values())
    players.sort(
        key=lambda p: p.get_points_within_n_gameweeks(next_gameweek, n)
        if hasattr(p, 'get_points_within_n_gameweeks')
        else sum(_match_adj(m) for m in _future_matches(p, n)),
        reverse=True,
    )
    return players


def write_predictions_csv(player_dict, next_gameweek, path=CSV_PATH, n=POINTS_HORIZON):
    players = sorted_players(player_dict, next_gameweek, n)
    with open(path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers(n))
        for player in players:
            writer.writerows(player_rows(player, n))
    print('Wrote %d player predictions to %s' % (len(players), path))
    return players


def _horizon_adj(player, n: int) -> float:
    if hasattr(player, 'get_points_within_n_gameweeks'):
        gws = [m.gameweek for m in (getattr(player, 'future_matches', None) or []) if getattr(m, 'gameweek', None) is not None]
        next_gw = min(gws) if gws else 1
        try:
            return float(player.get_points_within_n_gameweeks(next_gw, n) or 0)
        except Exception:
            pass
    return sum(_match_adj(m) for m in _future_matches(player, n))


def _fmt_transfer(transfer) -> dict:
    player_out = transfer.player_out
    player_in = transfer.player_in
    out_pts = _horizon_adj(player_out, POINTS_HORIZON)
    in_pts = _horizon_adj(player_in, POINTS_HORIZON)
    return {
        'out_name': _player_name(player_out),
        'out_team': _player_team(player_out),
        'out_pos': _player_pos(player_out),
        'out_price': '%.1f' % (float(getattr(transfer, 'selling_price', 0) or _player_price(player_out))),
        'in_name': _player_name(player_in),
        'in_team': _player_team(player_in),
        'in_pos': _player_pos(player_in),
        'in_price': '%.1f' % (float(getattr(transfer, 'purchase_price', 0) or _player_price(player_in))),
        'out_points': '%.1f' % out_pts,
        'in_points': '%.1f' % in_pts,
        'net_points': '%.1f' % (in_pts - out_pts),
    }


def _this_week_scores(opt) -> dict:
    scores = getattr(opt, 'this_week_scores', None) or {}
    if scores:
        return scores
    horizon = getattr(opt, 'horizon_plan', None) or []
    if horizon and 'horizon_gain' in horizon[0]:
        row = horizon[0]
        return {
            'hold_current': float(row.get('hold_current') or 0),
            'selected_current': float(row.get('selected_current') or 0),
            'hold_horizon': float(row.get('hold_horizon') or 0),
            'selected_horizon': float(row.get('selected_horizon') or 0),
            'current_gain': float(row.get('current_gain') or 0),
            'horizon_gain': float(row.get('horizon_gain') or 0),
        }
    return {}


def _esc(value) -> str:
    return html.escape('' if value is None else str(value))


def _chip_rows(opt):
    calendar = getattr(opt, 'chip_calendar', None)
    if calendar is None:
        return []
    rows = []
    mapping = (
        ('Wildcard', calendar.wildcards),
        ('Free Hit', calendar.free_hits),
        ('Bench Boost', calendar.bench_boosts),
        ('Triple Captain', calendar.triple_captains),
    )
    by_week = {}
    for label, weeks in mapping:
        for gameweek in weeks:
            by_week[int(gameweek)] = label
    for gameweek in sorted(by_week):
        rows.append({'gameweek': gameweek, 'chip': by_week[gameweek]})
    return rows


def _chip_reason_note(opt) -> str:
    reason = getattr(opt, 'chip_reason', None) or {}
    notes = []
    if reason.get('forced'):
        notes.append('Forced remaining chips before the half ends.')
    recovery = reason.get('recovery') or {}
    if recovery:
        if recovery.get('chose_free_hit'):
            notes.append('Same-week FH+WC: kept Free Hit because the squad recovers.')
        elif recovery.get('gameweek'):
            notes.append('Same-week FH+WC: wildcarded because recovery failed.')
    if reason.get('bench_boost_after_wildcard'):
        notes.append('Bench Boost is the week after Wildcard.')
    if reason.get('wildcard_gameweek'):
        notes.append('Wildcard GW %s.' % reason.get('wildcard_gameweek'))
    if reason.get('free_hit_gameweek'):
        notes.append('Free Hit GW %s.' % reason.get('free_hit_gameweek'))
    return ' '.join(notes)


def _html_chip_strategy(opt):
    rows = _chip_rows(opt)
    parts = ['<h2 style="margin:0 0 8px">Chip strategy</h2>']
    if not rows:
        parts.append('<p>No chips planned.</p>')
        return ''.join(parts)
    parts.append(
        '<table cellpadding="6" cellspacing="0" border="1" '
        'style="border-collapse:collapse;margin:0 0 12px;font-size:13px">'
        '<thead><tr style="background:#1f2937;color:#fff">'
        '<th>Gameweek</th><th>Chip</th></tr></thead><tbody>'
    )
    for row in rows:
        parts.append(
            '<tr><td>GW %s</td><td>%s</td></tr>'
            % (_esc(row['gameweek']), _esc(row['chip']))
        )
    parts.append('</tbody></table>')
    note = _chip_reason_note(opt)
    if note:
        parts.append('<p style="margin:0 0 16px;color:#444">%s</p>' % _esc(note))
    return ''.join(parts)


def _text_chip_strategy(opt):
    rows = _chip_rows(opt)
    lines = ['Chip strategy', '']
    if not rows:
        lines.append('No chips planned.')
    else:
        for row in rows:
            lines.append('GW %s: %s' % (row['gameweek'], row['chip']))
        note = _chip_reason_note(opt)
        if note:
            lines.append(note)
    return lines


def _html_transfer_strategy(opt, this_week):
    parts = ['<h2 style="margin:16px 0 8px">Transfer strategy</h2>']
    recommended = getattr(opt, 'recommended_squad', None) or []
    if recommended and getattr(opt, 'squad', None) and getattr(opt.squad, 'built_from_scratch', False):
        parts.append(
            '<p style="margin:0 0 10px">Recommended 15-man squad:</p><p>%s</p>'
            % _esc(', '.join(recommended))
        )
    horizon = getattr(opt, 'horizon_plan', None)
    scheme = None
    if horizon:
        scheme = horizon
    elif getattr(opt, 'transfers', None):
        scheme = opt.get_transfers()
    if not scheme:
        if this_week:
            parts.append('<p>See this gameweek\'s transfers below.</p>')
        else:
            parts.append('<p>No transfers planned.</p>')
        return ''.join(parts)
    parts.append(
        '<table cellpadding="6" cellspacing="0" border="1" '
        'style="border-collapse:collapse;margin:0 0 16px;font-size:13px">'
        '<thead><tr style="background:#1f2937;color:#fff">'
        '<th>Gameweek</th><th>Chip</th><th>Transfers</th></tr></thead><tbody>'
    )
    for entry in scheme:
        transfers = entry.get('transfers') or []
        if transfers and hasattr(transfers[0], 'player_out'):
            label = ', '.join(_esc(str(item)) for item in transfers)
        else:
            label = ', '.join(_esc(item) for item in transfers) if transfers else 'none'
        chip = entry.get('chip') or ''
        chip_label = {
            'wildcard': 'Wildcard',
            'free_hit': 'Free Hit',
            'bench_boost': 'Bench Boost',
            'triple_captain': 'Triple Captain',
        }.get(chip, chip) if chip else ''
        parts.append(
            '<tr><td>GW %s</td><td>%s</td><td>%s</td></tr>'
            % (_esc(entry.get('gameweek')), _esc(chip_label), label)
        )
    parts.append('</tbody></table>')
    return ''.join(parts)


def _text_squad_gain(opt) -> list:
    scores = _this_week_scores(opt)
    if not scores:
        return []
    return [
        'XI + captain (what the optimiser scores, not 15-man player totals):',
        '  this GW: hold %.1f -> %.1f (Δ %+.1f)' % (
            scores.get('hold_current') or 0,
            scores.get('selected_current') or 0,
            scores.get('current_gain') or 0,
        ),
        '  %dGW horizon: hold %.1f -> %.1f (Δ %+.1f)' % (
            POINTS_HORIZON,
            scores.get('hold_horizon') or 0,
            scores.get('selected_horizon') or 0,
            scores.get('horizon_gain') or 0,
        ),
        '',
    ]


def _html_squad_gain(opt) -> str:
    scores = _this_week_scores(opt)
    if not scores:
        return ''
    return (
        '<p style="margin:0 0 10px;color:#444">The optimiser scores starting XI + captain, '
        'not the sum of all 15 players. Per-row &quot;Player 5GW Δ&quot; is that player\'s '
        'raw next-%dGW points in minus out, paired by position — it can be negative while '
        'the squad XI still gains (bench, captain, or a cheaper starter).</p>'
        '<p style="margin:0 0 12px"><b>XI + captain this GW:</b> hold %.1f → %.1f (Δ %+.1f). '
        '<b>%dGW horizon:</b> hold %.1f → %.1f (Δ %+.1f).</p>'
        % (
            POINTS_HORIZON,
            scores.get('hold_current') or 0,
            scores.get('selected_current') or 0,
            scores.get('current_gain') or 0,
            POINTS_HORIZON,
            scores.get('hold_horizon') or 0,
            scores.get('selected_horizon') or 0,
            scores.get('horizon_gain') or 0,
        )
    )


def _text_transfer_strategy(opt, this_week):
    lines = ['Transfer strategy', '']
    recommended = getattr(opt, 'recommended_squad', None) or []
    if recommended and getattr(opt, 'squad', None) and getattr(opt.squad, 'built_from_scratch', False):
        lines.append('Recommended squad: %s' % ', '.join(recommended))
        lines.append('')
    horizon = getattr(opt, 'horizon_plan', None)
    scheme = horizon if horizon else (
        opt.get_transfers() if getattr(opt, 'transfers', None) else []
    )
    if not scheme:
        if not this_week:
            lines.append('No transfers planned.')
        return lines
    chip_labels = {
        'wildcard': 'Wildcard',
        'free_hit': 'Free Hit',
        'bench_boost': 'Bench Boost',
        'triple_captain': 'Triple Captain',
    }
    for entry in scheme:
        transfers = entry.get('transfers') or []
        if transfers and hasattr(transfers[0], 'player_out'):
            label = ', '.join(str(item) for item in transfers)
        else:
            label = ', '.join(str(item) for item in transfers) if transfers else 'none'
        chip = entry.get('chip')
        chip_label = chip_labels.get(chip, chip) if chip else ''
        suffix = ' [%s]' % chip_label if chip_label else ''
        lines.append('GW %s%s: %s' % (entry.get('gameweek'), suffix, label))
    return lines


def build_html_report(opt, applied=False, n=POINTS_HORIZON) -> str:
    gameweek = getattr(getattr(opt, 'squad', None), 'next_gameweek', None) or '?'
    this_week = opt.get_this_week_transfers() if getattr(opt, 'transfers', None) else []
    players = sorted_players(opt.player_dict, gameweek if gameweek != '?' else 1, n)
    status = 'applied' if applied else 'suggested, not applied'

    parts = [
        '<html><body style="font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#111;margin:16px">',
        '<h2 style="margin:0 0 8px">GW %s (%s)</h2>' % (_esc(gameweek), _esc(status)),
        _html_chip_strategy(opt),
        _html_transfer_strategy(opt, this_week),
        '<h2 style="margin:16px 0 8px">This gameweek transfers</h2>',
        _html_squad_gain(opt),
    ]
    if this_week:
        parts.append(
            '<table cellpadding="6" cellspacing="0" border="1" '
            'style="border-collapse:collapse;margin:0 0 20px;font-size:13px">'
            '<thead><tr style="background:#1f2937;color:#fff">'
            '<th>Out</th><th>Team</th><th>Pos</th><th>Sell</th>'
            '<th>In</th><th>Team</th><th>Pos</th><th>Buy</th>'
            '<th>Out 5GW</th><th>In 5GW</th><th>Player 5GW Δ</th>'
            '</tr></thead><tbody>'
        )
        for transfer in this_week:
            row = _fmt_transfer(transfer)
            parts.append(
                '<tr>'
                '<td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
                '<td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
                '<td>%s</td><td>%s</td><td>%s</td>'
                '</tr>' % tuple(_esc(row[k]) for k in (
                    'out_name', 'out_team', 'out_pos', 'out_price',
                    'in_name', 'in_team', 'in_pos', 'in_price',
                    'out_points', 'in_points', 'net_points',
                ))
            )
        parts.append('</tbody></table>')
    else:
        parts.append('<p>No transfers for this gameweek.</p>')

    shown = relevant_players(opt, players, n)
    bands = uses_minutes_bands(shown)
    gameweeks = window_gameweeks(shown, gameweek, n)
    parts.append(
        '<h2 style="margin:24px 0 8px">Player predictions (%d of %d)</h2>'
        '<p style="margin:0 0 10px;color:#444">Your squad, this week\'s and planned '
        'transfer players, and the top players in each position over the next %d gameweeks. '
        'Anyone else predicted under %.0f point%s over that window is left out.</p>'
        '<p style="margin:0 0 10px;color:#444">%s %s Prev pp90 is recent form: points per 90 '
        'over the last %d appearances.</p>'
        % (
            len(shown), len(players), n, EMAIL_MIN_POINTS,
            '' if EMAIL_MIN_POINTS == 1 else 's',
            _points_note(shown),
            (
                '<b>Mins</b> is the minutes-bands head: the chance, in %, of 0 / 1-59 / '
                '60+ minutes in that match.'
                if bands else '<b>Mins</b> is the predicted minutes.'
            ),
            LAST_5_WINDOW,
        )
    )
    parts.append(
        '<table cellpadding="4" cellspacing="0" border="1" '
        'style="border-collapse:collapse;font-size:11px;white-space:nowrap">'
        '<thead>'
        '<tr style="background:#1f2937;color:#fff">'
        '<th rowspan="2">Player</th><th rowspan="2">Team</th><th rowspan="2">Pos</th>'
        '<th rowspan="2">Price</th><th rowspan="2">Yours</th>'
        '<th rowspan="2">%dGW pts</th><th rowspan="2">Prev pp90</th>' % n
    )
    for week in gameweeks:
        parts.append('<th colspan="3">GW%d</th>' % week)
    parts.append('</tr><tr style="background:#374151;color:#fff">')
    for _ in range(n):
        parts.append('<th>Opp</th><th>Pts</th><th>%s</th>' % ('Mins 0/1-59/60+' if bands else 'Mins'))
    parts.append('</tr></thead><tbody>')
    tags = player_tags(opt)
    last_position = None
    for idx, player in enumerate(shown):
        position = _player_pos(player)
        if position != last_position:
            parts.append(
                '<tr style="background:#e5e7eb"><td colspan="%d"><b>%s</b></td></tr>'
                % (7 + 3 * n, _esc(POSITION_TITLES.get(position, position)))
            )
            last_position = position
        tag = tags.get(_player_id(player), '')
        style = 'background:%s' % ('#fef9c3' if tag else ('#f3f4f6' if idx % 2 else '#ffffff'))
        cells = email_player_row(player, n, tag, gameweeks=gameweeks)
        tds = ''.join('<td>%s</td>' % _esc(c) for c in cells)
        parts.append('<tr style="%s">%s</tr>' % (style, tds))
    parts.append('</tbody></table>')
    parts.append('</body></html>')
    return ''.join(parts)


# Players shown in the email besides your squad and transfer players.
EMAIL_TOP_PER_POSITION = {1: 4, 2: 8, 3: 10, 4: 6}
# Points over the window below which an unowned player is left out.
EMAIL_MIN_POINTS = 1.0
POSITION_TITLES = {'GKP': 'Goalkeepers', 'DEF': 'Defenders', 'MID': 'Midfielders', 'FWD': 'Forwards'}
_POSITION_ORDER = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}


def _player_id(player):
    try:
        return int(getattr(player, 'id', 0) or 0)
    except (TypeError, ValueError):
        return 0


def player_tags(opt) -> dict:
    """player id -> 'Yours', 'Buy', 'Sell' or 'Plan' (a later planned transfer)."""
    tags = {}
    squad = getattr(getattr(opt, 'squad', None), 'squad', None) or []
    for player in squad:
        tags[_player_id(player)] = 'Yours'
    for entry in getattr(opt, 'horizon_plan', None) or []:
        for transfer in entry.get('transfers') or []:
            for side in ('player_in', 'player_out'):
                player = getattr(transfer, side, None)
                if player is not None and _player_id(player) not in tags:
                    tags[_player_id(player)] = 'Plan'
    this_week = opt.get_this_week_transfers() if getattr(opt, 'transfers', None) else []
    for transfer in this_week or []:
        if getattr(transfer, 'player_in', None) is not None:
            tags[_player_id(transfer.player_in)] = 'Buy'
        if getattr(transfer, 'player_out', None) is not None:
            tags[_player_id(transfer.player_out)] = 'Sell'
    tags.pop(0, None)
    return tags


def relevant_players(opt, players, n: int):
    """Squad and transfer players, plus the top few per position, by position then points."""
    tags = player_tags(opt)
    points = {id(player): _horizon_adj(player, n) for player in players}
    kept = []
    taken = {}
    for player in sorted(players, key=lambda p: points[id(p)], reverse=True):
        position = _player_pos(player)
        limit = EMAIL_TOP_PER_POSITION.get(_POSITION_ORDER.get(position, 0), 0)
        if _player_id(player) in tags:
            kept.append(player)
            continue
        if points[id(player)] < EMAIL_MIN_POINTS or taken.get(position, 0) >= limit:
            continue
        taken[position] = taken.get(position, 0) + 1
        kept.append(player)
    kept.sort(key=lambda p: (_POSITION_ORDER.get(_player_pos(p), 9), -points[id(p)]))
    return kept


def _match_bands(match):
    bands = getattr(match, 'minutes_bands', None)
    if not bands or bands[0] is None:
        return None
    return tuple(float(value) for value in bands)


def uses_minutes_bands(players) -> bool:
    return any(
        _match_bands(match) is not None
        for player in players or []
        for match in (getattr(player, 'future_matches', None) or [])
    )


def _points_note(players) -> str:
    raw = any(
        getattr(match, 'points_basis', None) == 'raw'
        for player in players or []
        for match in (getattr(player, 'future_matches', None) or [])
    )
    from premierlearning.live_model import lightgbm_version, uses_lightgbm

    if raw and uses_lightgbm():
        version = lightgbm_version()
        news = (', and FPL\'s injury and suspension news' if version == 'v6' else '')
        return (
            '<b>Pts</b> is LightGBM %s\'s expected points for the match (gradient-boosted '
            'trees on the player\'s and both teams\' recent form%s). It already includes the '
            'risk of not playing, so the optimiser uses it as it is.' % (version, news)
        )
    if raw:
        return (
            '<b>Pts</b> is the blend model\'s expected points for the match (75% v2_log '
            'with the minutes-bands head, 25% v2_log with its 5-gameweek net). It already '
            'includes the risk of not playing, so the optimiser uses it as it is.'
        )
    return (
        '<b>Pts</b> is the score the optimiser uses: predicted points per 90 times '
        'effective minutes, times chance of playing when known.'
    )


BLANK = '-'


def _match_gameweek(match):
    try:
        return int(getattr(match, 'gameweek', None))
    except (TypeError, ValueError):
        return None


def window_gameweeks(players, start, n: int):
    """The n gameweeks the table shows: from the next deadline (or the earliest fixture)."""
    try:
        first = int(start)
    except (TypeError, ValueError):
        weeks = [
            week for player in players or []
            for week in (_match_gameweek(m) for m in (getattr(player, 'future_matches', None) or []))
            if week is not None
        ]
        first = min(weeks) if weeks else 1
    return list(range(first, first + int(n)))


def _minutes_cell(match) -> str:
    bands = _match_bands(match)
    if bands is None:
        return '%.0f' % _match_minutes(match)
    return '/'.join('%.0f' % (100.0 * value) for value in bands)


def email_player_row(player, n: int, tag='', gameweeks=None):
    """One row per player: who, then opponent, points and minutes in each gameweek.

    With gameweeks, columns are those gameweeks: a double shows both fixtures
    (opponents and minutes joined with " + ", points summed) and a blank shows
    "-". Without, columns are the player's next n fixtures (the old layout).
    """
    row = [
        _player_name(player),
        _player_team(player),
        _player_pos(player),
        '%.1f' % _player_price(player),
        tag,
        '%.1f' % _horizon_adj(player, n),
        '%.1f' % _last_n_appearance_pp90(player, LAST_5_WINDOW),
    ]
    if gameweeks is None:
        futs = _future_matches(player, n)
        groups = [[futs[i]] if i < len(futs) else [] for i in range(n)]
    else:
        futs = list(getattr(player, 'future_matches', None) or [])
        groups = [[m for m in futs if _match_gameweek(m) == int(week)] for week in gameweeks]
    for matches in groups:
        if not matches:
            row.extend([BLANK, '', ''] if gameweeks is not None else ['', '', ''])
            continue
        row.extend([
            ' + '.join(str(getattr(m, 'opponent', '') or '') for m in matches),
            '%.1f' % sum(_match_adj(m) for m in matches),
            ' + '.join(_minutes_cell(m) for m in matches),
        ])
    return row


def build_text_report(opt, applied=False, n=POINTS_HORIZON) -> str:
    gameweek = getattr(getattr(opt, 'squad', None), 'next_gameweek', None) or '?'
    this_week = opt.get_this_week_transfers() if getattr(opt, 'transfers', None) else []
    status = 'applied' if applied else 'suggested, not applied'
    lines = ['GW %s transfers (%s)' % (gameweek, status), '']
    lines.extend(_text_chip_strategy(opt))
    lines.append('')
    lines.extend(_text_transfer_strategy(opt, this_week))
    lines.append('')
    lines.extend(_text_squad_gain(opt))
    if this_week:
        for transfer in this_week:
            row = _fmt_transfer(transfer)
            lines.append(
                '%s (%s %s, sell %s, 5GW %s) -> %s (%s %s, buy %s, 5GW %s) player Δ %s'
                % (
                    row['out_name'], row['out_team'], row['out_pos'], row['out_price'],
                    row['out_points'],
                    row['in_name'], row['in_team'], row['in_pos'], row['in_price'],
                    row['in_points'],
                    row['net_points'],
                )
            )
    else:
        lines.append('No transfers for this gameweek.')
    return '\n'.join(lines)


def send_predictions_email(opt, applied=False, n=POINTS_HORIZON):
    to_addr = (os.environ.get('NOTIFY_EMAIL') or '').strip()
    if not to_addr:
        print('NOTIFY_EMAIL is not set; skipping predictions email')
        return
    from_addr = (os.environ.get('SES_FROM_EMAIL') or to_addr).strip()
    region = (
        os.environ.get('AWS_REGION')
        or os.environ.get('AWS_DEFAULT_REGION')
        or 'eu-west-2'
    )
    gameweek = getattr(getattr(opt, 'squad', None), 'next_gameweek', None) or '?'
    status = 'applied' if applied else 'suggested, not applied'
    subject = '[Premier Learning] GW%s predictions (%s)' % (gameweek, status)

    try:
        html_body = build_html_report(opt, applied=applied, n=n)
        text_body = build_text_report(opt, applied=applied, n=n)

        import boto3
        response = boto3.client('ses', region_name=region).send_email(
            Source='Premier Learning <%s>' % from_addr,
            Destination={'ToAddresses': [to_addr]},
            Message={
                'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                'Body': {
                    'Text': {'Data': text_body, 'Charset': 'UTF-8'},
                    'Html': {'Data': html_body, 'Charset': 'UTF-8'},
                },
            },
        )
        print(
            'Sent predictions email MessageId=%s to %s (GW %s, %s)'
            % (response.get('MessageId'), to_addr, gameweek, status)
        )
    except Exception as exc:
        print('Failed to send predictions email: %s' % exc)
