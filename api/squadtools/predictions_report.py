"""Player-prediction table for CSV and the Fargate results email."""
import csv
import html
import json
import os

from premierlearning.config_constants import POINTS_HORIZON
from premierlearning.helper_functions import implied_pp90
from premierlearning.player import LAST_5_WINDOW

POS_SHORT = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
CSV_PATH = 'top_players.csv'
SCORELINES_PATH = 'predicted_scorelines.json'


def _match_minutes(match) -> float:
    mins = getattr(match, 'minutes', None)
    if mins is None and getattr(match, 'game_stats', None):
        mins = match.game_stats.get('minutes', 0)
    return float(mins or 0)


def _match_adj(match) -> float:
    return float(getattr(match, 'points', 0) or 0)


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
    """Pooled points per 90 over the last n appearances (minutes > 0), same as model last_5."""
    played = [m for m in (getattr(player, 'matches', None) or []) if _match_minutes(m) > 0][-n:]
    total_mins = sum(_match_minutes(m) for m in played)
    if total_mins <= 0:
        return 0.0
    total_pts = sum(float(getattr(m, 'points', 0) or 0) for m in played)
    return total_pts * 90.0 / total_mins


def _avg_error(matches) -> float:
    if not matches:
        return 0.0
    errs = [float(getattr(m, 'points_error', 0.0) or 0.0) for m in matches]
    return sum(errs) / len(errs) if errs else 0.0


def csv_headers(n: int):
    headers = [
        'Player', 'Team', 'Pos', 'Price',
        'Pred', 'Pred mins', 'Prev pp90', 'Err',
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


def player_row(player, n: int):
    futs = _future_matches(player, n)
    pred_pts = sum(_match_adj(m) for m in futs)
    pred_mins = sum(_match_minutes(m) for m in futs)
    prev_pp90 = _last_n_appearance_pp90(player, LAST_5_WINDOW)
    row = [
        _player_name(player),
        _player_team(player),
        _player_pos(player),
        '%.1f' % _player_price(player),
        '%.1f' % (pred_pts / n if n else 0.0),
        '%.0f' % (pred_mins / n if n else 0.0),
        '%.1f' % prev_pp90,
        '%.2f' % _avg_error(futs),
    ]
    for i in range(n):
        if i < len(futs):
            match = futs[i]
            mins = _match_minutes(match)
            adj = _match_adj(match)
            row.extend([
                str(getattr(match, 'opponent', '') or ''),
                '%.1f' % implied_pp90(adj, mins),
                '%.0f' % mins,
                '%.1f' % adj,
            ])
        else:
            row.extend(['', '', '', ''])
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
            writer.writerow(player_row(player, n))
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
    net = _horizon_adj(player_in, POINTS_HORIZON) - _horizon_adj(player_out, POINTS_HORIZON)
    return {
        'out_name': _player_name(player_out),
        'out_team': _player_team(player_out),
        'out_pos': _player_pos(player_out),
        'out_price': '%.1f' % (float(getattr(transfer, 'selling_price', 0) or _player_price(player_out))),
        'in_name': _player_name(player_in),
        'in_team': _player_team(player_in),
        'in_pos': _player_pos(player_in),
        'in_price': '%.1f' % (float(getattr(transfer, 'purchase_price', 0) or _player_price(player_in))),
        'net_points': '%.1f' % net,
    }


def _esc(value) -> str:
    return html.escape('' if value is None else str(value))


def _load_scorelines(path=SCORELINES_PATH):
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding='utf-8') as file:
            rows = json.load(file)
        return rows if isinstance(rows, list) else []
    except Exception as exc:
        print('Could not read predicted scorelines: %s' % exc)
        return []


def _html_scorelines(rows):
    if not rows:
        return (
            '<p style="margin:24px 0 0;color:#444">No predicted scorelines '
            '(team expectancies did not write predicted_scorelines.json).</p>'
        )
    parts = [
        '<h2 style="margin:24px 0 8px">Predicted scorelines</h2>',
        '<p style="margin:0 0 10px;color:#444">Expected goals for remaining fixtures this season.</p>',
    ]
    gameweek = None
    for idx, row in enumerate(rows):
        gw = row.get('gameweek')
        if gw != gameweek:
            if gameweek is not None:
                parts.append('</tbody></table>')
            gameweek = gw
            parts.append(
                '<h3 style="margin:16px 0 6px">Gameweek %s</h3>'
                '<table cellpadding="4" cellspacing="0" border="1" '
                'style="border-collapse:collapse;font-size:12px;margin:0 0 8px">'
                '<thead><tr style="background:#1f2937;color:#fff">'
                '<th>Home</th><th>xG</th><th></th><th>xG</th><th>Away</th>'
                '</tr></thead><tbody>' % _esc(gw)
            )
        bg = '#f3f4f6' if idx % 2 else '#ffffff'
        parts.append(
            '<tr style="background:%s">'
            '<td>%s</td><td>%.2f</td><td>-</td><td>%.2f</td><td>%s</td>'
            '</tr>' % (
                bg,
                _esc(row.get('home')),
                float(row.get('home_xg') or 0),
                float(row.get('away_xg') or 0),
                _esc(row.get('away')),
            )
        )
    if gameweek is not None:
        parts.append('</tbody></table>')
    return ''.join(parts)


def _text_scorelines(rows):
    if not rows:
        return ['No predicted scorelines file.']
    lines = ['Predicted scorelines', '']
    gameweek = None
    for row in rows:
        gw = row.get('gameweek')
        if gw != gameweek:
            gameweek = gw
            lines.append('Gameweek %s' % gw)
        lines.append(
            '%s %.2f - %.2f %s'
            % (
                row.get('home'),
                float(row.get('home_xg') or 0),
                float(row.get('away_xg') or 0),
                row.get('away'),
            )
        )
    return lines


def build_html_report(opt, applied=False, n=POINTS_HORIZON) -> str:
    gameweek = getattr(getattr(opt, 'squad', None), 'next_gameweek', None) or '?'
    this_week = opt.get_this_week_transfers() if getattr(opt, 'transfers', None) else []
    players = sorted_players(opt.player_dict, gameweek if gameweek != '?' else 1, n)
    status = 'applied' if applied else 'suggested, not applied'

    parts = [
        '<html><body style="font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#111;margin:16px">',
        '<h2 style="margin:0 0 8px">GW %s transfers (%s)</h2>' % (_esc(gameweek), _esc(status)),
    ]
    if this_week:
        parts.append(
            '<table cellpadding="6" cellspacing="0" border="1" '
            'style="border-collapse:collapse;margin:0 0 20px;font-size:13px">'
            '<thead><tr style="background:#1f2937;color:#fff">'
            '<th>Out</th><th>Team</th><th>Pos</th><th>Sell</th>'
            '<th>In</th><th>Team</th><th>Pos</th><th>Buy</th><th>Net pts (5GW)</th>'
            '</tr></thead><tbody>'
        )
        for transfer in this_week:
            row = _fmt_transfer(transfer)
            parts.append(
                '<tr>'
                '<td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
                '<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
                '</tr>' % tuple(_esc(row[k]) for k in (
                    'out_name', 'out_team', 'out_pos', 'out_price',
                    'in_name', 'in_team', 'in_pos', 'in_price', 'net_points',
                ))
            )
        parts.append('</tbody></table>')
    else:
        parts.append('<p>No transfers for this gameweek.</p>')

    if getattr(opt, 'transfers', None):
        scheme = opt.get_transfers()
        if scheme and (len(scheme) > 1 or (scheme and scheme[0].get('transfers') and not this_week)):
            parts.append('<h3 style="margin:16px 0 8px">Multi-week scheme</h3><ul>')
            for entry in scheme:
                transfers = entry.get('transfers') or []
                label = ', '.join(_esc(t) for t in transfers) if transfers else 'none'
                parts.append('<li>GW %s: %s</li>' % (_esc(entry.get('gameweek')), label))
            parts.append('</ul>')

    parts.append(
        '<h2 style="margin:24px 0 8px">All player predictions (%d)</h2>'
        '<p style="margin:0 0 10px;color:#444">Adj is the minutes-scaled score used by the linear optimiser '
        '(pred mins ≥75 → 90, 60–75 → 75–85, below 60 unchanged). '
        'pp90 is the implied points-per-90 behind that adj score. '
        'Prev pp90 is pooled points/90 over the last %d appearances (minutes &gt; 0).</p>' % (len(players), LAST_5_WINDOW)
    )
    parts.append(
        '<table cellpadding="4" cellspacing="0" border="1" '
        'style="border-collapse:collapse;font-size:11px;white-space:nowrap">'
        '<thead>'
        '<tr style="background:#1f2937;color:#fff">'
        '<th rowspan="2">Player</th><th rowspan="2">Team</th><th rowspan="2">Pos</th>'
        '<th rowspan="2">Price</th><th rowspan="2">Pred</th><th rowspan="2">Pred mins</th>'
        '<th rowspan="2">Prev pp90</th><th rowspan="2">Err</th>'
    )
    for i in range(1, n + 1):
        parts.append('<th colspan="4">Match %d</th>' % i)
    parts.append('</tr><tr style="background:#374151;color:#fff">')
    for _ in range(n):
        parts.append('<th>Opp</th><th>pp90</th><th>Mins</th><th>Adj</th>')
    parts.append('</tr></thead><tbody>')

    for idx, player in enumerate(players):
        bg = '#f3f4f6' if idx % 2 else '#ffffff'
        cells = player_row(player, n)
        tds = ''.join('<td>%s</td>' % _esc(c) for c in cells)
        parts.append('<tr style="background:%s">%s</tr>' % (bg, tds))
    parts.append('</tbody></table>')
    parts.append(_html_scorelines(_load_scorelines()))
    parts.append('</body></html>')
    return ''.join(parts)


def build_text_report(opt, applied=False, n=POINTS_HORIZON) -> str:
    gameweek = getattr(getattr(opt, 'squad', None), 'next_gameweek', None) or '?'
    this_week = opt.get_this_week_transfers() if getattr(opt, 'transfers', None) else []
    status = 'applied' if applied else 'suggested, not applied'
    lines = ['GW %s transfers (%s)' % (gameweek, status), '']
    if this_week:
        for transfer in this_week:
            row = _fmt_transfer(transfer)
            lines.append(
                '%s (%s %s, sell %s) -> %s (%s %s, buy %s) net %s'
                % (
                    row['out_name'], row['out_team'], row['out_pos'], row['out_price'],
                    row['in_name'], row['in_team'], row['in_pos'], row['in_price'],
                    row['net_points'],
                )
            )
    else:
        lines.append('No transfers for this gameweek.')
    lines.append('')
    lines.extend(_text_scorelines(_load_scorelines()))
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
