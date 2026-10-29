"""Run premierlearning then squadtools in one process.

Environment:
  RUN_LEARNING=1          Train and persist players (default 0).
  APPLY_TRANSFERS=true    Submit transfers + starting XI to FPL.
  APPLY_TEAM_SELECTION=true  Submit captain/starters only.
  SUGGEST_TRANSFERS=true  Print suggestions without submitting (default if no apply flag).
  PREMIERLEARNING_ARGS    Extra argv for learning, e.g. "--horizon-reps 5"
                          (horizon-N and minutes average 5 fits; h1 points stay 1).
                          Add --horizon-only to skip minutes. Add --h1-poisson for
                          Poisson h1 + minutes/90 weights and no horizon net
                          (5 h1 fits). Pipeline/Fargate always prepends --update
                          so injury news is pulled live; pass --no-update to keep
                          the files copied into the image.
  APPLY_WITHIN_HOURS      If set, refuse apply unless the next GW deadline is within this many hours.
  LOGIN_ONLY=1            Skip learning and squadtools; only verify FPL login.
  FPL_STATUS_CHECK=1      Print HTTP/Playwright status for FPL URLs, then exit.
  SKIP_FPL_EGRESS_CHECK=1 Skip the runtime egress-IP / FPL 200 probe.
  LEARNING_DONE_QUEUE_URL If set, publish an SQS message after a successful persist (phase 2).
  APPLY_ON_DEADLINE_DAY=true  If set (default on Fargate), apply transfers when today is the next GW deadline (UK).
  PIPELINE_SCHEDULE_NAME  If set, move that EventBridge schedule to two hours before the following deadline.
  SCHEDULE_HOURS_BEFORE_DEADLINE  Hours before that deadline to book (default 2).
  SKIP_BENCH_FILL=true        Skip the second-pass bench refill (rotating XI only).
                              Default is first two outfield bench slots 0.25, other slots 0.
  NOTIFY_EMAIL            If set, email HTML player predictions (and suggested transfers) via SES.
  SES_FROM_EMAIL          SES From address (defaults to NOTIFY_EMAIL; must be a verified identity).
"""
import os
import shlex
import sys
from datetime import datetime, timedelta, timezone

from flask import Flask


def _flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


def _discover_egress_ip():
    import requests
    for url in ('https://checkip.amazonaws.com', 'https://ifconfig.me'):
        try:
            ip = (requests.get(url, timeout=10).text or '').strip().split()[0]
            if ip:
                return ip
        except Exception as exc:
            print('egress_ip lookup failed url=%s err=%s' % (url, exc))
    return 'unknown'


def _fpl_egress_preflight():
    """Discover the NAT public IP FPL will see, and refuse to continue on a 403.

    Fargate never uses a hardcoded public IP. Private-subnet default route points
    at the NAT ENI; AWS may assign a new public address if that instance is
    stopped and started. This probe reads whatever we egress as right now.
    """
    if _flag('SKIP_FPL_EGRESS_CHECK'):
        print('Skipping FPL egress check (SKIP_FPL_EGRESS_CHECK=1)')
        return
    import requests
    from premierlearning.fpl_apis import FPL_USER_AGENT, GENERAL

    egress_ip = _discover_egress_ip()
    print('egress_ip=%s (discovered at runtime, not hardcoded)' % egress_ip)
    try:
        response = requests.get(
            GENERAL, timeout=30, headers={'User-Agent': FPL_USER_AGENT},
        )
    except Exception as exc:
        raise SystemExit('FPL bootstrap unreachable from egress_ip=%s: %s' % (egress_ip, exc))
    print(
        'fpl_bootstrap=%s bytes=%s egress_ip=%s'
        % (response.status_code, len(response.content or b''), egress_ip)
    )
    if response.status_code == 403:
        raise SystemExit(
            'FPL returned 403 from egress_ip=%s. Stop/start the NAT EC2 to get a new '
            'auto-assigned public IP, then retry. Do not attach an Elastic IP.'
            % egress_ip
        )
    if response.status_code != 200:
        raise SystemExit(
            'FPL bootstrap HTTP %s from egress_ip=%s' % (response.status_code, egress_ip)
        )


def _next_deadline_event():
    from premierlearning.fpl_apis import GENERAL
    import requests
    from squadtools.fpl_deadline import next_deadline_event, parse_fpl_deadline
    events = requests.get(GENERAL, timeout=30).json().get('events') or []
    next_event = next_deadline_event(events)
    if next_event is None:
        return None, None
    return next_event, parse_fpl_deadline(next_event.get('deadline_time'))


def _deadline_guard():
    hours = os.environ.get('APPLY_WITHIN_HOURS', '').strip()
    if not hours:
        return
    limit = float(hours)
    next_event, deadline_dt = _next_deadline_event()
    if next_event is None:
        raise SystemExit('APPLY_WITHIN_HOURS is set but bootstrap-static has no current/next event')
    if not deadline_dt:
        raise SystemExit('Next event has no deadline_time')
    now = datetime.now(timezone.utc)
    remaining = (deadline_dt - now).total_seconds() / 3600.0
    print('Next GW %s deadline in %.1f hours (limit %.1f)' % (next_event.get('id'), remaining, limit))
    if remaining > limit:
        raise SystemExit(
            'Refusing to apply: next deadline is %.1f hours away (APPLY_WITHIN_HOURS=%s)'
            % (remaining, hours)
        )


def _should_apply_on_deadline_day():
    if not _flag('APPLY_ON_DEADLINE_DAY'):
        return False
    try:
        next_event, deadline_dt = _next_deadline_event()
    except Exception as exc:
        print('Could not read FPL deadline for APPLY_ON_DEADLINE_DAY: %s' % exc)
        return False
    from squadtools.fpl_deadline import is_transfer_deadline_day, to_london
    now = datetime.now(timezone.utc)
    if is_transfer_deadline_day(deadline_dt, now):
        print(
            'Transfer deadline day: GW %s deadline %s UK'
            % (
                next_event.get('id') if next_event else '?',
                to_london(deadline_dt).strftime('%Y-%m-%d %H:%M'),
            )
        )
        return True
    if next_event and deadline_dt:
        print(
            'Not transfer deadline day (next GW %s deadline %s UK)'
            % (next_event.get('id'), to_london(deadline_dt).strftime('%Y-%m-%d %H:%M'))
        )
    return False


def _publish_learning_done():
    queue_url = os.environ.get('LEARNING_DONE_QUEUE_URL', '').strip()
    if not queue_url:
        return
    try:
        import json
        import boto3
        from premierlearning.fpl_apis import get_next_gameweek
        body = {
            'run_id': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
            'gameweek': get_next_gameweek(),
        }
        boto3.client('sqs').send_message(QueueUrl=queue_url, MessageBody=json.dumps(body))
        print('Published learning-done message to SQS: %s' % body)
    except Exception as exc:
        raise SystemExit('Learning finished but SQS publish failed: %s' % exc)


def _learning_argv(extra=None):
    """Fargate always refreshes FPL bootstrap unless --no-update is set.

    Injury / chance-of-playing news changes without a finished gameweek, so the
    image's copied season JSON is not a safe cache on a live pipeline run.
    """
    extra = list(
        extra if extra is not None
        else shlex.split(os.environ.get('PREMIERLEARNING_ARGS', ''))
    )
    skip = {'--no-update', '-no-update'}
    if skip.intersection(extra):
        return [arg for arg in extra if arg not in skip]
    if '--update' not in extra:
        extra = ['--update'] + extra
    return extra


def run_learning():
    from premierlearning.runner import Runner
    extra = _learning_argv()
    sys.argv = ['premierlearning'] + extra
    print('Running premierlearning %s' % extra)
    runner = Runner()
    runner.run()
    _publish_learning_done()


def run_squadtools():
    apply_transfers = _flag('APPLY_TRANSFERS')
    apply_team_selection = _flag('APPLY_TEAM_SELECTION')
    suggest_transfers = _flag('SUGGEST_TRANSFERS')
    if not apply_transfers and _should_apply_on_deadline_day():
        apply_transfers = True
        suggest_transfers = False
        print('APPLY_ON_DEADLINE_DAY: applying transfers')
    if not apply_transfers and not apply_team_selection and not suggest_transfers:
        suggest_transfers = True
        print('No apply flag set; defaulting to SUGGEST_TRANSFERS=true')
    if apply_transfers or apply_team_selection:
        _deadline_guard()
    from squadtools.__main__ import main as squadtools_main
    print(
        'Running squadtools apply_transfers=%s apply_team_selection=%s suggest_transfers=%s'
        % (apply_transfers, apply_team_selection, suggest_transfers)
    )
    squadtools_main(
        apply_transfers=apply_transfers,
        apply_team_selection=apply_team_selection,
        suggest_transfers=suggest_transfers,
    )


def run_login_only():
    from premierlearning.fpl_auth import load_credentials
    from squadtools.logged_in_action_performer import get_user_jsons_login
    creds = load_credentials()
    print('LOGIN_ONLY: attempting FPL login as %s' % creds['EMAIL'])
    user_json = get_user_jsons_login(creds['EMAIL'], creds['PASSWORD'])
    print('LOGIN_ONLY loggedIn=%s manager_id=%s squad_size=%s' % (
        user_json['user']['loggedIn'],
        user_json['fpl_api']['manager_id'],
        len(user_json['user']['teamIDs'] or []),
    ))
    if not user_json['user']['loggedIn']:
        raise SystemExit('FPL login check failed')


def run_status_check():
    import requests
    from premierlearning.fpl_apis import FPL_ORIGIN, FPL_USER_AGENT, GENERAL, ME_API, OIDC_TOKEN_URL

    headers = {'User-Agent': FPL_USER_AGENT}
    urls = [
        FPL_ORIGIN + '/',
        GENERAL,
        ME_API,
        OIDC_TOKEN_URL,
    ]
    print('egress_ip=%s (discovered at runtime, not hardcoded)' % _discover_egress_ip())
    print('=== requests ===')
    for url in urls:
        try:
            response = requests.get(url, timeout=30, headers=headers, allow_redirects=True)
            interesting = {
                key: response.headers.get(key)
                for key in (
                    'server', 'via', 'cf-ray', 'cf-mitigated', 'x-cache',
                    'x-amz-cf-id', 'retry-after', 'content-type',
                )
                if response.headers.get(key)
            }
            print(
                'http_status=%s url=%s final=%s bytes=%s headers=%s head=%s'
                % (
                    response.status_code,
                    url,
                    response.url,
                    len(response.content or b''),
                    interesting,
                    (response.text or '')[:180].replace('\n', ' '),
                )
            )
        except Exception as exc:
            print('http_error url=%s err=%s' % (url, exc))

    print('=== playwright ===')
    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=['--disable-dev-shm-usage', '--no-sandbox', '--disable-gpu'],
        )
        page = browser.new_page(user_agent=FPL_USER_AGENT)
        try:
            response = page.goto(FPL_ORIGIN + '/', wait_until='domcontentloaded', timeout=45000)
            body = ''
            try:
                body = page.locator('body').inner_text(timeout=5000) or ''
            except Exception as exc:
                body = 'body_error %s' % exc
            print(
                'pw_status=%s pw_url=%s pw_bytes=%s pw_body=%s'
                % (
                    None if response is None else response.status,
                    page.url,
                    len(page.content()),
                    body[:180].replace('\n', ' | '),
                )
            )
            page.wait_for_timeout(8000)
            body2 = ''
            try:
                body2 = page.locator('body').inner_text(timeout=5000) or ''
            except Exception as exc:
                body2 = 'body_error %s' % exc
            print(
                'pw_status2=%s pw_url2=%s pw_bytes2=%s pw_body2=%s'
                % (
                    None if response is None else response.status,
                    page.url,
                    len(page.content()),
                    body2[:180].replace('\n', ' | '),
                )
            )
        finally:
            browser.close()
    print('FPL_STATUS_CHECK done')


def _schedule_next_deadline_run():
    """Book the following gameweek for two hours before its deadline.

    The morning cron is gone. This run is the one near a deadline, so it
    leaves the scheduler pointing at the deadline after that.
    """
    name = (os.environ.get('PIPELINE_SCHEDULE_NAME') or '').strip()
    if not name:
        return
    try:
        hours = float(os.environ.get('SCHEDULE_HOURS_BEFORE_DEADLINE', '2'))
    except ValueError:
        hours = 2.0
    try:
        from premierlearning.fpl_apis import GENERAL
        import requests
        from squadtools.fpl_deadline import (
            deadline_for_following_run,
            london_at_expression,
        )
        events = requests.get(GENERAL, timeout=30).json().get('events') or []
        now = datetime.now(timezone.utc)
        deadline = deadline_for_following_run(events, now)
        if deadline is None:
            print('No future FPL deadline; leaving schedule %s unchanged' % name)
            return
        run_at = deadline - timedelta(hours=hours)
        if run_at <= now:
            print('Next deadline run %s is already past; leaving schedule %s' % (run_at, name))
            return
        expression = london_at_expression(run_at)
        _update_pipeline_schedule(name, expression)
        print('Next Fargate run booked at %s UK (%s before GW deadline %s)' % (
            expression,
            hours,
            deadline.strftime('%Y-%m-%d %H:%M UTC'),
        ))
    except Exception as exc:
        print('Could not book the next deadline run: %s' % exc)


def _update_pipeline_schedule(name, expression):
    import boto3
    region = (
        os.environ.get('AWS_REGION')
        or os.environ.get('AWS_DEFAULT_REGION')
        or 'eu-west-2'
    )
    client = boto3.client('scheduler', region_name=region)
    current = client.get_schedule(Name=name)
    client.update_schedule(
        Name=name,
        GroupName=current.get('GroupName') or 'default',
        Description='Train and apply two hours before the next FPL deadline',
        ScheduleExpression=expression,
        ScheduleExpressionTimezone=current.get('ScheduleExpressionTimezone') or 'Europe/London',
        FlexibleTimeWindow=current['FlexibleTimeWindow'],
        Target=current['Target'],
        State='ENABLED',
        ActionAfterCompletion=current.get('ActionAfterCompletion') or 'NONE',
    )


def main():
    app = Flask(__name__)
    with app.app_context():
        if _flag('FPL_STATUS_CHECK'):
            run_status_check()
            return
        _fpl_egress_preflight()
        _schedule_next_deadline_run()
        if _flag('LOGIN_ONLY'):
            run_login_only()
            return
        if _flag('RUN_LEARNING'):
            run_learning()
        else:
            print('Skipping learning (RUN_LEARNING is not set)')
        run_apply = (
            _flag('APPLY_TRANSFERS')
            or _flag('APPLY_TEAM_SELECTION')
            or _flag('SUGGEST_TRANSFERS')
            or not _flag('RUN_LEARNING')
        )
        skip_apply = _flag('SKIP_SQUADTOOLS')
        if skip_apply:
            print('Skipping squadtools (SKIP_SQUADTOOLS=1)')
            return
        if run_apply:
            run_squadtools()
        else:
            print('Learning-only run; not starting squadtools')


if __name__ == '__main__':
    main()
