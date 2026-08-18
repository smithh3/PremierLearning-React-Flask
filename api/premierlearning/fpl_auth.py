import json
import os
import sys
import time
from pathlib import Path

import requests

from premierlearning.fpl_apis import (
    FPL_ORIGIN,
    FPL_USER_AGENT,
    ME_API,
    OIDC_CLIENT_ID,
    OIDC_TOKEN_URL,
)

CREDENTIALS_PATH = Path(__file__).resolve().parents[2] / '.vscode' / 'fantasy_login.json'


class FplLoginError(RuntimeError):
    pass


def credentials_path():
    override = os.environ.get('FPL_CREDENTIALS_PATH', '').strip()
    if override:
        return Path(override)
    return CREDENTIALS_PATH


def load_credentials():
    credentials = {}
    path = credentials_path()
    if path.is_file():
        with open(path, encoding='utf-8') as json_file:
            loaded = json.load(json_file)
            if isinstance(loaded, dict):
                credentials.update(loaded)

    env_map = {
        'EMAIL': 'FPL_EMAIL',
        'PASSWORD': 'FPL_PASSWORD',
        'REFRESH_TOKEN': 'FPL_REFRESH_TOKEN',
        'ID': 'FPL_ID',
    }
    for key, env_name in env_map.items():
        value = os.environ.get(env_name, '').strip()
        if value:
            credentials[key] = value

    if not credentials.get('EMAIL') or not credentials.get('PASSWORD'):
        raise FplLoginError(
            'FPL credentials missing. Set FPL_EMAIL and FPL_PASSWORD, or provide %s' % path
        )
    return credentials


def save_refresh_token(refresh_token):
    if not refresh_token:
        return
    current = None
    try:
        current = load_credentials().get('REFRESH_TOKEN')
    except FplLoginError:
        current = None
    if current == refresh_token:
        return

    path = credentials_path()
    write_json = path.is_file() or bool(os.environ.get('FPL_CREDENTIALS_PATH', '').strip())
    if write_json:
        try:
            data = {}
            if path.is_file():
                with open(path, encoding='utf-8') as json_file:
                    loaded = json.load(json_file)
                    if isinstance(loaded, dict):
                        data.update(loaded)
            data['REFRESH_TOKEN'] = refresh_token
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=4)
                json_file.write('\n')
            print('Saved FPL refresh token to %s' % path)
        except OSError as exc:
            print('Could not write FPL refresh token to %s: %s' % (path, exc))

    token_file = os.environ.get('FPL_REFRESH_TOKEN_FILE', '').strip()
    if token_file:
        try:
            token_path = Path(token_file)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(refresh_token, encoding='utf-8')
            print('Saved FPL refresh token to %s' % token_path)
        except OSError as exc:
            print('Could not write FPL_REFRESH_TOKEN_FILE: %s' % exc)

    secret_arn = os.environ.get('FPL_REFRESH_SECRET_ARN', '').strip()
    if secret_arn:
        _update_refresh_token_secret(secret_arn, refresh_token)


def _update_refresh_token_secret(secret_arn, refresh_token):
    try:
        import boto3
    except ImportError:
        print('boto3 is not installed; skipping Secrets Manager refresh-token update')
        return
    try:
        client = boto3.client('secretsmanager')
        current = client.get_secret_value(SecretId=secret_arn)
        payload = {}
        raw = current.get('SecretString') or '{}'
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            payload.update(loaded)
        if payload.get('REFRESH_TOKEN') == refresh_token:
            return
        payload['REFRESH_TOKEN'] = refresh_token
        client.put_secret_value(SecretId=secret_arn, SecretString=json.dumps(payload))
        print('Updated REFRESH_TOKEN in Secrets Manager')
    except Exception as exc:
        print('Could not update Secrets Manager refresh token: %s' % exc)


def auth_headers(client):
    headers = {
        'User-Agent': FPL_USER_AGENT,
        'Origin': FPL_ORIGIN,
        'Referer': '%s/' % FPL_ORIGIN,
    }
    token = getattr(client, 'fpl_access_token', None)
    if token:
        headers['X-API-Authorization'] = 'Bearer %s' % token
    return headers


def authed_get(client, url):
    return client.get(url, headers=auth_headers(client))


def authed_post(client, url, json=None):
    return client.post(url, json=json, headers=auth_headers(client))


def apply_tokens(client, tokens):
    client.fpl_access_token = tokens['access_token']
    client.fpl_refresh_token = tokens.get('refresh_token')
    client.headers.update(auth_headers(client))


def refresh_access_token(refresh_token):
    response = requests.post(
        OIDC_TOKEN_URL,
        data={
            'grant_type': 'refresh_token',
            'client_id': OIDC_CLIENT_ID,
            'refresh_token': refresh_token,
        },
        headers={
            'User-Agent': FPL_USER_AGENT,
            'Accept-Language': 'en',
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        timeout=30,
    )
    if not response.ok:
        print('FPL token refresh failed: %s %s' % (response.status_code, response.text[:300]))
        return None
    tokens = response.json()
    if not tokens.get('access_token'):
        print('FPL token refresh returned no access_token')
        return None
    tokens['method'] = 'refresh'
    return tokens


def _session_is_authed(client):
    try:
        response = authed_get(client, ME_API)
        print('GET %s -> %s' % (ME_API, response.status_code))
        if not response.ok:
            print(response.text[:500])
            return False
        player = (response.json() or {}).get('player') or {}
        return bool(player.get('entry'))
    except Exception as exc:
        print('Could not verify FPL session: %s' % exc)
        return False


def login(client, username, password):
    credentials = load_credentials()
    tokens = None
    stored_refresh = credentials.get('REFRESH_TOKEN')
    if stored_refresh:
        print('Refreshing stored FPL token')
        tokens = refresh_access_token(stored_refresh)
        if tokens is None:
            print('Stored FPL token was rejected; signing in with email/password')

    if tokens is None:
        print('Signing in to FPL with email/password')
        tokens = browser_login(username, password)

    apply_tokens(client, tokens)
    if tokens.get('method') == 'refresh' and not _session_is_authed(client):
        print('Refreshed token could not call /api/me/; signing in with email/password')
        tokens = browser_login(username, password)
        apply_tokens(client, tokens)

    new_refresh = tokens.get('refresh_token')
    if new_refresh:
        save_refresh_token(new_refresh)
    return tokens


def browser_login(username, password):
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise FplLoginError(
            "FPL login needs Playwright. Run: pip install playwright && playwright install chromium"
        ) from exc

    last_error = None
    for headless in ((True,) if _want_headless() else (False,)):
        try:
            return _browser_login_once(username, password, headless, PlaywrightTimeout, sync_playwright)
        except FplLoginError as exc:
            last_error = exc
            raise
    raise last_error


def _want_headless():
    env = os.getenv('FPL_LOGIN_HEADLESS', '').strip().lower()
    if env in ('1', 'true', 'yes'):
        return True
    if env in ('0', 'false', 'no'):
        return False
    return os.path.exists('/.dockerenv') or not sys.stdout.isatty()


def _chromium_launch_args():
    if _want_headless() or os.path.exists('/.dockerenv'):
        return ['--disable-dev-shm-usage', '--no-sandbox', '--disable-gpu']
    return []


def _browser_login_once(username, password, headless, PlaywrightTimeout, sync_playwright):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless, args=_chromium_launch_args())
        context = browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent=FPL_USER_AGENT,
        )
        page = context.new_page()
        popups = []
        page.on('popup', lambda p: popups.append(p))
        try:
            page.goto('%s/' % FPL_ORIGIN, wait_until='domcontentloaded')
            _wait_for_landing(page, popups)
            print('FPL landing url=%s' % page.url)
            _dismiss_cookie_banner(page, popups)
            _click_login(page, popups)
            target = _wait_for_idp(page, popups, PlaywrightTimeout)
            print('FPL login form url=%s' % target.url)
            _dismiss_cookie_banner(target, popups)
            _fill_first(target, popups, [
                '#username',
                "input[name='username']",
                "input[placeholder*='mail']",
                "input[placeholder*='Mail']",
                "input[type='email']",
                "input[name='email']",
                "input[name='login']",
                '#email',
            ], username)
            _fill_first(target, popups, [
                '#password',
                "input[placeholder*='assword']",
                "input[type='password']",
                "input[name='password']",
            ], password)
            _require_click(target, popups, [
                'button#btnSignIn',
                "button:has-text('Sign In')",
                "button[type='submit']",
                "input[type='submit']",
            ], timeout_ms=10000, force=False)
            _wait_for_login_success(page, popups, PlaywrightTimeout)

            oidc_user = _wait_for_oidc_user(page)
            access_token = oidc_user.get('access_token')
            if not access_token:
                raise FplLoginError('FPL login finished but no access token was stored')
            return {
                'access_token': access_token,
                'refresh_token': oidc_user.get('refresh_token'),
                'expires_in': oidc_user.get('expires_in', 3600),
                'method': 'browser',
            }
        finally:
            browser.close()


def _active_page(page, popups):
    for popup in reversed(popups):
        if not popup.is_closed():
            return popup
    return page


def _scopes(page):
    scopes = [page]
    try:
        scopes.extend(frame for frame in page.frames if frame != page.main_frame)
    except Exception:
        pass
    return scopes


def _find_visible(page, popups, selectors, timeout_ms):
    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        target = _active_page(page, popups)
        for scope in _scopes(target):
            for selector in selectors:
                try:
                    locator = scope.locator(selector).first
                    if locator.is_visible(timeout=300):
                        return locator
                except Exception:
                    continue
        page.wait_for_timeout(400)
    return None


def _dump_login_debug(page, popups, label):
    target = _active_page(page, popups)
    print('%s url=%s' % (label, getattr(target, 'url', '')))
    try:
        print('%s body=%s' % (label, (_page_text(target) or '')[:400]))
    except Exception as exc:
        print('%s body_error=%s' % (label, exc))


def _wait_for_landing(page, popups):
    selectors = (
        "button:has-text('Log in'), "
        "a:has-text('Log in'), "
        "#onetrust-accept-btn-handler, "
        "button:has-text('Accept All Cookies')"
    )
    try:
        page.wait_for_selector(selectors, timeout=45000)
    except Exception:
        _dump_login_debug(page, popups, 'landing wait failed')
        raise FplLoginError('FPL landing page did not show Log in or cookie banner (url=%s)' % page.url)
    try:
        page.wait_for_url('**/en/**', timeout=5000)
    except Exception:
        pass


def _click_login(page, popups):
    target = _active_page(page, popups)
    try:
        loc = target.get_by_role('button', name='Log in')
        loc.first.click(timeout=15000)
        print('Clicked Log in')
        return
    except Exception as exc:
        print('get_by_role Log in failed: %s' % exc)
    if _click_first(page, popups, [
        "button:has-text('Log in')",
        "a:has-text('Log in')",
        "button:has-text('Sign In')",
        "a:has-text('Sign In')",
        "a[href*='login']",
    ], timeout_ms=10000, force=False):
        print('Clicked Log in (selector fallback)')
        return
    _dump_login_debug(page, popups, 'log in click failed')
    raise FplLoginError('Could not click Log in (url=%s)' % getattr(target, 'url', ''))


def _click_first(page, popups, selectors, timeout_ms=4000, force=True):
    locator = _find_visible(page, popups, selectors, timeout_ms)
    if locator is None:
        return False
    locator.click(force=force)
    return True


def _require_click(page, popups, selectors, timeout_ms=4000, force=True):
    if _click_first(page, popups, selectors, timeout_ms=timeout_ms, force=force):
        return True
    target = _active_page(page, popups)
    raise FplLoginError(
        'Could not click %s (url=%s)' % (selectors[0], getattr(target, 'url', ''))
    )


def _wait_for_idp(page, popups, PlaywrightTimeout):
    deadline = time.time() + 30
    while time.time() < deadline:
        target = _active_page(page, popups)
        url = getattr(target, 'url', '') or ''
        if 'account.premierleague.com' in url:
            try:
                target.wait_for_load_state('domcontentloaded', timeout=8000)
            except PlaywrightTimeout:
                pass
            return target
        if _find_visible(target, popups, ['#username', "input[name='username']"], 400):
            return target
        page.wait_for_timeout(400)
    target = _active_page(page, popups)
    raise FplLoginError(
        'Log in did not open the FPL account page (url=%s)' % getattr(target, 'url', '')
    )


def _fill_first(page, popups, selectors, value):
    locator = _find_visible(page, popups, selectors, 15000)
    if locator is None:
        target = _active_page(page, popups)
        frame_urls = []
        try:
            frame_urls = [frame.url for frame in target.frames]
        except Exception:
            pass
        raise FplLoginError(
            'Could not find a field matching %s (url=%s frames=%s)' % (selectors[0], target.url, frame_urls)
        )
    locator.fill(value)


def _page_text(page):
    try:
        return (page.locator('body').inner_text(timeout=2000) or '').lower()
    except Exception:
        return ''


def _needs_password_update(page, popups):
    target = _active_page(page, popups)
    text = _page_text(target)
    phrases = (
        'update password',
        'change password',
        'reset password',
        'new password',
        'confirm password',
        'expired',
        'password has expired',
        'choose a new password',
    )
    if any(phrase in text for phrase in phrases):
        return True
    return _find_visible(target, popups, [
        "input[placeholder*='ew password']",
        "input[name='newPassword']",
        "input[name='new_password']",
        "input[id*='newPassword']",
        "input[id*='new-password']",
    ], 800) is not None


def _wait_for_login_success(page, popups, PlaywrightTimeout):
    try:
        page.wait_for_url('**://fantasy.premierleague.com/**', timeout=20000)
        return
    except PlaywrightTimeout:
        pass

    if _needs_password_update(page, popups):
        print('FPL is asking you to update your password in the browser window.')
        print('Complete that, then this login will continue automatically.')
        try:
            page.wait_for_url('**://fantasy.premierleague.com/**', timeout=180000)
            return
        except PlaywrightTimeout:
            pass

    current = _active_page(page, popups).url
    if 'account.premierleague.com' in current or 'login' in current.lower():
        raise FplLoginError(
            'FPL login did not complete (still on %s). Update FPL_PASSWORD or the stored password in %s if it changed.'
            % (current, credentials_path())
        )


COOKIE_SELECTORS = [
    '#onetrust-accept-btn-handler',
    '#accept-recommended-btn-handler',
    'button#onetrust-accept-btn-handler',
    '.onetrust-close-btn-handler',
    "button:has-text('Accept All Cookies')",
    "button:has-text('Accept all cookies')",
    "button:has-text('Accept All')",
    "button:has-text('I Accept')",
    "button:has-text('Allow all')",
    "button:has-text('Allow All')",
]


def _seed_cookie_consent(context):
    from datetime import datetime, timezone
    closed = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    cookies = []
    for url in ('https://fantasy.premierleague.com/', 'https://account.premierleague.com/'):
        cookies.append({
            'name': 'OptanonAlertBoxClosed',
            'value': closed,
            'url': url,
            'path': '/',
        })
    try:
        context.add_cookies(cookies)
    except Exception:
        pass


def _cookie_banner_visible(page, popups):
    return _find_visible(page, popups, [
        '#onetrust-banner-sdk',
        '#onetrust-consent-sdk .onetrust-pc-dark-filter',
        '#onetrust-accept-btn-handler',
    ], 600) is not None


def _hide_cookie_banner_js(page):
    script = """() => {
        const btn = document.querySelector('#onetrust-accept-btn-handler, #accept-recommended-btn-handler');
        if (btn) btn.click();
        for (const id of ['onetrust-banner-sdk', 'onetrust-pc-sdk']) {
            const el = document.getElementById(id);
            if (el) el.style.display = 'none';
        }
        const sdk = document.getElementById('onetrust-consent-sdk');
        if (sdk) sdk.style.display = 'none';
        document.body.style.overflow = 'auto';
        document.documentElement.style.overflow = 'auto';
    }"""
    try:
        page.evaluate(script)
    except Exception:
        pass
    try:
        for frame in page.frames:
            try:
                frame.evaluate(script)
            except Exception:
                continue
    except Exception:
        pass


def _dismiss_cookie_banner(page, popups, wait_ms=10000):
    deadline = time.time() + wait_ms / 1000.0
    accepted = False
    while time.time() < deadline:
        target = _active_page(page, popups)
        locator = _find_visible(target, popups, COOKIE_SELECTORS, 700)
        if locator is not None:
            try:
                locator.click()
                accepted = True
                print('Accepted cookies')
            except Exception:
                try:
                    locator.click(force=True)
                    accepted = True
                    print('Accepted cookies (force)')
                except Exception:
                    pass
            page.wait_for_timeout(600)
        if accepted:
            _hide_cookie_banner_js(target)
        if not _cookie_banner_visible(target, popups):
            return True
        page.wait_for_timeout(300)
    if accepted:
        _hide_cookie_banner_js(_active_page(page, popups))
    return False


def _wait_for_oidc_user(page):
    deadline = time.time() + 20
    while time.time() < deadline:
        oidc_user = page.evaluate(
            """() => {
                const key = Object.keys(localStorage).find(k => k.startsWith('oidc.user:'));
                if (!key) return null;
                try { return JSON.parse(localStorage.getItem(key)); }
                catch (e) { return null; }
            }"""
        )
        if oidc_user and oidc_user.get('access_token'):
            return oidc_user
        page.wait_for_timeout(500)
    raise FplLoginError('FPL login finished but the site did not store an OIDC token')


def fetch_me(client):
    last = None
    for _ in range(5):
        response = authed_get(client, ME_API)
        last = response
        print('GET %s -> %s' % (ME_API, response.status_code))
        if not response.ok:
            print(response.text[:500])
            time.sleep(1)
            continue
        payload = response.json()
        player = payload.get('player') or {}
        if player.get('entry'):
            return payload
        print('/api/me/ had no player.entry: %s' % list(payload.keys()))
        time.sleep(1)
    if last is not None and last.ok:
        return last.json()
    raise FplLoginError('Could not load /api/me/ after login')
