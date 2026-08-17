import json
import os
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


def load_credentials():
    with open(CREDENTIALS_PATH, encoding='utf-8') as json_file:
        return json.load(json_file)


def save_refresh_token(refresh_token):
    credentials = load_credentials()
    if credentials.get('REFRESH_TOKEN') == refresh_token:
        return
    credentials['REFRESH_TOKEN'] = refresh_token
    with open(CREDENTIALS_PATH, 'w', encoding='utf-8') as json_file:
        json.dump(credentials, json_file, indent=4)
        json_file.write('\n')


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
        return None
    tokens = response.json()
    if not tokens.get('access_token'):
        return None
    tokens['method'] = 'refresh'
    return tokens


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

    force_headless = os.getenv('FPL_LOGIN_HEADLESS', '').strip() in ('1', 'true', 'True')
    last_error = None
    for headless in ((True,) if force_headless else (False,)):
        try:
            return _browser_login_once(username, password, headless, PlaywrightTimeout, sync_playwright)
        except FplLoginError as exc:
            last_error = exc
            raise
    raise last_error


def _browser_login_once(username, password, headless, PlaywrightTimeout, sync_playwright):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent=FPL_USER_AGENT,
        )
        _seed_cookie_consent(context)
        page = context.new_page()
        popups = []
        page.on('popup', lambda p: popups.append(p))
        try:
            page.goto('%s/' % FPL_ORIGIN, wait_until='domcontentloaded')
            _dismiss_cookie_banner(page, popups)
            _click_first(page, popups, [
                "a:has-text('Log in')",
                "button:has-text('Log in')",
                "a:has-text('Sign In')",
                "button:has-text('Sign In')",
                "a[href*='login']",
            ])
            page.wait_for_timeout(2500)
            target = popups[-1] if popups else page
            try:
                target.wait_for_load_state('domcontentloaded', timeout=10000)
            except PlaywrightTimeout:
                pass
            _dismiss_cookie_banner(target, popups)
            _fill_first(target, popups, [
                "input[placeholder*='mail']",
                "input[placeholder*='Mail']",
                "input[type='email']",
                "input[name='email']",
                "input[name='login']",
                '#email',
                '#username',
            ], username)
            _fill_first(target, popups, [
                "input[placeholder*='assword']",
                "input[type='password']",
                "input[name='password']",
                '#password',
            ], password)
            _click_first(target, popups, [
                'button#btnSignIn',
                "button:has-text('Sign in')",
                "button:has-text('Sign In')",
                "button[type='submit']",
                "input[type='submit']",
            ])
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


def _click_first(page, popups, selectors):
    locator = _find_visible(page, popups, selectors, 4000)
    if locator is None:
        return False
    locator.click(force=True)
    return True


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
            'FPL login did not complete (still on %s). Update the stored password in .vscode/fantasy_login.json if it changed.' % current
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


def _dismiss_cookie_banner(page, popups, wait_ms=8000):
    deadline = time.time() + wait_ms / 1000.0
    while time.time() < deadline:
        target = _active_page(page, popups)
        locator = _find_visible(target, popups, COOKIE_SELECTORS, 700)
        if locator is not None:
            try:
                locator.click(force=True)
                print('Accepted cookies')
            except Exception:
                pass
            page.wait_for_timeout(400)
        _hide_cookie_banner_js(target)
        if not _cookie_banner_visible(target, popups):
            return True
        page.wait_for_timeout(300)
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
        if not response.ok:
            time.sleep(1)
            continue
        payload = response.json()
        player = payload.get('player') or {}
        if player.get('entry'):
            return payload
        time.sleep(1)
    if last is not None and last.ok:
        return last.json()
    raise FplLoginError('Could not load /api/me/ after login')
