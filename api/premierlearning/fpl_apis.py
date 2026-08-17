import requests


MANAGER_API = 'https://fantasy.premierleague.com/api/entry/%i/'
MANAGER_PICKS_API = 'https://fantasy.premierleague.com/api/entry/%i/event/%i/picks/'
MANAGER_TRANSFERS_API = 'https://fantasy.premierleague.com/api/entry/%i/transfers/'
MANAGER_GAMEWEEK_API = 'https://fantasy.premierleague.com/api/entry/%i/history/'

FPL_ORIGIN = 'https://fantasy.premierleague.com'
FPL_USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
)

OIDC_AUTHORITY = 'https://account.premierleague.com/as'
OIDC_TOKEN_URL = '%s/token' % OIDC_AUTHORITY
OIDC_CLIENT_ID = 'bfcbaf69-aade-4c1b-8f00-c1cb8a193030'
OIDC_REDIRECT_URI = '%s/' % FPL_ORIGIN
OIDC_SCOPE = 'openid profile email offline_access'

REDIRECT_URI = OIDC_REDIRECT_URI
LOGIN_URL = '%s/' % FPL_ORIGIN
PICK_TEAM_URL = '%s/my-team' % FPL_ORIGIN
PICK_TEAM_API = '%s/api/my-team' % FPL_ORIGIN
TRANSFERS_API = '%s/api/transfers/' % FPL_ORIGIN
ME_API = '%s/api/me/' % FPL_ORIGIN
GENERAL = '%s/api/bootstrap-static/' % FPL_ORIGIN


def get_next_gameweek():
    general = requests.get(GENERAL).json()
    return next(event['id'] for event in general['events'] if event['is_next'])

def get_current_gameweek():
    try:
        general = requests.get(GENERAL).json()
        return next(event['id'] for event in general['events'] if event['is_current'])
    except:
        return 0

_selectable_player_ids = None
UNFIT_STATUSES = ('u', 'n', 'i', 's')


def load_selectable_player_ids():
    global _selectable_player_ids
    if _selectable_player_ids is not None:
        return _selectable_player_ids
    elements = []
    try:
        elements = requests.get(GENERAL, timeout=20).json().get('elements') or []
    except Exception:
        elements = []
    if not elements:
        from premierlearning.config_constants import SEASON_YR
        from premierlearning.helper_functions import read_json
        fantasy = read_json('data/season_%i_%i/fantasy.json' % (SEASON_YR, SEASON_YR + 1)) or {}
        elements = fantasy.get('elements') or []
    selectable = set()
    for element in elements:
        if element.get('can_select') is False:
            continue
        if element.get('status') in UNFIT_STATUSES:
            continue
        chance = element.get('chance_of_playing_next_round')
        if chance is not None and chance <= 0:
            continue
        selectable.add(int(element['id']))
    _selectable_player_ids = selectable
    return selectable
