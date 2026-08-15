import requests


MANAGER_API = 'https://fantasy.premierleague.com/api/entry/%i/'
MANAGER_PICKS_API = 'https://fantasy.premierleague.com/api/entry/%i/event/%i/picks/'
MANAGER_TRANSFERS_API = 'https://fantasy.premierleague.com/api/entry/%i/transfers/'
MANAGER_GAMEWEEK_API = 'https://fantasy.premierleague.com/api/entry/%i/history/'

REDIRECT_URI = 'https://users.premierleague.com/'
LOGIN_URL = 'https://users.premierleague.com/accounts/login/'
PICK_TEAM_URL = 'https://fantasy.premierleague.com/my-team'
PICK_TEAM_API = 'https://fantasy.premierleague.com/api/my-team'
TRANSFERS_API = 'https://fantasy.premierleague.com/api/transfers/'
ME_API = 'https://fantasy.premierleague.com/api/me/'
GENERAL = 'https://fantasy.premierleague.com/api/bootstrap-static/'


def get_next_gameweek():
    general = requests.get(GENERAL).json()
    return next(event['id'] for event in general['events'] if event['is_next'])

def get_current_gameweek():
    try:
        general = requests.get(GENERAL).json()
        return next(event['id'] for event in general['events'] if event['is_current'])
    except:
        return 0
