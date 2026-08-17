import requests

from premierlearning.fpl_apis import *
from premierlearning.fpl_auth import (
    authed_get,
    authed_post,
    fetch_me,
    login,
)


def get_squad_ids(user_json):
    if 'picks' not in user_json:
        return None
    squad_ids = []
    for element in user_json['picks']:
        squad_ids.append(element['element'])
    return squad_ids

def get_user_jsons(manager_id):
    user_json = {
        'user': {
            'name': '',
            'loggedIn': False,
            'teamIDs': []
        },
        'fpl_api': {
            'manager_id': None,
            'manager_json': None,
            'picks': None
        }
    }
    with requests.Session() as client:
        manager = client.get(MANAGER_API % manager_id).json()
        if 'id' in manager:
            gw = get_current_gameweek()
            user_json['fpl_api']['manager_id'] = manager_id
            user_json['fpl_api']['manager_json'] = manager
            user_json['fpl_api']['picks'] = client.get(MANAGER_PICKS_API % (manager_id, gw)).json()

            user_json['user']['name'] = '%s %s' % (manager['player_first_name'], manager['player_last_name'])
            user_json['user']['loggedIn'] = True
            user_json['user']['teamIDs'] = get_squad_ids(user_json['fpl_api']['picks'])
    return user_json


def get_user_jsons_login(username, password):
    user_json = {
        'user': {
            'name': '',
            'loggedIn': False,
            'teamIDs': []
        },
        'fpl_api': {
            'manager_id': None,
            'manager_json': None,
            'picks': None
        }
    }
    with requests.Session() as client:
        login(client, username, password)
        me_payload = fetch_me(client)
        me_response = me_payload.get('player') or {}
        manager_id = me_response.get('entry')
        if not manager_id:
            return user_json
        user_json['fpl_api']['manager_id'] = manager_id
        user_json['fpl_api']['manager_json'] = authed_get(client, MANAGER_API % manager_id).json()
        user_json['fpl_api']['picks'] = authed_get(client, '%s/%i/' % (PICK_TEAM_API, manager_id)).json()

        user_json['user']['name'] = '%s %s' % (me_response.get('first_name', ''), me_response.get('last_name', ''))
        user_json['user']['loggedIn'] = True
        user_json['user']['teamIDs'] = get_squad_ids(user_json['fpl_api']['picks'])
    return user_json


def pick_team(request_payload, username, password):
    with requests.Session() as client:
        login(client, username, password)
        manager_id = fetch_me(client)['player']['entry']
        pick_response = authed_post(client, '%s/%i/' % (PICK_TEAM_API, manager_id), json=request_payload)
        print('POST %s/%s/ -> %s' % (PICK_TEAM_API, manager_id, pick_response.status_code))
        print(pick_response.text[:2000])
        if not pick_response.ok:
            raise RuntimeError(
                'FPL rejected the team selection (%s): %s' % (
                    pick_response.status_code,
                    pick_response.text[:500],
                )
            )
        return pick_response


def make_transfers(request_payload, username, password):
    with requests.Session() as client:
        login(client, username, password)
        make_transfers_response = authed_post(client, TRANSFERS_API, json=request_payload)
        print('POST %s -> %s' % (TRANSFERS_API, make_transfers_response.status_code))
        print(make_transfers_response.text[:2000])
        if not make_transfers_response.ok:
            raise RuntimeError(
                'FPL rejected the transfers (%s): %s' % (
                    make_transfers_response.status_code,
                    make_transfers_response.text[:500],
                )
            )
        return make_transfers_response
