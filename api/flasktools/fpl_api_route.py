import requests

from datetime import datetime
from flask import Blueprint, request, session
from flask_cors import CORS
from squadtools import OptimiseRunner, get_user_jsons
from persistence import DB_Handler
from premierlearning.fpl_apis import *

bp = Blueprint('login', __name__, url_prefix='/api')
CORS(bp)


@bp.route('/time', methods=(['GET']))
def get_time():
    return datetime.now()


@bp.route('/players', methods=(['GET']))
def get_all_players():
    db_handler = DB_Handler()
    players = db_handler.get_player_jsons()
    db_handler.close_connection()
    return {'players': players}


@bp.route('/general', methods=(['GET']))
def get_general():
    return requests.get(GENERAL).json()


@bp.route('/team', methods=(['GET']))
def optimise_by_login():
    if 'user_json' not in session:
        return {'transfers': None}
    optimise_runner = OptimiseRunner(session['user_json'])
    optimise_runner.run()
    return {'transfers': optimise_runner.get_transfers_json()}


@bp.route('/login', methods=(['POST']))
def login():
    request_data = request.get_json()
    session['user_json'] = get_user_jsons(int(request_data['manager_id']))
    return {'user_json': session['user_json']}


@bp.route('/logout', methods=(['GET']))
def logout():
    session['user_json'] = {
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
    return {}

@bp.route('/opt', methods=(['GET']))
def optimise_by_login_2():
    if 'user_json' not in session:
        return {'transfers': None, 'suggestedTeams': None}
    optimise_runner = OptimiseRunner(session['user_json']['fpl_api']['manager_id'])
    optimise_runner.run()
    transfers = optimise_runner.get_transfers_json()
    suggestedTeams = optimise_runner.get_future_starting_teams()
    return {
        'transfers': transfers,
        'suggestedTeams': suggestedTeams
    }
