import importlib.util
import sys
import types
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))


def _load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


premierlearning = types.ModuleType('premierlearning')
premierlearning.__path__ = [str(API_DIR / 'premierlearning')]
sys.modules['premierlearning'] = premierlearning

fpl_apis = _load_file_module(
    'premierlearning.fpl_apis',
    API_DIR / 'premierlearning' / 'fpl_apis.py',
)
sys.modules['premierlearning.fpl_apis'] = fpl_apis
fpl_auth = _load_file_module(
    'premierlearning.fpl_auth',
    API_DIR / 'premierlearning' / 'fpl_auth.py',
)
sys.modules['premierlearning.fpl_auth'] = fpl_auth

action_performer = _load_file_module(
    'logged_in_action_performer',
    Path(__file__).with_name('logged_in_action_performer.py'),
)
get_user_jsons_login = action_performer.get_user_jsons_login


def main():
    fantasy_login = fpl_auth.load_credentials()
    email = fantasy_login['EMAIL']
    password = fantasy_login['PASSWORD']
    print('Attempting FPL login as %s' % email)
    print('Using Ping Identity at %s' % fpl_apis.OIDC_AUTHORITY)

    user_json = get_user_jsons_login(email, password)
    print('\nget_user_jsons_login result')
    print('  loggedIn: %s' % user_json['user']['loggedIn'])
    print('  name: %r' % user_json['user']['name'])
    print('  manager_id: %s' % user_json['fpl_api']['manager_id'])
    team_ids = user_json['user']['teamIDs']
    print('  squad size: %s' % (len(team_ids) if team_ids else 0))
    print('  teamIDs: %s' % team_ids)
    if not user_json['user']['loggedIn']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
