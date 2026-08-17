import argparse
import json

from flask import Flask

from .logged_in_action_performer import get_user_jsons, get_user_jsons_login
from .optimise_runner import OptimiseRunner


def main(apply_transfers=False, apply_team_selection=False):
    app = Flask(__name__)

    with open('../.vscode/fantasy_login.json') as json_file:
        fantasy_login = json.load(json_file)
    with app.app_context():
        if apply_transfers or apply_team_selection:
            user_json = get_user_jsons_login(fantasy_login['EMAIL'], fantasy_login['PASSWORD'])
            if not user_json['user']['loggedIn']:
                raise SystemExit('FPL login failed')
            opt = OptimiseRunner(user_json['fpl_api']['manager_id'], user_json=user_json)
            if apply_team_selection and not apply_transfers:
                opt.populate_squad()
                opt.apply_this_week_team_selection(fantasy_login['EMAIL'], fantasy_login['PASSWORD'])
                return
            opt.run()
            if opt.transfers:
                print(opt.get_transfers())
            opt.apply_this_week_transfers(fantasy_login['EMAIL'], fantasy_login['PASSWORD'])
            return

        get_user_jsons(fantasy_login['ID'])
        opt = OptimiseRunner(fantasy_login['ID'])
        opt.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--apply-transfers',
        action='store_true',
        help='Log in and submit this gameweek\'s suggested transfers (also picks captain/starters)',
    )
    parser.add_argument(
        '--apply-team-selection',
        action='store_true',
        help='Log in and submit captain/starters only (no transfers)',
    )
    args = parser.parse_args()
    main(
        apply_transfers=args.apply_transfers,
        apply_team_selection=args.apply_team_selection,
    )
