import argparse

from flask import Flask

from premierlearning.fpl_auth import load_credentials
from .logged_in_action_performer import get_user_jsons, get_user_jsons_login
from .optimise_runner import OptimiseRunner
from .predictions_report import send_predictions_email


def main(apply_transfers=False, apply_team_selection=False, suggest_transfers=False):
    app = Flask(__name__)
    fantasy_login = load_credentials()
    with app.app_context():
        if apply_transfers or apply_team_selection or suggest_transfers:
            # Pre-GW1 public /event/1/picks/ is 404 until the season starts — need my-team via login
            user_json = get_user_jsons_login(fantasy_login['EMAIL'], fantasy_login['PASSWORD'])
            if not user_json['user']['loggedIn']:
                raise SystemExit('FPL login failed')
            opt = OptimiseRunner(user_json['fpl_api']['manager_id'], user_json=user_json)
            if apply_team_selection and not apply_transfers:
                opt.populate_squad()
                opt.apply_this_week_team_selection(fantasy_login['EMAIL'], fantasy_login['PASSWORD'])
                return
            opt.run()
            if suggest_transfers and not apply_transfers:
                this_week = opt.get_this_week_transfers()
                if this_week:
                    print('Suggested transfers for GW %s (not applied):' % opt.squad.next_gameweek)
                    for transfer in this_week:
                        print('  %s' % transfer)
                else:
                    print('No transfers suggested for this gameweek')
                if opt.transfers:
                    print('Full multi-week scheme:')
                    print(opt.get_transfers())
                send_predictions_email(opt, applied=False)
                return
            if opt.transfers:
                print(opt.get_transfers())
            opt.apply_this_week_transfers(fantasy_login['EMAIL'], fantasy_login['PASSWORD'])
            send_predictions_email(opt, applied=True)
            return

        manager_id = fantasy_login.get('ID')
        if not manager_id:
            raise SystemExit('Set FPL_ID (or ID in the credentials file) for public squad lookup')
        get_user_jsons(manager_id)
        opt = OptimiseRunner(manager_id)
        opt.run()
        send_predictions_email(opt, applied=False)


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
    parser.add_argument(
        '--suggest-transfers',
        action='store_true',
        help='Log in, load your squad (my-team), and print suggested transfers without submitting',
    )
    args = parser.parse_args()
    main(
        apply_transfers=args.apply_transfers,
        apply_team_selection=args.apply_team_selection,
        suggest_transfers=args.suggest_transfers,
    )
