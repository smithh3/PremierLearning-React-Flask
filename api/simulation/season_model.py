"""Fit historical prediction models once, then reuse them for every GW snapshot."""

import numpy

from premierlearning.clean_match import get_input_keys
from premierlearning.helper_functions import (
    effective_minutes_for_points,
    implied_pp90,
)
from premierlearning.learning import (
    HORIZON_REPS_FLAG,
    MINUTES_REPS_FLAG,
    fit_points_error_predictor,
    fit_reusable_predictor,
    parse_horizon_reps,
    parse_minutes_reps,
)
from premierlearning.player import MIN_APPEARANCES_TO_PREDICT
from simulation.historical_data import (
    HIST_PREDICT_SKIP_MAX_APPS,
    HIST_PREDICT_SKIP_MAX_POINTS,
)
from premierlearning.team_expectancy import (
    apply_team_expectancy_model,
    fit_team_expectancy_once,
)


class FixedSeasonModels:
    def __init__(
        self,
        team_model,
        points_model,
        minutes_model,
        horizon_model,
        points_error_model,
        horizon_gameweeks,
    ):
        self.team_model = team_model
        self.points_model = points_model
        self.minutes_model = minutes_model
        self.horizon_model = horizon_model
        self.points_error_model = points_error_model
        self.horizon_gameweeks = int(horizon_gameweeks)

    @classmethod
    def fit(cls, runner):
        """Train on pre-season data only; the supplied runner must be at GW1."""
        if int(runner.as_of_gameweek) != 1:
            raise ValueError('Fixed season models must be fitted from the GW1 cutoff')
        print('Training one fixed model bundle for the simulated season...')
        team_model = fit_team_expectancy_once(
            runner.past_seasons,
            runner.current_season,
        )
        apply_team_expectancy_model(
            team_model,
            runner.past_seasons,
            runner.current_season,
        )
        runner.process_player_matches()
        clean_matches = runner.populate_formatted_match_data()
        valid_clean_matches = [
            match for match in clean_matches if match['is_valid']
        ]
        minutes_clean_matches = [
            match
            for match in clean_matches
            if match.get('is_valid_for_minutes')
        ]
        h1_labels = numpy.array(
            [
                [runner._clean_match_points_per_90(match)]
                for match in valid_clean_matches
            ],
            dtype=float,
        )
        points_model = fit_reusable_predictor(
            valid_clean_matches,
            get_input_keys('total_points'),
            'total_points',
            True,
            output_learn=h1_labels,
            log_targets=True,
            reps=1,
        )
        minutes_reps = parse_minutes_reps()
        if minutes_reps > 1:
            print(
                'Minutes model: %s independent fits (%s); drop val MAE > 8%% over best'
                % (minutes_reps, MINUTES_REPS_FLAG)
            )
        minutes_model = fit_reusable_predictor(
            minutes_clean_matches,
            get_input_keys('minutes'),
            'minutes',
            True,
            reps=minutes_reps,
        )
        points_error_model = fit_points_error_predictor(
            valid_clean_matches,
            get_input_keys('total_points'),
            True,
        )
        horizon_training = runner.build_horizon_training_data(
            runner.horizon_gameweeks
        )
        horizon_reps = parse_horizon_reps()
        if horizon_reps > 1:
            print(
                'Horizon-%s model: %s independent fits (%s); '
                'h1 points stay 1 fit'
                % (runner.horizon_gameweeks, horizon_reps, HORIZON_REPS_FLAG)
            )
        horizon_model = fit_reusable_predictor(
            horizon_training['player_clean'],
            get_input_keys('total_points'),
            'total_points',
            True,
            output_learn=horizon_training['labels'],
            log_targets=True,
            reps=horizon_reps,
        )
        return cls(
            team_model,
            points_model,
            minutes_model,
            horizon_model,
            points_error_model,
            runner.horizon_gameweeks,
        )

    def prepare_runner(self, runner):
        """Apply fixed team weights, then rebuild causal player features."""
        apply_team_expectancy_model(
            self.team_model,
            runner.past_seasons,
            runner.current_season,
        )
        runner.process_player_matches()

    def predict(self, runner, already_prepared=False):
        if not already_prepared:
            self.prepare_runner(runner)
        future_matches = []
        future_clean_matches = []
        skipped_low_usage = 0
        for player in runner.current_season.players:
            if getattr(player, 'skip_historical_predict', False):
                skipped_low_usage += 1
                continue
            if player.matches_played < MIN_APPEARANCES_TO_PREDICT:
                for future_match in player.future_matches:
                    future_match.points = 0.0
                    future_match.game_stats['total_points'] = 0.0
                    future_match.game_stats['minutes'] = 0.0
                continue
            for future_match in player.future_matches:
                future_matches.append(future_match)
                future_clean_matches.append(future_match.to_clean_match())

        if skipped_low_usage:
            print(
                'Skipping predict for %d players with <=%s season appearances '
                'or <=%s season points'
                % (
                    skipped_low_usage,
                    HIST_PREDICT_SKIP_MAX_APPS,
                    int(HIST_PREDICT_SKIP_MAX_POINTS),
                )
            )

        if future_clean_matches:
            points_predictions = self.points_model.predict(future_clean_matches)
            minutes_predictions = self.minutes_model.predict(future_clean_matches)
            error_predictions = self.points_error_model.predict(
                future_clean_matches
            )
            for match, points_pred, minutes_pred, error in zip(
                future_matches,
                points_predictions,
                minutes_predictions,
                error_predictions,
            ):
                pp90 = max(0.0, float(points_pred[0]))
                minutes = min(90.0, max(0.0, float(minutes_pred[0])))
                points = pp90 * effective_minutes_for_points(minutes) / 90.0
                match.predicted_pp90 = pp90
                match.game_stats['minutes'] = minutes
                match.game_stats['total_points'] = points
                match.points = points
                match.points_error = float(error)
                match.set_minutes_from_stats()

        horizon_data = runner.build_horizon_predict_data(
            self.horizon_gameweeks
        )
        if horizon_data['player_clean']:
            horizon_predictions = self.horizon_model.predict(
                horizon_data['player_clean']
            )
            runner.redistribute_horizon_points(
                horizon_data['players'],
                horizon_data['fixture_windows'],
                horizon_predictions,
                self.horizon_gameweeks,
            )
        runner.sync_predicted_pp90()
        runner.current_season.calculate_player_points_over_next_5_gameweeks()
        return runner.current_season
