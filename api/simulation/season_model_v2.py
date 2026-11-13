"""Fit the learning_v2 points model once per season, then reuse it for every GW snapshot.

Each snapshot still rebuilds every feature from the matches before its own
deadline, so predictions use the latest form; only the weights stay at the GW1
fit. The vocab is frozen at fit time: build_dataset makes a fresh one per call,
and a fresh one would shift every embedding row. Players first seen after GW1
read as the unknown index.

Kept out of season_model.py on purpose: that file is part of the v1 cache hash.
"""

from premierlearning.learning import _parse_string_flag
from premierlearning.learning_v2.dataset import build_dataset, window_rate_batch
from premierlearning.learning_v2.learn import spread_window_rates, store_points
from premierlearning.learning_v2.plan import load_plan
from premierlearning.learning_v2.train import PATIENCE_FLOOR, fit_points


def _seasons(runner):
    return list(getattr(runner, 'past_seasons', []) or []) + [runner.current_season]


class FixedSeasonModelsV2:
    def __init__(self, plan, vocab, fitted, window_fitted=None, horizon_gameweeks=0):
        self.plan = plan
        self.vocab = vocab
        self.fitted = fitted
        self.window_fitted = window_fitted
        self.horizon_gameweeks = int(horizon_gameweeks)

    @classmethod
    def fit(cls, runner, plan_name=None, weighting=None, points_reps=1, loss_name=None,
            smear_mode='off', patience_floor=PATIENCE_FLOOR, horizon_gameweeks=0,
            minutes_weight=0.0, minutes_target='bands'):
        """Train on everything before the runner's gameweek (GW1: pre-season data only).

        Mid-season fits (--retrain-every) include this season's played
        gameweeks, with a fresh vocab (new players get embeddings) and team
        time blocks counted back from this fit's cutoff.
        """
        plan = load_plan(plan_name or _parse_string_flag('--model-plan', 'simple_h1'))
        print('Training one fixed learning_v2 %s model as of GW%s...' % (plan.name, runner.as_of_gameweek))
        runner.process_player_matches()
        train_batch, _predict, vocab = build_dataset(
            _seasons(runner), plan,
            predict_season=runner.current_season,
            cutoff_gameweek=runner._trained_through_gameweek(),
        )
        if not train_batch.matches:
            raise ValueError('learning_v2 found no training fixtures before GW%s' % runner.as_of_gameweek)
        vocab.freeze()
        print(
            '  %s train rows, %s players, %s teams (vocab frozen)'
            % (len(train_batch.matches), vocab.player_size - 1, vocab.team_size - 1)
        )
        fitted = fit_points(
            plan, vocab, train_batch,
            loss_name=loss_name, reps=points_reps, weighting=weighting,
            smear_mode=smear_mode, patience_floor=patience_floor,
            minutes_weight=minutes_weight, minutes_target=minutes_target,
        )
        window_fitted = None
        if horizon_gameweeks:
            window_batch, groups = window_rate_batch(train_batch, horizon_gameweeks)
            # The window rows copy h1's tensors; drop h1's before fitting again.
            del train_batch
            print('  %s window rows, %s GWs' % (len(window_batch.matches), horizon_gameweeks))
            window_fitted = fit_points(
                plan, vocab, window_batch,
                loss_name=loss_name, reps=points_reps, weighting=weighting,
                smear_mode=smear_mode, patience_floor=patience_floor, groups=groups,
            )
        return cls(plan, vocab, fitted, window_fitted, horizon_gameweeks)

    def prepare_runner(self, runner):
        """Rebuild causal player features, as Runner.run does before learning."""
        runner.process_player_matches()

    def predict(self, runner, already_prepared=False):
        if not already_prepared:
            self.prepare_runner(runner)
        _train, predict_batch, _vocab = build_dataset(
            _seasons(runner), self.plan,
            predict_season=runner.current_season,
            vocab=self.vocab,
            include_train=False,
        )
        store_points(predict_batch.matches, self.fitted.predict(predict_batch))
        if self.window_fitted is not None:
            spread_window_rates(
                predict_batch, self.window_fitted.predict(predict_batch),
                int(runner.current_season.next_gameweek), self.horizon_gameweeks,
            )
        runner.current_season.calculate_player_points_over_next_5_gameweeks()
        return runner.current_season
