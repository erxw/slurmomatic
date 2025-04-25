import numpy as np
import optuna
import submitit
import tempfile
import os
from sklearn.base import BaseEstimator, clone
from sklearn.utils import indexable
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring
from typing import Callable, Optional, Union, Dict, Any, List
from slurmomatic.core import is_slurm_available


def _evaluate_fold(estimator, X, y, train_idx, test_idx, scoring, error_score):
    try:
        estimator.fit(X[train_idx], y[train_idx])
        scorer = check_scoring(estimator, scoring)
        score = scorer(estimator, X[test_idx], y[test_idx])
        print(score)
        return score
    except Exception as e:
        if error_score == 'raise':
            raise e
        return error_score


class SlurmSearchCV(BaseEstimator):
    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: Callable[[optuna.Trial], Dict[str, Any]],
        n_trials: int = 10,
        scoring: Optional[Union[str, Callable]] = None,
        cv: Optional[int] = 5,
        error_score: Union[str, float] = np.nan,
        verbose: int = 0,
        random_state: Optional[int] = None,
        slurm_config: Optional[Dict[str, Any]] = None,
        use_slurm: bool = True
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_trials = n_trials
        self.scoring = scoring
        self.cv = cv
        self.error_score = error_score
        self.verbose = verbose
        self.random_state = random_state
        self.slurm_config = slurm_config or {}
        self.use_slurm = use_slurm

        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )

        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.cv_results_: List[Dict[str, Any]] = []

    def _objective(self, trial: optuna.Trial, X, y):
        params = self.param_distributions(trial)
        estimator = clone(self.estimator).set_params(**params)

        cv = check_cv(self.cv, y, classifier=False)
        splits = list(cv.split(X, y))

        print(is_slurm_available())

        executor = submitit.AutoExecutor(folder=tempfile.mkdtemp())

        executor.update_parameters(
            slurm_array_parallelism=5,  # Limit to 5 concurrent tasks

        )
#        executor.update_parameters(**self.slurm_config)
        jobs = []

        for train_idx, test_idx in splits:
            jobs.append(
                executor.submit(
                    _evaluate_fold,
                    estimator=clone(estimator),
                    X=X,
                    y=y,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    scoring=self.scoring,
                    error_score=self.error_score
                )
            )

        scores = [job.result() for job in jobs]

        mean_score = np.mean(scores)
        self.cv_results_.append({
            "params": params,
            "mean_test_score": mean_score,
            "fold_scores": scores,
        })

        return mean_score

    def fit(self, X, y=None, groups=None):
        X, y, _ = indexable(X, y, groups)

        self.study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials
        )

        self.best_params_ = self.study.best_params
        self.best_score_ = self.study.best_value

        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
        self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):
        if self.best_estimator_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        if self.best_estimator_ is None:
            raise RuntimeError("Call fit() before score().")
        scorer = check_scoring(self.best_estimator_, self.scoring)
        return scorer(self.best_estimator_, X, y)

    def get_cv_results(self):
        return self.cv_results_
