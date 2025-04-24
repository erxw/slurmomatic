import numpy as np
import optuna
from sklearn.base import BaseEstimator, clone
from sklearn.utils import indexable
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring
from typing import Callable, Optional, Union, Dict, Any, List
from sklearn.base import BaseEstimator, ClassifierMixin

from slurmomatic.ml import cross_validate


class SlurmSearchCV(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: Callable[[optuna.Trial], Dict[str, Any]],
        n_trials: int = 10,
        scoring: Optional[Union[str, Callable]] = None,
        cv: Optional[int] = 5,
        error_score: Union[str, float] = np.nan,
        return_train_score: bool = False,
        return_estimator: bool = False,
        use_slurm: bool = False,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_trials = n_trials
        self.scoring = scoring
        self.cv = cv
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.return_estimator = return_estimator
        self.use_slurm = use_slurm
        self.verbose = verbose
        self.random_state = random_state

        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )

        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.cv_results_: List[Dict[str, Any]] = []

    def _objective(self, trial: optuna.Trial, X, y, groups):
        # Sample hyperparameters
        params = self.param_distributions(trial)

        # Clone and configure estimator
        estimator = clone(self.estimator).set_params(**params)
        # Run CV with SLURM or locally
        result = cross_validate(
            estimator,
            X,
            y=y,
            groups=groups,
            scoring=self.scoring,
            cv=self.cv,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
            return_estimator=self.return_estimator,
            use_slurm=self.use_slurm,
            verbose=self.verbose,
        )

        # Extract mean score
        test_key = f'test_{self.scoring}' if isinstance(self.scoring, str) else 'test_score'
        if test_key not in result:
            test_key = 'test_score'  # fallback if key doesn't exist
#        test_key = f"test_{self.scoring}" if isinstance(self.scoring, str) else "test_score"
        score = np.mean(result[test_key])
        result["params"] = params
        result["mean_test_score"] = score
        self.cv_results_.append(result)

        return score

    def fit(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        check_cv(cv=self.cv, y=y, classifier=False)  # Validate CV

        self.study.optimize(
            lambda trial: self._objective(trial, X, y, groups),
            n_trials=self.n_trials
        )

        # Best results
        self.best_params_ = self.study.best_params
        self.best_score_ = self.study.best_value

        # Refit best estimator on full data
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
