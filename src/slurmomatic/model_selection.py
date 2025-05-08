import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import submitit
from sklearn.base import BaseEstimator, clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import check_scoring
from sklearn.model_selection import (
    ParameterGrid,
    ParameterSampler,
    check_cv,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils import indexable
import pandas as pd
from .core import is_slurm_available



def _fit_and_score(estimator, X_train, y_train, X_test, y_test, scorer, fit_params, return_train_score, return_estimator, error_score):
    """
    Fit the estimator on the training data and compute the score on the test data.

    Parameters
    ----------
    estimator : estimator object
        The estimator to fit.
    X_train : array-like
        Training data.
    y_train : array-like
        Training targets.
    X_test : array-like
        Test data.
    y_test : array-like
        Test targets.
    scorer : callable
        Scoring function.
    fit_params : dict
        Parameters to pass to the fit method.
    return_train_score : bool
        Whether to return the training score.
    return_estimator : bool
        Whether to return the fitted estimator.
    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.

    Returns
    -------
    result : dict
        Dictionary containing fit_time, score_time, test_score, and optionally train_score and estimator.
    """
    result = {}
    est = clone(estimator)
    try:
        start_time = time.time()
        est.fit(X_train, y_train, **fit_params)
        fit_time = time.time() - start_time

        start_time = time.time()
        test_score = scorer(est, X_test, y_test)
        score_time = time.time() - start_time

        result['fit_time'] = fit_time
        result['score_time'] = score_time
        result['test_score'] = test_score

        if return_train_score:
            result['train_score'] = scorer(est, X_train, y_train)
        if return_estimator:
            result['estimator'] = est

    except Exception as e:
        if error_score == 'raise':
            raise e
        else:
            warnings.warn(f"Estimator fit failed. The score on this train-test partition will be set to {error_score}. Details: \n{e}", RuntimeWarning)
            result['fit_time'] = np.nan
            result['score_time'] = np.nan
            result['test_score'] = error_score
            if return_train_score:
                result['train_score'] = error_score
            if return_estimator:
                result['estimator'] = None

    return result


def get_executor(folder: Path, **kwargs):
    if isinstance(folder, str):
        folder = Path(folder)
    folder.mkdir(parents = True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=folder)
    executor.update_parameters(**kwargs)
    return executor

def slurm_cross_validate(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    fit_params=None,
    return_train_score=False,
    return_estimator=False,
    error_score=np.nan,
    executor=None,
    verbose=0
):
    """
    Evaluate metric(s) by cross-validation, parallelized using SLURM via submitit.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,), default=None
        The target variable to try to predict in the case of supervised learning.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into train/test set.
    scoring : str, callable, list/tuple, or dict, default=None
        A single string or a callable to evaluate the predictions on the test set.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
    return_train_score : bool, default=False
        Whether to include train scores.
    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    executor : submitit.Executor, default=None
        A submitit executor for SLURM job submission. If None, a LocalExecutor is used.
    verbose : int, default=0
        The verbosity level.

    Returns
    -------
    scores : dict of float arrays of shape (n_splits,)
        Dictionary containing scores and timing metrics for each fold.
    """
    X, y, groups = indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=hasattr(estimator, "predict"))
    fit_params = fit_params if fit_params is not None else {}

    scorer = check_scoring(estimator, scoring=scoring)
    splits = list(cv.split(X, y, groups))

    def _job(train_idx, test_idx):
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
        else:
            X_train = X[train_idx]
            X_test = X[test_idx]
        if isinstance(y, pd.Series):
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
        else:
            y_train = y[train_idx]
            y_test = y[test_idx]
        return _fit_and_score(
            estimator,
            X_train,
            y_train,
            X_test,
            y_test,
            scorer,
            fit_params,
            return_train_score=return_train_score,
            return_estimator=return_estimator,
            error_score=error_score
        )
    label = "[slurm]" if is_slurm_available() else "[local]"
    if not executor:
        executor = get_executor('.slurm_logs')
    with executor.batch():
        print(f"{label} Submitting {len(splits)} job(s) as a SLURM batch...")
        jobs = [executor.submit(_job, train_idx, test_idx) for train_idx, test_idx in splits]
    print(f"{label} All jobs submitted. Waiting for results...")
    results = [job.result() for job in jobs]
    print(f"{label} All jobs complete.")

    output = {
        'fit_time': np.array([res['fit_time'] for res in results]),
        'score_time': np.array([res['score_time'] for res in results]),
        'test_score': np.array([res['test_score'] for res in results])
    }
    if return_train_score:
        output['train_score'] = np.array([res['train_score'] for res in results])

    if return_estimator:
        output['estimator'] = [res['estimator'] for res in results]

    return output

def slurm_cross_val_score(
    estimator, X, y=None, *, groups=None, scoring=None, cv=None,
    fit_params=None, error_score=np.nan, executor=None, verbose=0
):
    results = slurm_cross_validate(
        estimator, X, y,
        groups=groups,
        scoring=scoring,
        cv=cv,
        fit_params=fit_params,
        return_train_score=False,
        return_estimator=False,
        error_score=error_score,
        executor=executor,
        verbose=verbose
    )
    return results["test_score"]



class BaseSlurmSearchCV(BaseEstimator):
    def __init__(
        self,
        estimator: BaseEstimator,
        param_list: List[Dict[str, Any]],
        scoring=None,
        cv=5,
        executor: Optional[submitit.Executor] = None,
        random_state: Optional[int] = None
    ):
        self.estimator = estimator
        self.param_list = param_list
        self.scoring = scoring
        self.cv = cv
        self.executor = executor or submitit.AutoExecutor("slurm_logs")
        self.executor.update_parameters(timeout_min=60)
        self.random_state = random_state

    def _evaluate_params(self, X, y):
        def score_params(params):
            if self.random_state is not None:
                np.random.seed(self.random_state)
            est = clone(self.estimator).set_params(**params)
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            return {
                "params": params,
                "mean_test_score": np.mean(scores),
                "cv_scores": scores.tolist()
            }

        jobs = [self.executor.submit(score_params, p) for p in self.param_list]
        return [job.result() for job in jobs]

    def fit(self, X, y):
        results = self._evaluate_params(X, y)
        self.cv_results_ = results
        best = max(results, key=lambda r: r["mean_test_score"])
        self.best_params_ = best["params"]
        self.best_score_ = best["mean_test_score"]
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_).fit(X, y)
        return self

    def _check_is_fitted(self):
        if not hasattr(self, "best_estimator_"):
            raise AttributeError("You must call fit() before using this method.")

    def predict(self, X):
        self._check_is_fitted()
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        self._check_is_fitted()
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        self._check_is_fitted()
        return self.best_estimator_.score(X, y)

    def transform(self, X):
        self._check_is_fitted()
        return self.best_estimator_.transform(X)


class SlurmGridSearchCV(BaseSlurmSearchCV):
    def __init__(self, estimator, param_grid, scoring=None, cv=5, executor=None, **kwargs):
        self.param_grid = param_grid  # store for introspection
        param_list = list(ParameterGrid(param_grid))
        super().__init__(estimator, param_list, scoring=scoring, cv=cv, executor=executor, **kwargs)


class SlurmRandomizedSearchCV(BaseSlurmSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None, cv=5, executor=None, random_state=None, **kwargs):
        self.param_distributions = param_distributions  # store for introspection
        self.n_iter = n_iter
        param_list = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))
        super().__init__(estimator, param_list, scoring=scoring, cv=cv, executor=executor, random_state = random_state, **kwargs)
