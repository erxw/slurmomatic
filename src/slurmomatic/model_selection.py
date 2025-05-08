from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    ParameterGrid, ParameterSampler,
    check_cv, BaseCrossValidator, cross_val_score
)
import numpy as np
import submitit


def _get_param_list(
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, List[Any]]]],
    randomized: bool,
    n_iter: int,
    random_state: Optional[int]
) -> List[Dict[str, Any]]:
    """
    Generate a list of parameter combinations for hyperparameter search.

    Parameters
    ----------
    param_grid : dict or list of dict
        Grid or distribution of parameters.
    randomized : bool
        Whether to sample randomly or use grid.
    n_iter : int
        Number of samples if randomized.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    List[dict]
        List of parameter dictionaries.
    """
    if randomized:
        return list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state))
    return list(ParameterGrid(param_grid))

def get_slurm_executor(
    folder: str,
    *args, **kwargs
) -> submitit.AutoExecutor:
    """
    Configure a SLURM executor for job submission.

    Parameters
    ----------
    folder : str
        Folder for SLURM logs.

    Returns
    -------
    submitit.AutoExecutor
        Configured SLURM executor.
    """
    executor = submitit.AutoExecutor(folder)
    executor.update_parameters(
        *args, 
        **kwargs
    )
    return executor


### Core parallelizer
def dispatch_jobs(
    executor: submitit.AutoExecutor,
    func: Callable,
    args_list: List[Tuple]
) -> List[Any]:
    """
    Submit and execute a list of jobs in parallel via SLURM.

    Parameters
    ----------
    executor : submitit.AutoExecutor
        SLURM executor instance.
    func : Callable
        Function to run.
    args_list : list of tuple
        Arguments for each job.

    Returns
    -------
    List[Any]
        List of job results.
    """
    jobs = [executor.submit(func, *args) for args in args_list]
    return [job.result() for job in jobs]

### SLURM cross_val_score wrapper
def slurm_cross_val_score(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[int, BaseCrossValidator] = 5,
    scoring: Optional[Callable[[BaseEstimator, np.ndarray, np.ndarray], float]] = None,
    executor: Optional[submitit.AutoExecutor] = None
) -> List[float]:
    """
    Parallelized cross_val_score via SLURM.

    Parameters
    ----------
    estimator : BaseEstimator
    X : np.ndarray
    y : np.ndarray
    cv : int or cross-validator
    scoring : callable, optional
    executor : submitit.AutoExecutor, optional

    Returns
    -------
    List[float]
        Test scores from each fold.
    """
    cv = check_cv(cv)
    splits = list(cv.split(X, y))
    executor = executor or get_slurm_executor("./slurm_logs")

    def score_fold(train_idx, test_idx):
        est = clone(estimator)
        est.fit(X[train_idx], y[train_idx])
        return est.score(X[test_idx], y[test_idx]) if scoring is None else scoring(est, X[test_idx], y[test_idx])

    return dispatch_jobs(executor, score_fold, splits)

### SLURM cross_validate wrapper
def slurm_cross_validate(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[int, BaseCrossValidator] = 5,
    scoring: Optional[Callable[[BaseEstimator, np.ndarray, np.ndarray], float]] = None,
    return_train_score: bool = False,
    executor: Optional[submitit.AutoExecutor] = None
) -> List[Dict[str, float]]:
    """
    Parallelized cross_validate via SLURM.

    Returns test and optionally train scores for each fold.

    Returns
    -------
    List[dict]
        Each dict contains test_score (and optionally train_score).
    """
    cv = check_cv(cv)
    splits = list(cv.split(X, y))
    executor = executor or get_slurm_executor("./slurm_logs")

    def run_fold(train_idx, test_idx):
        est = clone(estimator)
        est.fit(X[train_idx], y[train_idx])
        result = {
            "test_score": scoring(est, X[test_idx], y[test_idx]) if scoring else est.score(X[test_idx], y[test_idx])
        }
        if return_train_score:
            result["train_score"] = est.score(X[train_idx], y[train_idx])
        return result

    return dispatch_jobs(executor, run_fold, splits)


### SLURM cross_val_predict (simplified version)
def slurm_cross_val_predict(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[int, BaseCrossValidator] = 5,
    executor: Optional[submitit.AutoExecutor] = None
) -> np.ndarray:
    """
    Parallelized cross_val_predict via SLURM.

    Returns
    -------
    np.ndarray
        Out-of-sample predictions for each input.
    """
    cv = check_cv(cv)
    splits = list(cv.split(X, y))
    executor = executor or get_slurm_executor("./slurm_logs")
    predictions = np.empty_like(y, dtype=object)

    def predict_fold(train_idx, test_idx):
        est = clone(estimator)
        est.fit(X[train_idx], y[train_idx])
        return test_idx, est.predict(X[test_idx])

    results = dispatch_jobs(executor, predict_fold, splits)
    for idxs, preds in results:
        predictions[idxs] = preds
    return predictions


### SlurmGridSearchCV
class SlurmGridSearchCV:
    """
    SLURM-parallelized version of GridSearchCV.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: Union[Dict[str, List[Any]], List[Dict[str, List[Any]]]],
        scoring: Optional[Callable[[BaseEstimator, np.ndarray, np.ndarray], float]] = None,
        cv: Union[int, BaseCrossValidator] = 5,
        executor: Optional[submitit.AutoExecutor] = None
    ):
        self.estimator = estimator
        self.param_grid = list(ParameterGrid(param_grid))
        self.cv = cv
        self.scoring = scoring
        self.executor = executor or get_slurm_executor("./slurm_logs")

    def fit(self, X: np.ndarray, y: np.ndarray):
        def evaluate(params):
            est = clone(self.estimator).set_params(**params)
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            return params, np.mean(scores)

        results = dispatch_jobs(self.executor, evaluate, [(p,) for p in self.param_grid])
        self.cv_results_ = results
        self.best_params_, self.best_score_ = max(results, key=lambda x: x[1])
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_).fit(X, y)
        return self


### SlurmRandomizedSearchCV
class SlurmRandomizedSearchCV:
    """
    SLURM-parallelized version of RandomizedSearchCV.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: Union[Dict[str, List[Any]], List[Dict[str, List[Any]]]],
        n_iter: int = 10,
        scoring: Optional[Callable[[BaseEstimator, np.ndarray, np.ndarray], float]] = None,
        cv: Union[int, BaseCrossValidator] = 5,
        executor: Optional[submitit.AutoExecutor] = None,
        random_state: Optional[int] = None
    ):
        self.estimator = estimator
        self.param_list = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))
        self.cv = cv
        self.scoring = scoring
        self.executor = executor or get_slurm_executor("./slurm_logs")

    def fit(self, X: np.ndarray, y: np.ndarray):
        def evaluate(params):
            est = clone(self.estimator).set_params(**params)
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            return params, np.mean(scores)

        results = dispatch_jobs(self.executor, evaluate, [(p,) for p in self.param_list])
        self.cv_results_ = results
        self.best_params_, self.best_score_ = max(results, key=lambda x: x[1])
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_).fit(X, y)
        return self
    
def slurm_nested_cross_val_score(
    estimator: BaseEstimator,
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, List[Any]]]],
    X: np.ndarray,
    y: np.ndarray,
    outer_cv: Union[int, BaseCrossValidator] = 5,
    inner_cv: Union[int, BaseCrossValidator] = 3,
    scoring: Optional[Callable[[BaseEstimator, np.ndarray, np.ndarray], float]] = None,
    executor: Optional[submitit.AutoExecutor] = None,
    randomized: bool = False,
    n_iter: int = 10,
    random_state: Optional[int] = None
) -> List[float]:
    """
    Perform nested cross-validation with SLURM for model evaluation.

    Parameters
    ----------
    estimator : BaseEstimator
        The base model to be trained.
    param_grid : dict or list of dict
        Hyperparameter grid or distributions for tuning.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    outer_cv : int or cross-validator
        Outer CV fold count or object.
    inner_cv : int or cross-validator
        Inner CV fold count or object.
    scoring : callable, optional
        Scoring function.
    executor : submitit.AutoExecutor, optional
        SLURM executor for job submission.
    randomized : bool
        Whether to use randomized search instead of grid.
    n_iter : int
        Number of iterations for randomized search.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    List[float]
        Test scores from each outer fold.
    """
    outer_cv = check_cv(outer_cv)
    inner_cv = check_cv(inner_cv)
    splits = list(outer_cv.split(X, y))
    executor = executor or get_slurm_executor("./slurm_logs")

    def evaluate_outer(train_idx, test_idx):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        params_list = _get_param_list(param_grid, randomized, n_iter, random_state)

        def evaluate_inner(params):
            est = clone(estimator).set_params(**params)
            scores = cross_val_score(est, X_train, y_train, cv=inner_cv, scoring=scoring)
            return params, np.mean(scores)

        inner_results = dispatch_jobs(executor, evaluate_inner, [(p,) for p in params_list])
        best_params, _ = max(inner_results, key=lambda x: x[1])

        final_model = clone(estimator).set_params(**best_params).fit(X_train, y_train)
        return scoring(final_model, X_test, y_test) if scoring else final_model.score(X_test, y_test)

    return dispatch_jobs(executor, evaluate_outer, splits)

def slurm_nested_cross_validate(
    estimator: BaseEstimator,
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, List[Any]]]],
    X: np.ndarray,
    y: np.ndarray,
    outer_cv: Union[int, BaseCrossValidator] = 5,
    inner_cv: Union[int, BaseCrossValidator] = 3,
    scoring: Optional[Callable[[BaseEstimator, np.ndarray, np.ndarray], float]] = None,
    executor: Optional[submitit.AutoExecutor] = None,
    randomized: bool = False,
    n_iter: int = 10,
    return_train_score: bool = False,
    random_state: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Perform nested cross-validation with SLURM and return detailed scores and best parameters.

    Parameters
    ----------
    estimator : BaseEstimator
        The base model to be trained.
    param_grid : dict or list of dict
        Hyperparameter grid or distributions for tuning.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    outer_cv : int or cross-validator
        Outer CV fold count or object.
    inner_cv : int or cross-validator
        Inner CV fold count or object.
    scoring : callable, optional
        Scoring function.
    executor : submitit.AutoExecutor, optional
        SLURM executor for job submission.
    randomized : bool
        Whether to use randomized search instead of grid.
    n_iter : int
        Number of iterations for randomized search.
    return_train_score : bool
        Whether to return inner (training) score.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    List[dict]
        Dictionary for each outer fold containing best params, test score, and optionally train score.
    """
    outer_cv = check_cv(outer_cv)
    inner_cv = check_cv(inner_cv)
    splits = list(outer_cv.split(X, y))
    executor = executor or get_slurm_executor("./slurm_logs")

    def evaluate_outer(train_idx, test_idx):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        params_list = _get_param_list(param_grid, randomized, n_iter, random_state)

        def evaluate_inner(params):
            est = clone(estimator).set_params(**params)
            scores = cross_val_score(est, X_train, y_train, cv=inner_cv, scoring=scoring)
            return params, np.mean(scores)

        inner_results = dispatch_jobs(executor, evaluate_inner, [(p,) for p in params_list])
        best_params, best_train_score = max(inner_results, key=lambda x: x[1])

        final_model = clone(estimator).set_params(**best_params).fit(X_train, y_train)
        test_score = scoring(final_model, X_test, y_test) if scoring else final_model.score(X_test, y_test)

        result = {
            "best_params": best_params,
            "test_score": test_score
        }
        if return_train_score:
            result["train_score"] = best_train_score
        return result

    return dispatch_jobs(executor, evaluate_outer, splits)