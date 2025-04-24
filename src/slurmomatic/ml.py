import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils import indexable
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring
from collections import defaultdict
from typing import Callable, Any, Dict, List, Optional, Union, Tuple
import time
from slurmomatic.core import slurmify


def _run_fold(
    estimator: BaseEstimator,
    X: Any,
    y: Any,
    scorer: Callable,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fit_params: dict,
    score_params: dict,
    return_train_score: bool = False,
    return_times: bool = False,
    return_estimator: bool = False,
    error_score: Union[str, float] = 0,
    verbose: int = 0,
) -> Dict[str, Any]:
    estimator = clone(estimator)

    try:
        start_time = time.time()
        estimator.fit(X[train_idx], y[train_idx], **fit_params)
        fit_time = time.time() - start_time

        start_time = time.time()
        test_score = scorer(estimator, X[test_idx], y[test_idx], **score_params)
        score_time = time.time() - start_time

        train_score = scorer(estimator, X[train_idx], y[train_idx], **score_params) if return_train_score else None

    except Exception as e:
        if error_score == "raise":
            raise e
        test_score = train_score = error_score
        fit_time = score_time = np.nan

    result = {
        "test_score": test_score,
        "train_score": train_score if return_train_score else None,
        "fit_time": fit_time if return_times else None,
        "score_time": score_time if return_times else None,
        "estimator": estimator if return_estimator else None,
    }

    if verbose > 0:
        print(f"[Fold] Test score: {test_score}")
        if return_train_score:
            print(f"[Fold] Train score: {train_score}")
        if return_times:
            print(f"[Fold] Fit: {fit_time:.2f}s, Score: {score_time:.2f}s")

    return result


@slurmify(slurm_array_parallelism=20)
def _slurm_run_fold(
    estimator: BaseEstimator,
    X: Any,
    y: Any,
    scorer: Callable,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fit_params: dict,
    score_params: dict,
    return_train_score: bool = False,
    return_times: bool = False,
    return_estimator: bool = False,
    error_score: Union[str, float] = 0,
    verbose: int = 0,
) -> Dict[str, Any]:
    
    return _run_fold(
        estimator=estimator,
        X=X,
        y=y,
        scorer=scorer,
        train_idx=train_idx,
        test_idx=test_idx,
        fit_params=fit_params,
        score_params=score_params,
        return_train_score=return_train_score,
        return_times=return_times,
        return_estimator=return_estimator,
        error_score=error_score,
        verbose=verbose,
    )


def _unpack_for_map_array(data: list[dict]) -> dict[list]:
    return {k: [d[k] for d in data] for k in data[0]}

def _submit_cv_folds(
    estimator,
    X,
    y,
    scorer,
    splits,
    *,
    return_train_score,
    return_estimator,
    error_score,
    verbose,
    use_slurm,
    fit_params=None,
    score_params=None,
):
    fit_params = fit_params or {}
    score_params = score_params or {}

    fold_inputs = []
    for train_idx, test_idx in splits:
        fold_inputs.append({
            "estimator": clone(estimator),
            "X": X,
            "y": y,
            "scorer": scorer,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "fit_params": fit_params,
            "score_params": score_params,
            "return_train_score": return_train_score,
            "return_times": True,
            "return_estimator": return_estimator,
            "error_score": error_score,
            "verbose": verbose,
        })

    if use_slurm:
        unpacked_array = _unpack_for_map_array(fold_inputs)
        results = _slurm_run_fold(**unpacked_array)
    else:
        results = [_run_fold(**kwargs) for kwargs in fold_inputs]

    return results


def _aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    aggregated = defaultdict(list)
    for res in results:
        for k, v in res.items():
            if v is not None:
                aggregated[k].append(v)
    return dict(aggregated)


def cross_validate(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_estimator=False,
    error_score=np.nan,
    use_slurm: bool = False,
) -> Dict[str, List[Any]]:
    """
    Custom cross-validation with optional SLURM parallelism.
    """
    X, y, groups = indexable(X, y, groups)
    cv = check_cv(cv=cv, y=y, classifier=False)
    scorer = check_scoring(estimator, scoring=scoring)

    splits = list(cv.split(X, y, groups))

    results = _submit_cv_folds(
        estimator=estimator,
        X=X,
        y=y,
        scorer=scorer,
        splits=splits,
        return_train_score=return_train_score,
        return_estimator=return_estimator,
        error_score=error_score,
        verbose=verbose,
        use_slurm=use_slurm,
        fit_params=fit_params,
    )

    return _aggregate_results(results)


def cross_val_score(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    error_score=np.nan,
    use_slurm: bool = False,
) -> np.ndarray:
    """
    Custom cross_val_score version using toggle_cross_validate.
    """
    results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        groups=groups,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        fit_params=fit_params,
        pre_dispatch=pre_dispatch,
        return_train_score=False,
        return_estimator=False,
        error_score=error_score,
        use_slurm=use_slurm,
    )

    # Use the scoring name if possible
    if isinstance(scoring, str):
        return np.array(results.get(f"test_{scoring}", results.get("test_score")))
    return np.array(results["test_score"])
