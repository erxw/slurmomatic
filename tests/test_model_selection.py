import numpy as np
import pytest
from unittest.mock import MagicMock
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from pathlib import Path
from slurmomatic.model_selection import (
    slurm_cross_val_score,
    slurm_cross_validate,
    slurm_cross_val_predict,
    slurm_nested_cross_val_score,
    slurm_nested_cross_validate,
    SlurmGridSearchCV,
    SlurmRandomizedSearchCV,
    get_slurm_executor,
    dispatch_jobs,
)


@pytest.fixture
def data():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    return X, y


@pytest.fixture
def estimator():
    return LogisticRegression(max_iter=500)


@pytest.fixture
def fake_executor():
    mock_executor = MagicMock()
    mock_executor.submit = lambda func, *args: MagicMock(result=lambda: func(*args))
    return mock_executor


def test_dispatch_jobs(fake_executor):
    def double(x): return x * 2
    results = dispatch_jobs(fake_executor, double, [(1,), (2,), (3,)])
    assert results == [2, 4, 6]


def test_slurm_cross_val_score(data, estimator, fake_executor):
    X, y = data
    scores = slurm_cross_val_score(estimator, X, y, cv=3, executor=fake_executor)
    assert len(scores) == 3
    assert all(0 <= s <= 1 for s in scores)


def test_slurm_cross_validate(data, estimator, fake_executor):
    X, y = data
    results = slurm_cross_validate(estimator, X, y, cv=3, return_train_score=True, executor=fake_executor)
    assert len(results) == 3
    for res in results:
        assert "test_score" in res
        assert "train_score" in res


def test_slurm_cross_val_predict(data, estimator, fake_executor):
    X, y = data
    preds = slurm_cross_val_predict(estimator, X, y, cv=3, executor=fake_executor)
    assert preds.shape == y.shape
    assert isinstance(preds[0], (int, float, np.integer, np.floating, np.ndarray))


def test_slurm_grid_search_cv(data, estimator, fake_executor):
    X, y = data
    param_grid = {"C": [0.1, 1.0]}
    search = SlurmGridSearchCV(estimator, param_grid, cv=3, executor=fake_executor)
    search.fit(X, y)
    assert hasattr(search, "best_params_")
    assert hasattr(search, "best_score_")
    assert hasattr(search, "best_estimator_")


def test_slurm_randomized_search_cv(data, estimator, fake_executor):
    X, y = data
    param_dist = {"C": [0.1, 1.0, 10.0]}
    search = SlurmRandomizedSearchCV(estimator, param_dist, n_iter=2, cv=3, executor=fake_executor, random_state=42)
    search.fit(X, y)
    assert hasattr(search, "best_params_")
    assert hasattr(search, "best_score_")
    assert hasattr(search, "best_estimator_")


def test_slurm_nested_cross_val_score(data, estimator, fake_executor):
    X, y = data
    param_grid = {"C": [0.1, 1.0]}
    scores = slurm_nested_cross_val_score(estimator, param_grid, X, y, outer_cv=3, inner_cv=2, executor=fake_executor)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)


def test_slurm_nested_cross_validate(data, estimator, fake_executor):
    X, y = data
    param_grid = {"C": [0.1, 1.0]}
    results = slurm_nested_cross_validate(estimator, param_grid, X, y, outer_cv=3, inner_cv=2, executor=fake_executor, return_train_score=True)
    assert len(results) == 3
    for res in results:
        assert "test_score" in res
        assert "train_score" in res
        assert "best_params" in res


def test_get_slurm_executor_config():
    executor = get_slurm_executor("/tmp/slurm_test", cpus=2, mem_gb=4, timeout_min=10)
    assert isinstance(executor.folder, (str, Path))

def test_empty_data_raises_error(estimator, fake_executor):
    X = np.empty((0, 5))
    y = np.empty((0,))
    with pytest.raises(ValueError):
        slurm_cross_val_score(estimator, X, y, cv=3, executor=fake_executor)

def test_invalid_param_grid_raises(data, estimator, fake_executor):
    X, y = data
    param_grid = {"C": []}  # Invalid: empty list
    with pytest.raises(ValueError):
        SlurmGridSearchCV(estimator, param_grid, executor=fake_executor).fit(X, y)

def test_invalid_scoring_function(data, estimator, fake_executor):
    X, y = data
    def bad_scoring_fn(model, X, y):
        raise RuntimeError("Invalid scorer")

    with pytest.raises(RuntimeError):
        slurm_cross_val_score(estimator, X, y, scoring=bad_scoring_fn, executor=fake_executor)


def test_pipeline_works_in_grid_search(data, fake_executor):
    X, y = data
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=200))
    ])
    param_grid = {"clf__C": [0.1, 1.0]}
    search = SlurmGridSearchCV(pipe, param_grid, cv=3, executor=fake_executor)
    search.fit(X, y)
    assert isinstance(search.best_score_, float)

def test_nested_cv_result_keys(data, estimator, fake_executor):
    X, y = data
    param_grid = {"C": [0.1, 1.0]}
    results = slurm_nested_cross_validate(estimator, param_grid, X, y, outer_cv=3, inner_cv=2, executor=fake_executor)
    for r in results:
        assert set(r.keys()) >= {"test_score", "best_params"}
def test_random_state_yields_same_results(data, estimator, fake_executor):
    X, y = data
    param_dist = {"C": [0.1, 1.0, 10]}
    r1 = SlurmRandomizedSearchCV(estimator, param_dist, n_iter=2, random_state=42, executor=fake_executor).fit(X, y)
    r2 = SlurmRandomizedSearchCV(estimator, param_dist, n_iter=2, random_state=42, executor=fake_executor).fit(X, y)
    assert r1.best_params_ == r2.best_params_

def test_single_param_grid_still_works(data, estimator, fake_executor):
    X, y = data
    param_grid = {"C": [1.0]}  # No tuning, just fitting
    search = SlurmGridSearchCV(estimator, param_grid, cv=3, executor=fake_executor)
    search.fit(X, y)
    assert isinstance(search.best_score_, float)
from sklearn.tree import DecisionTreeClassifier

def test_tree_model_in_cross_validate(data, fake_executor):
    X, y = data
    tree = DecisionTreeClassifier(max_depth=3)
    results = slurm_cross_validate(tree, X, y, cv=3, executor=fake_executor)
    assert all("test_score" in r for r in results)
def test_executor_none_triggers_default(data, estimator):
    X, y = data
    scores = slurm_cross_val_score(estimator, X, y, cv=3, executor=None)
    assert len(scores) == 3
def test_slurm_nested_cross_val_score_basic(data, estimator, fake_executor):
    X, y = data
    param_grid = {"C": [0.1, 1.0]}
    
    scores = slurm_nested_cross_val_score(
        estimator=estimator,
        param_grid=param_grid,
        X=X,
        y=y,
        outer_cv=3,
        inner_cv=2,
        executor=fake_executor
    )

    assert isinstance(scores, list)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
    assert all(0 <= s <= 1 for s in scores)
def test_slurm_nested_cross_validate_with_train_score(data, estimator, fake_executor):
    X, y = data
    param_grid = {"C": [0.1, 1.0]}

    results = slurm_nested_cross_validate(
        estimator=estimator,
        param_grid=param_grid,
        X=X,
        y=y,
        outer_cv=3,
        inner_cv=2,
        executor=fake_executor,
        return_train_score=True
    )

    assert isinstance(results, list)
    assert len(results) == 3
    for res in results:
        assert "test_score" in res
        assert "train_score" in res
        assert "best_params" in res
def test_slurm_nested_randomized_search(data, estimator, fake_executor):
    X, y = data
    param_grid = {"C": [0.1, 1.0, 10.0]}

    results = slurm_nested_cross_validate(
        estimator=estimator,
        param_grid=param_grid,
        X=X,
        y=y,
        outer_cv=2,
        inner_cv=2,
        executor=fake_executor,
        randomized=True,
        n_iter=2
    )

    assert len(results) == 2
    for res in results:
        assert "best_params" in res
        assert "test_score" in res
