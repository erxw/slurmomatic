import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from slurmomatic import slurm_cross_validate, slurm_cross_val_score, get_executor  # Replace 'your_module'


@pytest.fixture
def data():

    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return X, y


def test_slurm_cross_val_score_basic(data):
    X, y = data
    scores = slurm_cross_val_score(
        LogisticRegression(),
        X,
        y,
        cv=5
    )
    assert len(scores) == 5
    assert all(isinstance(score, float) for score in scores)


def test_slurm_cross_validate_default(data):
    X, y = data
    results = slurm_cross_validate(
        LogisticRegression(),
        X,
        y,
        cv=3
    )
    assert "test_score" in results
    assert results["test_score"].shape == (3,)
    assert "fit_time" in results
    assert "score_time" in results


def test_slurm_cross_validate_with_train_score(data):
    X, y = data
    results = slurm_cross_validate(
        LogisticRegression(),
        X,
        y,
        cv=3,
        return_train_score=True
    )
    assert "train_score" in results
    assert results["train_score"].shape == (3,)


def test_slurm_cross_validate_with_estimator(data):
    X, y = data
    results = slurm_cross_validate(
        LogisticRegression(),
        X,
        y,
        cv=2,
        return_estimator=True
    )
    assert "estimator" in results
    assert len(results["estimator"]) == 2
    assert all(hasattr(est, "predict") for est in results["estimator"])


def test_get_executor_creates_folder(tmp_path):
    folder = tmp_path / "executor_logs"
    executor = get_executor(folder, timeout_min=10)
    assert folder.exists()
    assert isinstance(executor, type(get_executor(".")))  # Same type check


def test_cross_val_score_with_custom_cv(data):
    X, y = data
    cv = StratifiedKFold(n_splits=4)
    scores = slurm_cross_val_score(
        LogisticRegression(),
        X,
        y,
        cv=cv
    )
    assert len(scores) == 4
    assert all(isinstance(s, float) for s in scores)


def test_cross_val_with_error_score_handling(data):
    X, y = data
    model = LogisticRegression(max_iter=1)  # Force underfitting or failure
    results = slurm_cross_validate(
        model,
        X,
        y,
        cv=2,
        error_score=0.0
    )
    assert "test_score" in results
    assert results["test_score"].shape == (2,)
