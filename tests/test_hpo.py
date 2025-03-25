from slurmomatic.core import slurmify
from slurmomatic.utils import batch
from slurmomatic.ml import cross_validate, cross_val_score
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from slurmomatic.hpo import SlurmSearchCV

def param_distributions(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 1, 10)
    }

def test_slurm_search_cv_fit():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    estimator = RandomForestClassifier(random_state=42)
    search = SlurmSearchCV(estimator, param_distributions, n_trials=5, scoring='accuracy', cv=3, random_state=42)
    search.fit(X, y)
    
    assert search.best_estimator_ is not None
    assert search.best_score_ is not None
    assert search.best_params_ is not None
    assert len(search.cv_results_) > 0

def test_slurm_search_cv_predict():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    estimator = RandomForestClassifier(random_state=42)
    search = SlurmSearchCV(estimator, param_distributions, n_trials=5, scoring='accuracy', cv=3, random_state=42)
    search.fit(X, y)
    
    predictions = search.predict(X)
    assert len(predictions) == len(y)
def test_slurm_search_cv_score():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    estimator = RandomForestClassifier(random_state=42)
    search = SlurmSearchCV(estimator, param_distributions, n_trials=5, scoring='accuracy', cv=3, random_state=42)
    search.fit(X, y)
    
    score = search.score(X, y)
    assert isinstance(score, float)

def test_slurm_search_cv_get_cv_results():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    estimator = RandomForestClassifier(random_state=42)
    search = SlurmSearchCV(estimator, param_distributions, n_trials=5, scoring='accuracy', cv=3, random_state=42)
    search.fit(X, y)
    
    cv_results = search.get_cv_results()
    assert isinstance(cv_results, list)
    assert len(cv_results) > 0
    assert 'params' in cv_results[0]
    assert 'mean_test_score' in cv_results[0]