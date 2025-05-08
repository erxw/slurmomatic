# SLURM Utilities for Parallel Machine Learning Workflows

This library provides a convenient interface to run Scikit-learn-compatible tasks (such as cross-validation, hyperparameter search, and job arrays) using [SLURM](https://slurm.schedmd.com/) . It can also fallback to local execution if SLURM is not available.

## Features

- `slurmify`: Decorator to run any function on SLURM or locally.
- `slurm_cross_validate`: SLURM-parallelized version of `cross_validate`.
- `slurm_cross_val_score`: SLURM-parallelized version of `cross_val_score`.
- `SlurmGridSearchCV` & `SlurmRandomizedSearchCV`: Parallel hyperparameter search using SLURM.
- `batch`: Utility to split inputs into consistent-size batches.
- `is_slurm_available`: Checks if SLURM is available on the system.

---

## Installation

```bash
uv pip install https://github.com/erxw/slurmomatic.git
```

## Usage

### @slurmify
Wraps any function and runs it via SLURM or locally. Function must have use_slurm: bool as an argument

```python
from slurm_utils import slurmify

@slurmify(folder="my_logs", slurm_array_parallelism=4)
def add(x, y, use_slurm=False):
    return x + y

results = add([1, 2, 3, 4], [10, 20, 30, 40], use_slurm=True)
print(results)  # [11, 22, 33, 44]
```

### slurm_cross_validate
SLURM-parallelized cross-validation.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from slurm_utils import slurm_cross_validate

X, y = make_classification(n_samples=200, n_features=20)
model = LogisticRegression()

results = slurm_cross_validate(model, X, y, cv=5, return_train_score=True)
print(results['test_score'])
print(results['train_score'])
```

### slurm_cross_val_score
Simplified version of slurm_cross_validate that returns only test scores.

```python
from slurm_utils import slurm_cross_val_score

scores = slurm_cross_val_score(model, X, y, cv=3)
print(scores)  # [0.91, 0.89, 0.92]
```


### SlurmGridSearchCV / SlurmRandomizedSearchCV
Drop-in replacements for GridSearchCV and RandomizedSearchCV, powered by SLURM.

```python
from slurm_utils import SlurmGridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

param_grid = {'C': [0.1, 1, 10]}
search = SlurmGridSearchCV(LogisticRegression(), param_grid, cv=StratifiedKFold(3), folder="grid_logs")
search.fit(X, y)

print(search.best_params_)
```


Same idea for:

```python
from slurm_utils import SlurmRandomizedSearchCV

param_dist = {'C': [0.01, 0.1, 1, 10]}
search = SlurmRandomizedSearchCV(LogisticRegression(), param_distributions=param_dist, n_iter=2)
search.fit(X, y)
```

### Nested Cross Validation (outer: cross_val_score; inner: SlurmGridSearchCV)
This uses Scikit-learn’s cross_val_score to evaluate a model after SLURM-powered grid search.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from slurm_utils import SlurmGridSearchCV

# Generate data
X, y = make_classification(n_samples=200, n_features=20, random_state=42)

# Use SlurmGridSearchCV to find best parameters
param_grid = {'C': [0.1, 1, 10]}
search = SlurmGridSearchCV(LogisticRegression(), param_grid, cv=3, folder="logs")

# Use cross_val_score with search estimator
scores = cross_val_score(search, X, y, cv=5)
print("Cross-validation scores:", scores)
```

### Nested Cross Validation (outer: slurm_cross_val_score; inner: GridSearchCV)
This uses GridSearchCV to find hyperparameters and then evaluates the best model using SLURM-based parallel CV scoring.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from slurm_utils import slurm_cross_val_score

# Generate data
X, y = make_classification(n_samples=200, n_features=20, random_state=42)

# Use standard GridSearchCV first
param_grid = {'C': [0.1, 1, 10]}
search = GridSearchCV(LogisticRegression(), param_grid, cv=3)

# Evaluate best model using SLURM-based parallel scoring
scores = slurm_cross_val_score(search, X, y, cv=5)
print("SLURM CV scores:", scores)

```

### batch
Splits multiple input lists into batches.

```python
from slurm_utils import batch

a = [1, 2, 3, 4]
b = ['a', 'b', 'c', 'd']

for ba, bb in batch(2, a, b):
    print(ba, bb)
# [1, 2] ['a', 'b']
# [3, 4] ['c', 'd']
```

### is_slurm_available
Returns True if SLURM is accessible on this system.

```python
from slurm_utils import is_slurm_available

print("SLURM detected?" , is_slurm_available())
```

## Notes
If SLURM is not available, all execution defaults to submitit.LocalExecutor.

SLURM logs will be stored in the folder specified (e.g., slurm_logs, grid_logs).

## Testing
Run all tests using pytest:

```bash
pytest tests/
```

## License
MIT License


