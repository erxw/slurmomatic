
# slurmomatic
slurmomatic is a Python library for seamless, distributed model evaluation and hyperparameter tuning using SLURM. It wraps scikit-learn-style workflows (cross_val_score, GridSearchCV, cross_val_predict, etc.) and executes them in parallel across SLURM clusters—or locally, if SLURM is unavailable. It also contains a slurmify decorator to turn any function into a slurm-deployable.

---


## Installation
```bash
uv pip install https://github.com/erxw/slurmomatic.git
```

---


## Project Structure
```bash
src/slurmomatic/
├── core.py              # SLURM job dispatching, parallelization primitives
├── model_selection.py   # Cross-validation and hyperparameter search utilities
├── utils.py             # Helper functions (SLURM detection, decorators, etc.)
└── __init__.py
```

---


## Features
Drop-in replacements for cross_val_score, cross_validate, cross_val_predict

Parallelized GridSearchCV and RandomizedSearchCV

SLURM-aware decorators for automatic job dispatch

Nested cross-validation with optional randomized search

Local fallback when SLURM is not available

---


## Quick Start

### Parallel Cross-Validation with SLURM

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from slurmomatic.model_selection import slurm_cross_val_score

X, y = make_classification(n_samples=1000, n_features=20)
clf = LogisticRegression(max_iter=500)

scores = slurm_cross_val_score(clf, X, y, cv=5)
print("SLURM CV scores:", scores)
```

---


### Grid Search with SLURM

```python
from slurmomatic.model_selection import SlurmGridSearchCV

param_grid = {"C": [0.1, 1.0, 10.0]}
search = SlurmGridSearchCV(clf, param_grid, cv=5)
search.fit(X, y)

print("Best score:", search.best_score_)
print("Best params:", search.best_params_)
```

---

### Nested Cross-Validation

```python
from slurmomatic.model_selection import slurm_nested_cross_val_score

nested_scores = slurm_nested_cross_val_score(
    estimator=clf,
    param_grid={"C": [0.1, 1.0]},
    X=X,
    y=y,
    outer_cv=5,
    inner_cv=3
)

print("Nested CV scores:", nested_scores)
```

---

### SLURM-aware Function Decorator

#### Important Note: Add use_slurm=False in Your Function Signatures
The slurmify decorator depends on a use_slurm keyword argument to decide whether to run via SLURM or locally. This should be present in your function signature:

```python
from slurmomatic import slurmify, batch
def my_function(x, y, use_slurm=False): ...
```

---

#### Example 1: Submitting a SLURM Job Array

```python
from slurmomatic import slurmify

@slurmify(slurm_array_parallelism=4, timeout_min=20)
def train(a: int, b: int, use_slurm: bool = False):
    print(f"Training with a={a}, b={b}")

# Run job array of 5 parallel job_arrays
train([1, 2, 3, 4, 5], [10, 20, 30, 40, 50], use_slurm=True)
```

---

#### Example 2: Submitting Multiple Individual Jobs

```python
from slurmomatic import slurmify

@slurmify(timeout_min=10)
def run_experiment(seed: int, use_slurm: bool = False):
    print(f"Running experiment with seed={seed}")

for seed in range(5):
    run_experiment(seed, use_slurm=True)
```
Each call submits its own SLURM job (or runs locally).

---

#### Example 3: Submitting Multiple Batches with Job Arrays
```python
from slurmomatic import slurmify, batch

@slurmify(slurm_array_parallelism=10, timeout_min=30)
def evaluate(x: int, y: int, use_slurm: bool = False):
    print(f"Evaluating with x={x}, y={y}")
    # Prepare large input lists

xs = list(range(1000))
ys = [1] * 1000

# Submit in batches of 200 using job arrays
for x_batch, y_batch in batch(200, xs, ys):
    evaluate(x_batch, y_batch, use_slurm=True)
```
This submits 5 SLURM job arrays, each with 200 jobs.

---

### @slurmify(...) Parameters
You can pass any SLURM submitit parameters directly to the decorator:
```python
@slurmify(timeout_min=30, cpus_per_task=4, gpus_per_node=1, partition="gpu")
```

Special key:

slurm_array_parallelism=10 → Triggers job array mode. 

---

### batch(batch_size: int, *args)
Utility to chunk long input lists into mini-batches.
```python
from slurmomatic import batch

for a_batch, b_batch in batch(100, list_a, list_b):
    train(a_batch, b_batch, use_slurm=True)
```

---

# 🛡️ Notes
✅ If SLURM is not available (sinfo not found or no job ID in environment), the jobs run locally using submitit.LocalExecutor.

---

### Testing
Run all unit tests using pytest:

```bash
pytest tests/
```

All SLURM jobs are mocked during tests using unittest.mock, so they run quickly and do not require a SLURM cluster.

---

### 🧠 SLURM Notes
By default, SLURM logs are saved to ./slurm_logs/. You can change this via the folder argument in most functions or decorators.

The SLURM detection logic checks for the SLURM_JOB_ID environment variable or runs sinfo to confirm availability.

If SLURM is not available, it automatically falls back to submitit.LocalExecutor.

---

### 📘 API Highlights
Function/Class	Description
slurm_cross_val_score	Parallelized cross_val_score
slurm_cross_validate	Parallelized cross_validate
slurm_cross_val_predict	Parallelized cross_val_predict
SlurmGridSearchCV	SLURM version of GridSearchCV
SlurmRandomizedSearchCV	SLURM version of RandomizedSearchCV
slurm_nested_cross_val_score	Nested CV for unbiased model evaluation
slurm_nested_cross_validate	Nested CV returning detailed fold metrics
slurmify	SLURM/Local decorator for standalone jobs

---

### License
MIT License

