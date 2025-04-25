# Slurmomatic

A lightweight Python decorator to conditionally submit functions as SLURM jobs (or job arrays), falling back to local execution when SLURM is not available.

## 🚀 Key Features

- 📦 **Drop-in simple**: Decorate any function with `@slurmify(...)`.
- 🔍 **Auto-detects SLURM**: Will submit jobs via SLURM if available, otherwise runs locally.
- ⚙️ **Unified interface**: Same code works on your laptop or cluster — no changes needed.
- 🧠 **Smart job control**: Supports both individual job submission and SLURM job arrays.

## 🔧 Requirements

- Python 3.10+
- [`submitit`](https://github.com/facebookincubator/submitit)

---

## 🧠 Usage

### Step 1: Import

```python
from slurmomatic import slurmify, batch
```
### Step 2: Decorate your function
Each decorated function must accept a use_slurm: bool argument.

--- 

# ✅ Example 1: Submitting a SLURM Job Array

```python
from slurmomatic import slurmify

@slurmify(slurm_array_parallelism=True, timeout_min=20)
def train(a: int, b: int, use_slurm: bool = False):
    print(f"Training with a={a}, b={b}")

# Run job array of 5 parallel job_arrays
train([1, 2, 3, 4, 5], [10, 20, 30, 40, 50], use_slurm=True)
```

---

# ✅ Example 2: Submitting Multiple Individual Jobs

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

# ✅ Example 3: Submitting Multiple Batches with Job Arrays
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

# 📦 @slurmify(...) Parameters
You can pass any SLURM submitit parameters directly to the decorator:
```python
@slurmify(timeout_min=30, cpus_per_task=4, gpus_per_node=1, partition="gpu")
```

Special key:

slurm_array_parallelism=10 → Triggers job array mode. 

---

# 🧰 batch(batch_size: int, *args)
Utility to chunk long input lists into mini-batches.
```python
from slurmomatic import batch

for a_batch, b_batch in batch(100, list_a, list_b):
    train(a_batch, b_batch, use_slurm=True)
```
---

# nested cross validation
```
python
from slurmomatic.core import slurmify
from slurmomatic.utils import batch
#from sklearn.model_selection import cross_val_score
from slurmomatic.ml import cross_validate, cross_val_score
from optuna.integration import OptunaSearchCV
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import optuna


X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

estimator = RandomForestClassifier(random_state=42)

outer_cv = 5

inner_cv = 3

param_distributions = {
    'n_estimators': optuna.distributions.IntDistribution( 10, 100),

    'max_depth': optuna.distributions.IntDistribution(1, 10)
}

search = OptunaSearchCV(estimator, param_distributions, n_trials=100, scoring='accuracy', cv=inner_cv, random_state=42)
scores = cross_val_score(estimator, X, y, cv=3, use_slurm=True)

print(scores) 

```


# 🛡️ Notes
✅ If SLURM is not available (sinfo not found or no job ID in environment), the jobs run locally using submitit.LocalExecutor.

### Todo: 
1. Need to add returns from jobs
2. Enable requeue

---

# 📜 License
MIT
