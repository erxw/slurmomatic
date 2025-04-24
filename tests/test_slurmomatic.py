import sys
import os
import pytest

# Allow importing slurmify from parent directory
from slurmomatic.utils import batch
from slurmomatic.core import is_slurm_available, slurmify


# test for batch
# tests for slurmify
# tests for cross_validate, cross_val_score
# tests for slurmsearchcv

# ----------------------------------------------------
# Dummy decorated functions
# ----------------------------------------------------
@slurmify(folder="test_logs")
def dummy_job(x, y, use_slurm=False):
    return x + y

@slurmify(folder="test_logs", slurm_array_parallelism=True)
def dummy_job_array(x, y, use_slurm=False):
    return x * y

# ----------------------------------------------------
# Tests for `batch`
# ----------------------------------------------------
def test_batch_basic():
    result = list(batch(2, [1, 2, 3, 4], ['a', 'b', 'c', 'd']))
    assert result == [([1, 2], ['a', 'b']), ([3, 4], ['c', 'd'])]

def test_batch_empty_input():
    result = list(batch(2))
    assert result == []

def test_batch_single_input():
    result = list(batch(2, [10, 20, 30]))
    assert result == [([10, 20],), ([30],)]

def test_batch_unequal_length_raises():
    with pytest.raises(ValueError):
        list(batch(2, [1, 2], [3]))

def test_batch_large_batch_size():
    result = list(batch(10, [1, 2, 3]))
    assert result == [([1, 2, 3],)]

def test_batch_size_one():
    result = list(batch(1, [1, 2], [3, 4]))
    assert result == [([1], [3]), ([2], [4])]

def test_batch_data_type_mismatch():
    result = list(batch(2, [1.0, 2.5, 3.0, 4.5], ["a", "b", "c", "d"]))
    assert result == [([1.0, 2.5], ["a", "b"]), ([3.0, 4.5], ["c", "d"])]


# ----------------------------------------------------
# Tests for `is_slurm_available`
# ----------------------------------------------------
def test_is_slurm_available_returns_bool():
    assert isinstance(is_slurm_available(), bool)


