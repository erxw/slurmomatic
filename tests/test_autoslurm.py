import sys
import os
import pytest

# Allow importing autoslurm from parent directory
from autoslurm.utils import batch
from autoslurm.core import is_slurm_available, autoslurm

# ----------------------------------------------------
# Dummy decorated functions
# ----------------------------------------------------
@autoslurm(folder="test_logs")
def dummy_job(x, y, use_slurm=False):
    return x + y

@autoslurm(folder="test_logs", slurm_array_parallelism=True)
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



# ----------------------------------------------------
# Tests for `autoslurm` decorated functions
# ----------------------------------------------------
def test_dummy_job_result_value():
    job = dummy_job(2, 3, use_slurm=False)
    assert job.result() == 5

def test_dummy_job_result_type():
    job = dummy_job(100, 50, use_slurm=False)
    assert isinstance(job.result(), int)

def test_dummy_job_array_correct_results():
    x = [1, 2, 3]
    y = [10, 20, 30]
    jobs = dummy_job_array(x, y, use_slurm=False)
    results = [job.result() for job in jobs]
    assert results == [10, 40, 90]

def test_dummy_job_array_single_element():
    jobs = dummy_job_array([4], [5], use_slurm=False)
    assert [job.result() for job in jobs] == [20]

def test_dummy_job_array_type_check():
    jobs = dummy_job_array([1], [2], use_slurm=False)
    assert all(isinstance(job.result(), int) for job in jobs)

def test_dummy_job_array_empty_lists():
    jobs = dummy_job_array([], [], use_slurm=False)
    assert jobs == []

def test_dummy_job_array_mismatched_lengths():
    with pytest.raises(ValueError, match="must have the same length"):
        dummy_job_array([1, 2], [3], use_slurm=False)

def test_dummy_job_array_invalid_type_inputs():
    with pytest.raises(ValueError, match="must be lists/tuples"):
        dummy_job_array(1, 2, use_slurm=False)