import os
import pytest
import builtins
from unittest.mock import MagicMock, patch
from slurmomatic import slurmify, is_slurm_available


def dummy_function(x, y, use_slurm=False):
    return x + y


def dummy_function_array(x, y, use_slurm=False):
    return x * y

def test_is_slurm_available_env_var(monkeypatch):
    monkeypatch.setitem(os.environ, "SLURM_JOB_ID", "12345")
    assert is_slurm_available()


def test_is_slurm_available_sinfo(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with patch("os.system", return_value=0):
        assert is_slurm_available()


def test_is_slurm_unavailable(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with patch("os.system", return_value=1):
        assert not is_slurm_available()

def test_slurmify_local_exec(monkeypatch):
    local_executor_mock = MagicMock()
    job_mock = MagicMock()
    job_mock.result.return_value = 5
    job_mock.job_id = 42
    local_executor_mock.return_value.submit.return_value = job_mock

    monkeypatch.setitem(builtins.__dict__, "print", lambda *a, **k: None)
    monkeypatch.setattr("slurmomatic.is_slurm_available", lambda: False)
    monkeypatch.setattr("submitit.LocalExecutor", local_executor_mock)

    decorated = slurmify(folder="mock_local_logs")(dummy_function)
    result = decorated(2, 3, use_slurm=True)

    assert result == [5]
    local_executor_mock.assert_called_once()

def test_slurmify_job_array(monkeypatch):
    local_executor_mock = MagicMock()
    job_mock = MagicMock()
    job_mock.result.side_effect = [6, 15]
    job_mock.job_id = 101
    local_executor_mock.return_value.map_array.return_value = [job_mock, job_mock]

    monkeypatch.setitem(builtins.__dict__, "print", lambda *a, **k: None)
    monkeypatch.setattr("slurmomatic.is_slurm_available", lambda: False)
    monkeypatch.setattr("submitit.LocalExecutor", local_executor_mock)

    decorated = slurmify(folder="mock_array_logs", slurm_array_parallelism=2)(dummy_function_array)
    results = decorated([2, 3], [3, 5], use_slurm=True)

    assert results == [6, 15]
    local_executor_mock.return_value.map_array.assert_called_once()

def test_slurmify_array_invalid_input_type(monkeypatch):
    monkeypatch.setitem(builtins.__dict__, "print", lambda *a, **k: None)

    decorated = slurmify(slurm_array_parallelism=2)(dummy_function_array)

    with pytest.raises(ValueError, match="must be lists/tuples"):
        decorated(3, [4, 5], use_slurm=True)


def test_slurmify_array_mismatched_lengths(monkeypatch):
    monkeypatch.setitem(builtins.__dict__, "print", lambda *a, **k: None)

    decorated = slurmify(slurm_array_parallelism=2)(dummy_function_array)

    with pytest.raises(ValueError, match="must have the same length"):
        decorated([1, 2, 3], [1, 2], use_slurm=True)
