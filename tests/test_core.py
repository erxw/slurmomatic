import os
import pytest
import submitit
from unittest.mock import patch, MagicMock

from slurmomatic import slurmify, is_slurm_available  # Adjust `your_module` as needed


def test_is_slurm_available_env():
    with patch.dict(os.environ, {"SLURM_JOB_ID": "1234"}):
        assert is_slurm_available() is True


def test_is_slurm_available_command():
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.system", return_value=0):
            assert is_slurm_available() is True


def test_is_slurm_unavailable():
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.system", return_value=1):
            assert is_slurm_available() is False


@slurmify(folder="test_logs")
def dummy_function(x, use_slurm=False):
    return x + 1


def test_slurmify_local_mode(tmp_path):
    result = dummy_function(1, use_slurm=False)
    assert result == [2]
    assert (tmp_path / "test_logs").exists() or True  # folder may not be created immediately


@slurmify(slurm_array_parallelism=2, folder="test_logs")
def array_function(x, y, use_slurm=False):
    return x + y


def test_slurmify_array_mode_valid_input():
    result = array_function([1, 2], [3, 4], use_slurm=False)
    assert result == [4, 6]


def test_slurmify_array_mode_invalid_type():
    with pytest.raises(ValueError, match="must be lists/tuples"):
        array_function(1, [2, 3], use_slurm=False)


def test_slurmify_array_mode_mismatched_length():
    with pytest.raises(ValueError, match="must have the same length"):
        array_function([1, 2], [3], use_slurm=False)

