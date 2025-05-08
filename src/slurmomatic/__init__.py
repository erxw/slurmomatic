from .core import slurmify, is_slurm_available
from .utils import batch
from .model_selection import (
    cross_val_score,
    slurm_cross_val_predict,
    slurm_cross_val_score,
    slurm_cross_validate,
    slurm_nested_cross_val_score,
    slurm_nested_cross_validate,
)

__all__ = [
    'is_slurm_available',
    'slurmify',
    'batch',
    'cross_val_score',
    'slurm_cross_val_predict',
    'slurm_cross_val_score',
    'slurm_cross_validate',
    'slurm_nested_cross_val_score',
    'slurm_nested_cross_validate',
]