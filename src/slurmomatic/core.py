import os
import inspect
from functools import wraps
from typing import Callable, Any

import submitit


def is_slurm_available() -> bool:
    """
    Check if SLURM is available on the system.
    This function checks for the presence of the SLURM_JOB_ID environment variable
    or verifies if the SLURM command `sinfo` can be executed successfully.
    Returns:
        bool: True if SLURM is available, False otherwise.
    """

    return "SLURM_JOB_ID" in os.environ or os.system("sinfo > /dev/null 2>&1") == 0


def slurmify(**slurm_kwargs):
    """
        Args:
            **slurm_kwargs: Additional keyword arguments to configure the SLURM job submission.
                    slurm_array_parallelism (int, optional): Enables job array mode if provided.
                    folder (str, optional): Submitit log directory (default: 'slurm_logs' or 'local_logs').

        Returns:
            Callable: A decorator that wraps the target function for SLURM job submission.

        Raises:
            ValueError: If `slurm_array_parallelism` is used and not all inputs (except 'use_slurm') are lists/tuples,
                        or if the input lists do not have the same length.

        Example:
            @slurm_entrypoint(slurm_array_parallelism=4, folder='my_logs')
            def my_function(x, y, use_slurm=False):
                # Function implementation
                pass"
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            sig = inspect.signature(fn)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            use_slurm = bound_args.arguments.get("use_slurm", False)
            slurm_array_parallelism = slurm_kwargs.get("slurm_array_parallelism")
            is_array = slurm_array_parallelism is not None
            is_remote = use_slurm and is_slurm_available()

            executor_class = submitit.AutoExecutor if is_remote else submitit.LocalExecutor
            executor_label = "SLURM" if is_remote else "local"
            folder = slurm_kwargs.get("folder", f"{executor_label.lower()}_logs")

            print(f"[slurmify] Using {executor_label}Executor. Logs in '{folder}'")

            executor = executor_class(folder=folder)
            executor.update_parameters(**slurm_kwargs)

            if is_array:
                arg_names = [k for k in bound_args.arguments if k != "use_slurm"]
                arg_lists = [bound_args.arguments[k] for k in arg_names]
                if not all(isinstance(arg, (list, tuple)) for arg in arg_lists):
                    raise ValueError("[slurmify] All inputs (except 'use_slurm') must be lists/tuples when slurm_array_parallelism is used.")

                if not all(len(arg_lists[0]) == len(arg) for arg in arg_lists):
                    raise ValueError("[slurmify] All input lists must have the same length.")

                jobs = executor.map_array(fn, *arg_lists)
                print(f"[slurmify] Submitted job array with job ids: {[job.job_id for job in jobs]}")
                results = [job.result() for job in jobs]
                return results

            else:
                jobs = [executor.submit(fn, *args, **kwargs)]
                print(f"[slurmify] Submitted job array with job ids: {[job.job_id for job in jobs]}")
                results = [job.result() for job in jobs]
                return results

        return wrapper
    return decorator
