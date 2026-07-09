import multiprocessing as mp
import os

DEFAULT_THREADING_THRESHOLD = 10_000


def resolve_cores(system_parameters: dict) -> int:
    """Resolves the requested number of cores, before clamping by ``set_cores``.

    The ``AEQ_CPUS`` environment variable wins over the project's ``parameters.yml``
    because the core count is a property of the machine, while the parameter file
    travels with the project. Values that cannot be interpreted as an integer
    resolve to 0 (all cores).
    """
    value = os.environ.get("AEQ_CPUS", system_parameters.get("cpus", 0))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def resolve_threading_threshold(system_parameters: dict) -> int:
    """Resolves the minimum array size for threaded execution of elementwise kernels.

    The ``AEQ_THREADING_THRESHOLD`` environment variable wins over the project's
    ``parameters.yml`` because the ideal threshold is a property of the machine,
    while the parameter file travels with the project.
    """
    value = os.environ.get("AEQ_THREADING_THRESHOLD", system_parameters.get("threading_threshold"))
    return DEFAULT_THREADING_THRESHOLD if value is None else int(value)


def set_cores(cores_count: int):
    if isinstance(cores_count, int):
        if cores_count < 0:
            return max(1, mp.cpu_count() + cores_count)
        if cores_count == 0:
            return mp.cpu_count()
        elif cores_count > 0:
            return min(mp.cpu_count(), cores_count)
    else:
        raise ValueError("Number of cores needs to be an integer")
