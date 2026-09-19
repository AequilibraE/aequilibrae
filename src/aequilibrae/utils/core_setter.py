from aequilibrae.utils.cython.openmp_helper import omp_get_max_threads
import os

DEFAULT_THREADING_THRESHOLD = 10_000

# Elementwise kernels are memory-bandwidth bound and their OpenMP regions last
# microseconds or milliseconds, so teams wider than this spend more time in barriers
# than computing.
ELEMENTWISE_CORES_CAP = 8


def clamp_cores(cores_count: int):
    """Clamps a requested core count to ``[1, omp_get_max_threads()]``.

    Zero means all available cores; negative values leave that many cores out.
    """
    if isinstance(cores_count, int):
        if cores_count < 0:
            return max(1, omp_get_max_threads() + cores_count)
        if cores_count == 0:
            return omp_get_max_threads()
        elif cores_count > 0:
            return min(omp_get_max_threads(), cores_count)
    else:
        raise ValueError("Number of cores needs to be an integer")


def resolve_cores(system_parameters: dict) -> int:
    """Resolves the number of cores for computation, clamped by ``clamp_cores``.

    The ``AEQ_CPUS`` environment variable wins over the project's ``parameters.yml``
    because the core count is a property of the machine, while the parameter file
    travels with the project. Values that cannot be interpreted as an integer
    resolve to the total number of available cores.
    """
    value = os.environ.get("AEQ_CPUS", system_parameters.get("cpus", omp_get_max_threads()))
    try:
        return clamp_cores(int(value))
    except (TypeError, ValueError):
        return omp_get_max_threads()


def resolve_threading_threshold(system_parameters: dict) -> int:
    """Resolves the minimum array size for threaded execution of elementwise kernels.

    The ``AEQ_THREADING_THRESHOLD`` environment variable wins over the project's
    ``parameters.yml`` because the ideal threshold is a property of the machine,
    while the parameter file travels with the project.
    """
    value = os.environ.get("AEQ_THREADING_THRESHOLD", system_parameters.get("threading_threshold"))
    return DEFAULT_THREADING_THRESHOLD if value is None else int(value)


def resolve_elementwise_cores(system_parameters: dict, cores: int) -> int:
    """Resolves the number of threads for elementwise (``parallel_numpy``/VDF) kernels.

    The ``AEQ_ELEMENTWISE_CPUS`` environment variable wins over the project's
    ``parameters.yml`` (``system: elementwise_cpus``) because the ideal team size
    is a property of the machine, while the parameter file travels with the
    project. Explicit values follow the same conventions as ``clamp_cores``.

    Unlike path finding, whose long regions amortise the barrier cost, these
    kernels never benefit from a full-machine team. When no explicit value is
    given, the team is capped at ``ELEMENTWISE_CORES_CAP`` threads (never more
    than ``cores``).
    """
    value = os.environ.get("AEQ_ELEMENTWISE_CPUS", system_parameters.get("elementwise_cpus"))
    if value is not None:
        try:
            return clamp_cores(int(value))
        except (TypeError, ValueError):
            pass
    return min(cores, ELEMENTWISE_CORES_CAP)
