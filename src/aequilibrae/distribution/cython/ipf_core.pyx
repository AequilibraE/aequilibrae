import warnings
from aequilibrae.utils.core_setter import clamp_cores

cimport cython
import numpy as np
from cython.parallel import parallel, prange, threadid
from libc.math cimport INFINITY
from libc.string cimport memset

# Rows given to a thread at a time: few enough to balance hybrid (P/E-core) CPUs, enough to amortise the scheduling
cdef enum:
    ROW_CHUNK = 16


def ipf_core(
        seed_matrix,
        target_productions,
        target_attractions,
        max_iterations=200,
        tolerance=0.001,
        cores = 0,
        warn=True
):
    """Balances a float32 or float64 seed matrix in place so its row and column totals match the targets.

    Only one factor per row and column changes during the iterations, and the matrix is scaled once at the end, which
    gives the iterates of the classic row-then-column IPF. It converges when no factor changed by more than
    ``tolerance`` (relative) in an iteration. ``cores`` is the number of threads, 0 for all of them.

    :Returns:
        **iterations** (:obj:`int`): Number of iterations performed

        **gap** (:obj:`float`): Convergence gap reached
    """
    seed = np.ascontiguousarray(seed_matrix)
    rows, columns, iterations, gap = ipf_factors(
        seed, target_productions, target_attractions, max_iterations, tolerance, cores, warn
    )
    _apply_factors(seed, rows, columns, clamp_cores(int(cores)))
    if seed is not seed_matrix:
        seed_matrix[:] = seed
    return iterations, gap


@cython.wraparound(False)
@cython.boundscheck(False)
def ipf_factors(
        const cython.floating[:, ::1] seed_matrix,
        const double[:] target_productions,
        const double[:] target_attractions,
        int max_iterations=200,
        double tolerance=0.001,
        int cores=0,
        bint warn=True,
):
    """Computes the factors that balance a seed matrix without modifying it. Arguments are those of ``ipf_core``.

    Returns the row factors, the column factors, the number of iterations and the gap. The balanced matrix is
    ``row_factors[:, None] * seed_matrix * column_factors[None, :]``.
    """
    # Each iteration is a single read-only pass over the seed: a row's factor comes from its dot product with the
    # column factors and, while the row is in cache, its scaled values are added to the thread's column totals
    cdef int i, tid, iteration = 0, cpus = clamp_cores(cores)  # OpenMP on Windows requires signed loop variables
    cdef Py_ssize_t j, t, n_cols = seed_matrix.shape[1]
    cdef double err = INFINITY
    cdef double[::1] row_fac = np.ones(seed_matrix.shape[0]), col_fac = np.ones(n_cols), thread_err = np.zeros(cpus)
    cdef double[:, ::1] thread_buf = np.empty((cpus, n_cols))
    assert target_productions.shape[0] == seed_matrix.shape[0] and target_attractions.shape[0] == n_cols

    with nogil:
        while iteration < max_iterations and err >= tolerance:
            iteration += 1
            memset(&thread_buf[0, 0], 0, cpus * n_cols * sizeof(double))
            with parallel(num_threads=cpus):
                tid = threadid()
                thread_err[tid] = 0
                for i in prange(seed_matrix.shape[0], schedule="dynamic", chunksize=ROW_CHUNK):
                    row_fac[i] = _scale_row(&seed_matrix[i, 0], &col_fac[0], n_cols, target_productions[i],
                                            row_fac[i], &thread_buf[tid, 0], &thread_err[tid])

            for i in prange(n_cols, num_threads=cpus, schedule="static"):
                for t in range(1, cpus):
                    thread_buf[0, i] += thread_buf[t, i]
            err = 0
            for t in range(cpus):
                err = max(err, thread_err[t])
            for j in range(n_cols):
                col_fac[j] = _factor(target_attractions[j], thread_buf[0, j], col_fac[j], &err)

    if err > tolerance and warn:
        warnings.warn(f"Could not reach convergence in {iteration} iterations: {err}", stacklevel=2)
    return np.asarray(row_fac), np.asarray(col_fac), iteration, err


@cython.wraparound(False)
@cython.boundscheck(False)
def _apply_factors(cython.floating[:, ::1] seed, const double[::1] row_fac, const double[::1] col_fac, int cpus):
    cdef int i
    cdef Py_ssize_t j
    for i in prange(seed.shape[0], nogil=True, num_threads=cpus, schedule="static"):
        for j in range(seed.shape[1]):
            seed[i, j] = <cython.floating> (row_fac[i] * seed[i, j] * col_fac[j])


@cython.cdivision(True)
cdef inline double _factor(double target, double total, double old, double *err) noexcept nogil:
    # Factor that scales a total to its target, zero without either, keeping the largest relative change in err
    if target <= 0 or total <= 0:
        return 0
    err[0] = max(err[0], max(target / total / old, old * total / target) - 1)
    return target / total


cdef inline double _scale_row(const cython.floating *row, const double *weights, Py_ssize_t n, double target,
                              double old, double *totals, double *err) noexcept nogil:
    # Factor that scales the weighted row to its target, adding the scaled row to the totals
    cdef double row_total = 0, factor
    cdef Py_ssize_t j
    if target <= 0:
        return 0
    for j in range(n):
        row_total += row[j] * weights[j]
    factor = _factor(target, row_total, old, err)
    for j in range(n):
        totals[j] += factor * row[j]
    return factor
