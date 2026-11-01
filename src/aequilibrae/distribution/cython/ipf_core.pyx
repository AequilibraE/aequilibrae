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
    cdef double[:, ::1] thread_buf = np.zeros((cpus, n_cols))
    assert target_productions.shape[0] == seed_matrix.shape[0] and target_attractions.shape[0] == n_cols

    with nogil:
        while iteration < max_iterations and err >= tolerance:
            iteration += 1
            with parallel(num_threads=cpus):
                tid = threadid()
                memset(&thread_buf[tid, 0], 0, n_cols * sizeof(double))
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
@cython.cdivision(True)
def ipf_block_factors(
        const cython.floating[:, ::1] seed_matrix,
        const double[:] target_productions,
        const double[:] target_attractions,
        const long long[:] groups,
        const double[:, :] block_targets,
        int max_iterations=200,
        double tolerance=0.001,
        int cores=0,
        block_bounds=(0, np.inf),
        bint warn=True,
):
    """Computes the factors that balance the rows, columns and blocks of a square seed matrix, without modifying it.

    Blocks are the pairs of row and column groups, such as districts, and ``block_targets`` holds their totals. Block
    factors start at one, within ``block_bounds``. Returns the row, column and block factors, the number of iterations
    and the gap. The balanced matrix is
    ``row_factors[i] * block_factors[groups[i], groups[j]] * seed_matrix[i, j] * column_factors[j]``.
    """
    # The pass of ipf_factors, with each row seeing the column factors times the block factors of its group. Rows go
    # in chunks of a single group, each with its own buffer, and the buffers are then summed by group
    cdef int k, tid, iteration = 0, cpus = clamp_cores(cores)
    order = np.argsort(groups, kind="stable")
    sorted_groups = np.asarray(groups)[order]
    within = np.arange(order.size) - np.searchsorted(sorted_groups, sorted_groups)
    starts = np.flatnonzero(within % max(ROW_CHUNK, -(-order.size // (8 * cpus))) == 0)

    cdef Py_ssize_t p, i, j, g, h, n_cols = seed_matrix.shape[1], n_groups = block_targets.shape[0]
    cdef double lower = block_bounds[0], upper = block_bounds[1], total, factor, err = INFINITY
    cdef long long[::1] rows_in_order = order, chunk_start = starts, chunk_end = np.r_[starts[1:], order.size]
    cdef long long[::1] chunk_group = sorted_groups[starts]
    cdef double[:, ::1] chunk_buf = np.zeros((starts.size, n_cols)), weights = np.zeros((n_groups, n_cols))
    cdef double[:, ::1] group_cols = np.zeros((n_groups, n_cols)), blk_tot = np.zeros((n_groups, n_groups))
    cdef double[:, ::1] blk_fac = np.clip(np.ones((n_groups, n_groups)), lower, upper)
    cdef double[::1] row_fac = np.ones(order.size), col_fac = np.ones(n_cols), thread_err = np.zeros(cpus)

    with nogil:
        while iteration < max_iterations and err >= tolerance:
            iteration += 1
            for g in range(n_groups):
                for j in range(n_cols):
                    weights[g, j] = col_fac[j] * blk_fac[g, groups[j]]

            with parallel(num_threads=cpus):
                tid = threadid()
                thread_err[tid] = 0
                for k in prange(chunk_start.shape[0], schedule="dynamic", chunksize=1):
                    memset(&chunk_buf[k, 0], 0, n_cols * sizeof(double))
                    for p in range(chunk_start[k], chunk_end[k]):
                        i = rows_in_order[p]
                        row_fac[i] = _scale_row(&seed_matrix[i, 0], &weights[chunk_group[k], 0], n_cols,
                                                target_productions[i], row_fac[i], &chunk_buf[k, 0], &thread_err[tid])

            for k in prange(n_cols, num_threads=cpus, schedule="static"):
                for g in range(n_groups):
                    group_cols[g, k] = 0
                for p in range(chunk_start.shape[0]):
                    group_cols[chunk_group[p], k] += chunk_buf[p, k]

            err = 0
            for k in range(cpus):
                err = max(err, thread_err[k])
            for j in range(n_cols):
                total = 0
                for g in range(n_groups):
                    total += blk_fac[g, groups[j]] * group_cols[g, j]
                col_fac[j] = _factor(target_attractions[j], total, col_fac[j], &err)
                for g in range(n_groups):
                    blk_tot[g, groups[j]] += col_fac[j] * group_cols[g, j]

            # Blocks without anything to scale keep their factor
            for g in range(n_groups):
                for h in range(n_groups):
                    if blk_tot[g, h] > 0:
                        factor = min(max(block_targets[g, h] / blk_tot[g, h], lower), upper)
                        if factor > 0:
                            err = max(err, max(factor / blk_fac[g, h], blk_fac[g, h] / factor) - 1)
                        blk_fac[g, h] = factor
                    blk_tot[g, h] = 0

    if err > tolerance and warn:
        warnings.warn(f"Could not reach convergence in {iteration} iterations: {err}", stacklevel=2)
    return np.asarray(row_fac), np.asarray(col_fac), np.asarray(blk_fac), iteration, err


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


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double _scale_row(const cython.floating *row, const double *weights, Py_ssize_t n, double target,
                              double old, double *totals, double *err) noexcept nogil:
    # Factor that scales the weighted row to its target, adding the scaled row to the totals. Independent
    # accumulators keep the floating-point pipeline busy without fast-math reassociation
    cdef double s0 = 0, s1 = 0, s2 = 0, s3 = 0, factor
    cdef Py_ssize_t j = 0
    if target <= 0:
        return 0
    while j + 4 <= n:
        s0 += row[j] * weights[j]
        s1 += row[j + 1] * weights[j + 1]
        s2 += row[j + 2] * weights[j + 2]
        s3 += row[j + 3] * weights[j + 3]
        j += 4
    while j < n:
        s0 += row[j] * weights[j]
        j += 1
    factor = _factor(target, (s0 + s1) + (s2 + s3), old, err)
    for j in range(n):
        totals[j] += factor * row[j]
    return factor
