from aequilibrae.utils.core_setter import clamp_cores

cimport cython
import numpy as np
from cython.parallel import parallel, prange, threadid
from libc.math cimport exp, isfinite, log


def deterrence_matrix(cost, function, alpha, beta, max_cost=-1.0, shift=0.0, out=None, cores=0):
    """Evaluates EXPO: exp(-beta * cost), POWER: cost ** -alpha or GAMMA: cost ** alpha * exp(-beta * cost).

    Cells get zero for missing or negative costs, zero costs (except for EXPO), costs at or above a positive
    ``max_cost`` and values that are not finite. ``shift`` is subtracted from the logarithm of every value.
    """
    cost = np.ascontiguousarray(cost, dtype=np.float32 if cost.dtype == np.float32 else np.float64)
    out = np.empty_like(cost) if out is None else out
    log_coefficient = -(alpha or 0) if function == "POWER" else alpha or 0
    _deterrence(cost, out, -(beta or 0), log_coefficient, function == "EXPO", max_cost, shift, clamp_cores(int(cores)))
    return out


@cython.wraparound(False)
@cython.boundscheck(False)
def weighted_cost_sums(const double[:, ::1] weights, const double[::1] row_factors, const double[::1] column_factors,
                       const double[:, ::1] cost, const long long[::1] groups, const double[:, :] block_factors,
                       int cores=0):
    """Sums of trips, trips times cost and trips times log-cost (of positive costs), for
    ``trips = row_factors[i] * block_factors[groups[i], groups[j]] * weights[i, j] * column_factors[j]``"""
    cdef int i, tid, cpus = clamp_cores(cores)
    cdef Py_ssize_t j, g
    cdef double trips, c, row_trips, row_cost, row_log
    cdef double[:, ::1] partial = np.zeros((cpus, 3))
    # The column factors seen by the rows of each group
    cdef double[:, ::1] columns = np.ascontiguousarray(
        np.asarray(column_factors) * np.asarray(block_factors)[:, np.asarray(groups)]
    )

    with nogil, parallel(num_threads=cpus):
        tid = threadid()
        for i in prange(weights.shape[0], schedule="dynamic", chunksize=16):
            g = groups[i]
            row_trips = 0
            row_cost = 0
            row_log = 0
            for j in range(weights.shape[1]):
                trips = weights[i, j] * columns[g, j]
                if trips > 0:
                    c = cost[i, j]
                    row_trips = row_trips + trips
                    row_cost = row_cost + trips * c
                    if c > 0:
                        row_log = row_log + trips * log(c)
            partial[tid, 0] += row_factors[i] * row_trips
            partial[tid, 1] += row_factors[i] * row_cost
            partial[tid, 2] += row_factors[i] * row_log
    return np.asarray(partial).sum(axis=0)


@cython.wraparound(False)
@cython.boundscheck(False)
def _deterrence(const cython.floating[:, ::1] cost, cython.floating[:, ::1] out, double cost_coefficient,
                double log_coefficient, bint zero_cost, double max_cost, double shift, int cpus):
    cdef int i
    cdef Py_ssize_t j
    cdef double c, value

    for i in prange(cost.shape[0], nogil=True, num_threads=cpus, schedule="static"):
        for j in range(cost.shape[1]):
            c = cost[i, j]
            value = 0
            # NaN fails every comparison
            if (c > 0 or (zero_cost and c == 0)) and (max_cost <= 0 or c < max_cost):
                value = exp(cost_coefficient * c + (log_coefficient * log(c) if log_coefficient != 0 else 0) - shift)
                if not isfinite(value):
                    value = 0
            out[i, j] = <cython.floating> value
