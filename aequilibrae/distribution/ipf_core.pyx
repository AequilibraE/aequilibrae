import warnings
import multiprocessing as mp

cimport cython
import numpy as np
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free



def ipf_core(seed_matrix, target_productions, target_attractions, max_iterations=200, tolerance=0.001, cores = 0):

    cdef int max_iter = max_iterations
    cdef double toler = tolerance
    cdef int cpus = cores

    assert target_attractions.shape[0] == seed_matrix.shape[1]
    assert target_productions.shape[0] == seed_matrix.shape[0]

    if cores < 0:
        cpus = max(1, mp.cpu_count() + cores)
    if cores == 0:
        cpus = mp.cpu_count()
    elif cores > 0:
        cpus = min(mp.cpu_count(), cores)

    mat_prod_tot = np.zeros_like(target_productions)
    factor_prod = np.zeros_like(target_productions)

    mat_attr_tot = np.zeros_like(target_attractions)
    factor_attr = np.zeros_like(target_attractions)

    cdef double [:] prod_tot = mat_prod_tot
    cdef double [:] prod_tgt = target_productions
    cdef double [:] prod_factor = factor_prod

    cdef double [:] attr_tot = mat_attr_tot
    cdef double [:] attr_tgt = target_attractions
    cdef double [:] attr_factor = factor_attr
    cdef double [:, :] flows = seed_matrix

    iter, err = _fratar(flows, prod_tot, prod_tgt, prod_factor, attr_tot, attr_tgt, attr_factor, max_iter, toler, cpus)
    if err > tolerance:
        warnings.warn(f"Could not reach convergence in {iter} iterations: {err}")
    return iter, err

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cdef  _fratar(double[:, :] flows,
                   double[:] prod_tot,
                   double[:] prod_tgt,
                   double[:] prod_factor,
                   double[:] attr_tot,
                   double[:] attr_tgt,
                   double[:] attr_factor,
                   int max_iter,
                   double toler,
                   int cpus) noexcept:

    cdef double err = 1.0
    cdef int iter = 0
    cdef long long i, j
    cdef long long I = flows.shape[0]
    cdef long long J = flows.shape[1]

    # we zero everyone that needs to be zero to be able to skip them in the future
    for i in prange(I, nogil=True):
        for j in range(J):
            if prod_tgt[i] + attr_tgt[j] == 0:
                flows[i, j] = 0

    for iter in range(max_iter):
        _total_prods(flows, prod_tgt, prod_tot, cpus)
        err = _factors(prod_tgt, prod_tot, prod_factor, cpus)
        for i in prange(I, nogil=True, num_threads=cpus):
            if prod_tgt[i] > 0:
                for j in range(J):
                    flows[i, j] = flows[i, j] * prod_factor[i]


        _total_attra(flows, prod_tgt, attr_tot, cpus)
        err = _factors(attr_tgt, attr_tot, attr_factor, cpus)
        for i in prange(I, nogil=True, num_threads=cpus):
            if prod_tgt[i] > 0:
                for j in range(J):
                    flows[i, j] = flows[i, j] * attr_factor[j]

        # FUNCTION TO COMPUTE THE ERROR
        err = _calc_err(prod_factor, attr_factor)
        if (err - 1) < toler:
            break

    return iter, err - 1

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void _total_attra(double[:, :] flows,
                        double[:] prod_tgt,
                        double[:] attr_tot,
                        int cpus) noexcept:

    cdef long long i, j, jk
    cdef double  *local_buf
    cdef long long I = flows.shape[0]
    cdef long long J = flows.shape[1]

    # Computes factors
    with nogil, parallel( num_threads=cpus):
        local_buf = <double *> malloc(sizeof(double) * J)
        for j in range(J):
            local_buf[j] = 0
            attr_tot[j] = 0

        for i in prange(I):
            if prod_tgt[i] == 0:
                continue
            for jk in range(J):
                # for jk in array_of_indices_of_non_zeros:
                local_buf[jk] += flows[i, jk]

        with gil:
            for j in range(J):
                attr_tot[j] += local_buf[j]

        free(local_buf)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void _total_prods(double[:, :] flows,
                        double[:] prod_tgt,
                        double[:] prod_tot,
                        int cpus) noexcept nogil:

    cdef long long i, j
    cdef long long I = flows.shape[0]
    cdef long long J = flows.shape[1]

    # Calculate the row totals (prods) from the flows
    for i in prange(I, num_threads=cpus):
        prod_tot[i] = 0
        if prod_tgt[i] == 0:
            continue
        for j in range(J):
            prod_tot[i] += flows[i, j]


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef double _factors(double[:] target,
                      double[:] total,
                      double[:] factor,
                      int cpus) noexcept:

    cdef long long i, I = target.shape[0]
    cdef double err = 1.0

    # Computes factors
    with nogil, parallel(num_threads=cpus):
        for i in prange(I):
            factor[i] = 0
            if target[i] > 0:
                if total[i] == 0:
                    err = -1.0
                else:
                    factor[i] = target[i] / total[i]
    return err

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef double _calc_err(double[:] p_factor,
                       double[:] a_factor) noexcept:

    cdef long long i, I = p_factor.shape[0]
    cdef long long j, J = a_factor.shape[0]
    cdef double err = 1.0

    # Production
    for i in range(I):
        if p_factor[i] > 0:
            err = err if err > 1 / p_factor[i] else 1 / p_factor[i]
        err = err if err > p_factor[i] else p_factor[i]

    # Attraction
    for j in range(J):
        if a_factor[j] > 0:
            err = err if err > 1 / a_factor[j] else 1 / a_factor[j]
        err = err if err > a_factor[j] else a_factor[j]
    return err
