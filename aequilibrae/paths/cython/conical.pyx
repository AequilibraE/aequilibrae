from libc.math cimport sqrt
from cython.parallel import prange


def conical(congested_times, link_flows, capacity, fftime, alpha, beta, cores):
    cdef int c = cores

    cdef double [:] congested_view = congested_times
    cdef double [:] link_flows_view = link_flows
    cdef double [:] capacity_view = capacity
    cdef double [:] fftime_view = fftime
    cdef double [:] alpha_view = alpha
    cdef double [:] beta_view = beta

    conical_cython(congested_view, link_flows_view, capacity_view, fftime_view, alpha_view, beta_view, c)


def delta_conical(dbpr, link_flows, capacity, fftime, alpha, beta, cores):
    cdef int c = cores

    cdef double [:] dbpr_view = dbpr
    cdef double [:] link_flows_view = link_flows
    cdef double [:] capacity_view = capacity
    cdef double [:] fftime_view = fftime
    cdef double [:] alpha_view = alpha
    cdef double [:] beta_view = beta

    dconical_cython(dbpr_view, link_flows_view, capacity_view, fftime_view, alpha_view, beta_view, c)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void conical_cython(
    double[:] congested_time,
    double[:] link_flows,
    double [:] capacity,
    double [:] fftime,
    double[:] alpha,
    double [:] beta,
    int cores
) noexcept:
    cdef long long i
    cdef long long l = congested_time.shape[0]
    cdef double reserve, radical

    for i in prange(l, nogil=True, num_threads=cores):
        if link_flows[i] <= 0:
            congested_time[i] = fftime[i]
            continue

        # `reserve` is the unused share of capacity, which is what shapes the cone
        reserve = 1.0 - link_flows[i] / capacity[i]
        radical = sqrt(alpha[i] * alpha[i] * reserve * reserve + beta[i] * beta[i])
        congested_time[i] = fftime[i] * (2.0 + radical - alpha[i] * reserve - beta[i])


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void dconical_cython(
    double[:] deltaresult,
    double[:] link_flows,
    double [:] capacity,
    double [:] fftime,
    double[:] alpha,
    double [:] beta,
    int cores
) noexcept:
    cdef long long i
    cdef long long l = deltaresult.shape[0]
    cdef double reserve, radical

    for i in prange(l, nogil=True, num_threads=cores):
        if link_flows[i] <= 0:
            deltaresult[i] = fftime[i]
            continue

        reserve = 1.0 - link_flows[i] / capacity[i]
        radical = sqrt(alpha[i] * alpha[i] * reserve * reserve + beta[i] * beta[i])
        deltaresult[i] = fftime[i] * alpha[i] * (1.0 - alpha[i] * reserve / radical) / capacity[i]
