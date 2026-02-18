from libc.math cimport pow, sqrt
from cython.parallel import prange


def akcelik(congested_times, link_flows, capacity, fftime, alpha, tau, cores):
    cdef int c = cores

    cdef double [:] congested_view = congested_times
    cdef double [:] link_flows_view = link_flows
    cdef double [:] capacity_view = capacity
    cdef double [:] fftime_view = fftime
    cdef double [:] alpha_view = alpha
    cdef double [:] tau_view = tau

    akcelik_cython(congested_view, link_flows_view, capacity_view, fftime_view, alpha_view, tau_view, c)


def delta_akcelik(d_akcelik, link_flows, capacity, fftime, alpha, tau, cores):
    cdef int c = cores

    cdef double [:] d_akcelik_view = d_akcelik
    cdef double [:] link_flows_view = link_flows
    cdef double [:] capacity_view = capacity
    cdef double [:] fftime_view = fftime
    cdef double [:] alpha_view = alpha
    cdef double [:] tau_view = tau

    dakcelik_cython(d_akcelik_view, link_flows_view, capacity_view, fftime_view, alpha_view, tau_view, c)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void akcelik_cython(
    double[:] congested_time,
    const double[:] link_flows,
    const double [:] capacity,
    const double [:] fftime,
    const double [:] alpha,
    const double[:] tau,
    const int cores
) noexcept:
    # tau is redefined as 8 * tau
    cdef long long i
    cdef long long l = congested_time.shape[0]

    cdef:
        double voc = 0.0
        double z = 0.0

    for i in prange(l, nogil=True, num_threads=cores):
        if link_flows[i] > 0:
            voc = link_flows[i] / capacity[i]
            z = voc - 1.0

            congested_time[i] = (
                fftime[i]  # t_o
                + alpha[i] * (
                    z + sqrt(
                         z * z  # z^2
                         + tau[i] * voc / capacity[i]
                     )
                )
            )
        else:
            congested_time[i] = fftime[i]


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void dakcelik_cython(
    double[:] deltaresult,
    const double [:] link_flows,
    const double [:] capacity,
    const double [:] fftime,
    const double [:] alpha,
    const double[:] tau,
    const int cores
) noexcept:
    cdef long long i
    cdef long long l = deltaresult.shape[0]

    for i in prange(l, nogil=True, num_threads=cores):
        if link_flows[i] > 0:
            deltaresult[i] = alpha[i] * (
                0.5 * tau[i] - capacity[i] + link_flows[i]
            ) / (
                capacity[i] * sqrt(pow(capacity[i] - link_flows[i], 2) + tau[i] * link_flows[i])
            ) + (alpha[i] / capacity[i])

        else:
            deltaresult[i] = fftime[i]
