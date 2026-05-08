from libc.math cimport pow, log
from cython.parallel import prange


def inrets(congested_times, link_flows, capacity, fftime, alpha, beta, cores):
    cdef int c = cores

    cdef double [:] congested_view = congested_times
    cdef double [:] link_flows_view = link_flows
    cdef double [:] capacity_view = capacity
    cdef double [:] fftime_view = fftime
    cdef double [:] alpha_view = alpha
    cdef double [:] beta_view = beta

    inrets_cython(congested_view, link_flows_view, capacity_view, fftime_view, alpha_view, beta_view, c)


def delta_inrets(dbpr, link_flows, capacity, fftime, alpha, beta, cores):
    cdef int c = cores

    cdef double [:] dbpr_view = dbpr
    cdef double [:] link_flows_view = link_flows
    cdef double [:] capacity_view = capacity
    cdef double [:] fftime_view = fftime
    cdef double [:] alpha_view = alpha
    cdef double [:] beta_view = beta

    dinrets_cython(dbpr_view, link_flows_view, capacity_view, fftime_view, alpha_view, beta_view, c)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void inrets_cython(
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

    for i in prange(l, nogil=True, num_threads=cores):
        if link_flows[i] > 0:
            if link_flows[i] > capacity[i]:
                congested_time[i] = fftime[i] * (
                    (1.1 - alpha[i])/0.1) * (
                    pow(link_flows[i] / capacity[i], 2))
            else:
                congested_time[i] = fftime[i] * (
                    1.1 - (alpha[i]*(link_flows[i] / capacity[i]))) / (
                    1.1 - (link_flows[i] / capacity[i]))
        else:
            congested_time[i] = fftime[i]


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void dinrets_cython(
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

    for i in prange(l, nogil=True, num_threads=cores):
        if link_flows[i] > 0:
            if link_flows[i] > capacity[i]:
                deltaresult[i] = fftime[i] * (
                    (-20)*(alpha[i]-1.1)*link_flows[i]) / (
                    pow(capacity[i], 2))
            else:
                deltaresult[i] = fftime[i] * (
                    (-110)*(alpha[i]-1)*capacity[i]) / (
                    pow((11*capacity[i])-(10*link_flows[i]), 2))

        else:
            deltaresult[i] = fftime[i]


def integral_inrets(integral, link_flows, capacity, fftime, alpha, beta, cores):
    """Per-link Beckmann integral ``Z_l = ∫_0^{x_l} c_l(s) ds`` for the
    INRETS cost function. Used to report the OpenPath-style ``objective`` /
    ``best_lower_bound`` / ``best_rgap`` convergence metrics.
    """
    cdef int c = cores

    cdef double [:] integral_view = integral
    cdef double [:] link_flows_view = link_flows
    cdef double [:] capacity_view = capacity
    cdef double [:] fftime_view = fftime
    cdef double [:] alpha_view = alpha
    cdef double [:] beta_view = beta

    integral_inrets_cython(
        integral_view, link_flows_view, capacity_view,
        fftime_view, alpha_view, beta_view, c,
    )


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void integral_inrets_cython(
    double[:] integral,
    double[:] link_flows,
    double [:] capacity,
    double [:] fftime,
    double[:] alpha,
    double [:] beta,
    int cores
) noexcept:
    # Piecewise INRETS integral (with u = x/cap):
    #
    #   for x ≤ cap:  c(x) = fftime · (1.1 − α u) / (1.1 − u)
    #     ∫c dx = cap · fftime · [ α u − 1.1 (1−α) ln(1.1 − u) + 1.1 (1−α) ln(1.1) ]
    #
    #   for x > cap:  c(x) = fftime · ((1.1 − α)/0.1) · u²
    #     Z(x) = Z(cap) + cap · fftime · ((1.1 − α)/0.1) · (u³ − 1) / 3
    #     where Z(cap) = cap · fftime · [ α − 1.1 (1−α) (ln(0.1) − ln(1.1)) ]
    cdef long long i
    cdef long long l = integral.shape[0]
    cdef double x, u, cap_i, alpha_i, fft_i, z_cap, log_11

    log_11 = log(1.1)

    for i in prange(l, nogil=True, num_threads=cores):
        if link_flows[i] > 0:
            x = link_flows[i]
            cap_i = capacity[i]
            alpha_i = alpha[i]
            fft_i = fftime[i]
            u = x / cap_i
            if x > cap_i:
                # Closed form at u = 1: ln(1.1 − 1) = ln(0.1)
                z_cap = cap_i * fft_i * (
                    alpha_i - 1.1 * (1.0 - alpha_i) * (log(0.1) - log_11)
                )
                integral[i] = z_cap + cap_i * fft_i * (
                    (1.1 - alpha_i) / 0.1
                ) * (pow(u, 3.0) - 1.0) / 3.0
            else:
                integral[i] = cap_i * fft_i * (
                    alpha_i * u
                    - 1.1 * (1.0 - alpha_i) * log(1.1 - u)
                    + 1.1 * (1.0 - alpha_i) * log_11
                )
        else:
            integral[i] = 0.0
