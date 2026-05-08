from libc.math cimport pow, sqrt, asinh
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

    for i in prange(l, nogil=True, num_threads=cores):
        if link_flows[i] > 0:

            congested_time[i] = fftime[i] * (
                sqrt(pow(alpha[i], 2) * pow(1 - link_flows[i] / capacity[i], 2)\
                + pow(beta[i], 2)) - alpha[i] * (
                1 - link_flows[i] / capacity[i]) - beta[i] + 2)
        else:
            congested_time[i] = fftime[i]


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

    for i in prange(l, nogil=True, num_threads=cores):
        if link_flows[i] > 0:
            deltaresult[i] = fftime[i] * ((alpha[i] / capacity[i]) - (
                    (pow(alpha[i], 2) * (1 - link_flows[i] / capacity[i])) / (
                    capacity[i] * sqrt(pow(alpha[i], 2) * pow(
                    1 - link_flows[i] / capacity[i], 2) + pow(beta[i], 2)))))

        else:
            deltaresult[i] = fftime[i]


def integral_conical(integral, link_flows, capacity, fftime, alpha, beta, cores):
    """Per-link Beckmann integral ``Z_l = ∫_0^{x_l} c_l(s) ds`` for the
    Conical (Spiess 1990) cost function. Used to report the OpenPath-style
    ``objective`` / ``best_lower_bound`` / ``best_rgap`` convergence metrics.
    """
    cdef int c = cores

    cdef double [:] integral_view = integral
    cdef double [:] link_flows_view = link_flows
    cdef double [:] capacity_view = capacity
    cdef double [:] fftime_view = fftime
    cdef double [:] alpha_view = alpha
    cdef double [:] beta_view = beta

    integral_conical_cython(
        integral_view, link_flows_view, capacity_view,
        fftime_view, alpha_view, beta_view, c,
    )


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void integral_conical_cython(
    double[:] integral,
    double[:] link_flows,
    double [:] capacity,
    double [:] fftime,
    double[:] alpha,
    double [:] beta,
    int cores
) noexcept:
    # Conical cost (with u = x/cap, v = 1 − u):
    #   c(x) = fftime · [ √(α² v² + β²) − α v − β + 2 ]
    #
    # Integrating from 0 to x (i.e. v from 1 down to 1−u):
    #   ∫ √(α² v² + β²) dv = (v/2) √(α² v² + β²) + (β² / (2α)) sinh⁻¹(α v / β)
    #
    # Substituting back and collecting:
    #   Z_l = fftime · [ (2 − β) · x  − α · (x − x²/(2·cap))
    #                    − cap · ( S(1 − u) − S(1) ) ]
    # with S(v) = (v/2) √(α² v² + β²) + (β² / (2α)) sinh⁻¹(α v / β).
    cdef long long i
    cdef long long l = integral.shape[0]
    cdef double x, u, v_low, v_high, s_low, s_high, alpha_i, beta_i, fft_i, cap_i

    for i in prange(l, nogil=True, num_threads=cores):
        if link_flows[i] > 0:
            x = link_flows[i]
            cap_i = capacity[i]
            alpha_i = alpha[i]
            beta_i = beta[i]
            fft_i = fftime[i]
            u = x / cap_i
            v_high = 1.0
            v_low = 1.0 - u
            s_high = 0.5 * v_high * sqrt(alpha_i * alpha_i * v_high * v_high + beta_i * beta_i) \
                     + (beta_i * beta_i / (2.0 * alpha_i)) * asinh(alpha_i * v_high / beta_i)
            s_low = 0.5 * v_low * sqrt(alpha_i * alpha_i * v_low * v_low + beta_i * beta_i) \
                    + (beta_i * beta_i / (2.0 * alpha_i)) * asinh(alpha_i * v_low / beta_i)
            integral[i] = fft_i * (
                (2.0 - beta_i) * x
                - alpha_i * (x - x * x / (2.0 * cap_i))
                - cap_i * (s_low - s_high)
            )
        else:
            integral[i] = 0.0
