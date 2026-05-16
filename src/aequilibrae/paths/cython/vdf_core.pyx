import cython
from libc.math cimport pow, sqrt
from cython.parallel import prange

# ------------------------------------------------------------------------------------------------
#                             BPR FUNCTION AND DERIVATIVE
# ------------------------------------------------------------------------------------------------
def bpr(congested_times, link_flows, capacity, fftime, alpha, beta, cores):
    cdef int c = cores

    cdef double [:] congested_view = congested_times
    cdef double [:] link_flows_view = link_flows
    cdef double [:] capacity_view = capacity
    cdef double [:] fftime_view = fftime
    cdef double [:] alpha_view = alpha
    cdef double [:] beta_view = beta

    with nogil:
        bpr_cython(congested_view, link_flows_view, capacity_view, fftime_view, alpha_view, beta_view, c)


def delta_bpr(dbpr, link_flows, capacity, fftime, alpha, beta, cores):
    cdef int c = cores

    cdef double [:] dbpr_view = dbpr
    cdef double [:] link_flows_view = link_flows
    cdef double [:] capacity_view = capacity
    cdef double [:] fftime_view = fftime
    cdef double [:] alpha_view = alpha
    cdef double [:] beta_view = beta

    with nogil:
        dbpr_cython(dbpr_view, link_flows_view, capacity_view, fftime_view, alpha_view, beta_view, c)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void bpr_cython(
    double[:] congested_time,
    double[:] link_flows,
    double [:] capacity,
    double [:] fftime,
    double[:] alpha,
    double [:] beta,
    int cores
) noexcept nogil:
    cdef long long i
    cdef long long l = congested_time.shape[0]

    # TODO: Use prange with use_threads_if when Cython 3.1 is released
    for i in range(l):
        if link_flows[i] > 0:
            congested_time[i] = fftime[i] * (1 + alpha[i] * (pow(link_flows[i] / capacity[i], beta[i])))
        else:
            congested_time[i] = fftime[i]


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void dbpr_cython(
    double[:] deltaresult,
    double[:] link_flows,
    double [:] capacity,
    double [:] fftime,
    double[:] alpha,
    double [:] beta,
    int cores
) noexcept nogil:
    cdef long long i
    cdef long long l = deltaresult.shape[0]

    # TODO: Use prange with use_threads_if when Cython 3.1 is released
    for i in range(l):
        if link_flows[i] > 0:
            deltaresult[i] = fftime[i] * (
                alpha[i] * beta[i] * (pow(link_flows[i] / capacity[i], beta[i]-1))
            ) / capacity[i]
        else:
            deltaresult[i] = fftime[i]


# ------------------------------------------------------------------------------------------------
#                             BPR2 FUNCTION AND DERIVATIVE
# ------------------------------------------------------------------------------------------------


def bpr2(congested_times, link_flows, capacity, fftime, alpha, beta, cores):
    cdef int c = cores

    cdef double [:] congested_view = congested_times
    cdef double [:] link_flows_view = link_flows
    cdef double [:] capacity_view = capacity
    cdef double [:] fftime_view = fftime
    cdef double [:] alpha_view = alpha
    cdef double [:] beta_view = beta

    bpr2_cython(congested_view, link_flows_view, capacity_view, fftime_view, alpha_view, beta_view, c)


def delta_bpr2(dbpr2, link_flows, capacity, fftime, alpha, beta, cores):
    cdef int c = cores

    cdef double [:] dbpr2_view = dbpr2
    cdef double [:] link_flows_view = link_flows
    cdef double [:] capacity_view = capacity
    cdef double [:] fftime_view = fftime
    cdef double [:] alpha_view = alpha
    cdef double [:] beta_view = beta

    dbpr2_cython(dbpr2_view, link_flows_view, capacity_view, fftime_view, alpha_view, beta_view, c)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void bpr2_cython(
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
                congested_time[i] = fftime[i] * (1 + alpha[i] * (
                    pow(link_flows[i] / capacity[i], 2*beta[i])))
            else:
                congested_time[i] = fftime[i] * (1 + alpha[i] * (
                    pow(link_flows[i] / capacity[i], beta[i])))
        else:
            congested_time[i] = fftime[i]


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void dbpr2_cython(
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
                deltaresult[i] = fftime[i] * (alpha[i] * 2 * beta[i] * (
                    pow(link_flows[i] / capacity[i], (2*beta[i])-1))) / (
                    capacity[i])
            else:
                deltaresult[i] = fftime[i] * (alpha[i] * beta[i] * (
                    pow(link_flows[i] / capacity[i], beta[i]-1))) / capacity[i]
        else:
            deltaresult[i] = fftime[i]


# ------------------------------------------------------------------------------------------------
#                             CONICAL FUNCTION AND DERIVATIVE
# ------------------------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------------------------
#                             INRETS FUNCTION AND DERIVATIVE
# ------------------------------------------------------------------------------------------------

def inrets(congested_times, link_flows, capacity, fftime, alpha, cores):
    cdef int c = cores

    cdef double [:] congested_view = congested_times
    cdef double [:] link_flows_view = link_flows
    cdef double [:] capacity_view = capacity
    cdef double [:] fftime_view = fftime
    cdef double [:] alpha_view = alpha

    inrets_cython(congested_view, link_flows_view, capacity_view, fftime_view, alpha_view, c)


def delta_inrets(dbpr, link_flows, capacity, fftime, alpha, cores):
    cdef int c = cores

    cdef double [:] dbpr_view = dbpr
    cdef double [:] link_flows_view = link_flows
    cdef double [:] capacity_view = capacity
    cdef double [:] fftime_view = fftime
    cdef double [:] alpha_view = alpha

    dinrets_cython(dbpr_view, link_flows_view, capacity_view, fftime_view, alpha_view, c)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void inrets_cython(
    double[:] congested_time,
    double[:] link_flows,
    double [:] capacity,
    double [:] fftime,
    double[:] alpha,
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

# ------------------------------------------------------------------------------------------------
#                             AKCELIK FUNCTION AND DERIVATIVE
# ------------------------------------------------------------------------------------------------



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

