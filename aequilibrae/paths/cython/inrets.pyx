from libc.math cimport pow
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
cpdef void inrets_cython(double[:] congested_time,
                    double[:] link_flows,
                    double [:] capacity,
                    double [:] fftime,
                    double[:] alpha,
                    double [:] beta,
                    int cores) noexcept:
    cdef long long i
    cdef long long l = congested_time.shape[0]

    for i in prange(l, nogil=True, num_threads=cores):
        if link_flows[i] > 0:
            if link_flows[i] > capacity[i]:
                congested_time[i] = fftime[i] * (
                    (1.1 - alpha[i])/0.1) * (
                    pow(link_flows[i] / capacity[i],2) )
            else:
                congested_time[i] = fftime[i] * (
                    1.1 - (alpha[i]*(link_flows[i] / capacity[i]))) / (
                    1.1 - (link_flows[i] / capacity[i]) )
        else:
            congested_time[i] = fftime[i]

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void dinrets_cython(double[:] deltaresult,
                        double[:] link_flows,
                        double [:] capacity,
                        double [:] fftime,
                        double[:] alpha,
                        double [:] beta,
                        int cores) noexcept:
    cdef long long i
    cdef long long l = deltaresult.shape[0]

    for i in prange(l, nogil=True, num_threads=cores):
        if link_flows[i] > 0:
            if link_flows[i] > capacity[i]:
                deltaresult[i] = fftime[i] * (
                    (-20)*(alpha[i]-1.1)*link_flows[i]) / (
                    pow(capacity[i],2))
            else:
                deltaresult[i] = fftime[i] * (
                    (-110)*(alpha[i]-1)*capacity[i]) / (
                    pow((11*capacity[i])-(10*link_flows[i]),2))

        else:
            deltaresult[i] = fftime[i]
