from libc.math cimport pow
from cython.parallel import prange

def bpr(congested_times, link_flows, capacity, fftime, alpha, beta, cores):
    cdef int c = cores

    cdef double [:] congested_view = congested_times
    cdef double [:] link_flows_view = link_flows
    cdef double [:] capacity_view = capacity
    cdef double [:] fftime_view = fftime
    cdef double [:] alpha_view = alpha
    cdef double [:] beta_view = beta

    bpr_cython(congested_view, link_flows_view, capacity_view, fftime_view, alpha_view, beta_view, c)

def delta_bpr(dbpr, link_flows, capacity, fftime, alpha, beta, cores):
    cdef int c = cores

    cdef double [:] dbpr_view = dbpr
    cdef double [:] link_flows_view = link_flows
    cdef double [:] capacity_view = capacity
    cdef double [:] fftime_view = fftime
    cdef double [:] alpha_view = alpha
    cdef double [:] beta_view = beta

    dbpr_cython(dbpr_view, link_flows_view, capacity_view, fftime_view, alpha_view, beta_view, c)

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void bpr_cython(double[:] congested_time,
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
        congested_time[i] = fftime[i] * (1 + alpha[i] * (pow(link_flows[i] / capacity[i], beta[i])))
      else:
        congested_time[i] = fftime[i]

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void dbpr_cython(double[:] deltaresult,
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
        deltaresult[i] = fftime[i] * (alpha[i] * beta[i] * (pow(link_flows[i] / capacity[i], beta[i]-1)))/ capacity[i]
    else:
        deltaresult[i] = fftime[i]
