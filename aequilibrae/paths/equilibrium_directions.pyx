import numpy as np
from libc.math cimport pow
from cython.parallel import prange


def mu_nu_computations(prev_minus_cur, aon_minus_cur, vdf_der):
    cdef double [:,:] prev_minus_cur_view = prev_minus_cur
    cdef double [:,:] aon_minus_cur_view = aon_minus_cur
    cdef double [:] vdf_der_view = vdf_der
    return cmu_nu_computations(prev_minus_cur_view, aon_minus_cur_view, vdf_der_view)


cpdef double cmu_nu_computations(double[:, :] prev_minus_cur,
                               double[:, :] aon_minus_cur,
                               double [:] vdf_der):
    cdef long long i, j
    cdef long long l = prev_minus_cur.shape[0]
    cdef long long m = prev_minus_cur.shape[1]
    cdef double munu = 0

    for j in range(m):
        for i in prange(l, nogil=True):
            result[i, j] = flow1[i, j] * step_size + flow2[i, j] * (k - step_size)

    return munu

def composition(result, flow1, flow2, stepsize):
    cdef double stp

    stp = stepsize
    cdef double [:,:] result_view = result
    cdef double [:,:] flow1_view = flow1
    cdef double [:,:] flow2_view = flow2

    ccomposition(result_view, flow1_view, flow2_view, stp)

cpdef void ccomposition(double[:,:] result,
                        double[:,:] flow1,
                        double [:,:] flow2,
                        double step_size):
  cdef long long i, j
  cdef long long l = result.shape[0]
  cdef long long m = result.shape[1]
  cdef double k = 1.0

  for j in range(m):
    for i in prange(l, nogil=True):
         result[i, j] = flow1[i, j] * step_size + flow2[i, j] * (k - step_size)


def subtraction(result, flow1, flow2):
    cdef double [:,:] result_view = result
    cdef double [:,:] flow1_view = flow1
    cdef double [:,:] flow2_view = flow2

    csubtraction(result_view, flow1_view, flow2_view)

cpdef void csubtraction(double[:,:] result,
                        double[:,:] flow1,
                        double [:,:] flow2):
  cdef long long i, j
  cdef long long l = result.shape[0]
  cdef long long m = result.shape[1]

  for j in range(m):
    for i in prange(l, nogil=True):
         result[i, j] = flow1[i, j] - flow2[i, j]




def multiplication(result, flow1, flow2):
    cdef double [:,:] result_view = result
    cdef double [:,:] flow1_view = flow1
    cdef double [:,:] flow2_view = flow2

    cmultiplication(result_view, flow1_view, flow2_view)

cpdef void cmultiplication(double[:,:] result,
                        double[:,:] flow1,
                        double [:,:] flow2):
  cdef long long i, j
  cdef long long l = result.shape[0]
  cdef long long m = result.shape[1]

  for j in range(m):
    for i in prange(l, nogil=True):
         result[i, j] = flow1[i, j] * flow2[i, j]
