from cython.parallel import prange

def sum_axis1(totals, multiples):

    cdef double [:] totals_view = totals
    cdef double [:, :] multiples_view = multiples

    sum_axis1_cython(totals_view, multiples_view)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void sum_axis1_cython(double[:] totals,
                            double[:, :] multiples):
  cdef long long i, j
  cdef long long l = totals.shape[0]
  cdef long long k = multiples.shape[1]

  for i in prange(l, nogil=True):
      totals[i] = 0
      for j in range(k):
          totals[i] += multiples[i, j]




def linear_combination(results, array1, array2, stepsize):
    cdef double stpsz

    stpsz = float(stepsize)
    cdef double [:, :] results_view = results
    cdef double [:, :] array1_view = array1
    cdef double [:, :] array2_view = array2

    linear_combination_cython(stpsz, results_view, array1_view, array2_view)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void linear_combination_cython(double stepsize,
                                     double[:, :] results,
                                     double[:, :] array1,
                                     double[:, :] array2):
  cdef long long i, j
  cdef long long l = results.shape[0]
  cdef long long k = results.shape[1]

  for i in prange(l, nogil=True):
      for j in range(k):
          results[i, j] = array1[i, j] * stepsize + array2[i, j] * (1.0 - stepsize)




def linear_combination_skims(results, array1, array2, stepsize):
    cdef double stpsz

    stpsz = float(stepsize)
    cdef double [:, :, :] results_view = results
    cdef double [:, :, :] array1_view = array1
    cdef double [:, :, :] array2_view = array2

    linear_combination_skims_cython(stpsz, results_view, array1_view, array2_view)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void linear_combination_skims_cython(double stepsize,
                                           double[:, :,:] results,
                                           double[:, :, :] array1,
                                           double[:, :, :] array2):
    cdef long long i, j, k
    cdef long long a = results.shape[0]
    cdef long long b = results.shape[1]
    cdef long long c = results.shape[2]

    for k in range(c):
        for j in range(b):
            for i in prange(a, nogil=True):
                results[i, j, k] = array1[i, j, k] * stepsize + array2[i, j, k] * (1.0 - stepsize)




def triple_linear_combination(results, array1, array2, array3, stepsizes):

    cdef double [:, :] results_view = results
    cdef double [:, :] array1_view = array1
    cdef double [:, :] array2_view = array2
    cdef double [:, :] array3_view = array3
    cdef double [:] stpsz_view = stepsizes

    triple_linear_combination_cython(stpsz_view, results_view, array1_view, array2_view, array3_view)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void triple_linear_combination_cython(double [:] stepsizes,
                                            double[:, :] results,
                                            double[:, :] array1,
                                            double[:, :] array2,
                                            double[:, :] array3):
    cdef long long i, j
    cdef long long l = results.shape[0]
    cdef long long k = results.shape[1]

    for j in range(k):
        for i in prange(l, nogil=True):
            results[i, j] = array1[i, j] * stepsizes[0] + array2[i, j] * stepsizes[1]  + array3[i, j] * stepsizes[2]



def triple_linear_combination_skims(results, array1, array2, array3, stepsizes):

    cdef double [:, :, :] results_view = results
    cdef double [:, :, :] array1_view = array1
    cdef double [:, :, :] array2_view = array2
    cdef double [:, :, :] array3_view = array3
    cdef double [:] stpsz_view = stepsizes

    triple_linear_combination_cython_skims(stpsz_view, results_view, array1_view, array2_view, array3_view)

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void triple_linear_combination_cython_skims(double [:] stepsizes,
                                                  double[:, :, :] results,
                                                  double[:, :, :] array1,
                                                  double[:, :, :] array2,
                                                  double[:, :, :] array3):
    cdef long long i, j, k
    cdef long long a = results.shape[0]
    cdef long long b = results.shape[1]
    cdef long long c = results.shape[2]

    for k in range(c):
        for j in range(b):
            for i in prange(a, nogil=True):
                results[i, j, k] = array1[i, j, k] * stepsizes[0] + array2[i, j, k] * stepsizes[1]  + \
                                   array3[i, j, k] * stepsizes[2]



def copy_one_dimension(target, source):

    cdef double [:] target_view = target
    cdef double [:] source_view = source

    copy_one_dimension_cython(target_view, source_view)

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void copy_one_dimension_cython(double[:] target,
                                      double[:] source):
    cdef long long i
    cdef long long l = target.shape[0]

    for i in prange(l, nogil=True):
        target[i] = source[i]



def copy_two_dimensions(target, source):

    cdef double [:, :] target_view = target
    cdef double [:, :] source_view = source

    copy_two_dimensions_cython(target_view, source_view)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void copy_two_dimensions_cython(double[:, :] target,
                                      double[:, :] source):
    cdef long long i, j
    cdef long long l = target.shape[0]
    cdef long long k = target.shape[1]

    for j in range(k):
        for i in prange(l, nogil=True):
            target[i, j] = source[i, j]




def copy_three_dimensions(target, source):

    cdef double [:, :, :] target_view = target
    cdef double [:, :, :] source_view = source

    copy_three_dimensions_cython(target_view, source_view)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void copy_three_dimensions_cython(double[:, :, :] target,
                                        double[:, :, :] source):
    cdef long long i, j, k
    cdef long long a = target.shape[0]
    cdef long long b = target.shape[1]
    cdef long long c = target.shape[2]

    for k in range(c):
        for j in range(b):
            for i in prange(a, nogil=True):
                target[i, j, k] = source[i, j, k]
