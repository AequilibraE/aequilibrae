from cython.parallel import prange

def sum_axis1(totals, multiples, cores):
    cdef int c = cores
    cdef double [:] totals_view = totals
    cdef double [:, :] multiples_view = multiples

    sum_axis1_cython(totals_view, multiples_view, c)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void sum_axis1_cython(double[:] totals,
                            double[:, :] multiples,
                            int cores) noexcept:
  cdef long long i, j
  cdef long long l = totals.shape[0]
  cdef long long k = multiples.shape[1]

  for i in prange(l, nogil=True, num_threads=cores):
      totals[i] = 0
      for j in range(k):
          totals[i] += multiples[i, j]



def sum_a_times_b_minus_c(array1, array2, array3, cores):
    cdef int c = cores
    cdef double result
    cdef double [:] array1_view = array1
    cdef double [:] array2_view = array2
    cdef double [:] array3_view = array3

    result = sum_a_times_b_minus_c_cython(array1_view, array2_view, array3_view, c)
    return result

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef double sum_a_times_b_minus_c_cython(double[:] array1,
                                          double[:] array2,
                                          double[:] array3,
                                          int cores) noexcept:
    cdef long long i
    cdef double row_result
    cdef double result = 0.0
    cdef long long l = array1.shape[0]

    for i in prange(l, nogil=True, num_threads=cores):
        row_result = array1[i] * (array2[i] - array3[i])
        result += row_result

    return result

def linear_combination_1d(results, array1, array2, stepsize, cores):
    cdef double stpsz
    cdef int c = cores

    stpsz = float(stepsize)
    cdef double [:] results_view = results
    cdef double [:] array1_view = array1
    cdef double [:] array2_view = array2

    linear_combination_cython_1d(stpsz, results_view, array1_view, array2_view, c)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void linear_combination_cython_1d(double stepsize,
                                        double[:] results,
                                        double[:] array1,
                                        double[:] array2,
                                        int cores) noexcept:
    cdef long long i
    cdef long long l = results.shape[0]

    for i in prange(l, nogil=True, num_threads=cores):
        results[i] = array1[i] * stepsize + array2[i] * (1.0 - stepsize)


def linear_combination(results, array1, array2, stepsize, cores):
    cdef double stpsz
    cdef int c = cores

    stpsz = float(stepsize)
    cdef double [:, :] results_view = results
    cdef double [:, :] array1_view = array1
    cdef double [:, :] array2_view = array2

    linear_combination_cython(stpsz, results_view, array1_view, array2_view, c)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void linear_combination_cython(double stepsize,
                                     double[:, :] results,
                                     double[:, :] array1,
                                     double[:, :] array2,
                                     int cores) noexcept:
    cdef long long i, j
    cdef long long l = results.shape[0]
    cdef long long k = results.shape[1]

    for j in range(k):
        for i in prange(l, nogil=True, num_threads=cores):
            results[i, j] = array1[i, j] * stepsize + array2[i, j] * (1.0 - stepsize)



def linear_combination_skims(results, array1, array2, stepsize, cores):
    cdef double stpsz
    cdef int c = cores

    stpsz = float(stepsize)
    cdef double [:, :, :] results_view = results
    cdef double [:, :, :] array1_view = array1
    cdef double [:, :, :] array2_view = array2

    linear_combination_skims_cython(stpsz, results_view, array1_view, array2_view, c)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void linear_combination_skims_cython(double stepsize,
                                           double[:, :,:] results,
                                           double[:, :, :] array1,
                                           double[:, :, :] array2,
                                           int cores) noexcept:
    cdef long long i, j, k
    cdef long long a = results.shape[0]
    cdef long long b = results.shape[1]
    cdef long long c = results.shape[2]

    for i in prange(a, nogil=True, num_threads=cores):
        for k in range(c):
            for j in range(b):
                results[i, j, k] = array1[i, j, k] * stepsize + array2[i, j, k] * (1.0 - stepsize)




def triple_linear_combination(results, array1, array2, array3, stepsizes, cores):
    cdef int c = cores

    cdef double [:, :] results_view = results
    cdef double [:, :] array1_view = array1
    cdef double [:, :] array2_view = array2
    cdef double [:, :] array3_view = array3
    cdef double [:] stpsz_view = stepsizes

    triple_linear_combination_cython(stpsz_view, results_view, array1_view, array2_view, array3_view, c)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void triple_linear_combination_cython(double [:] stepsizes,
                                            double[:, :] results,
                                            double[:, :] array1,
                                            double[:, :] array2,
                                            double[:, :] array3,
                                            int cores) noexcept:
    cdef long long i, j
    cdef long long l = results.shape[0]
    cdef long long k = results.shape[1]

    for i in prange(l, nogil=True, num_threads=cores):
        for j in range(k):
            results[i, j] = array1[i, j] * stepsizes[0] + array2[i, j] * stepsizes[1]  + array3[i, j] * stepsizes[2]



def triple_linear_combination_skims(results, array1, array2, array3, stepsizes, cores):
    cdef int c = cores

    cdef double [:, :, :] results_view = results
    cdef double [:, :, :] array1_view = array1
    cdef double [:, :, :] array2_view = array2
    cdef double [:, :, :] array3_view = array3
    cdef double [:] stpsz_view = stepsizes

    triple_linear_combination_cython_skims(stpsz_view, results_view, array1_view, array2_view, array3_view, c)

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void triple_linear_combination_cython_skims(double [:] stepsizes,
                                                  double[:, :, :] results,
                                                  double[:, :, :] array1,
                                                  double[:, :, :] array2,
                                                  double[:, :, :] array3,
                                                  int cores) noexcept:
    cdef long long i, j, k
    cdef long long a = results.shape[0]
    cdef long long b = results.shape[1]
    cdef long long c = results.shape[2]

    for i in prange(a, nogil=True, num_threads=cores):
        for k in range(c):
            for j in range(b):
                results[i, j, k] = array1[i, j, k] * stepsizes[0] + array2[i, j, k] * stepsizes[1]  + \
                                   array3[i, j, k] * stepsizes[2]



def copy_one_dimension(target, source, cores):
    cdef int c = cores

    cdef double [:] target_view = target
    cdef double [:] source_view = source

    copy_one_dimension_cython(target_view, source_view, c)

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void copy_one_dimension_cython(double[:] target,
                                     double[:] source,
                                     int cores) noexcept:
    cdef long long i
    cdef long long l = target.shape[0]

    for i in prange(l, nogil=True, num_threads=cores):
        target[i] = source[i]



def copy_two_dimensions(target, source, cores):
    cdef int c = cores

    cdef double [:, :] target_view = target
    cdef double [:, :] source_view = source

    copy_two_dimensions_cython(target_view, source_view, c)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void copy_two_dimensions_cython(double[:, :] target,
                                      double[:, :] source,
                                      int cores) noexcept:
    cdef long long i, j
    cdef long long l = target.shape[0]
    cdef long long k = target.shape[1]

    for j in range(k):
        for i in prange(l, nogil=True, num_threads=cores):
            target[i, j] = source[i, j]




def copy_three_dimensions(target, source, cores):
    cdef int c = cores

    cdef double [:, :, :] target_view = target
    cdef double [:, :, :] source_view = source

    copy_three_dimensions_cython(target_view, source_view, c)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void copy_three_dimensions_cython(double[:, :, :] target,
                                        double[:, :, :] source,
                                        int cores) noexcept:
    cdef long long i, j, k
    cdef long long a = target.shape[0]
    cdef long long b = target.shape[1]
    cdef long long c = target.shape[2]

    for i in prange(a, nogil=True, num_threads=cores):
        for k in range(c):
            for j in range(b):
                target[i, j, k] = source[i, j, k]



def assign_link_loads(actual_links, compressed_links, crosswalk, cores):
    cdef int c = cores

    cdef double [:, :] actual_view = actual_links
    cdef double [:, :] compressed_view = compressed_links
    cdef long long [:] crosswalk_view = crosswalk

    assign_link_loads_cython(actual_view, compressed_view, crosswalk_view, c)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void assign_link_loads_cython(double[:, :] actual,
                                    double[:, :] compressed,
                                    long long[:] crosswalk,
                                    int cores) noexcept:
    cdef long long i, j, k
    cdef long long links = actual.shape[0]
    cdef long long n = actual.shape[1]

    for i in prange(links, nogil=True, num_threads=cores):
        for j in range(n):
            k = crosswalk[i]
            actual[i, j] = compressed[k, j]


def aggregate_link_costs(actual_costs, compressed_costs, crosswalk):
    cdef double [:] actual_view = actual_costs
    cdef double [:] compressed_view = compressed_costs
    cdef long long [:] crosswalk_view = crosswalk

    aggregate_link_costs_cython(actual_view, compressed_view, crosswalk_view)


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void aggregate_link_costs_cython(double[:] actual,
                                       double[:] compressed,
                                       long long[:] crosswalk) noexcept:
    cdef long long i, j, k
    cdef long long links = actual.shape[0]
    cdef long long c_l = compressed.shape[0]

    for i in range(c_l):
        compressed[i] = 0

    for i in range(links):
        k = crosswalk[i]
        if k < c_l:
            compressed[k] += actual[i]
