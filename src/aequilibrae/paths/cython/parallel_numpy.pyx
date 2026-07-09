# cython: boundscheck=False, wraparound=False, embedsignature=True
cimport cython

from cython.parallel cimport prange
from libcpp cimport bool


cpdef cython.floating[::1] sum_axis1(
    cython.floating[::1] out,
    const cython.floating[:, ::1] multiples,
    int cores,
    Py_ssize_t threading_threshold=1000000,
) noexcept nogil:
    cdef:
        Py_ssize_t l = out.shape[0]
        Py_ssize_t k = multiples.shape[1]
        Py_ssize_t i, j
        bool use_threads = l > threading_threshold and threading_threshold >= 0

    assert multiples.shape[0] == l, "mismatched shape"

    for i in prange(l, nogil=True, use_threads_if=use_threads):
        out[i] = 0
        for j in range(k):
            out[i] += multiples[i, j]

    return out


cpdef cython.floating sum_a_times_b_minus_c(
    const cython.floating[::1] array1,
    const cython.floating[::1] array2,
    const cython.floating[::1] array3,
    int cores,
    Py_ssize_t threading_threshold=1000000,
) noexcept nogil:
    cdef:
        Py_ssize_t l = array1.shape[0]
        Py_ssize_t i
        cython.floating result = 0.0
        bool use_threads = l > threading_threshold and threading_threshold >= 0

    assert l == array2.shape[0] == array3.shape[0]

    for i in prange(l, nogil=True, use_threads_if=use_threads):
        result += array1[i] * (array2[i] - array3[i])

    return result


cpdef cython.floating[::1] linear_combination_1d(
    cython.floating[::1] results,
    const cython.floating[::1] array1,
    const cython.floating[::1] array2,
    const cython.floating stepsize,
    int cores,
    Py_ssize_t threading_threshold=1000000,
) noexcept nogil:
    cdef:
        Py_ssize_t l = results.shape[0]
        Py_ssize_t i
        bool use_threads = l > threading_threshold and threading_threshold >= 0

    assert l == array1.shape[0] == array2.shape[0]

    for i in prange(l, nogil=True, use_threads_if=use_threads):
        results[i] = array1[i] * stepsize + array2[i] * (1.0 - stepsize)

    return results


cpdef cython.floating[:, ::1] linear_combination(
    cython.floating[:, ::1] results,
    const cython.floating[:, ::1] array1,
    const cython.floating[:, ::1] array2,
    const cython.floating stepsize,
    int cores,
    Py_ssize_t threading_threshold=1000000,
) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef Py_ssize_t l = results.shape[0]
    cdef Py_ssize_t k = results.shape[1]
    cdef bool use_threads = l * k > threading_threshold and threading_threshold >= 0

    assert array1.shape[0] == l and array2.shape[0] == l, "mismatched shape"
    assert array1.shape[1] == k and array2.shape[1] == k, "mismatched shape"

    for i in prange(l, nogil=True, use_threads_if=use_threads):
        for j in range(k):
            results[i, j] = array1[i, j] * stepsize + array2[i, j] * (1.0 - stepsize)

    return results


cpdef cython.floating[:, :, ::1] linear_combination_skims(
    cython.floating[:, :, ::1] results,
    const cython.floating[:, :, ::1] array1,
    const cython.floating[:, :, ::1] array2,
    const cython.floating stepsize,
    int cores,
    Py_ssize_t threading_threshold=1000000,
) noexcept nogil:
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t a = results.shape[0]
    cdef Py_ssize_t b = results.shape[1]
    cdef Py_ssize_t c = results.shape[2]
    cdef bool use_threads = a * b * c > threading_threshold and threading_threshold >= 0

    assert array1.shape[0] == a and array2.shape[0] == a, "mismatched shape"
    assert array1.shape[1] == b and array2.shape[1] == b, "mismatched shape"
    assert array1.shape[2] == c and array2.shape[2] == c, "mismatched shape"

    for i in prange(a, nogil=True, num_threads=cores, use_threads_if=use_threads):
        for k in range(c):
            for j in range(b):
                results[i, j, k] = array1[i, j, k] * stepsize + array2[i, j, k] * (1.0 - stepsize)

    return results


cpdef cython.floating[:, ::1] triple_linear_combination(
    cython.floating[:, ::1] results,
    const cython.floating[:, ::1] array1,
    const cython.floating[:, ::1] array2,
    const cython.floating[:, ::1] array3,
    const cython.floating[::1] stepsizes,
    int cores,
    Py_ssize_t threading_threshold=1000000,
) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef Py_ssize_t l = results.shape[0]
    cdef Py_ssize_t k = results.shape[1]
    cdef bool use_threads = l * k > threading_threshold and threading_threshold >= 0

    assert stepsizes.shape[0] == 3, "expected 3 stepsizes"
    assert array1.shape[0] == l and array2.shape[0] == l and array3.shape[0] == l, "mismatched shape"
    assert array1.shape[1] == k and array2.shape[1] == k and array3.shape[1] == k, "mismatched shape"

    for i in prange(l, nogil=True, num_threads=cores, use_threads_if=use_threads):
        for j in range(k):
            results[i, j] = array1[i, j] * stepsizes[0] + array2[i, j] * stepsizes[1] + array3[i, j] * stepsizes[2]

    return results


cpdef cython.floating[:, :, ::1] triple_linear_combination_skims(
    cython.floating[:, :, ::1] results,
    const cython.floating[:, :, ::1] array1,
    const cython.floating[:, :, ::1] array2,
    const cython.floating[:, :, ::1] array3,
    const cython.floating[::1] stepsizes,
    int cores,
    Py_ssize_t threading_threshold=1000000,
) noexcept nogil:
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t a = results.shape[0]
    cdef Py_ssize_t b = results.shape[1]
    cdef Py_ssize_t c = results.shape[2]
    cdef bool use_threads = a * b * c > threading_threshold and threading_threshold >= 0

    assert stepsizes.shape[0] == 3, "expected 3 stepsizes"
    assert array1.shape[0] == a and array2.shape[0] == a and array3.shape[0] == a, "mismatched shape"
    assert array1.shape[1] == b and array2.shape[1] == b and array3.shape[1] == b, "mismatched shape"
    assert array1.shape[2] == c and array2.shape[2] == c and array3.shape[2] == c, "mismatched shape"

    for i in prange(a, nogil=True, num_threads=cores, use_threads_if=use_threads):
        for k in range(c):
            for j in range(b):
                results[i, j, k] = array1[i, j, k] * stepsizes[0] + array2[i, j, k] * stepsizes[1]  + \
                                   array3[i, j, k] * stepsizes[2]

    return results


cpdef cython.floating[::1] copy_one_dimension(
    cython.floating[::1] target,
    const cython.floating[::1] source,
    int cores,
    Py_ssize_t threading_threshold=1000000,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t l = target.shape[0]
    cdef bool use_threads = l > threading_threshold and threading_threshold >= 0

    assert source.shape[0] == l, "mismatched shape"

    for i in prange(l, nogil=True, num_threads=cores, use_threads_if=use_threads):
        target[i] = source[i]

    return target


cpdef cython.floating[:, ::1] copy_two_dimensions(
    cython.floating[:, ::1] target,
    const cython.floating[:, ::1] source,
    int cores,
    Py_ssize_t threading_threshold=1000000,
) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef Py_ssize_t l = target.shape[0]
    cdef Py_ssize_t k = target.shape[1]
    cdef bool use_threads = l * k > threading_threshold and threading_threshold >= 0

    assert source.shape[0] == l and source.shape[1] == k, "mismatched shape"

    for i in prange(l, nogil=True, num_threads=cores, use_threads_if=use_threads):
        for j in range(k):
            target[i, j] = source[i, j]

    return target


cpdef cython.floating[:, :, ::1] copy_three_dimensions(
    cython.floating[:, :, ::1] target,
    const cython.floating[:, :, ::1] source,
    int cores,
    Py_ssize_t threading_threshold=1000000,
) noexcept nogil:
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t a = target.shape[0]
    cdef Py_ssize_t b = target.shape[1]
    cdef Py_ssize_t c = target.shape[2]
    cdef bool use_threads = a * b * c > threading_threshold and threading_threshold >= 0

    assert source.shape[0] == a and source.shape[1] == b and source.shape[2] == c, "mismatched shape"

    for i in prange(a, nogil=True, num_threads=cores, use_threads_if=use_threads):
        for k in range(c):
            for j in range(b):
                target[i, j, k] = source[i, j, k]

    return target


cpdef cython.floating[:] aggregate_link_costs(
    const cython.floating[::1] actual,
    cython.floating[:] compressed,  # Non-contiguous
    const long long[::1] crosswalk
) noexcept nogil:
    cdef Py_ssize_t i
    cdef long long k
    cdef Py_ssize_t links = actual.shape[0]
    cdef Py_ssize_t c_l = compressed.shape[0]

    assert crosswalk.shape[0] == links, "mismatched shape"

    for i in range(c_l):
        compressed[i] = 0

    # Sequential: multiple links may map to the same compressed link, so this
    # accumulation cannot be safely parallelised.
    for i in range(links):
        k = crosswalk[i]
        if k < c_l:
            compressed[k] += actual[i]

    return compressed


cpdef cython.floating[:, ::1] assign_link_loads(
    cython.floating[:, ::1] actual,
    const cython.floating[:, ::1] compressed,
    const long long[::1] crosswalk,
    int cores,
    Py_ssize_t threading_threshold=1000000,
) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef long long k
    cdef Py_ssize_t links = actual.shape[0]
    cdef Py_ssize_t n = actual.shape[1]
    cdef bool use_threads = links * n > threading_threshold and threading_threshold >= 0

    assert crosswalk.shape[0] == links, "mismatched shape"
    assert compressed.shape[1] == n, "mismatched shape"

    for i in prange(links, nogil=True, num_threads=cores, use_threads_if=use_threads):
        for j in range(n):
            k = crosswalk[i]
            actual[i, j] = compressed[k, j]

    return actual
