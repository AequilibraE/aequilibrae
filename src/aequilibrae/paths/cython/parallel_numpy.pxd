cimport cython


cpdef cython.floating[::1] sum_axis1(
    cython.floating[::1] out,
    const cython.floating[:, ::1] multiples,
    int cores,
    Py_ssize_t threading_threshold=*,
) noexcept nogil

cpdef cython.floating sum_a_times_b_minus_c(
    const cython.floating[::1] array1,
    const cython.floating[::1] array2,
    const cython.floating[::1] array3,
    int cores,
    Py_ssize_t threading_threshold=*,
) noexcept nogil

cpdef cython.floating[::1] linear_combination_1d(
    cython.floating[::1] results,
    const cython.floating[::1] array1,
    const cython.floating[::1] array2,
    const cython.floating stepsize,
    int cores,
    Py_ssize_t threading_threshold=*,
) noexcept nogil

cpdef cython.floating[:, ::1] linear_combination(
    cython.floating[:, ::1] results,
    const cython.floating[:, ::1] array1,
    const cython.floating[:, ::1] array2,
    const cython.floating stepsize,
    int cores,
    Py_ssize_t threading_threshold=*,
) noexcept nogil

cpdef cython.floating[:, :, ::1] linear_combination_skims(
    cython.floating[:, :, ::1] results,
    const cython.floating[:, :, ::1] array1,
    const cython.floating[:, :, ::1] array2,
    const cython.floating stepsize,
    int cores,
    Py_ssize_t threading_threshold=*,
) noexcept nogil

cpdef cython.floating[:, ::1] triple_linear_combination(
    cython.floating[:, ::1] results,
    const cython.floating[:, ::1] array1,
    const cython.floating[:, ::1] array2,
    const cython.floating[:, ::1] array3,
    const cython.floating[::1] stepsizes,
    int cores,
    Py_ssize_t threading_threshold=*,
) noexcept nogil

cpdef cython.floating[:, :, ::1] triple_linear_combination_skims(
    cython.floating[:, :, ::1] results,
    const cython.floating[:, :, ::1] array1,
    const cython.floating[:, :, ::1] array2,
    const cython.floating[:, :, ::1] array3,
    const cython.floating[::1] stepsizes,
    int cores,
    Py_ssize_t threading_threshold=*,
) noexcept nogil

cpdef cython.floating[::1] copy_one_dimension(
    cython.floating[::1] target,
    const cython.floating[::1] source,
    int cores,
    Py_ssize_t threading_threshold=*,
) noexcept nogil

cpdef cython.floating[:, ::1] copy_two_dimensions(
    cython.floating[:, ::1] target,
    const cython.floating[:, ::1] source,
    int cores,
    Py_ssize_t threading_threshold=*,
) noexcept nogil

cpdef cython.floating[:, :, ::1] copy_three_dimensions(
    cython.floating[:, :, ::1] target,
    const cython.floating[:, :, ::1] source,
    int cores,
    Py_ssize_t threading_threshold=*,
) noexcept nogil

cpdef cython.floating[:, ::1] assign_link_loads(
    cython.floating[:, ::1] actual,
    const cython.floating[:, ::1] compressed,
    const long long[::1] crosswalk,
    int cores,
    Py_ssize_t threading_threshold=*,
) noexcept nogil

cpdef cython.floating[:] aggregate_link_costs(
    const cython.floating[::1] actual,
    cython.floating[:] compressed,
    const long long[::1] crosswalk
) noexcept nogil
