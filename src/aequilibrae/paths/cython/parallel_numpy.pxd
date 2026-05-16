cimport cython

cpdef void sum_axis1_cython(double[:] totals, double[:, :] multiples, int cores) noexcept nogil

cpdef double sum_a_times_b_minus_c_cython(
    double[:] array1,
    double[:] array2,
    double[:] array3,
    int cores
) noexcept nogil

cpdef void linear_combination_cython_1d(
    double stepsize,
    double[:] results,
    double[:] array1,
    double[:] array2,
    int cores
) noexcept nogil

cpdef void linear_combination_cython(
    double stepsize,
    double[:, :] results,
    double[:, :] array1,
    double[:, :] array2,
    int cores
) noexcept nogil

cpdef void linear_combination_skims_cython(
    double stepsize,
    double[:, :, :] results,
    double[:, :, :] array1,
    double[:, :, :] array2,
    int cores
) noexcept

cpdef void triple_linear_combination_cython(
    double[:] stepsizes,
    double[:, :] results,
    double[:, :] array1,
    double[:, :] array2,
    double[:, :] array3,
    int cores
) noexcept

cpdef void triple_linear_combination_cython_skims(
    double[:] stepsizes,
    double[:, :, :] results,
    double[:, :, :] array1,
    double[:, :, :] array2,
    double[:, :, :] array3,
    int cores
) noexcept

cpdef void copy_one_dimension_cython(double[:] target, double[:] source, int cores) noexcept

cpdef void copy_two_dimensions_cython(double[:, :] target, double[:, :] source, int cores) noexcept

cpdef void copy_three_dimensions_cython(double[:, :, :] target, double[:, :, :] source, int cores) noexcept

cpdef void assign_link_loads_cython(
    cython.floating[:, :] actual,
    cython.floating[:, :] compressed,
    const long long[:] crosswalk,
    int cores
) noexcept

cpdef void aggregate_link_costs_cython(double[:] actual, double[:] compressed, const long long[:] crosswalk) noexcept

