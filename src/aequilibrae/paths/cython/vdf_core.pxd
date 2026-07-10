cimport cython

cpdef void bpr_cython(
    double[:] congested_time,
    double[:] link_flows,
    double [:] capacity,
    double [:] fftime,
    double[:] alpha,
    double [:] beta,
    int cores
) noexcept nogil

cpdef void dbpr_cython(
    double[:] deltaresult,
    double[:] link_flows,
    double [:] capacity,
    double [:] fftime,
    double[:] alpha,
    double [:] beta,
    int cores
) noexcept nogil

cpdef void bpr2_cython(
    double[:] congested_time,
    double[:] link_flows,
    double [:] capacity,
    double [:] fftime,
    double[:] alpha,
    double [:] beta,
    int cores
) noexcept

cpdef void dbpr2_cython(
    double[:] deltaresult,
    double[:] link_flows,
    double [:] capacity,
    double [:] fftime,
    double[:] alpha,
    double [:] beta,
    int cores
) noexcept

cpdef void conical_cython(
    double[:] congested_time,
    double[:] link_flows,
    double [:] capacity,
    double [:] fftime,
    double[:] alpha,
    double [:] beta,
    int cores
) noexcept

cpdef void dconical_cython(
    double[:] deltaresult,
    double[:] link_flows,
    double [:] capacity,
    double [:] fftime,
    double[:] alpha,
    double [:] beta,
    int cores
) noexcept

cpdef void inrets_cython(
    double[:] congested_time,
    double[:] link_flows,
    double [:] capacity,
    double [:] fftime,
    double[:] alpha,
    int cores
) noexcept

cpdef void dinrets_cython(
    double[:] deltaresult,
    double[:] link_flows,
    double [:] capacity,
    double [:] fftime,
    double[:] alpha,
    int cores
) noexcept

cpdef void akcelik_cython(
    double[:] congested_time,
    const double[:] link_flows,
    const double [:] capacity,
    const double [:] fftime,
    const double [:] alpha,
    const double[:] tau,
    const double[:] length,
    const int cores
) noexcept

cpdef void akcelik_cython(
    double[:] congested_time,
    const double[:] link_flows,
    const double [:] capacity,
    const double [:] fftime,
    const double [:] alpha,
    const double[:] tau,
    const double[:] length,
    const int cores
) noexcept
