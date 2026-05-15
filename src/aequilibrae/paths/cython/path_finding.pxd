from libc.stddef cimport size_t

from aequilibrae.utils.cython.bridge cimport AeqLogClosure

cdef extern from "path_finding.hpp" namespace "aequilibrae::paths::cpp" nogil:
    size_t dijkstra[Queue](
        const size_t origin,
        const size_t max_size,
        const double *costs,
        const size_t *csr,
        const size_t *fs,
        size_t *predecessors,
        AeqLogClosure *closure,
    ) noexcept
