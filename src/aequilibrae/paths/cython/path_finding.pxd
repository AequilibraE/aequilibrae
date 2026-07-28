from libc.stddef cimport size_t

from aequilibrae.utils.cython.bridge cimport AeqLogClosure

cdef extern from "path_finding.hpp" namespace "aequilibrae::paths::cpp" nogil:

    const size_t SENTINEL

    cdef enum class Heuristic:
        HAVERSINE
        EQUIRECTANGULAR

    ctypedef double (*HeuristicFn)(double lat1, double lon1, double lat2,
                                    double lon2, void *data) noexcept nogil

    double haversine_heuristic(double lat1, double lon1, double lat2,
                                double lon2, void *data) noexcept nogil
    double equirectangular_heuristic(double lat1, double lon1, double lat2,
                                      double lon2, void *data) noexcept nogil

    size_t dijkstra[Queue](
        const size_t origin,
        const size_t max_size,
        const double *costs,
        const size_t *csr,
        const size_t *fs,
        size_t *predecessors,
        const size_t *ids,
        size_t *connectors,
        size_t *reached_first,
        const unsigned char *destinations,
        long long destination_count,
        AeqLogClosure *closure,
    ) noexcept nogil

    void a_star[Queue](
        size_t origin,
        size_t destination,
        const size_t max_size,
        const double *costs,
        const size_t *csr,
        const size_t *fs,
        const size_t *nodes_to_indices,
        const double *lats,
        const double *lons,
        size_t *predecessors,
        const size_t *ids,
        size_t *connectors,
        HeuristicFn heur,
        void *heuristic_data,
        AeqLogClosure *closure,
    ) noexcept nogil
