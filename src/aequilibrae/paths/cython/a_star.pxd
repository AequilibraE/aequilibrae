from libc.stddef cimport size_t
from aequilibrae.paths.cython.context cimport CppNodeBasedContext, CppTurnBasedContext
from aequilibrae.paths.cython.queries cimport CppSearchQuery
from aequilibrae.paths.cython.search_results cimport CppMutableSearchResults


cdef extern from "heuristics.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    cdef cppclass CppEuclideanContext "aequilibrae::paths::cpp::mvp::EuclideanContext":
        CppEuclideanContext() noexcept
        size_t node_count
        const double *x
        const double *y
        double scale
        double distance(size_t, size_t) noexcept

    cdef cppclass CppHaversineContext "aequilibrae::paths::cpp::mvp::HaversineContext":
        CppHaversineContext() noexcept
        size_t node_count
        const double *latitudes
        const double *longitudes
        const double *cos_latitudes
        double scale
        double distance(size_t, size_t) noexcept


cdef class EuclideanContext:
    cdef const double[::1] x, y
    cdef readonly size_t node_count
    cdef readonly double scale
    cdef CppEuclideanContext view(self) noexcept nogil


cdef class HaversineContext:
    cdef const double[::1] latitudes, longitudes, cos_latitudes
    cdef readonly size_t node_count
    cdef readonly double scale
    cdef CppHaversineContext view(self) noexcept nogil


ctypedef fused HeuristicContext:
    EuclideanContext
    HaversineContext

ctypedef fused CppRoutingContext:
    CppNodeBasedContext
    CppTurnBasedContext

ctypedef fused CppHeuristicContext:
    CppEuclideanContext
    CppHaversineContext


cdef extern from "a_star.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    void node_euclidean "aequilibrae::paths::cpp::mvp::a_star"[Queue](
        const CppNodeBasedContext &, const CppSearchQuery &, size_t,
        const CppEuclideanContext &, const CppMutableSearchResults &,
    ) noexcept
    void node_haversine "aequilibrae::paths::cpp::mvp::a_star"[Queue](
        const CppNodeBasedContext &, const CppSearchQuery &, size_t,
        const CppHaversineContext &, const CppMutableSearchResults &,
    ) noexcept
    void turn_euclidean "aequilibrae::paths::cpp::mvp::a_star"[Queue](
        const CppTurnBasedContext &, const CppSearchQuery &, size_t,
        const CppEuclideanContext &, const CppMutableSearchResults &,
    ) noexcept
    void turn_haversine "aequilibrae::paths::cpp::mvp::a_star"[Queue](
        const CppTurnBasedContext &, const CppSearchQuery &, size_t,
        const CppHaversineContext &, const CppMutableSearchResults &,
    ) noexcept
