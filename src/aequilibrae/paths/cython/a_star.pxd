from libc.stddef cimport size_t
from aequilibrae.paths.cython.context cimport CppNodeBasedContext, CppTurnBasedContext
from aequilibrae.paths.cython.search_query cimport CppSearchQuery
from aequilibrae.paths.cython.search_results cimport CppMutableSearchResults
from aequilibrae.paths.cython.workspaces cimport CppAStarWorkspace


cdef extern from "heuristics.hpp" namespace "aequilibrae::paths::cpp::routing" nogil:
    cdef cppclass CppEuclideanContext "aequilibrae::paths::cpp::routing::EuclideanContext"[T]:
        CppEuclideanContext() noexcept
        size_t node_count
        const T *x
        const T *y
        T scale
        T distance(size_t, size_t) noexcept

    cdef cppclass CppHaversineContext "aequilibrae::paths::cpp::routing::HaversineContext"[T]:
        CppHaversineContext() noexcept
        size_t node_count
        const T *latitudes
        const T *longitudes
        const T *cos_latitudes
        T scale
        T distance(size_t, size_t) noexcept


cdef class EuclideanContext:
    cdef const double[::1] x, y
    cdef readonly size_t node_count
    cdef readonly double scale
    cdef CppEuclideanContext[double] view(self) noexcept nogil


cdef class HaversineContext:
    cdef const double[::1] latitudes, longitudes, cos_latitudes
    cdef readonly size_t node_count
    cdef readonly double scale
    cdef CppHaversineContext[double] view(self) noexcept nogil


ctypedef fused HeuristicContext:
    EuclideanContext
    HaversineContext

ctypedef fused CppRoutingContext:
    CppNodeBasedContext[double]
    CppTurnBasedContext[double]

ctypedef fused CppHeuristicContext:
    CppEuclideanContext[double]
    CppHaversineContext[double]


cdef extern from "a_star.hpp" namespace "aequilibrae::paths::cpp::routing" nogil:
    void cpp_a_star "aequilibrae::paths::cpp::routing::a_star"(
        const CppNodeBasedContext[double] &, const CppSearchQuery &, size_t,
        const CppEuclideanContext[double] &, const CppMutableSearchResults[double] &, const CppAStarWorkspace[double] &,
    ) noexcept
    void cpp_a_star "aequilibrae::paths::cpp::routing::a_star"(
        const CppNodeBasedContext[double] &, const CppSearchQuery &, size_t,
        const CppHaversineContext[double] &, const CppMutableSearchResults[double] &, const CppAStarWorkspace[double] &,
    ) noexcept
    void cpp_a_star "aequilibrae::paths::cpp::routing::a_star"(
        const CppTurnBasedContext[double] &, const CppSearchQuery &, size_t,
        const CppEuclideanContext[double] &, const CppMutableSearchResults[double] &, const CppAStarWorkspace[double] &,
    ) noexcept
    void cpp_a_star "aequilibrae::paths::cpp::routing::a_star"(
        const CppTurnBasedContext[double] &, const CppSearchQuery &, size_t,
        const CppHaversineContext[double] &, const CppMutableSearchResults[double] &, const CppAStarWorkspace[double] &,
    ) noexcept
