from aequilibrae.paths.cython.context cimport (
    CppNodeBasedContext, CppTurnBasedContext, NodeBasedContext, TurnBasedContext,
)
from aequilibrae.paths.cython.queries cimport CppSearchQuery, SearchQuery
from aequilibrae.paths.cython.basic_path_finding cimport HeapType
from aequilibrae.paths.cython.search_results cimport CppMutableSearchResults, SearchResults


ctypedef fused RoutingContext:
    NodeBasedContext
    TurnBasedContext


cdef HeapType routing_heap_from_name(object heap) except *
cdef void run_dijkstra(
    RoutingContext context,
    const CppSearchQuery &query,
    const CppMutableSearchResults &results,
    HeapType heap_type,
) noexcept nogil


cdef extern from "dijkstra.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    void cpp_dijkstra "aequilibrae::paths::cpp::mvp::dijkstra"[Queue](
        const CppNodeBasedContext &context,
        const CppSearchQuery &query,
        const CppMutableSearchResults &results,
    ) noexcept

    void cpp_turn_dijkstra "aequilibrae::paths::cpp::mvp::dijkstra"[Queue](
        const CppTurnBasedContext &context,
        const CppSearchQuery &query,
        const CppMutableSearchResults &results,
    ) noexcept
