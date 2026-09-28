from aequilibrae.paths.cython.context cimport (
    CppNodeBasedContext, CppTurnBasedContext, NodeBasedContext, TurnBasedContext,
)
from aequilibrae.paths.cython.queries cimport CppSearchQuery, SearchQuery
from aequilibrae.paths.cython.search_results cimport CppMutableSearchResults, SearchResults


cdef enum HeapType:
    FOUR_ARY_HEAP
    PAIRING_HEAP
    STD_PRIORITY_QUEUE


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


cdef extern from "dijkstra.hpp" namespace "aequilibrae::paths::cpp::routing" nogil:
    void cpp_dijkstra "aequilibrae::paths::cpp::routing::dijkstra"[Queue](
        const CppNodeBasedContext &context,
        const CppSearchQuery &query,
        const CppMutableSearchResults &results,
    ) noexcept

    void cpp_turn_dijkstra "aequilibrae::paths::cpp::routing::dijkstra"[Queue](
        const CppTurnBasedContext &context,
        const CppSearchQuery &query,
        const CppMutableSearchResults &results,
    ) noexcept
