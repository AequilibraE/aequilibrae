from aequilibrae.paths.cython.graph_context cimport (
    CppNodeBasedContext, CppTurnBasedContext, NodeBasedContext, TurnBasedContext,
)
from aequilibrae.paths.cython.queries cimport CppSearchQuery, SearchQuery
from aequilibrae.paths.cython.search_results cimport CppMutableSearchResults, SearchResults


ctypedef fused RoutingContext:
    NodeBasedContext
    TurnBasedContext


cdef extern from "dijkstra.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    void cpp_dijkstra "aequilibrae::paths::cpp::mvp::dijkstra"[Queue](
        const CppNodeBasedContext &context,
        const CppSearchQuery &query,
        CppMutableSearchResults results,
    ) noexcept

    void cpp_turn_dijkstra "aequilibrae::paths::cpp::mvp::dijkstra"[Queue](
        const CppTurnBasedContext &context,
        const CppSearchQuery &query,
        CppMutableSearchResults results,
    ) noexcept
