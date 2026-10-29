from aequilibrae.paths.cython.context cimport (
    CppNodeBasedContext, CppTurnBasedContext, NodeBasedContext, TurnBasedContext,
)
from aequilibrae.paths.cython.queries cimport CppSearchQuery, SearchQuery
from aequilibrae.paths.cython.search_results cimport CppMutableSearchResults, SearchResults
from aequilibrae.paths.cython.workspaces cimport CppSearchWorkspace, SearchWorkspace


ctypedef fused RoutingContext:
    NodeBasedContext
    TurnBasedContext


cdef extern from "dijkstra.hpp" namespace "aequilibrae::paths::cpp::routing" nogil:
    void cpp_dijkstra "aequilibrae::paths::cpp::routing::dijkstra"(
        const CppNodeBasedContext &context,
        const CppSearchQuery &query,
        const CppMutableSearchResults &results,
        const CppSearchWorkspace &workspace,
    ) noexcept

    void cpp_dijkstra "aequilibrae::paths::cpp::routing::dijkstra"(
        const CppTurnBasedContext &context,
        const CppSearchQuery &query,
        const CppMutableSearchResults &results,
        const CppSearchWorkspace &workspace,
    ) noexcept
