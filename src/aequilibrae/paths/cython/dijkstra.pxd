from aequilibrae.paths.cython.context cimport (
    CppNodeBasedContext, CppTurnBasedContext, NodeBasedContext, TurnBasedContext,
)
from aequilibrae.paths.cython.search_query cimport CppSearchQuery, SearchQuery
from aequilibrae.paths.cython.search_results cimport CppMutableSearchResults, SearchResults
from aequilibrae.paths.cython.workspaces cimport CppSearchWorkspace, SearchWorkspace


ctypedef fused RoutingContext:
    NodeBasedContext
    TurnBasedContext


cdef extern from "dijkstra.hpp" namespace "aequilibrae::paths::cpp::routing" nogil:
    void cpp_dijkstra "aequilibrae::paths::cpp::routing::dijkstra"(
        const CppNodeBasedContext[double] &context,
        const CppSearchQuery &query,
        const CppMutableSearchResults[double] &results,
        const CppSearchWorkspace[double] &workspace,
    ) noexcept

    void cpp_dijkstra "aequilibrae::paths::cpp::routing::dijkstra"(
        const CppTurnBasedContext[double] &context,
        const CppSearchQuery &query,
        const CppMutableSearchResults[double] &results,
        const CppSearchWorkspace[double] &workspace,
    ) noexcept
