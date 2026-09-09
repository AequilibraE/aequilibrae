from libc.stddef cimport size_t

from aequilibrae.paths.cython.graph_context cimport CppNodeBasedContext, CppTurnBasedContext, NodeBasedContext, TurnBasedContext
from aequilibrae.paths.cython.search_results cimport CppSearchResults


ctypedef fused RoutingContext:
    NodeBasedContext
    TurnBasedContext


cdef extern from "path_finding.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    void cpp_dijkstra "aequilibrae::paths::cpp::mvp::dijkstra"[Queue](
        const CppNodeBasedContext& context,
        size_t origin,
        CppSearchResults& results,
    ) noexcept


    void cpp_turn_dijkstra "aequilibrae::paths::cpp::mvp::dijkstra"[Queue](
        const CppTurnBasedContext& context,
        size_t origin,
        CppSearchResults& results,
    ) noexcept
