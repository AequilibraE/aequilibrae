# cython: language_level=3str
from aequilibrae.paths.cython.route_choice_set_results cimport RouteChoiceSetResults
from aequilibrae.paths.cython.route_choice_link_loading_results cimport LinkLoadingResults

from libcpp.vector cimport vector

from aequilibrae.paths.cython.route_choice_types cimport (
    RouteCandidate,
    RouteCandidateSet_t,
    RouteVec_t,
)
from aequilibrae.paths.cython.a_star cimport CppEuclideanContext, CppHaversineContext
from aequilibrae.paths.cython.workspaces cimport CppAStarWorkspace
from aequilibrae.paths.cython.context cimport CppNodeBasedContext, CppTurnBasedContext
from aequilibrae.paths.cython.search_query cimport CppSearchQuery
from aequilibrae.paths.cython.search_results cimport CppMutableSearchResults
from libc.stdint cimport *
from libc.stddef cimport size_t


cdef struct RouteChoiceHeuristic:
    bint a_star
    bint haversine
    CppEuclideanContext[double] euclidean_context
    CppHaversineContext[double] haversine_context


cdef class RouteChoiceSet:
    cdef:
        const double [::1] cost_view
        object routing
        object coordinates
        object node_ids
        bint turn_based
        bint has_turn_costs
        object graph
        readonly object centroids
        readonly object mapping
        readonly object network_mapping
        const Py_ssize_t[::1] network_link_to_graph
        long long num_nodes
        long long num_links

        const Py_ssize_t[::1] mapping_idx
        const Py_ssize_t[::1] mapping_data

        readonly RouteChoiceSetResults results
        readonly LinkLoadingResults ll_results

    cdef object import_dataframe(
        self,
        object df,
        RouteVec_t &routes,
        vector[bint] &mask,
    )
    cdef void map_routes(
        self,
        object table,
        const RouteVec_t &routes,
        RouteVec_t &graph_routes,
    )
    cdef void recost_routes(
        self,
        object table,
        const RouteVec_t &routes,
        const RouteVec_t &graph_routes,
        vector[vector[double]] &turns,
        vector[double] &costs,
    )

    cdef RouteCandidate *trace_route(
        RouteChoiceSet self,
        const CppMutableSearchResults[double] &result,
        size_t destination,
        bint save_turns
    ) noexcept nogil

    cdef void path_find(
        RouteChoiceSet self,
        CppSearchQuery &query,
        const CppMutableSearchResults[double] &result,
        const CppAStarWorkspace[double] &workspace,
        const CppNodeBasedContext[double] &node_context,
        const CppTurnBasedContext[double] &turn_context,
        size_t destination,
        const RouteChoiceHeuristic &heuristic
    ) noexcept nogil

    cdef void bfsle(
        RouteChoiceSet self,
        RouteCandidateSet_t &route_set,
        long origin_index,
        long dest_index,
        unsigned int max_routes,
        unsigned int max_depth,
        unsigned int max_misses,
        double [::1] thread_cost,
        CppSearchQuery &query,
        const CppMutableSearchResults[double] &result,
        const CppAStarWorkspace[double] &workspace,
        const CppNodeBasedContext[double] &node_context,
        const CppTurnBasedContext[double] &turn_context,
        const RouteChoiceHeuristic &heuristic,
        double penalty,
        unsigned int seed,
        bint save_turns
    ) noexcept nogil

    cdef void link_penalisation(
        RouteChoiceSet self,
        RouteCandidateSet_t &route_set,
        long origin_index,
        long dest_index,
        unsigned int max_routes,
        unsigned int max_depth,
        unsigned int max_misses,
        double [::1] thread_cost,
        CppSearchQuery &query,
        const CppMutableSearchResults[double] &result,
        const CppAStarWorkspace[double] &workspace,
        const CppNodeBasedContext[double] &node_context,
        const CppTurnBasedContext[double] &turn_context,
        const RouteChoiceHeuristic &heuristic,
        double penalty,
        unsigned int seed,
        bint save_turns
    ) noexcept nogil


cdef class Checkpoint:
    cdef:
        public object where
        public object schema
        public object partition_cols
