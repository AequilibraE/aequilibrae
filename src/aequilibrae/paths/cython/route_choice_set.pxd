# cython: language_level=3str
from aequilibrae.paths.cython.route_choice_set_results cimport RouteChoiceSetResults
from aequilibrae.paths.cython.route_choice_link_loading_results cimport LinkLoadingResults

from libcpp.vector cimport vector

from aequilibrae.paths.cython.route_choice_types cimport RouteCandidateSet_t
from aequilibrae.paths.cython.context cimport CppNodeBasedContext, CppTurnBasedContext
from aequilibrae.paths.cython.queries cimport CppSearchQuery
from aequilibrae.paths.cython.search_results cimport CppMutableSearchResults
from libc.stdint cimport *


cdef class RouteChoiceSet:
    cdef:
        const double [::1] cost_view
        object routing
        bint turn_based
        bint has_turn_costs
        long long [::1] nodes_to_indices_view
        const long long [::1] graph_compressed_id_view
        long long num_nodes
        long long num_links

        unsigned int [:] mapping_idx
        int64_t [::] mapping_data
        const int64_t [::] link_id_direction

        readonly RouteChoiceSetResults results
        readonly LinkLoadingResults ll_results

    cdef void path_find(
        RouteChoiceSet self,
        CppSearchQuery &query,
        const CppMutableSearchResults &result,
        const CppNodeBasedContext &node_context,
        const CppTurnBasedContext &turn_context
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
        const CppMutableSearchResults &result,
        const CppNodeBasedContext &node_context,
        const CppTurnBasedContext &turn_context,
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
        const CppMutableSearchResults &result,
        const CppNodeBasedContext &node_context,
        const CppTurnBasedContext &turn_context,
        double penalty,
        unsigned int seed,
        bint save_turns
    ) noexcept nogil


cdef class Checkpoint:
    cdef:
        public object where
        public object schema
        public object partition_cols
