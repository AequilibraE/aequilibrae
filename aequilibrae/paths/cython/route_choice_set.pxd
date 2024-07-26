# cython: language_level=3str
from aequilibrae.paths.results import PathResults
from aequilibrae.paths.cython.route_choice_set_results cimport RouteChoiceSetResults
from aequilibrae.paths.cython.route_choice_link_loading_results cimport LinkLoadingResults

from libcpp.vector cimport vector

from aequilibrae.paths.cython.route_choice_types cimport RouteSet_t


cdef class RouteChoiceSet:
    cdef:
        double [:] cost_view
        long long [:] graph_fs_view
        long long [:] b_nodes_view
        long long [:] nodes_to_indices_view
        double [:] lat_view
        double [:] lon_view
        long long [:] ids_graph_view
        long long [:] graph_compressed_id_view
        long long [:] compressed_link_ids
        long long num_nodes
        long long num_links
        long long zones
        bint block_flows_through_centroids
        bint a_star

        unsigned int [:] mapping_idx
        unsigned int [:] mapping_data

        readonly RouteChoiceSetResults results
        readonly LinkLoadingResults ll_results

    cdef void path_find(
        RouteChoiceSet self,
        long origin_index,
        long dest_index,
        double [:] scratch_cost,
        long long [:] thread_predecessors,
        long long [:] thread_conn,
        long long [:] thread_b_nodes,
        long long [:] thread_reached_first
    ) noexcept nogil

    cdef void bfsle(
        RouteChoiceSet self,
        RouteSet_t &route_set,
        long origin_index,
        long dest_index,
        unsigned int max_routes,
        unsigned int max_depth,
        unsigned int max_misses,
        double [:] thread_cost,
        long long [:] thread_predecessors,
        long long [:] thread_conn,
        long long [:] thread_b_nodes,
        long long [:] _thread_reached_first,
        double penatly,
        unsigned int seed
    ) noexcept nogil

    cdef void link_penalisation(
        RouteChoiceSet self,
        RouteSet_t &route_set,
        long origin_index,
        long dest_index,
        unsigned int max_routes,
        unsigned int max_depth,
        unsigned int max_misses,
        double [:] thread_cost,
        long long [:] thread_predecessors,
        long long [:] thread_conn,
        long long [:] thread_b_nodes,
        long long [:] _thread_reached_first,
        double penatly,
        unsigned int seed
    ) noexcept nogil

    # @staticmethod
    # cdef vector[vector[double] *] *compute_path_files(
    #     vector[pair[long long, long long]] &ods,
    #     vector[RouteSet_t *] &results,
    #     vector[vector[long long] *] &link_union_set,
    #     vector[vector[double] *] &prob_set,
    #     unsigned int cores
    # ) noexcept nogil

    # cdef vector[double] *apply_link_loading(RouteChoiceSet self, double[:, :] matrix_view) noexcept nogil
    # cdef vector[double] *apply_link_loading_from_path_files(RouteChoiceSet self, double[:, :] matrix_view, vector[vector[double] *] &path_files) noexcept nogil
    # cdef apply_link_loading_func(RouteChoiceSet self, vector[double] *ll, int cores)

    # cdef vector[double] *apply_select_link_loading(
    #     RouteChoiceSet self,
    #     COO sparse_mat,
    #     double[:, :] matrix_view,
    #     unordered_set[long] &select_link_set
    # ) noexcept nogil


cdef class Checkpoint:
    cdef:
        public object where
        public object schema
        public object partition_cols
