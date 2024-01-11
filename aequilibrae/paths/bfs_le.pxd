# cython: language_level=3str
from aequilibrae.paths.results import PathResults
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map


cdef class RouteChoice:
    cdef:
        int num
        readonly object res
        unsigned int depth

        double [:] cost_view
        long long [:] graph_fs_view
        long long [:] b_nodes_view
        long long [:] nodes_to_indices_view
        double [:] lat_view
        double [:] lon_view
        long long [:] predecessors_view
        long long [:] ids_graph_view
        long long [:] conn_view
    cdef void path_find(RouteChoice self, long origin_index, long dest_index, double [:] scratch_cost) noexcept nogil
    # cdef unordered_map[unordered_set[long long] *, vector[long long] *] *generate_route_set(RouteChoice self, long origin_index, long dest_index, unsigned int max_routes, unsigned int max_depth, double [:] scratch_cost) noexcept nogil
    cdef unordered_set[vector[long long] *] *generate_route_set(RouteChoice self, long origin_index, long dest_index, unsigned int max_routes, unsigned int max_depth, double [:] scratch_cost) noexcept nogil
