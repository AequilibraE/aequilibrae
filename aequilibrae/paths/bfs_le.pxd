# cython: language_level=3str
from aequilibrae.paths.results import PathResults


cpdef float cube(float x)

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
    cdef void c_helloworld(RouteChoice self) noexcept nogil
    cpdef helloworld(self)
    cdef void generate_route_set(RouteChoice self, long origin_index, long dest_index, unsigned int max_depth) noexcept nogil
