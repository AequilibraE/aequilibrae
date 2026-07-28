from aequilibrae.utils.cython.bridge cimport AeqLogClosure
from aequilibrae.paths.cython.path_finding cimport Heuristic

cdef void blocking_centroid_flows(int action,
                                  long long orig,
                                  long long centroids,
                                  long long [:] fs,
                                  long long [:] temp_b_nodes,
                                  const long long [:] real_b_nodes) noexcept nogil


cdef int path_finding(
    long origin,
    unsigned char [::1] destinations,
    long long destination_count,
    double[::1] graph_costs,
    long long [::1] csr_indices,
    long long [::1] graph_fs,
    long long [::1] pred,
    const long long [::1] ids,
    long long [::1] connectors,
    long long [::1] reached_first,
    HeapType heap=*,
    AeqLogClosure *closure=*,
) noexcept nogil

cdef void path_finding_a_star(
    long origin,
    long destination,
    double[::1] graph_costs,
    long long [::1] csr_indices,
    long long [::1] graph_fs,
    long long [::1] nodes_to_indices,
    const double [::1] lats,
    const double [::1] lons,
    long long [::1] pred,
    const long long [::1] ids,
    long long [::1] connectors,
    Heuristic heuristic,
    HeapType heap=*,
    AeqLogClosure *closure=*
) noexcept nogil

cdef enum HeapType:
    FOUR_ARY_HEAP
    PAIRING_HEAP
    STD_PRIORITY_QUEUE

cdef int _path_finding_arc_based_core(
    long origin,
    unsigned char [:] destinations,
    long long destination_count,
    double[:] graph_costs,
    const long long [:] csr_indices,
    const long long [:] graph_fs,
    long long [:] arc_pred,
    const long long [:] ids,
    const long long [:] a_nodes,
    long long [:] node_pred,
    long long [:] connectors,
    long long [:] reached_first,
    double [:] node_turn_penalties,
    const long long [:] turn_fs,
    const long long [:] turn_to_arcs,
    const double [:] turn_penalties,
    bint allow_uturns,
    double [:] arc_turn_penalties,
    double *node_costs,
) noexcept nogil

cpdef int path_finding_arc_based(
    long origin,
    unsigned char [:] destinations,
    long long destination_count,
    double[:] graph_costs,
    const long long [:] csr_indices,
    const long long [:] graph_fs,
    long long [:] arc_pred,
    const long long [:] ids,
    const long long [:] a_nodes,
    long long [:] node_pred,
    long long [:] connectors,
    long long [:] reached_first,
    double [:] node_turn_penalties,
    const long long [:] turn_fs,
    const long long [:] turn_to_arcs,
    const double [:] turn_penalties,
    bint allow_uturns,
    double [:] arc_turn_penalties
) noexcept nogil
