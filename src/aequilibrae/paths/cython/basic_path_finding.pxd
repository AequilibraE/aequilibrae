cdef void blocking_centroid_flows(int action,
                                  long long orig,
                                  long long centroids,
                                  long long [:] fs,
                                  long long [:] temp_b_nodes,
                                  const long long [:] real_b_nodes) noexcept nogil

cpdef int path_finding(
    long origin,
    unsigned char [:] destinations,
    long long destination_count,
    double[:] graph_costs,
    long long [:] csr_indices,
    long long [:] graph_fs,
    long long [:] pred,
    const long long [:] ids,
    long long [:] connectors,
    long long [:] reached_first
) noexcept nogil

ctypedef enum Heuristic:
    HAVERSINE
    EQUIRECTANGULAR

cpdef void path_finding_a_star(long origin,
                               long destination,
                               double[:] graph_costs,
                               long long [:] csr_indices,
                               long long [:] graph_fs,
                               long long [:] nodes_to_indices,
                               const double [:] lats,
                               const double [:] lons,
                               long long [:] pred,
                               const long long [:] ids,
                               long long [:] connectors,
                               int heuristic) noexcept nogil

