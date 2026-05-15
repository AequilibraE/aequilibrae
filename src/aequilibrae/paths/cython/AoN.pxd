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

