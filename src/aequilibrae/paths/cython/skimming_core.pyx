cimport cython
from libc.math cimport INFINITY
from cython.parallel cimport parallel, prange, threadid
import numpy as np
from aequilibrae.paths.results.skim_results import SkimResults
from aequilibrae.paths.cython.basic_path_finding cimport blocking_centroid_flows, path_finding
from aequilibrae.paths.multi_threaded_paths import MultiThreadedPaths

def skimming_parallel(graph, result, long cores):
    """OpenMP-parallel skimming over all valid centroids.

    Runs one Dijkstra per origin inside a single ``with nogil, parallel``
    block, eliminating the per-origin Python ThreadPool dispatch overhead
    that ``NetworkSkimming.execute`` paid before. Each OpenMP thread uses
    its own slice of the per-thread aux arrays (indexed by ``threadid()``)
    and its own persistent priority queue.

    Returns a list of (origin, message) tuples for any centroid that could
    not be processed. Successful origins return an empty list.
    """

    result = SkimResults()
    result.prepare(graph)
    num_skims = len(graph.skim_fields)
    num_nodes = result.nodes

    aux_result = MultiThreadedPaths()
    aux_result.prepare(graph, cores, num_nodes, num_skims)
    aux_result.temporary_skims = np.zeros((cores, num_nodes, num_skims), dtype=ftype)

    if result._graph_id != graph._id:
        raise ValueError("Results object not prepared. Use --> results.prepare(graph)")

    cdef:
        long long compact_nodes = graph.compact_num_nodes + 1
        long long zones = graph.num_zones
        long long block_flows_through_centroids = graph.block_centroid_flows
        long long skims = result.num_skims

    # Pre-resolve centroids -> compact indices on the Python side. We also
    # filter out any centroid that has no outgoing edges so the parallel
    # kernel can be a tight loop with no branching for malformed inputs.
    centroids = list(graph.centroids)
    compact_nodes_to_indices = graph.compact_nodes_to_indices
    compact_fs = graph.compact_fs
    valid_origin_indices = []
    skipped = []
    for _orig in centroids:
        _ci = int(compact_nodes_to_indices[_orig])
        if _ci < 0 or _ci >= compact_nodes:
            skipped.append((_orig, f"Centroid {_orig} is outside the compact graph"))
            continue
        if compact_fs[_ci] == compact_fs[_ci + 1]:
            skipped.append((_orig, f"Centroid {_orig} has no outgoing edges"))
            continue
        valid_origin_indices.append(_ci)

    cdef long long n_origins = len(valid_origin_indices)
    if n_origins == 0:
        return skipped

    cdef long long [:] origin_idx_view = np.asarray(valid_origin_indices, dtype=np.int64)

    # Graph views (shared, read-only across threads).
    cdef long long [:] graph_fs_view = compact_fs
    cdef double [:] g_view = graph.compact_cost
    cdef const long long [:] ids_graph_view = graph.compact_graph.id.to_numpy(copy=False)
    cdef const long long [:] original_b_nodes_view = graph.compact_graph.b_node.to_numpy(copy=False)
    cdef double [:, :] graph_skim_view = graph.compact_skims[:, :]

    # Output skim cube (origin_index, dest_zone, skim).
    cdef double [:, :, :] final_skim_view = result.skims.matrix_view

    # Per-thread aux state (sliced by threadid inside the parallel region).
    cdef long long [:, :] predecessors_mat = aux_result.predecessors
    cdef long long [:, :] reached_first_mat = aux_result.reached_first
    cdef long long [:, :] connectors_mat = aux_result.connectors
    cdef long long [:, :] b_nodes_mat = aux_result.temp_b_nodes
    cdef double [:, :, :] skim_mat = aux_result.temporary_skims

    # Empty destinations array (we never use early exit for skimming).
    cdef unsigned char [:] destinations = np.zeros(0, dtype=np.uint8)

    cdef:
        long long i, oi, w
        int tid

    with nogil, parallel(num_threads=cores):
        tid = threadid()

        for i in prange(n_origins, schedule="guided"):
            oi = origin_idx_view[i]

            if block_flows_through_centroids:
                blocking_centroid_flows(0, oi, zones, graph_fs_view,
                                        b_nodes_mat[tid], original_b_nodes_view)

            w = path_finding(oi,
                             destinations,
                             -1,
                             g_view,
                             b_nodes_mat[tid],
                             graph_fs_view,
                             predecessors_mat[tid],
                             ids_graph_view,
                             connectors_mat[tid],
                             reached_first_mat[tid])

            skim_multiple_fields(oi,
                                 compact_nodes,
                                 zones,
                                 skims,
                                 skim_mat[tid],
                                 predecessors_mat[tid],
                                 connectors_mat[tid],
                                 graph_skim_view,
                                 reached_first_mat[tid],
                                 w,
                                 final_skim_view[oi, :, :])

            if block_flows_through_centroids:
                blocking_centroid_flows(1, oi, zones, graph_fs_view,
                                        b_nodes_mat[tid], original_b_nodes_view)

    return skipped

def skimming_single_origin(origin, graph, result, aux_result, curr_thread):
    """
    :param origin:
    :param graph:
    :param results:
    :return:
    """
    cdef long long nodes, orig, origin_index, block_flows_through_centroids, skims, zones, b
    # We transform the python variables in Cython variables
    orig = origin
    origin_index = graph.compact_nodes_to_indices[orig]

    graph_fs = graph.compact_fs
    if result._graph_id != graph._id:
        raise ValueError("Results object not prepared. Use --> results.prepare(graph)")

    if orig not in graph.centroids:
        raise ValueError("Centroid " + str(orig) + " is outside the range of zones in the graph")

    if origin_index > graph.compact_num_nodes:
        raise ValueError("Centroid " + str(orig) + " does not exist in the graph")

    if graph_fs[origin_index] == graph_fs[origin_index + 1]:
        raise ValueError("Centroid " + str(orig) + " does not exist in the graph")

    nodes = graph.compact_num_nodes + 1
    zones = graph.num_zones
    block_flows_through_centroids = graph.block_centroid_flows
    skims = result.num_skims

    # In order to release the GIL for this procedure, we create all the
    # memory views we will need

    # views from the graph
    cdef long long [:] graph_fs_view = graph_fs
    cdef double [:] g_view = graph.compact_cost
    cdef const long long [:] ids_graph_view = graph.compact_graph.id.to_numpy(copy=False)
    cdef const long long [:] original_b_nodes_view = graph.compact_graph.b_node.to_numpy(copy=False)
    cdef double [:, :] graph_skim_view = graph.compact_skims[:, :]

    cdef double [:, :] final_skim_matrices_view = result.skims.matrix_view[origin_index, :, :]

    # views from the aux-result object
    cdef long long [:] predecessors_view = aux_result.predecessors[curr_thread, :]
    cdef long long [:] reached_first_view = aux_result.reached_first[curr_thread, :]
    cdef long long [:] conn_view = aux_result.connectors[curr_thread, :]
    cdef long long [:] b_nodes_view = aux_result.temp_b_nodes[curr_thread, :]
    cdef double [:, :] skim_matrix_view = aux_result.temporary_skims[curr_thread, :, :]

    # Destination set
    cdef unsigned char [:] destinations = np.array([], dtype=bool)

    # Now we do all procedures with NO GIL
    with nogil:
        if block_flows_through_centroids:  # Unblocks the centroid if that is the case
            b = 0
            blocking_centroid_flows(b,
                                    origin_index,
                                    zones,
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)
        w = path_finding(origin_index,
                         destinations,
                         -1,  # destination index to disable early exit
                         g_view,
                         b_nodes_view,
                         graph_fs_view,
                         predecessors_view,
                         ids_graph_view,
                         conn_view,
                         reached_first_view)

        skim_multiple_fields(origin_index,
                             nodes,
                             zones,  # ???????????????
                             skims,
                             skim_matrix_view,
                             predecessors_view,
                             conn_view,
                             graph_skim_view,
                             reached_first_view,
                             w,
                             final_skim_matrices_view)
        if block_flows_through_centroids:  # Unblocks the centroid if that is the case
            b = 1
            blocking_centroid_flows(b,
                                    origin_index,
                                    zones,
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)
    return orig

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)  # turn of bounds-checking for entire function
cpdef void skim_multiple_fields(long origin,
                                long nodes,
                                long zones,
                                long skims,
                                double[:, :] node_skims,
                                long long [:] pred,
                                long long [:] conn,
                                double[:, :] graph_costs,
                                long long [:] reached_first,
                                long found,
                                double [:, :] final_skims) noexcept nogil:
    cdef long long i, node, predecessor, connector, j

    # sets all skims to infinity
    for i in range(nodes):
        for j in range(skims):
            node_skims[i, j] = INFINITY

    # Zeroes the intrazonal cost
    for j in range(skims):
        node_skims[origin, j] = 0

    # Cascade skimming
    for i in range(1, found + 1):
        node = reached_first[i]

        # captures how we got to that node
        predecessor = pred[node]
        connector = conn[node]

        for j in range(skims):
            node_skims[node, j] = node_skims[predecessor, j] + graph_costs[connector, j]

    for i in range(zones):
        for j in range(skims):
            final_skims[i, j] = node_skims[i, j]

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void _copy_skims(
        double[:, :] skim_matrix,  # Skim matrix_procedures computed from one origin to all nodes
        double[:, :] final_skim_matrix
) noexcept nogil:  # Skim matrix_procedures computed for one origin to all other centroids only

    cdef long i, j
    cdef long N = final_skim_matrix.shape[0]
    cdef long skims = final_skim_matrix.shape[1]

    for i in range(N):
        for j in range(skims):
            final_skim_matrix[i, j] = skim_matrix[i, j]

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)  # turn of bounds-checking for entire function
cpdef void skim_single_path(long origin,
                            long nodes,
                            long skims,
                            double[:, :] node_skims,
                            long long [:] pred,
                            long long [:] conn,
                            double[:, :] graph_costs,
                            long long [:] reached_first,
                            long found) noexcept nogil:
    cdef long long i, node, predecessor, connector, j

    # sets all skims to infinity
    for i in range(nodes):
        for j in range(skims):
            node_skims[i, j] = INFINITY

    # Zeroes the intrazonal cost
    for j in range(skims):
        node_skims[origin, j] = 0

    # Cascade skimming
    for i in range(1, found + 1):
        node = reached_first[i]

        # captures how we got to that node
        predecessor = pred[node]
        connector = conn[node]

        for j in range(skims):
            node_skims[node, j] = node_skims[predecessor, j] + graph_costs[connector, j]
