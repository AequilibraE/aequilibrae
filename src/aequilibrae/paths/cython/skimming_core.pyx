cimport cython
from libc.math cimport INFINITY
from cython.parallel cimport parallel, prange, threadid
import numpy as np
from aequilibrae.paths.cython.basic_path_finding cimport (
    blocking_centroid_flows,
    path_finding,
    _path_finding_arc_based_core,
    path_finding_hybrid,
)


def skimming_parallel(graph, result, long cores):
    """OpenMP-parallel skimming over all valid centroids.

    Runs one Dijkstra per origin inside a single ``with nogil, parallel``
    block, eliminating the per-origin Python ThreadPool dispatch overhead
    that ``NetworkSkimming.execute`` paid before. Each OpenMP thread uses
    its own slice of the per-thread aux arrays (indexed by ``threadid()``),
    while ``path_finding`` or ``_path_finding_arc_based_core`` is invoked
    once per origin within the parallel loop.

    Returns a list of (origin, message) tuples for any centroid that could
    not be processed. Successful origins return an empty list.
    """

    if result._graph_id != graph._id:
        raise ValueError("Results object not prepared. Use --> results.prepare(graph)")

    cdef:
        long long compact_nodes = graph.compact_num_nodes + 1
        long long compact_links = graph.compact_num_links + 1
        long long zones = graph.num_zones
        long long block_flows_through_centroids = graph.block_centroid_flows
        long long skims = result.num_skims
        Py_ssize_t i, j
        long long oi, w
        int tid
        bint use_turn_restrictions = graph.has_turn_restrictions
        bint allow_uturns = graph._allow_path_uturns if use_turn_restrictions else False


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
    cdef long long [::1] graph_fs_view = compact_fs
    cdef double [::1] g_view = graph.compact_cost
    cdef const long long [::1] ids_graph_view = graph.compact_graph.id.to_numpy(copy=False)
    cdef const long long [::1] original_b_nodes_view = graph.compact_graph.b_node.to_numpy(copy=False)
    cdef double [:, ::1] graph_skim_view = graph.compact_skims[:, :]

    # Turn restriction views (if applicable).
    cdef long long [:] turn_fs_view
    cdef long long [:] turn_to_arcs_view
    cdef double [:] turn_penalties_view
    cdef const long long [:] a_nodes_view
    cdef const long long [:] first_ctx_view
    cdef const long long [:] last_ctx_view
    cdef const long long [::1] penalty_skim_indices_view
    cdef bint turn_penalty_skims = False

    cdef long long [:, ::1] arc_pred_mat
    cdef double [:, ::1] arc_turn_pen_mat
    cdef double [:, ::1] node_turn_pen_mat
    cdef double [:, ::1] node_costs_mat
    cdef double [:, :, ::1] arc_skims_memo_mat
    cdef long long [:, ::1] arc_visited_mat
    cdef long long [:, ::1] arc_stack_mat

    if use_turn_restrictions:
        if graph.compact_turn_fs.shape[0] < graph.compact_num_links + 1:
            raise ValueError("Turn restriction CSR is not sized for the compact graph. Re-run Graph.prepare_graph()")
        turn_fs = graph.compact_turn_fs
        turn_to = graph.compact_turn_to_arcs if graph.compact_turn_to_arcs.size else np.zeros(1, dtype=np.int64)
        turn_pen = graph.compact_turn_penalties if graph.compact_turn_penalties.size else np.zeros(1, dtype=np.float64)

        _pen_fields = graph.turn_skim_fields if graph.turn_skim_fields else (
            [graph.cost_field] if graph.cost_field else []
        )
        pen_idx = np.array(
            [graph.skim_fields.index(f) for f in _pen_fields if f in graph.skim_fields], dtype=np.int64
        )
        turn_penalty_skims = skims > 0 and pen_idx.shape[0] > 0
        if pen_idx.shape[0] == 0:
            pen_idx = np.zeros(1, dtype=np.int64)

        turn_fs_view = turn_fs
        turn_to_arcs_view = turn_to
        turn_penalties_view = turn_pen
        penalty_skim_indices_view = pen_idx
        a_nodes_view = graph.compact_graph.a_node.to_numpy(copy=False)

        # Boundary contexts are sized to the compact graph.
        first_ctx_view = graph._compact_first_node
        last_ctx_view = graph._compact_last_node

        arc_pred_mat = np.zeros((cores, compact_links), dtype=np.int64)
        arc_turn_pen_mat = np.zeros((cores, compact_links), dtype=np.float64)
        node_turn_pen_mat = np.zeros((cores, compact_nodes), dtype=np.float64)
        node_costs_mat = np.zeros((cores, compact_nodes), dtype=np.float64)
        # Sized on use_turn_restrictions alone, not on skims: skim_arc_based_paths walks
        # arc_visited/arc_stack even when there are no skim columns to accumulate.
        arc_skims_memo_mat = np.zeros((cores, compact_links, skims), dtype=np.float64)
        arc_visited_mat = np.zeros((cores, compact_links), dtype=np.int64)
        arc_stack_mat = np.zeros((cores, compact_links), dtype=np.int64)
    else:
        turn_fs_view = np.zeros(1, dtype=np.int64)
        turn_to_arcs_view = np.zeros(1, dtype=np.int64)
        turn_penalties_view = np.zeros(1, dtype=np.float64)
        penalty_skim_indices_view = np.zeros(1, dtype=np.int64)
        a_nodes_view = np.zeros(1, dtype=np.int64)
        first_ctx_view = original_b_nodes_view
        last_ctx_view = original_b_nodes_view
        arc_pred_mat = np.zeros((1, 1), dtype=np.int64)
        arc_turn_pen_mat = np.zeros((1, 1), dtype=np.float64)
        node_turn_pen_mat = np.zeros((1, 1), dtype=np.float64)
        node_costs_mat = np.zeros((1, 1), dtype=np.float64)
        arc_skims_memo_mat = np.zeros((1, 1, 1), dtype=np.float64)
        arc_visited_mat = np.zeros((1, 1), dtype=np.int64)
        arc_stack_mat = np.zeros((1, 1), dtype=np.int64)

    cdef const unsigned char [:] stateful_view
    cdef const long long [:] rep_arc_view
    cdef bint use_hybrid = False
    if use_turn_restrictions:
        stateful_view = graph.compact_stateful
        rep_arc_view = graph.compact_rep_arc
        use_hybrid = bool(graph.use_hybrid)
    else:
        stateful_view = np.zeros(1, dtype=np.uint8)
        rep_arc_view = np.zeros(1, dtype=np.int64)

    # Output skim cube (origin_index, dest_zone, skim).
    cdef double [:, :, :] final_skim_view = result.skims.matrix_view

    # Per-thread aux state (sliced by threadid inside the parallel region).
    cdef long long [:, ::1] predecessors_mat = np.zeros((cores, compact_nodes), dtype=np.int64)
    cdef long long [:, ::1] reached_first_mat = np.zeros((cores, compact_nodes), dtype=np.int64)
    cdef long long [:, ::1] connectors_mat = np.zeros((cores, compact_nodes), dtype=np.int64)
    cdef double [:, :, ::1] skim_mat = np.zeros((cores, compact_nodes, skims), dtype=np.float64)
    cdef unsigned char [::1] destinations = np.empty(0, dtype=np.uint8)
    cdef long long[:, ::1] b_nodes_mat = np.tile(graph.compact_graph.b_node.to_numpy(copy=False), (cores, 1))
    # Set if a prefix walk overruns its stack, which can only happen if arc_pred holds a
    # cycle. Checked once the parallel region ends so the failure is loud.
    truncated_arr = np.zeros(cores, dtype=np.int64)
    cdef long long [::1] truncated_view = truncated_arr

    with nogil, parallel(num_threads=cores):
        tid = threadid()

        for i in prange(n_origins, schedule="guided"):
            oi = origin_idx_view[i]

            if use_turn_restrictions:
                if use_hybrid:
                    w = path_finding_hybrid(
                        oi,
                        destinations,
                        -1,
                        g_view,
                        original_b_nodes_view,
                        graph_fs_view,
                        a_nodes_view,
                        stateful_view,
                        rep_arc_view,
                        predecessors_mat[tid],
                        connectors_mat[tid],
                        reached_first_mat[tid],
                        node_costs_mat[tid],
                        node_turn_pen_mat[tid],
                        arc_pred_mat[tid],
                        arc_turn_pen_mat[tid],
                        turn_fs_view,
                        turn_to_arcs_view,
                        turn_penalties_view,
                        allow_uturns,
                        block_flows_through_centroids,
                        zones,
                        first_ctx_view,
                        last_ctx_view,
                    )
                else:
                    w = _path_finding_arc_based_core(
                        oi,
                        destinations,
                        -1,
                        g_view,
                        original_b_nodes_view,
                        graph_fs_view,
                        arc_pred_mat[tid],
                        ids_graph_view,
                        a_nodes_view,
                        predecessors_mat[tid],
                        connectors_mat[tid],
                        reached_first_mat[tid],
                        node_turn_pen_mat[tid],
                        turn_fs_view,
                        turn_to_arcs_view,
                        turn_penalties_view,
                        allow_uturns,
                        arc_turn_pen_mat[tid],
                        &node_costs_mat[tid, 0],
                        block_flows_through_centroids,
                        zones,
                        first_ctx_view,
                        last_ctx_view,
                    )
                truncated_view[tid] += skim_arc_based_paths(
                    oi,
                    zones,
                    skims,
                    skim_mat[tid],
                    arc_pred_mat[tid],
                    connectors_mat[tid],
                    graph_skim_view,
                    arc_turn_pen_mat[tid],
                    penalty_skim_indices_view if turn_penalty_skims else penalty_skim_indices_view[:0],
                    arc_skims_memo_mat[tid],
                    arc_visited_mat[tid],
                    arc_stack_mat[tid],
                    oi + 1,
                )
                _copy_skims(skim_mat[tid, :zones, :], final_skim_view[oi, :, :])
            else:
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

    if truncated_arr.sum() > 0:
        raise RuntimeError(
            "Arc predecessor chain exceeded the number of arcs during skimming, which means the "
            "shortest path tree contains a cycle. Skims from this run are not valid."
        )

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
    cdef long long [::1] graph_fs_view = graph_fs
    cdef double [::1] g_view = graph.compact_cost
    cdef const long long [::1] ids_graph_view = graph.compact_graph.id.to_numpy(copy=False)
    cdef const long long [::1] original_b_nodes_view = graph.compact_graph.b_node.to_numpy(copy=False)
    cdef double [:, ::1] graph_skim_view = graph.compact_skims[:, :]

    cdef double [:, ::1] final_skim_matrices_view = result.skims.matrix_view[origin_index, :, :]

    # views from the aux-result object
    cdef long long [::1] predecessors_view = aux_result.predecessors[curr_thread, :]
    cdef long long [::1] reached_first_view = aux_result.reached_first[curr_thread, :]
    cdef long long [::1] conn_view = aux_result.connectors[curr_thread, :]
    cdef long long [::1] b_nodes_view = aux_result.temp_b_nodes[curr_thread, :]
    cdef double [:, ::1] skim_matrix_view = aux_result.temporary_skims[curr_thread, :, :]

    # Destination set
    cdef unsigned char [::1] destinations = np.array([], dtype=bool)

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


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)  # turn of bounds-checking for entire function
cpdef void skim_single_path_with_turn_penalties(long origin,
                                                long nodes,
                                                long skims,
                                                double[:, :] node_skims,
                                                long long[:] pred,
                                                long long[:] conn,
                                                double[:, :] graph_costs,
                                                long long[:] reached_first,
                                                long found,
                                                double [:] node_turn_penalties,
                                                const long long [:] penalty_indices) noexcept nogil:
    """
    Like skim_single_path but adds accumulated node turn penalties to every skim
    field listed in *penalty_indices*.  This lets callers apply the same turn
    penalty to several skim columns at once (e.g. cost + any other field that
    shares the same units as the turn penalty).
    """
    cdef long long i, node, predecessor, connector, j, k

    # sets all skims to infinity
    for i in range(nodes):
        for j in range(skims):
            node_skims[i, j] = INFINITY

    # Zeroes the intrazonal cost
    for j in range(skims):
        node_skims[origin, j] = 0

    # Cascade skimming: base link costs first, then turn penalty on each
    # specified skim field
    for i in range(1, found + 1):
        node = reached_first[i]

        # captures how we got to that node
        predecessor = pred[node]
        connector = conn[node]

        for j in range(skims):
            node_skims[node, j] = node_skims[predecessor, j] + graph_costs[connector, j]
        # node_turn_penalties holds the *cumulative* penalty to reach each node and
        # node_skims[predecessor] already includes the predecessor's share, so only
        # the incremental penalty of the final turn is added here.
        for k in range(<long>penalty_indices.shape[0]):
            node_skims[node, penalty_indices[k]] += node_turn_penalties[node] - node_turn_penalties[predecessor]


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef int skim_arc_based_paths(
    long long origin,
    long long dest_count,
    long long skims,
    double[:, :] node_skims,
    const long long[:] arc_pred,
    const long long[:] connectors,
    const double[:, :] graph_costs,
    const double[:] arc_turn_penalties,
    const long long[:] penalty_indices,
    double[:, :] arc_skims_memo,
    long long[:] arc_visited,
    long long[:] arc_stack,
    long long run_id,
) noexcept nogil:
    """
    Skims paths using arc-based memoized prefix accumulation from destination connectors.
    Traverses each arc in the shortest path tree at most once per origin, avoiding quadratic
    backtracking overhead while strictly preserving arc-specific costs and turn penalties.
    """
    cdef long long d, j, k, current_arc, curr, arc, p, stack_top
    # The prefix walk pushes each arc at most once, so overrunning the stack means arc_pred
    # holds a cycle. Bound the walk and report it rather than skimming a truncated path.
    cdef int truncated = 0
    cdef double total_turn

    for d in range(dest_count):
        for j in range(skims):
            node_skims[d, j] = INFINITY

    if 0 <= origin < dest_count:
        for j in range(skims):
            node_skims[origin, j] = 0.0

    for d in range(dest_count):
        if d == origin:
            continue
        current_arc = connectors[d]
        if current_arc < 0:
            continue

        # Walk backwards pushing unvisited arcs onto arc_stack
        stack_top = 0
        curr = current_arc
        while curr >= 0 and arc_visited[curr] != run_id and stack_top < arc_stack.shape[0]:
            arc_stack[stack_top] = curr
            stack_top += 1
            curr = arc_pred[curr]
        if curr >= 0 and arc_visited[curr] != run_id:
            truncated = 1

        # Unwind stack forwards, accumulating prefix totals
        while stack_top > 0:
            stack_top -= 1
            arc = arc_stack[stack_top]
            p = arc_pred[arc]
            if p >= 0 and arc_visited[p] == run_id:
                for j in range(skims):
                    arc_skims_memo[arc, j] = arc_skims_memo[p, j] + graph_costs[arc, j]
            else:
                for j in range(skims):
                    arc_skims_memo[arc, j] = graph_costs[arc, j]
            arc_visited[arc] = run_id

        for j in range(skims):
            node_skims[d, j] = arc_skims_memo[current_arc, j]

        if penalty_indices.shape[0] > 0:
            total_turn = arc_turn_penalties[connectors[d]]
            for k in range(penalty_indices.shape[0]):
                node_skims[d, penalty_indices[k]] += total_turn

    return truncated
