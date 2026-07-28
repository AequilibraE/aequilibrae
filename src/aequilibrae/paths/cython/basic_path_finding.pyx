"""
Original Algorithm for Shortest path (Dijkstra with a 4-ary heap) was written by François Pacull
<francois.pacull@architecture-performance.fr> under license: MIT, (C) 2022
"""

"""
TODO:
LIST OF ALL THE THINGS WE NEED TO DO TO NOT HAVE TO HAVE nodes 1..n as CENTROIDS. ARBITRARY NUMBERING
- Checks of weather the centroid we are computing path from is a centroid and/or exists in the graph
- Re-write function **network_loading** on the part of loading flows to centroids
"""
cimport cython
from libc.math cimport INFINITY, sin, cos, asin, sqrt, pi
from libc.stdlib cimport malloc, free
from libc.stddef cimport size_t
from libc.stdint cimport int64_t
from libcpp.vector cimport vector

from aequilibrae.paths.cython.pq_heap_types cimport (
    FourAryHeap,
    PairingHeap,
    StdPriorityQueueAdapter,
    ElementState,
    NOT_IN_HEAP,
    SCANNED,
)
from aequilibrae.paths.cython.path_finding cimport (
    dijkstra,
    a_star,
    HeuristicFn,
    haversine_heuristic,
    equirectangular_heuristic,
    Heuristic,
)
from aequilibrae.utils.cython.bridge cimport AeqLogClosure


HEAP_MAP = {"4ary": FOUR_ARY_HEAP, "pairing": PAIRING_HEAP, "std": STD_PRIORITY_QUEUE}

HEURISTIC_MAP = {"haversine": Heuristic.HAVERSINE, "equirectangular": Heuristic.EQUIRECTANGULAR}

# Sentinel value stored in arc_pred for arcs originating from the source node.
# Backtracking loops terminate when they encounter this value (< 0 and != -1).
cdef long long ORIGIN_ARC_SENTINEL = -2


cdef int[:] return_an_int_view(input) noexcept nogil:
    cdef int [:] critical_links_view = input
    return critical_links_view

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cdef void blocking_centroid_flows(int action,
                                  long long orig,
                                  long long centroids,
                                  long long [:] fs,
                                  long long [:] temp_b_nodes,
                                  const long long [:] real_b_nodes) noexcept nogil:
    cdef long long i

    if action == 1:  # We are unblocking
        for i in range(fs[centroids]):
            temp_b_nodes[i] = real_b_nodes[i]
    else:  # We are blocking:
        for i in range(fs[centroids]):
            temp_b_nodes[i] = orig

        for i in range(fs[orig], fs[orig + 1]):
            temp_b_nodes[i] = real_b_nodes[i]

# ######################################################################################################################
########################################################################################################################
# Original Dijkstra implementation by François Pacull, taken from https://github.com/Edsger-dev/priority_queues
# Old Numpy Buffers were replaces with latest memory views interface to allow for the release of the GIL
# Path tracking arrays and skim arrays were also added to it
########################################################################################################################
# ######################################################################################################################

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)  # turn of bounds-checking for entire function
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
    HeapType heap=FOUR_ARY_HEAP,
    AeqLogClosure *closure=NULL
) noexcept nogil:
    cdef:
        size_t origin_vert = <size_t>origin
        size_t max_size = <size_t>pred.shape[0]
        const double *costs_ptr = &graph_costs[0]
        const size_t *csr_ptr = <const size_t*>&csr_indices[0]
        const size_t *fs_ptr = <const size_t*>&graph_fs[0]
        size_t *pred_ptr = <size_t*>&pred[0]
        const size_t *ids_ptr = <const size_t*>&ids[0]
        size_t *conn_ptr = <size_t*>&connectors[0]
        size_t *reached_ptr = <size_t*>&reached_first[0]
        # When early exit is disabled the destination mask is never read and
        # may be empty; don't form a pointer into a zero-length buffer.
        const unsigned char *dest_ptr = &destinations[0] if destination_count >= 0 else NULL

    if heap == PAIRING_HEAP:
        return <int>dijkstra[PairingHeap](origin_vert, max_size, costs_ptr, csr_ptr, fs_ptr, pred_ptr,
                                          ids_ptr, conn_ptr, reached_ptr, dest_ptr, destination_count, closure)
    elif heap == STD_PRIORITY_QUEUE:
        return <int>dijkstra[StdPriorityQueueAdapter](origin_vert, max_size, costs_ptr, csr_ptr, fs_ptr, pred_ptr,
                                                      ids_ptr, conn_ptr, reached_ptr, dest_ptr, destination_count, closure)
    else:
        return <int>dijkstra[FourAryHeap](origin_vert, max_size, costs_ptr, csr_ptr, fs_ptr, pred_ptr,
                                          ids_ptr, conn_ptr, reached_ptr, dest_ptr, destination_count, closure)

cdef int _HAVERSINE = 0
cdef int _EQUIRECTANGULAR = 1

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)  # turn of bounds-checking for entire function
cpdef void dfs(long origin,
               long long [:] csr_indices,
               long long [:] graph_fs,
               long long [:] pred) noexcept nogil:

    cdef:
        size_t tail_vert_idx, head_vert_idx  # indices
        unsigned int M = pred.shape[0]
        vector[size_t] visited
        size_t origin_vert = <size_t>origin

    for i in range(M):
        pred[i] = -1

    pred[origin_vert] = 0
    # initialization of the list of nodes to be analysed
    visited.push_back(origin_vert)

    # main loop
    while not visited.empty():
        tail_vert_idx = visited.back()
        visited.pop_back()

        # loop on outgoing edges
        for idx in range(<size_t>graph_fs[tail_vert_idx], <size_t>graph_fs[tail_vert_idx + 1]):
            head_vert_idx = <size_t>csr_indices[idx]
            if pred[head_vert_idx] < 0:
                pred[head_vert_idx] = tail_vert_idx
                visited.push_back(head_vert_idx)

    visited.clear()



@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cdef void path_finding_a_star(long origin,
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
                              HeapType heap=FOUR_ARY_HEAP,
                              AeqLogClosure *closure=NULL) noexcept nogil:
    cdef:
        HeuristicFn heur_fn
        void* heur_data
        double cos_lat1_local
        size_t origin_vert = <size_t>origin
        size_t destination_vert = <size_t>destination
        size_t max_size = <size_t>pred.shape[0]
        const double *costs_ptr = &graph_costs[0]
        const size_t *csr_ptr = <const size_t*>&csr_indices[0]
        const size_t *fs_ptr = <const size_t*>&graph_fs[0]
        const size_t *nti_ptr = <const size_t*>&nodes_to_indices[0]
        const double *lats_ptr = &lats[0]
        const double *lons_ptr = &lons[0]
        size_t *pred_ptr = <size_t*>&pred[0]
        const size_t *ids_ptr = <const size_t*>&ids[0]
        size_t *conn_ptr = <size_t*>&connectors[0]

    if heuristic == Heuristic.HAVERSINE:
        heur_fn = haversine_heuristic
        cos_lat1_local = cos(lats[<size_t>destination if destination != -1 else 0] * pi / 180.0)
        heur_data = <void*>&cos_lat1_local
    else:
        heur_fn = equirectangular_heuristic
        heur_data = NULL

    if heap == PAIRING_HEAP:
        a_star[PairingHeap](origin_vert, destination_vert, max_size, costs_ptr, csr_ptr, fs_ptr, nti_ptr,
                            lats_ptr, lons_ptr, pred_ptr, ids_ptr, conn_ptr, heur_fn, heur_data, closure)
    elif heap == STD_PRIORITY_QUEUE:
        a_star[StdPriorityQueueAdapter](origin_vert, destination_vert, max_size, costs_ptr, csr_ptr, fs_ptr, nti_ptr,
                                        lats_ptr, lons_ptr, pred_ptr, ids_ptr, conn_ptr, heur_fn, heur_data, closure)
    else:
        a_star[FourAryHeap](origin_vert, destination_vert, max_size, costs_ptr, csr_ptr, fs_ptr, nti_ptr,
                            lats_ptr, lons_ptr, pred_ptr, ids_ptr, conn_ptr, heur_fn, heur_data, closure)


# ######################################################################################################################
########################################################################################################################
# Arc-based Dijkstra implementation for turn restrictions
# In arc-based Dijkstra, the state space is arcs (edges) instead of nodes.
# This allows modeling turn restrictions and penalties between consecutive arcs.
########################################################################################################################
# ######################################################################################################################


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
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
) noexcept nogil:
    """
    Arc-based Dijkstra's algorithm with turn restrictions.

    Unlike the standard node-based Dijkstra, this operates on arcs (edges).
    The state is (current_arc, cost) and transitions happen from arc to arc
    with optional turn penalties.

    Arguments:
        origin: Origin node index
        destinations: Boolean array marking destination nodes
        destination_count: Number of destinations (-1 for all)
        graph_costs: Cost of each arc
        csr_indices: B-nodes for each arc (CSR format)
        graph_fs: Forward star indices for nodes
        arc_pred: Output - predecessor arc for each arc (for path reconstruction)
        ids: Arc IDs
        a_nodes: A-node (tail) for each arc
        node_pred: Output - predecessor node for each node (for compatibility)
        connectors: Output - incoming arc for each node (for compatibility)
        reached_first: Order in which nodes were reached
        node_turn_penalties: Output - cumulative turn penalty to reach each node
        turn_fs: CSR index into explicit custom turn entries per incoming arc
        turn_to_arcs: Target arcs for each explicit custom turn
        turn_penalties: Custom turn penalties (INFINITY = prohibited turn)
        allow_uturns: Whether U-turn transitions are allowed by default
        arc_turn_penalties: Output - Turn penalty incurred to enter each arc
        node_costs: Scratch array of size num_nodes for node label costs

    Returns:
        Number of nodes reached
    """
    cdef unsigned int num_nodes = node_pred.shape[0]
    cdef unsigned int num_arcs = arc_pred.shape[0]
    cdef:
        size_t current_arc, next_arc, current_node, idx, turn_idx
        size_t restriction_start, restriction_end
        double current_cost, next_cost, turn_penalty
        double current_turn_cost
        FourAryHeap pqueue
        ElementState arc_state
        size_t origin_vert = <size_t>origin
        int found = 0
        bint has_explicit_entry
        unsigned int i

    # Only the node-indexed outputs are reset. ``arc_pred`` and ``arc_turn_penalties``
    # deliberately keep whatever the previous origin left behind on this thread's
    # buffers, which is safe: every backtrack starts at ``connectors[i]``, which is -1
    # unless node i was reached by *this* run. A reached node's arc was extracted from
    # the heap, so it was inserted by this run, so its arc_pred/arc_turn_penalties
    # entries were written by this run. Chains terminate at ORIGIN_ARC_SENTINEL.
    for i in range(num_nodes):
        node_pred[i] = -1
        connectors[i] = -1
        node_costs[i] = INFINITY

    # Initialize heap for arcs
    pqueue.init_heap(<size_t>num_arcs)

    # Insert all outgoing arcs from origin with their base costs
    # Place origin at reached_first[0] for compatibility with skim_single_path
    node_costs[origin_vert] = 0.0
    reached_first[0] = origin_vert
    found = 1
    node_turn_penalties[origin_vert] = 0.0

    if destination_count > 0 and destinations[origin_vert]:
        destinations[origin_vert] = False
        destination_count = destination_count - 1
        if destination_count == 0:
            return found - 1
    for idx in range(<size_t>graph_fs[origin_vert], <size_t>graph_fs[origin_vert + 1]):
        if graph_costs[idx] < INFINITY:
            pqueue.insert(idx, graph_costs[idx])
            arc_pred[idx] = ORIGIN_ARC_SENTINEL
            arc_turn_penalties[idx] = 0.0

    # Main loop - process arcs by best known arc-label cost.
    while not pqueue.is_empty():
        current_arc = pqueue.extract_min()
        current_cost = pqueue.element_key(current_arc)

        # Get the head node of the current arc
        current_node = <size_t>csr_indices[current_arc]

        # Update node information if this is a better path to current_node
        if current_cost < node_costs[current_node]:
            node_costs[current_node] = current_cost
            reached_first[found] = current_node
            found += 1
            # Store cumulative turn penalties to reach this node
            node_turn_penalties[current_node] = arc_turn_penalties[current_arc]

            # Store node predecessor and connector for compatibility with node-based routines
            node_pred[current_node] = a_nodes[current_arc]
            connectors[current_node] = ids[current_arc]

            # Check early exit condition
            if destination_count > 0 and destinations[current_node]:
                destinations[current_node] = False
                destination_count = destination_count - 1
                if destination_count == 0:
                    break

        # Explore possible outgoing arcs from current node.
        #
        # Representation details:
        # - graph_fs/current_node enumerate *all* physically possible next arcs.
        # - (turn_fs, turn_to_arcs, turn_penalties) hold only explicit custom turns,
        #   sorted by to-arc for each from-arc.
        # - Missing explicit entry means default turn penalty of 0.0.
        #
        # This sparse representation keeps memory small when only a tiny fraction of
        # turns are restricted/penalized.
        current_turn_cost = arc_turn_penalties[current_arc]
        restriction_start = <size_t>turn_fs[current_arc]
        restriction_end = <size_t>turn_fs[current_arc + 1]
        for idx in range(<size_t>graph_fs[current_node], <size_t>graph_fs[current_node + 1]):
            next_arc = idx

            turn_penalty = 0.0
            has_explicit_entry = False
            # Resolve explicit custom turn entry (if present) for (current_arc -> next_arc).
            # Entries are sorted by next_arc so we can early-break when surpassed.
            for turn_idx in range(restriction_start, restriction_end):
                if <size_t>turn_to_arcs[turn_idx] == next_arc:
                    turn_penalty = turn_penalties[turn_idx]
                    has_explicit_entry = True
                    break
                if <size_t>turn_to_arcs[turn_idx] > next_arc:
                    break

            # Explicit turn entries (penalties or prohibitions) take priority over the
            # global u-turn ban.  Only apply the ban when no explicit entry covers this
            # transition - so a user-defined penalty can still permit a u-turn even when
            # allow_uturns is False.
            if not allow_uturns and not has_explicit_entry and csr_indices[next_arc] == a_nodes[current_arc]:
                continue

            # INFINITY marks prohibited turns.
            if turn_penalty == INFINITY:
                continue

            arc_state = pqueue.effective_state(next_arc)
            if arc_state != SCANNED:
                next_cost = current_cost + graph_costs[next_arc] + turn_penalty

                if next_cost == INFINITY:
                    continue
                elif arc_state == NOT_IN_HEAP:
                    pqueue.insert(next_arc, next_cost)
                    arc_pred[next_arc] = current_arc
                    arc_turn_penalties[next_arc] = current_turn_cost + turn_penalty
                elif pqueue.element_key(next_arc) > next_cost:
                    pqueue.decrease_key(next_arc, next_cost)
                    arc_pred[next_arc] = current_arc
                    arc_turn_penalties[next_arc] = current_turn_cost + turn_penalty

    return found - 1


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
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
) noexcept nogil:
    """Arc-based Dijkstra wrapper that allocates its own node label cost scratch array."""
    cdef unsigned int num_nodes = node_pred.shape[0]
    cdef:
        double *node_costs = <double *>malloc(num_nodes * sizeof(double))
        int found = 0

    if node_costs == NULL:
        return 0

    found = _path_finding_arc_based_core(
        origin,
        destinations,
        destination_count,
        graph_costs,
        csr_indices,
        graph_fs,
        arc_pred,
        ids,
        a_nodes,
        node_pred,
        connectors,
        reached_first,
        node_turn_penalties,
        turn_fs,
        turn_to_arcs,
        turn_penalties,
        allow_uturns,
        arc_turn_penalties,
        node_costs,
    )

    free(node_costs)
    return found
