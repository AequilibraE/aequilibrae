# cython: language_level=3
"""Legacy public path queries. Assignment uses aon_context instead."""

import numpy as np
from libc.stdint cimport int64_t

from aequilibrae.paths.cython.skimming_core cimport (
    skim_single_path,
    skim_single_path_with_turn_penalties,
)
from aequilibrae.paths.cython.basic_path_finding cimport (
    blocking_centroid_flows,
    path_finding,
    path_finding_a_star,
    path_finding_arc_based,
    HeapType,
)
from aequilibrae.paths.cython.path_finding cimport Heuristic
from aequilibrae.utils.cython.bridge cimport Bridge, AeqLogClosure
from aequilibrae.paths.cython.basic_path_finding import HEURISTIC_MAP, HEAP_MAP


def available_heaps() -> list:
    """Return the available priority queue implementations for path queries."""
    return list(HEAP_MAP.keys())


def _resolve_heap(result):
    return HEAP_MAP[result._heap]


def path_computation(origin: int, destination: int, results, bridge: Bridge | None = None):
    """
    :param graph: AequilibraE graph. Needs to have been set with number of centroids and list of skims (if any)
    :param results: AequilibraE Matrix properly set for computation using matrix.computational_view([matrix list])
    :param skimming: if we will skim for all nodes or not
    """
    cdef int64_t nodes, orig, dest, p, b, origin_index, dest_index, connector, zones
    cdef long skims, block_flows_through_centroids
    cdef bint early_exit_bint = results.early_exit
    cdef bint use_turn_restrictions = False
    cdef bint allow_uturns = False
    cdef long long [:] penalty_skim_indices_view
    results.origin = origin
    results.destination = destination
    orig = origin
    dest = destination
    graph = results.graph

    # A* currently relies on node states + geometric heuristic and is not combined
    # with arc-state turn restrictions in this implementation.
    if graph.has_turn_restrictions and not results.a_star:
        use_turn_restrictions = True
        allow_uturns = graph.allow_path_uturns
    elif graph.has_turn_restrictions and results.a_star:
        raise RuntimeError("Turn restrictions and A* are not compatible.")
    origin_index = graph.nodes_to_indices[orig]
    dest_index = graph.nodes_to_indices[dest]

    # nodes_to_indices maps unknown nodes to -1. Without these guards A* would silently target
    # node 0 and Dijkstra's early exit would target an arbitrary node.
    if origin_index < 0:
        raise ValueError(f"Origin {orig} does not exist in the graph")
    if dest_index < 0:
        raise ValueError(f"Destination {dest} does not exist in the graph")

    # We transform the python variables in Cython variables
    nodes = graph.num_nodes
    zones = graph.num_zones

    # initializes skim_matrix for output
    # initializes predecessors  and link connectors for output
    results.predecessors.fill(-1)
    results.connectors.fill(-1)
    skims = len(graph.skim_fields)

    # In order to release the GIL for this procedure, we create all the
    # memory views we will need
    cdef double [::1] g_view = graph.cost
    cdef const long long [:] original_b_nodes_view = graph.graph.b_node.to_numpy(copy=False)
    cdef long long [::1] graph_fs_view = graph.fs
    cdef double [:, :] graph_skim_view = graph.skims
    cdef const long long [::1] ids_graph_view = graph.graph.id.to_numpy(copy=False)
    cdef const long long [:] a_nodes_view = graph.graph.a_node.to_numpy(copy=False)
    block_flows_through_centroids = graph.block_centroid_flows

    cdef long long [::1] predecessors_view = results.predecessors
    cdef long long [::1] conn_view = results.connectors
    cdef double [:, :] skim_matrix_view = results._skimming_array
    cdef long long [::1] reached_first_view = results.reached_first

    new_b_nodes = graph.graph.b_node.values.copy()
    cdef long long [::1] b_nodes_view = new_b_nodes

    # Turn restriction state (full graph CSR structures + per-call scratch arrays)
    cdef long long [:] turn_fs_view
    cdef long long [:] turn_to_arcs_view
    cdef double [:] turn_penalties_view
    cdef long long [:] arc_pred_view
    cdef double [:] node_turn_penalties_view
    cdef double [:] arc_turn_penalties_view

    if use_turn_restrictions:
        arc_pred = np.empty(graph.num_links, dtype=graph.default_types('int'))
        arc_pred_view = arc_pred
        node_turn_penalties = np.empty(graph.num_nodes, dtype=graph.default_types('float'))
        node_turn_penalties_view = node_turn_penalties
        arc_turn_penalties = np.empty(graph.num_links, dtype=graph.default_types('float'))
        arc_turn_penalties_view = arc_turn_penalties

        turn_fs_view = graph.turn_fs
        turn_to_arcs_view = graph.turn_to_arcs
        turn_penalties_view = graph.turn_penalties

        # Compute which skim fields receive turn penalties.
        # Default (empty turn_skim_fields) falls back to [cost_field] for backward compatibility.
        _pen_fields = graph.turn_skim_fields if graph.turn_skim_fields else (
            [graph.cost_field] if graph.cost_field else []
        )
        _pen_idx = np.array(
            [graph.skim_fields.index(f) for f in _pen_fields if f in graph.skim_fields],
            dtype=np.int64,
        )
        penalty_skim_indices_view = _pen_idx

    cdef HeapType heap_type = _resolve_heap(results)
    cdef Bridge br = bridge
    cdef AeqLogClosure *closure = br.c if br is not None else <AeqLogClosure*>NULL

    cdef bint a_star_bint = results.a_star
    cdef const double [::1] lat_view
    cdef const double [::1] lon_view
    cdef long long [::1] nodes_to_indices_view
    cdef Heuristic heuristic
    if results.a_star:
        lat_view = graph.lonlat_index.lat.to_numpy(copy=False)
        lon_view = graph.lonlat_index.lon.to_numpy(copy=False)
        nodes_to_indices_view = graph.nodes_to_indices
        heuristic = HEURISTIC_MAP[results._heuristic]

    # Destination set
    cdef unsigned char [::1] destinations
    if early_exit_bint and not a_star_bint:
        destinations = np.zeros(nodes, dtype=bool)
        destinations[dest_index] = True
    else:
        destinations = np.zeros(1, dtype=bool)

    # Now we do all procedures with NO GIL
    with nogil:
        if block_flows_through_centroids and not use_turn_restrictions:
            # Unblocks the centroid if that is the case. With turn restrictions the
            # automatic connector-to-connector turn bans replace b-node patching.
            b = 0
            blocking_centroid_flows(b,
                                    origin_index,
                                    zones,
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)

        if a_star_bint:
            path_finding_a_star(
                origin_index,
                dest_index,
                g_view,
                b_nodes_view,
                graph_fs_view,
                nodes_to_indices_view,
                lat_view,
                lon_view,
                predecessors_view,
                ids_graph_view,
                conn_view,
                heuristic,
                heap_type,
                closure
            )
        elif use_turn_restrictions:
            # Arc-based shortest path reconstruction still writes node-level
            # predecessor/connectors for compatibility with existing result APIs.
            w = path_finding_arc_based(
                origin_index,
                destinations,
                1 if early_exit_bint else -1,
                g_view,
                original_b_nodes_view,
                graph_fs_view,
                arc_pred_view,
                ids_graph_view,
                a_nodes_view,
                predecessors_view,
                conn_view,
                reached_first_view,
                node_turn_penalties_view,
                turn_fs_view,
                turn_to_arcs_view,
                turn_penalties_view,
                allow_uturns,
                arc_turn_penalties_view,
            )
        else:
            w = path_finding(origin_index,
                             destinations,
                             1 if early_exit_bint else -1,
                             g_view,
                             b_nodes_view,
                             graph_fs_view,
                             predecessors_view,
                             ids_graph_view,
                             conn_view,
                             reached_first_view,
                             heap_type,
                             closure)

        if skims > 0 and not a_star_bint:
            if use_turn_restrictions and penalty_skim_indices_view.shape[0] > 0:
                skim_single_path_with_turn_penalties(origin_index,
                                                     nodes,
                                                     skims,
                                                     skim_matrix_view,
                                                     predecessors_view,
                                                     conn_view,
                                                     graph_skim_view,
                                                     reached_first_view,
                                                     w,
                                                     node_turn_penalties_view,
                                                     penalty_skim_indices_view)
            else:
                skim_single_path(origin_index,
                                 nodes,
                                 skims,
                                 skim_matrix_view,
                                 predecessors_view,
                                 conn_view,
                                 graph_skim_view,
                                 reached_first_view,
                                 w)

        if block_flows_through_centroids and not use_turn_restrictions:
            # Restores the b-nodes if they were patched for centroid blocking
            b = 1
            blocking_centroid_flows(b,
                                    origin_index,
                                    zones,
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)

    path: np.ndarray | None = None
    path_nodes: np.ndarray | None = None
    path_link_directions: np.ndarray | None = None
    milepost: np.ndarray | None = None

    if predecessors_view[dest_index] >= 0:
        # Materialise the columns once. Reading them from the DataFrame inside the loop dominates
        # the runtime of this function, as each access re-boxes the whole column.
        link_ids = graph.graph.link_id.to_numpy(copy=False)
        directions = graph.graph.direction.to_numpy(copy=False)
        all_connectors = []
        link_directions = []
        all_nodes = [dest_index]
        mileposts = []

        if use_turn_restrictions:
            # Arc-based backtracking: walk predecessor arcs until the origin
            # sentinel (< 0) is reached.
            connector = conn_view[dest_index]
            while connector >= 0:
                all_connectors.append(link_ids[connector])
                link_directions.append(directions[connector])
                mileposts.append(g_view[connector])
                all_nodes.append(a_nodes_view[connector])
                connector = arc_pred[connector]
        else:
            p = dest_index
            if p != origin_index:
                while p != origin_index:
                    p = predecessors_view[p]
                    connector = conn_view[dest_index]
                    all_connectors.append(link_ids[connector])
                    link_directions.append(directions[connector])
                    mileposts.append(g_view[connector])
                    all_nodes.append(p)
                    dest_index = p

        if all_connectors:
            path = np.asarray(all_connectors, graph.default_types('int'))[::-1]
            path_nodes = graph.all_nodes[np.asarray(all_nodes, graph.default_types('int'))][::-1]
            path_link_directions = np.asarray(link_directions, graph.default_types('int'))[::-1]
            mileposts.append(0)
            milepost = np.cumsum(mileposts[::-1])

        del all_nodes
        del all_connectors
        del mileposts

    return path, path_nodes, path_link_directions, milepost


def update_path_trace(results, destination, graph):
    # type: (PathResults, int, Graph) -> (None)
    """
    If `results.early_exit` is `True`, early exit will be enabled if the path is to be recomputed.
    If `results.a_star` is `True`, A* will be used if the path is to be recomputed.

    :param graph: AequilibraE graph. Needs to have been set with number of centroids and list of skims (if any)
    :param results: AequilibraE Matrix properly set for computation using matrix.computational_view([matrix list])
    :param destination: New destination for path computation
    """
    cdef long long p, origin_index, dest_index, connector
    results.destination = destination
    if destination == results.origin:
        results.milepost = np.array([0], dtype=np.float32)
        results.path_nodes = np.array([results.origin], dtype=np.int32)
    else:
        dest_index = graph.nodes_to_indices[destination]
        origin_index = graph.nodes_to_indices[results.origin]
        results.milepost = None
        results.path_nodes = None

        # If the predecessor is -1 and early exit was enabled we cannot differentiate between an unreachable node and
        # one we just didn't see yet. We need to recompute the tree with the new destination If `a_star` was enabled
        # then the stored tree has no guarantees and may not be useful due to the heuristic used TODO: revisit with
        # heuristic specific reuse logic
        # With turn restrictions, the node-level shortest path tree is not reusable for
        # arc-level path reconstruction. Always recompute to guarantee legal turn sequences.
        if graph.has_turn_restrictions:
            results.compute_path(results.origin, destination, early_exit=results.early_exit, a_star=results.a_star)
            return
        if results.predecessors[dest_index] == -1 and results._early_exit or results._a_star:
            results.compute_path(results.origin, destination, early_exit=results.early_exit, a_star=results.a_star)

        # By the invariant hypothesis presented at
        # https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Proof_of_correctness Dijkstra's algorithm produces the
        # shortest path tree for all scanned nodes or vertices, even if it exited early.
        if results.predecessors[dest_index] >= 0:
            # Materialise the columns once. Reading them from the DataFrame inside the loop dominates
            # the runtime of this function, as each access re-boxes the whole column.
            link_ids = graph.graph.link_id.to_numpy(copy=False)
            directions = graph.graph.direction.to_numpy(copy=False)
            costs = graph.cost
            predecessors = results.predecessors
            connectors = results.connectors
            all_connectors = []
            link_directions = []
            all_nodes = [dest_index]
            mileposts = []
            p = dest_index
            if p != origin_index:
                while p != origin_index:
                    p = predecessors[p]
                    connector = connectors[dest_index]
                    all_connectors.append(link_ids[connector])
                    link_directions.append(directions[connector])
                    mileposts.append(costs[connector])
                    all_nodes.append(p)
                    dest_index = p
                results.path = np.asarray(all_connectors, graph.default_types('int'))[::-1]
                results.path_link_directions = np.asarray(link_directions, graph.default_types('int'))[::-1]
                results.path_nodes = graph.all_nodes[np.asarray(all_nodes, graph.default_types('int'))][::-1]
                mileposts.append(0)
                results.milepost = np.cumsum(mileposts[::-1])
        else:
            results.path = None
            results.path_nodes = None
            results.path_link_directions = None
            results.milepost = None
