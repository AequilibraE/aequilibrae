# cython: language_level=3
cimport cython

import os
import numpy as np
from libc.string cimport memset
from aequilibrae.paths.cython.skimming_core cimport skim_single_path, _copy_skims
from aequilibrae.paths.cython.basic_path_finding cimport (
    blocking_centroid_flows,
    path_finding,
    path_finding_a_star,
)
from aequilibrae.paths.cython.basic_path_finding import HEURISTIC_MAP
from aequilibrae.paths.cython.path_file_saving import save_path_file

include 'parameters.pxi'


def one_to_all(origin, matrix, graph, result, aux_result, curr_thread):
    # type: (int, AequilibraeMatrix, Graph, AssignmentResults, MultiThreadedAoN, int) -> int
    cdef long nodes, orig, block_flows_through_centroids, classes, b, origin_index, zones, links
    cdef int skims

    # Origin index is the index of the matrix we are assigning
    # this is used as index for the skim matrices
    # orig is the ID of the actual centroid
    # Is is used to actual path computation and to refer to outputs of path computation

    orig = origin
    origin_index = graph.compact_nodes_to_indices[orig]

    # We transform the python variables in Cython variables
    nodes = graph.compact_num_nodes
    links = graph.compact_num_links

    skims = len(graph.skim_fields)

    zones = graph.num_zones
    block_flows_through_centroids = graph.block_centroid_flows

    # In order to release the GIL for this procedure, we create all the
    # memory views we will need
    cdef double [:, :] demand_view = matrix.matrix_view[origin_index, :, :]
    classes = matrix.matrix_view.shape[2]

    # Destination set
    cdef long long nnz_destinations = 0
    cdef unsigned char [:] destinations

    tmp = np.zeros(nodes, dtype=bool)
    if not skims:
        nonzero = matrix.matrix_view[origin_index, :, :].sum(axis=1).nonzero()[0]
        tmp[nonzero] = True
        nnz_destinations = len(nonzero)
    else:
        tmp[graph.nodes_to_indices[graph.centroids]] = True
        nnz_destinations = zones

    destinations = tmp

    # If there's no demand, disable early exit. We could let this fall through an immediately exit the path finding, but
    # this case should never happen in assignment so this is a little more flexible.
    if nnz_destinations == 0:
        destinations = np.array([], dtype=bool)
        nnz_destinations = -1

    # views from the graph
    cdef long long [:] graph_fs_view = graph.compact_fs
    cdef double [:] g_view = graph.compact_cost
    cdef const long long [:] ids_graph_view = graph.compact_graph.id.to_numpy(copy=False)
    cdef const long long [:] original_b_nodes_view = graph.compact_graph.b_node.to_numpy(copy=False)

    if skims > 0:
        gskim = graph.compact_skims
        tskim = aux_result.temporary_skims[curr_thread, :, :]
        fskm = result.skims.matrix_view[origin_index, :, :]
    else:
        gskim = np.zeros((1, 1))
        tskim = np.zeros((1, 1))
        fskm = np.zeros((1, 1))

    cdef double [:, :] graph_skim_view = gskim
    cdef double [:, :] skim_matrix_view = tskim
    cdef double [:, :] final_skim_matrices_view = fskm

    # views from the aux-result object
    cdef long long [:] predecessors_view = aux_result.predecessors[curr_thread, :]
    cdef long long [:] reached_first_view = aux_result.reached_first[curr_thread, :]
    cdef long long [:] conn_view = aux_result.connectors[curr_thread, :]
    cdef double [:, :] link_loads_view = aux_result.temp_link_loads[curr_thread, :, :]
    cdef long long [:] b_nodes_view = aux_result.temp_b_nodes[curr_thread, :]

    # path saving file paths
    cdef bint write_feather = True
    if result.save_path_file:
        write_feather = result.write_feather
        if write_feather:
            base_string = os.path.join(result.path_file_dir, f"o{origin_index}.feather")
            index_string = os.path.join(result.path_file_dir, f"o{origin_index}_indexdata.feather")
        else:
            base_string = os.path.join(result.path_file_dir, f"o{origin_index}.parquet")
            index_string = os.path.join(result.path_file_dir, f"o{origin_index}_indexdata.parquet")

    cdef:
        double [:, :, :] sl_od_matrix_view
        double [:, :, :] sl_link_loading_view
        unsigned char [:] has_flow_mask
        long long[:, :] link_list
        bint select_link = False

    if result._selected_links:
        has_flow_mask = aux_result.has_flow_mask[curr_thread, :]
        sl_od_matrix_view = aux_result.temp_sl_od_matrix[curr_thread, :, origin_index, :, :]
        sl_link_loading_view = aux_result.temp_sl_link_loading[curr_thread, :, :, :]
        link_list = aux_result.select_links[:, :]  # Read only, don't need to slice on curr_thread
        select_link = True

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
                         -1 if skims > 0 else nnz_destinations,
                         g_view,
                         b_nodes_view,
                         graph_fs_view,
                         predecessors_view,
                         ids_graph_view,
                         conn_view,
                         reached_first_view)

        if block_flows_through_centroids:  # Re-blocks the centroid if that is the case
            b = 1
            blocking_centroid_flows(b,
                                    origin_index,
                                    zones,
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)

        if skims > 0:
            skim_single_path(origin_index,
                             nodes,
                             skims,
                             skim_matrix_view,
                             predecessors_view,
                             conn_view,
                             graph_skim_view,
                             reached_first_view,
                             w)
            _copy_skims(skim_matrix_view,
                        final_skim_matrices_view)

        # If we aren't doing SL analysis we use a fast cascade assignment in the 'network_loading' method.
        # However, if we are doing SL analysis, we have to walk the entire path for each OD pair anyway
        # Even if cascading is more efficient, we can do the link loading concurrently while executing SL loading
        # which reduces the amount of repeated work we would do if they were separate
        # Note: 1 corresponds to select link analysis, 0 means no select link
        if select_link:
            # Do SL and network loading at once
            sl_network_loading(link_list, demand_view, predecessors_view, conn_view, link_loads_view, sl_od_matrix_view,
                               sl_link_loading_view, has_flow_mask, classes)
        else:
            # do ONLY regular loading
            network_loading(
                classes,
                demand_view,
                predecessors_view,
                conn_view,
                link_loads_view
            )

    if result.save_path_file:
        save_path_file(
            origin_index,
            links,
            zones,
            predecessors_view,
            conn_view,
            base_string,
            index_string,
            write_feather
        )
    return origin


def path_computation(origin, destination, results):
    # type: (int, int, PathResults) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]
    """
    :param graph: AequilibraE graph. Needs to have been set with number of centroids and list of skims (if any)
    :param results: AequilibraE Matrix properly set for computation using matrix.computational_view([matrix list])
    :param skimming: if we will skim for all nodes or not
    """
    cdef long long nodes, orig, dest, p, b, origin_index, dest_index, connector, zones
    cdef long skims, block_flows_through_centroids
    cdef bint early_exit_bint = results.early_exit
    results.origin = origin
    results.destination = destination
    orig = origin
    dest = destination
    graph = results.graph
    origin_index = graph.nodes_to_indices[orig]
    dest_index = graph.nodes_to_indices[dest]

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
    cdef double [:] g_view = graph.cost
    cdef const long long [:] original_b_nodes_view = graph.graph.b_node.to_numpy(copy=False)
    cdef long long [:] graph_fs_view = graph.fs
    cdef double [:, :] graph_skim_view = graph.skims
    cdef const long long [:] ids_graph_view = graph.graph.id.to_numpy(copy=False)
    block_flows_through_centroids = graph.block_centroid_flows

    cdef long long [:] predecessors_view = results.predecessors
    cdef long long [:] conn_view = results.connectors
    cdef double [:, :] skim_matrix_view = results._skimming_array
    cdef long long [:] reached_first_view = results.reached_first

    new_b_nodes = graph.graph.b_node.values.copy()
    cdef long long [:] b_nodes_view = new_b_nodes

    cdef bint a_star_bint = results.a_star
    cdef const double [:] lat_view
    cdef const double [:] lon_view
    cdef long long [:] nodes_to_indices_view
    cdef int heuristic
    if results.a_star:
        lat_view = graph.lonlat_index.lat.to_numpy(copy=False)
        lon_view = graph.lonlat_index.lon.to_numpy(copy=False)
        nodes_to_indices_view = graph.nodes_to_indices
        heuristic = HEURISTIC_MAP[results._heuristic]

    # Destination set
    cdef unsigned char [:] destinations
    if early_exit_bint and not a_star_bint:
        destinations = np.zeros(nodes, dtype=bool)
        destinations[dest_index] = True
    else:
        destinations = np.zeros(1, dtype=bool)

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
                heuristic
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
                             reached_first_view)

        if skims > 0 and not a_star_bint:
            skim_single_path(origin_index,
                             nodes,
                             skims,
                             skim_matrix_view,
                             predecessors_view,
                             conn_view,
                             graph_skim_view,
                             reached_first_view,
                             w)

        if block_flows_through_centroids:  # Unblocks the centroid if that is the case
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
        all_connectors = []
        link_directions = []
        all_nodes = [dest_index]
        mileposts = []
        p = dest_index
        if p != origin_index:
            while p != origin_index:
                p = predecessors_view[p]
                connector = conn_view[dest_index]
                all_connectors.append(graph.graph.link_id.values[connector])
                link_directions.append(graph.graph.direction.values[connector])
                mileposts.append(g_view[connector])
                all_nodes.append(p)
                dest_index = p
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
    print(f"started updateing trace to {destination}")
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
        if results.predecessors[dest_index] == -1 and results._early_exit or results._a_star:
            results.compute_path(results.origin, destination, early_exit=results.early_exit, a_star=results.a_star)

        # By the invariant hypothesis presented at
        # https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Proof_of_correctness Dijkstra's algorithm produces the
        # shortest path tree for all scanned nodes. That is if a node was scanned, its shortest path has been found,
        # even if we exited early. As the un-scanned nodes are marked as unreachable this invariant holds.
        if results.predecessors[dest_index] >= 0:
            all_connectors = []
            link_directions = []
            all_nodes = [dest_index]
            mileposts = []
            p = dest_index
            if p != origin_index:
                while p != origin_index:
                    p = results.predecessors[p]
                    connector = results.connectors[dest_index]
                    all_connectors.append(graph.graph.link_id.values[connector])
                    link_directions.append(graph.graph.direction.values[connector])
                    mileposts.append(graph.cost[connector])
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



@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)  # turn off bounds-checking for entire function
cpdef void network_loading(
    long classes,
    double[:, :] demand,
    long long [:] pred,
    long long [:] conn,
    double[:, :] link_loads
) noexcept nogil:
    cdef long long i, j, predecessor, connector, node
    cdef long long zones = demand.shape[0]

    # Traditional loading, without cascading
    for i in range(zones):
        node = i

        predecessor = pred[node]
        connector = conn[node]
        while predecessor >= 0:
            for j in range(classes):
                link_loads[connector, j] += demand[i, j]

            connector = conn[predecessor]
            predecessor = pred[predecessor]


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cdef void sl_network_loading(
    long long [:, :] selected_links,
    double [:, :] demand,
    long long [:] pred,
    long long [:] conn,
    double [:, :] link_loads,
    double [:, :, :] sl_od_matrix,
    double [:, :, :] sl_link_loading,
    unsigned char [:] has_flow_mask,
    long classes
) noexcept nogil:
    # VARIABLES:
    #   selected_links: 2d memoryview. Each row corresponds to a set of selected links specified by the user
    #   demand:         The input demand matrix for a given origin. The first index corresponds to destination,
    #                   second is the class
    #   pred:           The list of predecessor nodes, i.e. given a node, referencing that node's index in pred
    #                   yields the previous node in the minimum spanning tree
    #   conn:           The list of links which connect predecessor nodes. referencing it by the predecessor yields
    #                   the link it used to connect the two nodes
    # link_loads:       Stores the loading on each link. Equivalent to link_loads in network_loading
    # temp_sl_od_matrix:     Stores the OD matrix for each set of selected links sliced for the given origin
    # The indices are:  set of links, destination, class
    # temp_sl_link_loading:  Stores the loading on the Selected links, and the paths which use the selected links
    #                   The indices are: set of links, link_id, class)
    # has_flow_mask:    An array which acts as a flag for which links were used in retracing a given OD path
    # classes:          the number of subclasses of vehicles for the given TrafficClass
    #
    # Executes regular loading, while keeping track of SL links
    cdef:
        int i, j, k, l, dests = demand.shape[0], xshape = has_flow_mask.shape[0]
        long long predecessor, connection
        bint found
    for j in range(dests):
        memset(&has_flow_mask[0], 0, xshape * sizeof(unsigned char))
        connection = conn[j]
        predecessor = pred[j]

        # Walk the path and mark all used links in the has_flow_mask
        while predecessor >= 0:
            for k in range(classes):
                link_loads[connection, k] += demand[j, k]
            has_flow_mask[connection] = 1
            connection = conn[predecessor]
            predecessor = pred[predecessor]
        # Now iterate through each SL set and see if any of their links were marked
        for i in range(selected_links.shape[0]):
            # We check to see if the given OD path marked any of our selected links
            found = 0
            l = 0
            while l < selected_links.shape[1] and found == 0:
                # Checks to see if the current set of selected links has finished
                # NOTE: -1 is a default value for the selected_links array. It cannot be a link id, if -1 turns up
                # There is either a serious bug, or the program has reached the end of a set of links in SL.
                # This lets us early exit from the loop without needing to iterate through the rest of the array
                # Which just has placeholder values
                if selected_links[i][l] == -1:
                    break
                if has_flow_mask[selected_links[i][l]] != 0:
                    found = 1
                l += 1
            if found == 0:
                continue
            for k in range(classes):
                sl_od_matrix[i, j, k] = demand[j, k]
            connection = conn[j]
            predecessor = pred[j]
            while predecessor >= 0:
                for k in range(classes):
                    sl_link_loading[i, connection, k] += demand[j, k]
                connection = conn[predecessor]
                predecessor = pred[predecessor]


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void put_path_file_on_disk(unsigned int orig,
                                 unsigned int [:] pred,
                                 long long [:] predecessors,
                                 unsigned int [:] conn,
                                 long long [:] connectors,
                                 long long [:] all_nodes,
                                 unsigned int [:] origins_to_write,
                                 unsigned int [:] nodes_to_write) noexcept nogil:
    cdef long long i
    cdef long long k = pred.shape[0]

    for i in range(k):
        origins_to_write[i] = orig
        nodes_to_write[i] = all_nodes[i]
        pred[i] = all_nodes[predecessors[i]]
        conn[i] = connectors[i]
