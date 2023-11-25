# cython: language_level=3
import os

cimport numpy as np
from libcpp cimport bool
# include 'parameters.pxi'
include 'basic_path_finding.pyx'
include 'bpr.pyx'
include 'bpr2.pyx'
include 'conical.pyx'
include 'inrets.pyx'
include 'parallel_numpy.pyx'
include 'path_file_saving.pyx'
include 'graph_building.pyx'

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

    #We transform the python variables in Cython variables
    nodes = graph.compact_num_nodes
    links = graph.compact_num_links

    skims = len(graph.skim_fields)


    zones = graph.num_zones
    block_flows_through_centroids = graph.block_centroid_flows

    # In order to release the GIL for this procedure, we create all the
    # memory views we will need
    cdef double [:, :] demand_view = matrix.matrix_view[origin_index, :, :]
    classes = matrix.matrix_view.shape[2]

    # views from the graph
    cdef long long [:] graph_fs_view = graph.compact_fs
    cdef double [:] g_view = graph.compact_cost
    cdef long long [:] ids_graph_view = graph.compact_graph.id.values
    cdef long long [:] all_nodes_view = graph.compact_all_nodes
    cdef long long [:] original_b_nodes_view = graph.compact_graph.b_node.values

    if skims > 0:
        gskim = graph.compact_skims
        tskim = aux_result.temporary_skims[curr_thread, :, :]
        fskm = result.skims.matrix_view[origin_index, :, :]
    else:
        gskim = np.zeros((1,1))
        tskim = np.zeros((1,1))
        fskm = np.zeros((1,1))

    cdef double [:, :] graph_skim_view = gskim
    cdef double [:, :] skim_matrix_view = tskim
    cdef double [:, :] final_skim_matrices_view = fskm

    # views from the result object
    cdef long long [:] no_path_view = result.no_path[origin_index, :]

    # views from the aux-result object
    cdef long long [:] predecessors_view = aux_result.predecessors[curr_thread, :]
    cdef long long [:] reached_first_view = aux_result.reached_first[curr_thread, :]
    cdef long long [:] conn_view = aux_result.connectors[curr_thread, :]
    cdef double [:, :] link_loads_view = aux_result.temp_link_loads[curr_thread, :, :]
    cdef double [:, :] node_load_view = aux_result.temp_node_loads[curr_thread, :, :]
    cdef long long [:] b_nodes_view = aux_result.temp_b_nodes[curr_thread, :]

    # path saving file paths
    cdef string path_file_base
    cdef string path_index_file_base
    cdef bool save_paths = False
    cdef bool write_feather = True
    if result.save_path_file:
        save_paths = True
        write_feather = result.write_feather
        if write_feather:
            base_string = os.path.join(result.path_file_dir, f"o{origin_index}.feather")
            index_string = os.path.join(result.path_file_dir, f"o{origin_index}_indexdata.feather")
        else:
            base_string = os.path.join(result.path_file_dir, f"o{origin_index}.parquet")
            index_string = os.path.join(result.path_file_dir, f"o{origin_index}_indexdata.parquet")
        path_file_base = base_string.encode('utf-8')
        path_index_file_base = index_string.encode('utf-8')

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
    #Now we do all procedures with NO GIL
    with nogil:
        if block_flows_through_centroids: # Unblocks the centroid if that is the case
            b = 0
            blocking_centroid_flows(b,
                                    origin_index,
                                    zones,
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)

        w = path_finding(origin_index,
                         -1,  # destination index to disable early exit
                         g_view,
                         b_nodes_view,
                         graph_fs_view,
                         predecessors_view,
                         ids_graph_view,
                         conn_view,
                         reached_first_view)

        if block_flows_through_centroids: # Re-blocks the centroid if that is the case
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
            # do ONLY reular loading (via cascade assignment)
            network_loading(classes,
                            demand_view,
                            predecessors_view,
                            conn_view,
                            link_loads_view,
                            no_path_view,
                            reached_first_view,
                            node_load_view,
                            w)

    if result.save_path_file == True:
        save_path_file(origin_index, links, zones, predecessors_view, conn_view, path_file_base, path_index_file_base, write_feather)
    return origin

def path_computation(origin, destination, graph, results):
    # type: (int, int, Graph, PathResults) -> (None)
    """
    :param graph: AequilibraE graph. Needs to have been set with number of centroids and list of skims (if any)
    :param results: AequilibraE Matrix properly set for computation using matrix.computational_view([matrix list])
    :param skimming: if we will skim for all nodes or not
    """
    cdef ITYPE_t nodes, orig, dest, p, b, origin_index, dest_index, connector, zones
    cdef long i, j, skims, a, block_flows_through_centroids
    cdef bint early_exit_bint = results.early_exit

    results.origin = origin
    results.destination = destination
    orig = origin
    dest = destination
    origin_index = graph.nodes_to_indices[orig]
    dest_index = graph.nodes_to_indices[dest]
    if results.__graph_id__ != graph.__id__:
        raise ValueError("Results object not prepared. Use --> results.prepare(graph)")


    #We transform the python variables in Cython variables
    nodes = graph.num_nodes
    zones = graph.num_zones

    # initializes skim_matrix for output
    # initializes predecessors  and link connectors for output
    results.predecessors.fill(-1)
    results.connectors.fill(-1)
    skims = len(graph.skim_fields)

    #In order to release the GIL for this procedure, we create all the
    #memmory views we will need
    cdef double [:] g_view = graph.cost
    cdef long long [:] original_b_nodes_view = graph.graph.b_node.values
    cdef long long [:] graph_fs_view = graph.fs
    cdef double [:, :] graph_skim_view = graph.skims
    cdef long long [:] ids_graph_view = graph.graph.id.values
    block_flows_through_centroids = graph.block_centroid_flows

    cdef long long [:] predecessors_view = results.predecessors
    cdef long long [:] conn_view = results.connectors
    cdef double [:, :] skim_matrix_view = results._skimming_array
    cdef long long [:] reached_first_view = results.reached_first

    new_b_nodes = graph.graph.b_node.values.copy()
    cdef long long [:] b_nodes_view = new_b_nodes

    cdef bint a_star_bint = results.a_star
    cdef double [:] lat_view
    cdef double [:] lon_view
    cdef long long [:] nodes_to_indices_view
    cdef Heuristic heuristic
    if results.a_star:
        lat_view = graph.lonlat_index.lat.values
        lon_view = graph.lonlat_index.lon.values
        nodes_to_indices_view = graph.nodes_to_indices
        heuristic = HEURISTIC_MAP[results._heuristic]


    #Now we do all procedures with NO GIL
    with nogil:
        if block_flows_through_centroids: # Unblocks the centroid if that is the case
            b = 0
            blocking_centroid_flows(b,
                                    origin_index,
                                    zones,
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)

        if a_star_bint:
            w = path_finding_a_star(origin_index,
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
                                    reached_first_view,
                                    heuristic)
        else:
            w = path_finding(origin_index,
                             dest_index if early_exit_bint else -1,
                             g_view,
                             b_nodes_view,
                             graph_fs_view,
                             predecessors_view,
                             ids_graph_view,
                             conn_view,
                             reached_first_view)


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

        if block_flows_through_centroids: # Unblocks the centroid if that is the case
            b = 1
            blocking_centroid_flows(b,
                                    origin_index,
                                    zones,
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)

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
            results.path = np.asarray(all_connectors, graph.default_types('int'))[::-1]
            results.path_nodes = graph.all_nodes[np.asarray(all_nodes, graph.default_types('int'))][::-1]
            results.path_link_directions = np.asarray(link_directions, graph.default_types('int'))[::-1]
            mileposts.append(0)
            results.milepost =  np.cumsum(mileposts[::-1])

            del all_nodes
            del all_connectors
            del mileposts
    else:
        results.path = None
        results.path_nodes = None
        results.path_link_directions = None
        results.milepost = None

def update_path_trace(results, destination, graph):
    # type: (PathResults, int, Graph) -> (None)
    """
    If `results.early_exit` is `True`, early exit will be enabled if the path is to be recomputed.
    If `results.a_star` is `True`, A* will be used if the path is to be recomputed.

    :param graph: AequilibraE graph. Needs to have been set with number of centroids and list of skims (if any)
    :param results: AequilibraE Matrix properly set for computation using matrix.computational_view([matrix list])
    :param skimming: if we will skim for all nodes or not
    :param early_exit: Exit Dijkstra's once the destination has been found if the shortest path tree must be reconstructed.
    """
    cdef p, origin_index, dest_index, connector

    results.destination = destination
    if destination == results.origin:
        results.milepost = np.array([0], dtype=np.float32)
        results.path_nodes = np.array([results.origin], dtype=np.int32)
    else:
        dest_index = graph.nodes_to_indices[destination]
        origin_index = graph.nodes_to_indices[results.origin]
        results.milepost = None
        results.path_nodes = None

        # If the predecessor is -1 and early exit was enabled we cannot differentiate between
        # an unreachable node and one we just didn't see yet. We need to recompute the tree with the new destination
        # If `a_star` was enabled then the stored tree has no guarantees and may not be useful due to the heuristic used
        # TODO: revisit with heuristic specific reuse logic
        if results.predecessors[dest_index] == -1 and results._early_exit or results._a_star:
            results.compute_path(results.origin, destination, early_exit=results.early_exit, a_star=results.a_star)

        # By the invariant hypothesis presented at https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Proof_of_correctness
        # Dijkstra's algorithm produces the shortest path tree for all scanned nodes. That is if a node was scanned,
        # its shortest path has been found, even if we exited early. As the un-scanned nodes are marked as unreachable this
        # invariant holds.
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


def skimming_single_origin(origin, graph, result, aux_result, curr_thread):
    """
    :param origin:
    :param graph:
    :param results:
    :return:
    """
    cdef long long nodes, orig, origin_index, i, block_flows_through_centroids, skims, zones, b
    #We transform the python variables in Cython variables
    orig = origin
    origin_index = graph.compact_nodes_to_indices[orig]

    graph_fs = graph.compact_fs
    if result.__graph_id__ != graph.__id__:

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
    cdef long long [:] ids_graph_view = graph.compact_graph.id.values
    cdef long long [:] original_b_nodes_view = graph.compact_graph.b_node.values
    cdef double [:, :] graph_skim_view = graph.compact_skims[:, :]

    cdef double [:, :] final_skim_matrices_view = result.skims.matrix_view[origin_index, :, :]

    # views from the aux-result object
    cdef long long [:] predecessors_view = aux_result.predecessors[curr_thread, :]
    cdef long long [:] reached_first_view = aux_result.reached_first[curr_thread, :]
    cdef long long [:] conn_view = aux_result.connectors[curr_thread, :]
    cdef long long [:] b_nodes_view = aux_result.temp_b_nodes[curr_thread, :]
    cdef double [:, :] skim_matrix_view = aux_result.temporary_skims[curr_thread, :, :]

    #Now we do all procedures with NO GIL
    with nogil:
        if block_flows_through_centroids: # Unblocks the centroid if that is the case
            b = 0
            blocking_centroid_flows(b,
                                    origin_index,
                                    zones,
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)
        w = path_finding(origin_index,
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
                             zones, # ???????????????
                             skims,
                             skim_matrix_view,
                             predecessors_view,
                             conn_view,
                             graph_skim_view,
                             reached_first_view,
                             w,
                             final_skim_matrices_view)
        if block_flows_through_centroids: # Unblocks the centroid if that is the case
            b = 1
            blocking_centroid_flows(b,
                                    origin_index,
                                    zones,
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)
    return orig
