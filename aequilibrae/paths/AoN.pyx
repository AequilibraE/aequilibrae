""" -----------------------------------------------------------------------------------------------------------
 Package:    AequilibraE
 Name:       Core path computation algorithms
 Purpose:    Implement shortest path and network loading routines
 Original Author:  Pedro Camargo (c@margo.co)
 Contributors:
 Last edited by: Pedro Camrgo
 Website:    www.AequilibraE.com
 Repository:  https://github.com/AequilibraE/AequilibraE
 Created:    15/09/2013
 Updated:    24/04/2018
 Copyright:   (c) AequilibraE authors
 Licence:     See LICENSE.TXT
 -----------------------------------------------------------------------------------------------------------"""
import os

# cython: language_level=3
cimport numpy as np
from libcpp cimport bool

# include 'parameters.pxi'
include 'basic_path_finding.pyx'
include 'bpr.pyx'
include 'conical.pyx'
include 'parallel_numpy.pyx'
include 'path_file_saving.pyx'
# include 'path_file_saving_nogil.pyx'


from .__version__ import binary_version as VERSION_COMPILED

def one_to_all(origin, matrix, graph, result, aux_result, curr_thread):
    cdef long nodes, orig, i, block_flows_through_centroids, classes, b, origin_index, zones, posit, posit1, links
    cdef int critical_queries = 0
    cdef int path_file = 0
    cdef int skims
    cdef int link_extract_queries, query_type

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

    if VERSION_COMPILED != graph.__version__:
        raise ValueError('This graph was created for a different version of AequilibraE. Please re-create it')

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
        tskim = aux_result.temporary_skims[:, :, curr_thread]
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
    cdef long long [:] predecessors_view = aux_result.predecessors[:, curr_thread]
    cdef long long [:] reached_first_view = aux_result.reached_first[:, curr_thread]
    cdef long long [:] conn_view = aux_result.connectors[:, curr_thread]
    cdef double [:, :] link_loads_view = aux_result.temp_link_loads[:, :, curr_thread]
    cdef double [:, :] node_load_view = aux_result.temp_node_loads[:, :, curr_thread]
    cdef long long [:] b_nodes_view = aux_result.temp_b_nodes[:, curr_thread]

    # path saving file paths
    cdef string path_file_base
    cdef string path_index_file_base
    cdef bool save_paths = False
    cdef bool write_feather = True
    if result.save_path_file == True:
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
                         g_view,
                         b_nodes_view,
                         graph_fs_view,
                         predecessors_view,
                         ids_graph_view,
                         conn_view,
                         reached_first_view)

        network_loading(classes,
                        demand_view,
                        predecessors_view,
                        conn_view,
                        link_loads_view,
                        no_path_view,
                        reached_first_view,
                        node_load_view,
                        w)

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

        if block_flows_through_centroids: # Re-blocks the centroid if that is the case
            b = 1
            blocking_centroid_flows(b,
                                    origin_index,
                                    zones,
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)

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

    results.origin = origin
    results.destination = destination
    orig = origin
    dest = destination
    origin_index = graph.nodes_to_indices[orig]
    dest_index = graph.nodes_to_indices[dest]
    if results.__graph_id__ != graph.__id__:
        raise ValueError("Results object not prepared. Use --> results.prepare(graph)")

    if VERSION_COMPILED != graph.__version__:
        raise ValueError('This graph was created for a different version of AequilibraE. Please re-create it')

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

    if predecessors_view[dest_index] > 0:
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


def update_path_trace(results, destination, graph):
    # type: (PathResults, int, Graph) -> (None)
    """
    :param graph: AequilibraE graph. Needs to have been set with number of centroids and list of skims (if any)
    :param results: AequilibraE Matrix properly set for computation using matrix.computational_view([matrix list])
    :param skimming: if we will skim for all nodes or not
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
        if results.predecessors[dest_index] > 0:
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

    if VERSION_COMPILED != graph.__version__:
        raise ValueError('This graph was created for a different version of AequilibraE. Please re-create it')

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
    cdef long long [:] predecessors_view = aux_result.predecessors[:, curr_thread]
    cdef long long [:] reached_first_view = aux_result.reached_first[:, curr_thread]
    cdef long long [:] conn_view = aux_result.connectors[:, curr_thread]
    cdef long long [:] b_nodes_view = aux_result.temp_b_nodes[:, curr_thread]
    cdef double [:, :] skim_matrix_view = aux_result.temporary_skims[:, :, curr_thread]

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
