"""
 -----------------------------------------------------------------------------------------------------------
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
 -----------------------------------------------------------------------------------------------------------
 """

"""
TODO:
LIST OF ALL THE THINGS WE NEED TO DO TO NOT HAVE TO HAVE nodes 1..n as CENTROIDS. ARBITRARY NUMBERING
- Checks of weather the centroid we are computing path from is a centroid and/or exists in the graph
- Re-write function **network_loading** on the part of loading flows to centroids
"""

cimport numpy as np

# include 'parameters.pxi'
include 'basic_path_finding.pyx'
from libc.stdlib cimport abort, malloc, free
from __version__ import binary_version as VERSION_COMPILED

def one_to_all(origin, matrix, graph, result, aux_result, curr_thread):
    cdef long nodes, orig, i, block_flows_through_centroids, classes, b, origin_index, zones, posit, posit1
    cdef int critical_queries = 0
    cdef int link_extract_queries, query_type

    # Origin index is the index of the matrix we are assigning
    # this is used as index for the skim matrices
    # orig is the ID of the actual centroid
    # Is is used to actual path computation and to refer to outputs of path computation

    orig = origin
    origin_index = graph.nodes_to_indices[orig]

    if VERSION_COMPILED != graph.__version__:
        raise ValueError('This graph was created for a different version of AequilibraE. Please re-create it')

    if result.critical_links['save']:
        critical_queries = len(result.critical_links['queries'])
        aux_link_flows = np.zeros(result.links, ITYPE)
    else:
        aux_link_flows = np.zeros(1, ITYPE)

    if result.link_extraction['save']:
        link_extract_queries = len(result.link_extraction['queries'])

    nodes = graph.num_nodes
    zones = graph.num_zones
    block_flows_through_centroids = graph.block_centroid_flows

    # In order to release the GIL for this procedure, we create all the
    # memory views we will need
    cdef double [:, :] demand_view = matrix.matrix_view[origin_index, :, :]
    classes = matrix.matrix_view.shape[2]

    # views from the graph
    cdef long long [:] graph_fs_view = graph.fs
    cdef double [:] g_view = graph.cost
    cdef long long [:] ids_graph_view = graph.ids
    cdef long long [:] all_nodes_view = graph.all_nodes
    cdef long long [:] original_b_nodes_view = graph.graph['b_node']
    cdef double [:, :] graph_skim_view = graph.skims

    # views from the result object
    cdef double [:, :] final_skim_matrices_view = result.skims.matrix_view[origin_index, :, :]
    cdef long long [:] no_path_view = result.no_path[origin_index, :]

    # views from the aux-result object
    cdef long long [:] predecessors_view = aux_result.predecessors[:, curr_thread]
    cdef double [:, :] skim_matrix_view = aux_result.temporary_skims[:, :, curr_thread]
    cdef long long [:] reached_first_view = aux_result.reached_first[:, curr_thread]
    cdef long long [:] conn_view = aux_result.connectors[:, curr_thread]
    cdef double [:, :] link_loads_view = aux_result.temp_link_loads[:, :, curr_thread]
    cdef double [:, :] node_load_view = aux_result.temp_node_loads[:, :, curr_thread]
    cdef long long [:] b_nodes_view = aux_result.temp_b_nodes[:, curr_thread]

    # path file variables
    # 'origin', 'node', 'predecessor', 'connector'
    posit = origin_index * graph.num_nodes * result.path_file['save']
    posit1 = posit + graph.num_nodes

    cdef unsigned int [:] pred_view = result.path_file['results'].predecessor[posit:posit1]
    cdef unsigned int [:] c_view = result.path_file['results'].connector[posit:posit1]
    cdef unsigned int [:] o_view = result.path_file['results'].origin[posit:posit1]
    cdef unsigned int [:] n_view = result.path_file['results'].node[posit:posit1]

    # select link variables
    cdef double [:, :] sel_link_view = result.critical_links['results'].matrix_view[origin_index,:,:]
    cdef long long [:] aux_link_flows_view = aux_link_flows

    #Now we do all procedures with NO GIL
    with nogil:
        if block_flows_through_centroids: # Unblocks the centroid if that is the case
            b = 0
            blocking_centroid_flows(b,
                                    origin_index,
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

        if block_flows_through_centroids: # Re-blocks the centroid if that is the case
            b = 1
            blocking_centroid_flows(b,
                                    origin_index,
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)

        _copy_skims(skim_matrix_view,
                    final_skim_matrices_view)

    if result.path_file['save']:
        with nogil:
            put_path_file_on_disk(orig,
                                  pred_view,
                                  predecessors_view,
                                  c_view,
                                  conn_view,
                                  all_nodes_view,
                                  o_view,
                                  n_view)

    for i in range(critical_queries):
        critical_links_view = return_an_int_view(result.path_file['queries']['elements'][i])
        query_type = 0
        if result.path_file['queries'][ type][i] == "or":
            query_type = 1
        with nogil:
            perform_select_link_analysis(orig,
                                         classes,
                                         demand_view,
                                         predecessors_view,
                                         conn_view,
                                         aux_link_flows_view,
                                         sel_link_view,
                                         query_type)

    return origin

def path_computation(origin, destination, graph, results):
    # type: (int, int, Graph, PathResults) -> (None)
    """
    :param graph: AequilibraE graph. Needs to have been set with number of centroids and list of skims (if any)
    :param results: AequilibraE Matrix properly set for computation using matrix.computational_view([matrix list])
    :param skimming: if we will skim for all nodes or not
    """
    cdef ITYPE_t nodes, orig, dest, p, b, origin_index, dest_index, connector
    cdef long i, j, skims, a, block_flows_through_centroids

    results.origin = origin
    results.destination = destination
    orig = origin
    dest = destination
    origin_index = graph.nodes_to_indices[orig]
    dest_index = graph.nodes_to_indices[dest]
    if results.__graph_id__ != graph.__id__:
        return "Results object not prepared. Use --> results.prepare(graph)"

    # Consistency checks
    # if origin >= graph.fs.shape[0]:
    #     raise ValueError ("Node " + str(origin) + " is outside the range of nodes in the graph")

    if VERSION_COMPILED != graph.__version__:
        return 'This graph was created for a different version of AequilibraE. Please re-create it'

    #We transform the python variables in Cython variables
    nodes = graph.num_nodes

     # initializes skim_matrix for output
    # initializes predecessors  and link connectors for output
    results.predecessors.fill(-1)
    results.connectors.fill(-1)
    skims = results.num_skims

    #In order to release the GIL for this procedure, we create all the
    #memmory views we will need
    cdef double [:] g_view = graph.cost
    cdef long long [:] original_b_nodes_view = graph.graph['b_node']
    cdef long long [:] graph_fs_view = graph.fs
    cdef double [:, :] graph_skim_view = graph.skims
    cdef long long [:] ids_graph_view = graph.ids
    block_flows_through_centroids = graph.block_centroid_flows

    cdef long long [:] predecessors_view = results.predecessors
    cdef long long [:] conn_view = results.connectors
    cdef double [:, :] skim_matrix_view = results.skims
    cdef long long [:] reached_first_view = results.reached_first

    new_b_nodes = graph.b_node.copy()
    cdef long long [:] b_nodes_view = new_b_nodes

    #Now we do all procedures with NO GIL
    with nogil:
        if block_flows_through_centroids: # Unblocks the centroid if that is the case
            b = 0
            blocking_centroid_flows(b,
                                    origin_index,
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
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)

    if predecessors_view[dest_index] > 0:
        all_connectors = []
        all_nodes = [dest_index]
        mileposts = []
        p = dest_index
        if p != origin_index:
            while p != origin_index:
                p = predecessors_view[p]
                connector = conn_view[dest_index]
                all_connectors.append(graph.graph['link_id'][connector])
                mileposts.append(g_view[connector])
                all_nodes.append(p)
                dest_index = p
            results.path = np.asarray(all_connectors, graph.default_types('int'))[::-1]
            results.path_nodes = graph.all_nodes[np.asarray(all_nodes, graph.default_types('int'))][::-1]
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
            all_nodes = [dest_index]
            mileposts = []
            p = dest_index
            if p != origin_index:
                while p != origin_index:
                    p = results.predecessors[p]
                    connector = results.connectors[dest_index]
                    all_connectors.append(graph.graph['link_id'][connector])
                    mileposts.append(graph.cost[connector])
                    all_nodes.append(p)
                    dest_index = p
                results.path = np.asarray(all_connectors, graph.default_types('int'))[::-1]
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
    origin_index = graph.nodes_to_indices[orig]

    graph_fs = graph.fs

    if result.__graph_id__ != graph.__id__:
        return "Results object not prepared. Use --> results.prepare(graph)"

    if orig not in graph.centroids:
        return "Centroid " + str(orig) + " is outside the range of zones in the graph"

    if orig > graph.num_nodes:
        return "Centroid " + str(orig) + " does not exist in the graph"

    if graph_fs[orig] == graph_fs[orig + 1]:
        return "Centroid " + str(orig) + " does not exist in the graph"

    if VERSION_COMPILED != graph.__version__:
        return 'This graph was created for a different version of AequilibraE. Please re-create it'

    nodes = graph.num_nodes + 1
    zones = graph.num_zones
    block_flows_through_centroids = graph.block_centroid_flows
    skims = result.num_skims

    # In order to release the GIL for this procedure, we create all the
    # memory views we will need

    # views from the graph
    cdef long long [:] graph_fs_view = graph.fs
    cdef double [:] g_view = graph.cost
    cdef long long [:] ids_graph_view = graph.ids
    cdef long long [:] original_b_nodes_view = graph.b_node
    cdef double [:, :] graph_skim_view = graph.skims[:, :]

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
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)
    return orig