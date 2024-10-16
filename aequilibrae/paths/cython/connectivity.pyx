import threading
from multiprocessing.dummy import Pool as ThreadPool

import cython
from libcpp.vector cimport vector

from aequilibrae.paths.cython.parameters import ITYPE
from aequilibrae.paths.multi_threaded_paths import MultiThreadedPaths

ctypedef vector[ITYPE_t]* disconn_vec

def connectivity_multi_threaded(tester):
    graph = tester.graph
    cores = tester.cores
    signal = tester.connectivity
    
    aux_result = MultiThreadedPaths()
    aux_result.prepare_(graph, cores, graph.compact_num_nodes + 1)
    
    cdef:
        long zones = graph.num_zones
        vector[disconn_vec] all_disconnected = vector[disconn_vec](zones)

    pool = ThreadPool(cores)
    all_threads = {"count": 0, "run": 0}
    for i, orig in enumerate(list(graph.centroids)):
        args = (orig, graph, aux_result, all_threads, signal)
        pool.apply_async(connectivity_single_threaded, args=args)
    pool.close()
    pool.join()

    signal.emit(["text connectivity", "Saving Outputs"])
    signal.emit(["finished_threaded_procedure", None])


cdef disconn_vec connectivity_single_threaded(origin, graph, aux_result, all_threads, signal):
    if threading.get_ident() in all_threads:
        core_id = all_threads[threading.get_ident()]
    else:
        all_threads[threading.get_ident()] = all_threads["count"]
        core_id = all_threads["count"]
        all_threads["count"] += 1

    cdef:
        ITYPE_t i, b, thread_num
        int orig = origin
        int core = core_id
        long long block_flows_through_centroids = graph.block_centroid_flows
        long long [:] origin_index = graph.compact_nodes_to_indices
        int zones = graph.num_zones
        vector[ITYPE_t] *disconnected = new vector[ITYPE_t]()

        long long [:] graph_fs_view = graph.compact_fs
        double [:] g_view = graph.compact_cost
        long long [:] ids_graph_view = graph.compact_graph.id.values
        long long [:] original_b_nodes_view = graph.compact_graph.b_node.values

        # views from the aux-result object
        long long [:] predecessors_view = aux_result.predecessors[core_id, :]
        long long [:] reached_first_view = aux_result.reached_first[core_id, :]
        long long [:] conn_view = aux_result.connectors[core_id, :]
        long long [:] b_nodes_view = aux_result.temp_b_nodes[core_id, :]

    with nogil:
        if block_flows_through_centroids: # Unblocks the centroid if that is the case
            b = 0
            blocking_centroid_flows(b,
                                    origin_index[orig],
                                    zones,
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)

        w = path_finding(origin_index[orig],
                         -1,  # destination index to disable early exit
                         g_view,
                         b_nodes_view,
                         graph_fs_view,
                         predecessors_view,
                         ids_graph_view,
                         conn_view,
                         reached_first_view)
        if block_flows_through_centroids: # Unblocks the centroid if that is the case
            b = 1
            blocking_centroid_flows(b,
                                    origin_index[orig],
                                    zones,
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)
        for i in range(zones):
            if predecessors_view[i] == -1:
                disconnected.push_back(i)

    signal.emit(["zones finalized", all_threads["count"]])
    signal.emit(["text connectivity", f"{all_threads['count']} / {zones}"])

    return disconnected