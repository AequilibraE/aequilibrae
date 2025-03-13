import threading
from multiprocessing.dummy import Pool as ThreadPool

import cython
import numpy as np
import pandas as pd

from aequilibrae.paths.multi_threaded_paths import MultiThreadedPaths


def connectivity_multi_threaded(tester):
    graph = tester.graph
    cores = tester.cores
    signal = tester.connectivity

    aux_result = MultiThreadedPaths()
    aux_result.prepare_(graph, cores, graph.compact_num_nodes + 1)

    pool = ThreadPool(cores)
    all_threads = {"count": 0, "run": 0}
    results = {"disconnected": []}

    disconn_array = np.zeros((cores, graph.num_zones, 2), dtype=ITYPE)
    for i, orig in enumerate(list(graph.centroids)):
        args = (orig, graph, aux_result, disconn_array, all_threads, results, signal)
        pool.apply_async(connectivity_single_threaded, args=args)
    pool.close()
    pool.join()

    signal.emit(["text connectivity", "Saving Outputs"])
    signal.emit(["finished_threaded_procedure", None])

    if len(results["disconnected"]) > 0:
        disconn = np.vstack(results["disconnected"]).astype(np.int64)
        disconn[:, 1] = graph.centroids[disconn[:, 1]]
    else:
        disconn = []

    return pd.DataFrame(disconn, columns=["origin", "destination"])


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cdef connectivity_single_threaded(origin, graph, aux_result, disconn_array, all_threads, results, signal):
    if threading.get_ident() in all_threads:
        core_id = all_threads[threading.get_ident()]
    else:
        all_threads[threading.get_ident()] = all_threads["count"]
        core_id = all_threads["count"]
        all_threads["count"] += 1

    cdef:
        ITYPE_t i, b, k
        long orig = origin
        long long block_flows_through_centroids = graph.block_centroid_flows
        long long [:] origin_index = graph.compact_nodes_to_indices
        int zones = graph.num_zones

        long long [:] graph_fs_view = graph.compact_fs
        long long [:] original_b_nodes_view = graph.compact_graph.b_node.values

        # views from the aux-result object
        long long [:] predecessors_view = aux_result.predecessors[core_id, :]
        long long [:] b_nodes_view = aux_result.temp_b_nodes[core_id, :]
        long long [:, :] disconn_view = disconn_array[core_id, :, :]

    with nogil:
        if block_flows_through_centroids:  # Blocks the centroid if that is the case
            b = 0
            blocking_centroid_flows(b,
                                    origin_index[orig],
                                    zones,
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)

        dfs(origin_index[orig],
            b_nodes_view,
            graph_fs_view,
            predecessors_view)

        if block_flows_through_centroids:  # Unblocks the centroid if that is the case
            b = 1
            blocking_centroid_flows(b,
                                    origin_index[orig],
                                    zones,
                                    graph_fs_view,
                                    b_nodes_view,
                                    original_b_nodes_view)

        k = 0
        for i in range(zones):
            if predecessors_view[i] < 0:
                disconn_view[k, 0] = orig
                disconn_view[k, 1] = i
                k+= 1

    signal.emit(["zones finalized", all_threads["count"]])
    signal.emit(["text connectivity", f"{all_threads['count']} / {zones}"])

    if k > 0:
        results["disconnected"].append(np.array(disconn_array[core_id, :k, :]))
