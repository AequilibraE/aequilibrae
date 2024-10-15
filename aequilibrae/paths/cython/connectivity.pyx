from cython.operator cimport dereference as d
from cython.parallel import parallel, prange
cimport numpy as np
from libcpp.vector cimport vector
from openmp cimport omp_get_thread_num
from libc.stdlib cimport malloc, free
from aequilibrae.paths.cython.parameters import ITYPE
import cython

ctypedef vector[pair_t]* disconn_pair

cdef struct pair_t:
    long long origin
    long long destination


def connectivity_multi_threaded(graph, aux_result, cores):
    cdef:
        long long i, b, thread_num
        int c_cores = cores
        long long block_flows_through_centroids = graph.block_centroid_flows
        long long [:] origin_index = graph.compact_nodes_to_indices
        int zones = graph.num_zones
        int nodes = graph.compact_num_nodes + 1
        long long [:] centroids = graph.centroids


        long long [:] graph_fs_view = graph.compact_fs
        double [:] g_view = graph.compact_cost
        long long [:] ids_graph_view = graph.compact_graph.id.values
        long long [:] original_b_nodes_view = graph.compact_graph.b_node.values
        long long b_size = graph.compact_graph.b_node.shape[0]

        # views from the aux-result object
        long long [:, :] predecessors_view = aux_result.predecessors
        long long [:, :] reached_first_view = aux_result.reached_first
        long long [:, :] conn_view = aux_result.connectors
        long long [:, :] b_nodes_view = aux_result.temp_b_nodes

        vector[disconn_pair] all_disconnected = vector[disconn_pair](zones)

    #Now we do all procedures with NO GIL
    with nogil, parallel(num_threads=c_cores):
        thread_num = omp_get_thread_num()
        for i in prange(zones, schedule="dynamic"):
            # if block_flows_through_centroids: # Unblocks the centroid if that is the case
            #     b = 0
                # blocking_centroid_flows(b,
                #                         origin_index[centroids[i]],
                #                         zones,
                #                         graph_fs_view,
                #                         b_nodes_view[thread_num, :],
                #                         original_b_nodes_view)
            # w = path_finding(origin_index[centroids[i]],
            #                  -1,  # destination index to disable early exit
            #                  g_view,
            #                  b_nodes_view[thread_num, :],
            #                  graph_fs_view,
            #                  predecessors_view[thread_num, :],
            #                  ids_graph_view,
            #                  conn_view[thread_num, :],
            #                  reached_first_view[thread_num, :])
            #
            # if block_flows_through_centroids: # Unblocks the centroid if that is the case
            #     b = 1
            #     blocking_centroid_flows(b,
            #                             origin_index,
            #                             zones,
            #                             graph_fs_view,
            #                             b_nodes_view[thread_num, :],
            #                             original_b_nodes_view)
            #
            all_disconnected[i] = disconnected(omp_get_thread_num(),
                                               centroids[i],
                                               zones,
                                               conn_view
                                                  )

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cdef disconn_pair disconnected(int thread_num,
                               long long origin,
                               long long zones,
                               long long [:, :] connectors) noexcept nogil:

    cdef:
        long long i
        vector[pair_t] *pairs = new vector[pair_t]()

    for i in range(zones):
        if connectors[thread_num, i] == NULL_IDX:
            create_pair(origin, i, d(pairs))
    return pairs


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cdef void create_pair(long long origin, long long destination, vector[pair_t] &pairs) noexcept nogil:
    cdef pair_t pair
    
    pair.origin = origin
    pair.destination = destination
    pairs.push_back(pair)
