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

from aequilibrae.paths.cython.pq_heap_types cimport FourAryHeap, PairingHeap, StdPriorityQueueAdapter
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
