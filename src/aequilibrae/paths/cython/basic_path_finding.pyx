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
from libc.math cimport sin, cos, asin, sqrt, pi
from libcpp.vector cimport vector

include 'pq_4ary_heap.pyx'

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
cpdef int path_finding(
        long origin,
        unsigned char [:] destinations,
        long long destination_count,
        double[:] graph_costs,
        long long [:] csr_indices,
        long long [:] graph_fs,
        long long [:] pred,
        const long long [:] ids,
        long long [:] connectors,
        long long [:] reached_first
) noexcept nogil:
    cdef unsigned int M = pred.shape[0]
    cdef:
        size_t tail_vert_idx, head_vert_idx, idx  # indices
        DTYPE_t tail_vert_val, head_vert_val  # vertex travel times
        PriorityQueue pqueue  # binary heap
        ElementState vert_state  # vertex state
        size_t origin_vert = <size_t> origin
        ITYPE_t found = 0

    for i in range(M):
        pred[i] = -1
        connectors[i] = -1
        reached_first[i] = -1

    # initialization of the heap elements
    # all nodes have INFINITY key and NOT_IN_HEAP state
    init_heap(&pqueue, <size_t> M)

    # the key is set to zero for the origin vertex,
    # which is inserted into the heap
    insert(&pqueue, origin_vert, 0.0)

    # main loop
    while pqueue.size > 0:
        tail_vert_idx = extract_min(&pqueue)
        reached_first[found] = tail_vert_idx
        found += 1

        if destination_count < 0:
            pass  # early exit is disabled
        elif destination_count > 0 and destinations[tail_vert_idx]:
            destinations[tail_vert_idx] = False
            destination_count = destination_count - 1

        # If we've just found the last destination, we can exit here. No need to explore any more edges
        if destination_count == 0:
            # If we wish to reuse the tree we've constructed in update_path_trace we need to mark the un-scanned
            # nodes as unreachable. The nodes not in the heap (NOT_IN_HEAP) are already -1
            for idx in range(pqueue.length):
                if pqueue.Elements[idx].state == IN_HEAP:
                    pred[idx] = -1
                    connectors[idx] = -1
            break

        tail_vert_val = pqueue.Elements[tail_vert_idx].key

        # loop on outgoing edges
        for idx in range(<size_t> graph_fs[tail_vert_idx], <size_t> graph_fs[tail_vert_idx + 1]):
            head_vert_idx = <size_t> csr_indices[idx]
            vert_state = pqueue.Elements[head_vert_idx].state
            if vert_state != SCANNED:
                head_vert_val = tail_vert_val + graph_costs[idx]
                if head_vert_val == INFINITY:
                    continue
                elif vert_state == NOT_IN_HEAP:
                    insert(&pqueue, head_vert_idx, head_vert_val)
                    pred[head_vert_idx] = tail_vert_idx
                    connectors[head_vert_idx] = ids[idx]
                elif pqueue.Elements[head_vert_idx].key > head_vert_val:
                    decrease_key(&pqueue, head_vert_idx, head_vert_val)
                    pred[head_vert_idx] = tail_vert_idx
                    connectors[head_vert_idx] = ids[idx]

    free_heap(&pqueue)
    return found - 1

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
        vector[ITYPE_t] visited
        size_t origin_vert = <size_t> origin

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
        for idx in range(<size_t> graph_fs[tail_vert_idx], <size_t> graph_fs[tail_vert_idx + 1]):
            head_vert_idx = <size_t> csr_indices[idx]
            if pred[head_vert_idx] < 0:
                pred[head_vert_idx] = tail_vert_idx
                visited.push_back(head_vert_idx)

    visited.clear()


cdef int _HAVERSINE = 0
cdef int _EQUIRECTANGULAR = 1

HEURISTIC_MAP = {"haversine": _HAVERSINE, "equirectangular": _EQUIRECTANGULAR}

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cdef inline double haversine_heuristic(double lat1, double lon1, double lat2, double lon2, void * data) noexcept nogil:
    """
    A haversine heuristic written to minimise expensive (trig) operations.

    Arguments:
        **lat1** (:obj:`double`): Latitude of destination
        **lon1** (:obj:`double`): Longitude of destination
        **lat2** (:obj:`double`): Latitude of node to evaluate
        **lon2** (:obj:`double`): Longitude of node to evaluate
        **data** (:obj:`void*`): This void pointer should hold a precomputed cos(lat1) as a double

    Returns the distance between (lat1, lon1) and (lat2, lon2).
    """
    cdef:
        double cos_lat1 = (<double *> data)[0]  # Cython doesn't support c-style derefs, use array notation instead
        double dlat = lat2 - lat1
        double dlon = lon2 - lon1
        double sin_dlat = sin(dlat / 2)
        double sin_dlon = sin(dlon / 2)
        double a = sin_dlat * sin_dlat + cos_lat1 * cos(lat2) * sin_dlon * sin_dlon
    return 2.0 * 6371000.0 * asin(sqrt(a))

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cdef inline double equirectangular_heuristic(
        double lat1,
        double lon1,
        double lat2,
        double lon2,
        void * _data
) noexcept nogil:
    """
    A Equirectangular approximation heuristic, useful for small distances.
    Not admissible for large distances. A* may not return the optimal path with this heuristic.

    Arguments:
        **lat1** (:obj:`double`): Latitude of destination
        **lon1** (:obj:`double`): Longitude of destination
        **lat2** (:obj:`double`): Latitude of node to evaluate
        **lon2** (:obj:`double`): Longitude of node to evaluate
        **data** (:obj:`void*`): Unused void pointer, for compatibility with other heuristics

    Returns the distance between (lat1, lon1) and (lat2, lon2).

    Reference: https://www.movable-type.co.uk/scripts/latlong.html
    """
    cdef:
        double x = (lon2 - lon1) * cos((lat1 + lat2) / 2.0)
        double y = (lat2 - lat1)
    return 6371000.0 * sqrt(x * x + y * y)

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cpdef void path_finding_a_star(long origin,
                               long destination,
                               double[:] graph_costs,
                               long long [:] csr_indices,
                               long long [:] graph_fs,
                               long long [:] nodes_to_indices,
                               const double [:] lats,
                               const double [:] lons,
                               long long [:] pred,
                               const long long [:] ids,
                               long long [:] connectors,
                               int heuristic) noexcept nogil:
    """
    Based on the pseudocode presented at https://en.wikipedia.org/wiki/A*_search_algorithm#Pseudocode
    The following variables have been renamed to be consistent with out Dijkstra's implementation
        - openSet: pqueue
        - cameFrom: pred
        - fScore: pqueue.Elements[idx].key, for some idx
    """

    cdef unsigned int M = pred.shape[0]

    cdef:
        size_t current, neighbour, idx  # indices
        DTYPE_t tentative_gScore  # vertex travel times
        PriorityQueue pqueue  # binary heap
        size_t origin_vert = <size_t> origin
        size_t destination_vert = <size_t> destination if destination != -1 else 0
        double *gScore = <double *> malloc(M * sizeof(double))

    cdef:
        double deg2rad = pi / 180.0
        double lat1_rad = lats[destination_vert] * deg2rad
        double lon1_rad = lons[destination_vert] * deg2rad
        double h, cos_lat1 = cos(lat1_rad)
        double (*heur)(double, double, double, double, void *) noexcept nogil
        void * data

    if heuristic == _HAVERSINE:
        heur = haversine_heuristic
        data = <void *> &cos_lat1
    else:  # heuristic == _EQUIRECTANGULAR:
        heur = equirectangular_heuristic
        data = <void *> 0

    for i in range(M):
        pred[i] = -1
        connectors[i] = -1
        gScore[i] = INFINITY

    # initialization of the heap elements
    # all nodes have INFINITY key and NOT_IN_HEAP state
    init_heap(&pqueue, <size_t> M)

    # the key is set to zero for the origin vertex,
    # which is inserted into the heap
    insert(&pqueue, origin_vert, 0.0)
    gScore[origin_vert] = 0.0

    # main loop
    while pqueue.size > 0:
        current = extract_min(&pqueue)

        if current == destination_vert:
            break

        # loop on outgoing edges
        for idx in range(<size_t> graph_fs[current], <size_t> graph_fs[current + 1]):
            neighbour = <size_t> csr_indices[idx]

            tentative_gScore = gScore[current] + graph_costs[idx]
            if tentative_gScore < gScore[neighbour]:
                pred[neighbour] = current
                connectors[neighbour] = ids[idx]
                gScore[neighbour] = tentative_gScore

                h = heur(lat1_rad, lon1_rad, lats[neighbour] * deg2rad, lons[neighbour] * deg2rad, data)

                # Unlike Dijkstra's we can remove a node from the heap and rediscover it with a cheaper path
                if pqueue.Elements[neighbour].state != IN_HEAP:
                    insert(&pqueue, neighbour, tentative_gScore + h)
                else:
                    decrease_key(&pqueue, neighbour, tentative_gScore + h)

    free_heap(&pqueue)
    free(gScore)
