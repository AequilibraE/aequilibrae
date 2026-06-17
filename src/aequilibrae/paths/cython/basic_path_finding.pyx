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
from libc.string cimport memset
from libc.stdlib cimport malloc, free
from libc.stddef cimport size_t
from libc.stdint cimport int64_t
from libcpp.vector cimport vector

from aequilibrae.paths.cython.pq_heap_types cimport FourAryHeap
from aequilibrae.paths.cython.path_finding cimport (
    dijkstra,
    a_star,
    HeuristicFn,
    haversine_heuristic,
    equirectangular_heuristic,
)


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
cdef void _copy_skims(
    double[:, :] skim_matrix,  # Skim matrix_procedures computed from one origin to all nodes
    double[:, :] final_skim_matrix
) noexcept nogil:  # Skim matrix_procedures computed for one origin to all other centroids only

    cdef long i, j
    cdef long N = final_skim_matrix.shape[0]
    cdef long skims = final_skim_matrix.shape[1]

    for i in range(N):
        for j in range(skims):
            final_skim_matrix[i, j]=skim_matrix[i, j]


cdef int[:] return_an_int_view(input) noexcept nogil:
    cdef int [:] critical_links_view = input
    return critical_links_view


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


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)  # turn of bounds-checking for entire function
cdef void skim_single_path(long origin,
                           long nodes,
                           long skims,
                           double[:, :] node_skims,
                           long long [:] pred,
                           long long [:] conn,
                           double[:, :] graph_costs,
                           long long [:] reached_first,
                           long found) noexcept nogil:
    cdef long long i, node, predecessor, connector, j

    # sets all skims to infinity
    for i in range(nodes):
        for j in range(skims):
            node_skims[i, j] = INFINITY

    # Zeroes the intrazonal cost
    for j in range(skims):
        node_skims[origin, j] = 0

    # Cascade skimming
    for i in range(1, found + 1):
        node = reached_first[i]

        # captures how we got to that node
        predecessor = pred[node]
        connector = conn[node]

        for j in range(skims):
            node_skims[node, j] = node_skims[predecessor, j] + graph_costs[connector, j]


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)  # turn of bounds-checking for entire function
cpdef void skim_multiple_fields(long origin,
                                long nodes,
                                long zones,
                                long skims,
                                double[:, :] node_skims,
                                long long [:] pred,
                                long long [:] conn,
                                double[:, :] graph_costs,
                                long long [:] reached_first,
                                long found,
                                double [:, :] final_skims) noexcept nogil:
    cdef long long i, node, predecessor, connector, j

    # sets all skims to infinity
    for i in range(nodes):
        for j in range(skims):
            node_skims[i, j] = INFINITY

    # Zeroes the intrazonal cost
    for j in range(skims):
        node_skims[origin, j] = 0

    # Cascade skimming
    for i in range(1, found + 1):
        node = reached_first[i]

        # captures how we got to that node
        predecessor = pred[node]
        connector = conn[node]

        for j in range(skims):
            node_skims[node, j] = node_skims[predecessor, j] + graph_costs[connector, j]

    for i in range(zones):
        for j in range(skims):
            final_skims[i, j] = node_skims[i, j]


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
    return <int>dijkstra[FourAryHeap](
        <size_t>origin,
        <size_t>pred.shape[0],
        &graph_costs[0],
        <const size_t*>&csr_indices[0],
        <const size_t*>&graph_fs[0],
        <size_t*>&pred[0],
        <const size_t*>&ids[0],
        <size_t*>&connectors[0],
        <size_t*>&reached_first[0],
        &destinations[0],
        destination_count,
        NULL)


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


cdef enum Heuristic:
    HAVERSINE
    EQUIRECTANGULAR

HEURISTIC_MAP = {"haversine": HAVERSINE, "equirectangular": EQUIRECTANGULAR}


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
                               Heuristic heuristic) noexcept nogil:
    cdef HeuristicFn heur_fn
    cdef void* heur_data
    cdef double cos_lat1_local

    if heuristic == HAVERSINE:
        heur_fn = haversine_heuristic
        cos_lat1_local = cos(lats[<size_t>destination if destination != -1 else 0] * pi / 180.0)
        heur_data = <void*>&cos_lat1_local
    else:
        heur_fn = equirectangular_heuristic
        heur_data = NULL

    a_star[FourAryHeap](
        <size_t>origin,
        <size_t>destination,
        <size_t>pred.shape[0],
        &graph_costs[0],
        <const size_t*>&csr_indices[0],
        <const size_t*>&graph_fs[0],
        <const size_t*>&nodes_to_indices[0],
        &lats[0],
        &lons[0],
        <size_t*>&pred[0],
        <const size_t*>&ids[0],
        <size_t*>&connectors[0],
        heur_fn,
        heur_data,
        NULL)
