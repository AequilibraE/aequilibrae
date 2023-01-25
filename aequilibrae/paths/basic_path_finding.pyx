"""
Original Algorithm for Shortest path (Dijkstra with a 4-ary heap) was written by François Pacull <francois.pacull@architecture-performance.fr> under license: MIT, (C) 2022
"""

"""
TODO:
LIST OF ALL THE THINGS WE NEED TO DO TO NOT HAVE TO HAVE nodes 1..n as CENTROIDS. ARBITRARY NUMBERING
- Checks of weather the centroid we are computing path from is a centroid and/or exists in the graph
- Re-write function **network_loading** on the part of loading flows to centroids
"""
cimport cython
from libc.math cimport isnan, INFINITY
from libc.stdio cimport printf
from libc.string cimport memset
from libc.stdlib cimport malloc, free

include 'pq_4ary_heap.pyx'

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False) # turn of bounds-checking for entire function
cpdef void network_loading(long classes,
                           double[:, :] demand,
                           long long [:] pred,
                           long long [:] conn,
                           double[:, :] link_loads,
                           long long [:] no_path,
                           long long [:] reached_first,
                           double [:, :] node_load,
                           long found) nogil:

    cdef long long i, j, predecessor, connector, node
    cdef long long zones = demand.shape[0]
    cdef long long N = node_load.shape[0]

# Traditional loading, without cascading
#    for i in range(zones):
#        node = i
#        predecessor = pred[node]
#        connector = conn[node]
#        while predecessor >= 0:
#            for j in range(classes):
#                link_loads[connector, j] += demand[i, j]
#
#            predecessor = pred[predecessor]
#            connector = conn[predecessor]

    # Clean the node load array
    for i in range(N):
        node_load[i] = 0

    # Loads the demand to the centroids
    for j in range(classes):
        for i in range(zones):
            if not isnan(demand[i, j]):
                node_load[i, j] = demand[i, j]

    #Recursively cascades to the origin
    for i in range(found, 0, -1):
        node = reached_first[i]

        # captures how we got to that node
        predecessor = pred[node]
        connector = conn[node]

        # loads the flow to the links for each class
        for j in range(classes):
            link_loads[connector, j] += node_load[node, j]
            # Cascades the load from the node to their predecessor
            node_load[predecessor, j] += node_load[node, j]

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cdef void _copy_skims(double[:,:] skim_matrix,  #Skim matrix_procedures computed from one origin to all nodes
                      double[:,:] final_skim_matrix) nogil:  #Skim matrix_procedures computed for one origin to all other centroids only

    cdef long i, j
    cdef long N = final_skim_matrix.shape[0]
    cdef long skims = final_skim_matrix.shape[1]

    for i in range(N):
        for j in range(skims):
            final_skim_matrix[i,j]=skim_matrix[i,j]


cdef return_an_int_view(input):
    cdef int [:] critical_links_view = input
    return critical_links_view


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
cdef void perform_select_link_analysis(long origin,
                                        long long [:] selected_links,
                                        double [:, :] demand,
                                        long long [:] pred,
                                        long long [:] conn,
                                        double [:, :] sl_od_matrix,
                                        double [:, :] sl_link_loading,
                                       int [:] tmp_flow,
                                       long classes) nogil:
# origin: Origin of the path
# demand: Demand matrix of size Destinations x classes. Stores the loading on the given OD pair for the given class
# pred: Stores the predecessor to a node at a given index e.g. in path 2, 3, 4 indexing into pred[4] would return 3.
# conn: Stores the link connecting the given index (predecessor) to its next node
#e.g: in sioux falls, for origin 0, the node 1's predecessor is 0. Referencing conn[1] will return the link 0 (link 1)
# conn gives the link that connects the current predecessor to its predecessor
# sl_od_loading: Destination x classes size matrix. Stores 0 if the selected links aren't used in the path, demand otherwise
# sl_link_loading: Stores the demand loading on each individual link across all links in the project and all paths
#Dimension: num_links by 1
# NOTE: Each call of this function will only do the sl_link_loading for the given origin to all destinations.
# selected links: An array of link_id's (from the compressed graph) which are being examined in the SL pipeline
#tmp flow: array of zeros as long as the number of links in compressed graph
    cdef:
        int i, j, k, idx, dests = demand.shape[0], xshape = tmp_flow.shape[0]
        long long predecessor, connection, lid, link
        double tot
    for j in range(dests):
        tot = 0
        k = 0
        while tot == 0 and k < classes:
            tot += demand[j, k]
            k += 1
        if tot == 0:
            continue
        # for k in range(classes):
        #     tot += demand[j, k]
        # if tot == 0:
        #     continue
        memset(&tmp_flow[0], 0, xshape * sizeof(int))
        # for i in range(conn.shape[0]):
        #     #TODO: check if memset is faster than rewalking path
        #     if conn[i] != -1:
        #         for k in range(classes):
        #             tmp_flow[conn[i], k] = 0
        predecessor = j
        connection = conn[predecessor]
        predecessor = pred[predecessor]
        while predecessor >= 0:
            tmp_flow[connection] = 1
            connection = conn[predecessor]
            predecessor = pred[predecessor]
        # for k in range(classes):
        for i in range(selected_links.shape[0]):
            lid = selected_links[i]
            #TODO: CONFIRM BEHAVIOUR OF CLASSES, swap the classes
            if tmp_flow[lid] != 0:
                for k in range(classes):
                    # sl_od_loading[j, :] = demand[j, :]
                    sl_od_matrix[j, k] = demand[j, k]
                predecessor = j
                connection = conn[predecessor]
                predecessor = pred[predecessor]
                while predecessor >= 0:
                    for k in range(classes):
                        sl_link_loading[connection, k] += demand[j, k]
                    connection = conn[predecessor]
                    predecessor = pred[predecessor]
                break
                #once at least one link in the set is shown to be in the current destination, the load along that path is added.
                #There is no need to check the remaining links - this would add extra demand that isn't there

                # for idx in range(conn.shape[0]):
                #     if conn[idx] != -1:
                #         link = conn[idx]
                #         sl_link_loading[link, :] += demand[j, :]


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
                                 unsigned int [:] nodes_to_write) nogil:
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
                                  long long [:] real_b_nodes) nogil:
    cdef long long i

    if action == 1: # We are unblocking
        for i in range(fs[centroids]):
            temp_b_nodes[i] = real_b_nodes[i]
    else: # We are blocking:
        for i in range(fs[centroids]):
            temp_b_nodes[i] = orig

        for i in range(fs[orig], fs[orig + 1]):
            temp_b_nodes[i] = real_b_nodes[i]


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False) # turn of bounds-checking for entire function
cdef void skim_single_path(long origin,
                           long nodes,
                           long skims,
                           double[:, :] node_skims,
                           long long [:] pred,
                           long long [:] conn,
                           double[:, :] graph_costs,
                           long long [:] reached_first,
                           long found) nogil:
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
@cython.boundscheck(False) # turn of bounds-checking for entire function
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
                                double [:,:] final_skims) nogil:
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

# ###########################################################################################################################
#############################################################################################################################
# Original Dijkstra implementation by François Pacull, taken from https://github.com/Edsger-dev/priority_queues
# Old Numpy Buffers were replaces with latest memory views interface to allow for the release of the GIL
# Path tracking arrays and skim arrays were also added to it
#############################################################################################################################
# ###########################################################################################################################

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False) # turn of bounds-checking for entire function
cpdef int path_finding(long origin,
                       double[:] graph_costs,
                       long long [:] csr_indices,
                       long long [:] graph_fs,
                       long long [:] pred,
                       long long [:] ids,
                       long long [:] connectors,
                       long long [:] reached_first) nogil:

    cdef unsigned int N = graph_costs.shape[0]
    cdef unsigned int M = pred.shape[0]


    cdef:
        size_t tail_vert_idx, head_vert_idx, idx  # indices
        DTYPE_t tail_vert_val, head_vert_val  # vertex travel times
        PriorityQueue pqueue  # binary heap
        ElementState vert_state  # vertex state
        size_t origin_vert = <size_t>origin
        ITYPE_t found = 0

    for i in range(M):
        pred[i] = -1
        connectors[i] = -1
        reached_first[i] = -1

    # initialization of the heap elements
    # all nodes have INFINITY key and NOT_IN_HEAP state
    init_heap(&pqueue, <size_t>N)

    # the key is set to zero for the origin vertex,
    # which is inserted into the heap
    insert(&pqueue, origin_vert, 0.0)

    # main loop
    while pqueue.size > 0:
        tail_vert_idx = extract_min(&pqueue)
        tail_vert_val = pqueue.Elements[tail_vert_idx].key
        reached_first[found] = tail_vert_idx
        found += 1

        # loop on outgoing edges
        for idx in range(<size_t>graph_fs[tail_vert_idx], <size_t>graph_fs[tail_vert_idx + 1]):
            head_vert_idx = <size_t>csr_indices[idx]
            vert_state = pqueue.Elements[head_vert_idx].state
            if vert_state != SCANNED:
                head_vert_val = tail_vert_val + graph_costs[idx]
                if vert_state == NOT_IN_HEAP:
                    insert(&pqueue, head_vert_idx, head_vert_val)
                    pred[head_vert_idx] = tail_vert_idx
                    connectors[head_vert_idx] = ids[idx]
                elif pqueue.Elements[head_vert_idx].key > head_vert_val:
                    decrease_key(&pqueue, head_vert_idx, head_vert_val)
                    pred[head_vert_idx] = tail_vert_idx
                    connectors[head_vert_idx] = ids[idx]

    free_heap(&pqueue)
    return found - 1
