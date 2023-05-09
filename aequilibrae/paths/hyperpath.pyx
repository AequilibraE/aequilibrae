""" 
An implementation of Spiess and Florian's hyperpath generating algorithm.

reference: Spiess, H. and Florian, M. (1989). Optimal strategies: A new 
assignment model for transit networks. Transportation Research Part B 23(2), 
83-102.
"""

import numpy as np
cimport numpy as cnp

# from edsger.commons import DTYPE_PY, DTYPE_INF_PY
# from edsger.commons cimport (
#     DTYPE_INF, MIN_FREQ, UNLABELED, SCANNED, DTYPE_t, ElementState)
# cimport edsger.pq_4ary_dec_0b as pq  # priority queue



cpdef void compute_SF_in(
    cnp.uint32_t[::1] csc_indptr,
    cnp.uint32_t[::1] csc_edge_idx,
    DTYPE_t[::1] c_a_vec,
    DTYPE_t[::1] f_a_vec,
    cnp.uint32_t[::1] tail_indices,
    cnp.uint32_t[::1] head_indices,
    cnp.uint32_t[::1] demand_indices,
    DTYPE_t[::1] demand_values,
    DTYPE_t[::1] v_a_vec,
    DTYPE_t[::1] u_i_vec,
    int vertex_count,
    int dest_vert_index,
):

    cdef:
        int edge_count = tail_indices.shape[0]

    # initialization
    u_i_vec[<size_t>dest_vert_index] = 0.0

    # vertex properties
    f_i_vec = np.zeros(vertex_count, dtype=DTYPE_PY)  # vertex frequency (inverse of the maximum delay)
    u_j_c_a_vec = DTYPE_INF_PY * np.ones(edge_count, dtype=DTYPE_PY)    
    v_i_vec = np.zeros(vertex_count, dtype=DTYPE_PY)  # vertex volume
    
    # edge properties
    h_a_vec = np.zeros(edge_count, dtype=bool)  # edge belonging to hyperpath

    # first pass #
    # ---------- #

    _SF_in_first_pass_full(
        csc_indptr,
        csc_edge_idx,
        c_a_vec,
        f_a_vec,
        tail_indices,
        u_i_vec,
        f_i_vec,
        u_j_c_a_vec,
        h_a_vec,
        dest_vert_index
    )

    # second pass #
    # ----------- #

    cdef:
        DTYPE_t u_r, v_a_new, v_i, u_i
        size_t i, h_a_count
        cnp.uint32_t vert_idx 

    v_i_vec = np.zeros(vertex_count, dtype=DTYPE_PY)  # vertex volume

    u_r = DTYPE_INF_PY
    for i, vert_idx in enumerate(demand_indices):

        v_i_vec[<size_t>vert_idx] = demand_values[i]
        u_i = u_i_vec[<size_t>vert_idx]

        if u_i < u_r:

            u_r = u_i

    # if the destination can be reached from any of the origins
    if u_r < DTYPE_INF_PY:

        # make sure f_i values are not zero
        f_i_vec = np.where(
            f_i_vec < MIN_FREQ_PY, MIN_FREQ_PY, f_i_vec
        )

        # sort the links with descreasing order of u_j + c_a
        h_a_count = h_a_vec.sum()
        masked_a = np.ma.array(-u_j_c_a_vec, mask=~h_a_vec)
        edge_indices = np.argsort(masked_a).astype(np.uint32)

        _SF_in_second_pass(
            edge_indices,
            tail_indices,
            head_indices,
            v_i_vec,
            v_a_vec,
            f_i_vec,
            f_a_vec,
            h_a_count
        )


cdef void _SF_in_first_pass_full(
    cnp.uint32_t[::1] csc_indptr, 
    cnp.uint32_t[::1] csc_edge_idx,
    DTYPE_t[::1] c_a_vec,
    DTYPE_t[::1] f_a_vec,
    cnp.uint32_t[::1] tail_indices,
    DTYPE_t[::1] u_i_vec,
    DTYPE_t[::1] f_i_vec,
    DTYPE_t[::1] u_j_c_a_vec,
    cnp.uint8_t[::1] h_a_vec,
    int dest_vert_index,
) nogil:
    """All vertices are visited."""

    cdef:
        int edge_count = tail_indices.shape[0]
        pq.PriorityQueue pqueue
        ElementState edge_state
        size_t i, edge_idx, tail_vert_idx
        DTYPE_t u_j_c_a, u_i, f_i, beta, u_i_new, f_a

    # initialization of the heap elements 
    # all nodes have INFINITY key and UNLABELED state
    pq.init_pqueue(&pqueue, <size_t>edge_count, <size_t>edge_count)

    # only the incoming edges of the target vertex are inserted into the 
    # priority queue
    for i in range(<size_t>csc_indptr[<size_t>dest_vert_index], 
        <size_t>csc_indptr[<size_t>(dest_vert_index + 1)]):
        edge_idx = csc_edge_idx[i]
        pq.insert(&pqueue, edge_idx, c_a_vec[edge_idx])
        u_j_c_a_vec[edge_idx] = c_a_vec[edge_idx]

    # first pass
    while pqueue.size > 0:

        edge_idx = pq.extract_min(&pqueue)
        u_j_c_a = pqueue.Elements[edge_idx].key
        tail_vert_idx = <size_t>tail_indices[edge_idx]
        u_i = u_i_vec[tail_vert_idx]

        if u_i >= u_j_c_a:

            f_i = f_i_vec[tail_vert_idx]

            # compute the beta coefficient
            if (u_i < DTYPE_INF) | (f_i > 0.0):

                beta = f_i * u_i

            else:

                beta = 1.0

            # update u_i
            f_a = f_a_vec[edge_idx]
            u_i_new = (beta + f_a * u_j_c_a) / (f_i + f_a)
            u_i_vec[tail_vert_idx] = u_i_new

            # update f_i
            f_i_vec[tail_vert_idx] = f_i + f_a

            # add the edge to hyperpath
            h_a_vec[edge_idx] = 1

        else:

            u_i_new = u_i

        # loop on incoming edges
        for i in range(<size_t>csc_indptr[tail_vert_idx], 
            <size_t>csc_indptr[tail_vert_idx + 1]):

            edge_idx = csc_edge_idx[i]
            edge_state = pqueue.Elements[edge_idx].state

            if edge_state != SCANNED:

                # u_j of current edge = u_i of outgoing edge
                u_j_c_a = u_i_new + c_a_vec[edge_idx]

                if edge_state == UNLABELED:

                    pq.insert(&pqueue, edge_idx, u_j_c_a)
                    u_j_c_a_vec[edge_idx] = u_j_c_a 

                elif (pqueue.Elements[edge_idx].key > u_j_c_a):

                    pq.decrease_key(&pqueue, edge_idx, u_j_c_a)
                    u_j_c_a_vec[edge_idx] = u_j_c_a

    pq.free_pqueue(&pqueue)


cdef void _SF_in_second_pass(
    cnp.uint32_t[::1] edge_indices,
    cnp.uint32_t[::1] tail_indices,
    cnp.uint32_t[::1] head_indices,
    DTYPE_t[::1] v_i_vec,
    DTYPE_t[::1] v_a_vec,
    DTYPE_t[::1] f_i_vec,
    DTYPE_t[::1] f_a_vec,
    size_t h_a_count
) nogil:

    cdef:
        size_t i, edge_idx, vert_idx
        DTYPE_t v_i, f_i, f_a, v_a_new

    for i in range(h_a_count):

        edge_idx = <size_t>edge_indices[i]
        vert_idx = <size_t>tail_indices[edge_idx]

        v_i = v_i_vec[vert_idx]
        f_i = f_i_vec[vert_idx]
        f_a = f_a_vec[edge_idx]

        # update v_a
        v_a_new = v_i * f_a / f_i
        v_a_vec[edge_idx] = v_a_new
        v_i_vec[<size_t>head_indices[edge_idx]] += v_a_new