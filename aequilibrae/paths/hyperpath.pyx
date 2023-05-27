""" 
An implementation of Spiess and Florian's hyperpath generating algorithm.

reference: Spiess, H. and Florian, M. (1989). Optimal strategies: A new 
assignment model for transit networks. Transportation Research Part B 23(2), 
83-102.
"""

cimport cython
import numpy as np
cimport numpy as cnp


ctypedef cnp.float64_t DATATYPE_t
DATATYPE_PY = np.float64

cdef DATATYPE_t DATATYPE_INF = <DATATYPE_t>np.finfo(dtype=DATATYPE_PY).max
DATATYPE_INF_PY = DATATYPE_INF

# infinite frequency is defined here numerically
# this must be a very large number depending on the precision on the computation
# INF_FREQ << DATATYPE_INF
cdef DATATYPE_t INF_FREQ = 1.0e+20
INF_FREQ_PY = INF_FREQ

# smallest frequency
# WARNING: this must be small but not too small 
# 1 / MIN_FREQ << DATATYPE_INF
cdef DATATYPE_t MIN_FREQ 
MIN_FREQ = 1.0 / INF_FREQ
MIN_FREQ_PY =  MIN_FREQ

# a very small time interval
cdef DATATYPE_t A_VERY_SMALL_TIME_INTERVAL
A_VERY_SMALL_TIME_INTERVAL = 1.0e+08 * MIN_FREQ
A_VERY_SMALL_TIME_INTERVAL_PY = A_VERY_SMALL_TIME_INTERVAL

include 'pq_4ary_heap.pyx'  # priority queue


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void _coo_tocsc_uint32(
    cnp.uint32_t [::1] Ai,
    cnp.uint32_t [::1] Aj,
    cnp.uint32_t [::1] Ax,   
    cnp.uint32_t [::1] Bp,
    cnp.uint32_t [::1] Bi,
    cnp.uint32_t [::1] Bx) nogil:

    cdef:
        size_t i, col, dest
        size_t n_vert = <size_t>(Bp.shape[0] - 1)
        size_t n_edge = <size_t>Bi.shape[0]
        cnp.uint32_t temp, cumsum, last

    for i in range(n_edge):
        Bp[<size_t>Aj[i]] += 1

    cumsum = 0
    for i in range(n_vert):
        temp = Bp[i]
        Bp[i] = cumsum
        cumsum += temp
    Bp[<size_t>n_vert] = <cnp.uint32_t>n_edge 

    for i in range(n_edge):
        col  = <size_t>Aj[i]
        dest = <size_t>Bp[col]
        Bi[dest] = Ai[i]
        Bx[dest] = Ax[i]
        Bp[col] += 1

    last = 0
    for i in range(n_vert + 1):
        temp = Bp[i]
        Bp[i] = last
        last = temp


cpdef convert_graph_to_csc_uint32(edges, tail, head, data, vertex_count):
    """
    Convert an edge dataframe in COO format into CSC format.

    The data vector is of uint32 type.

    Parameters
    ----------
    edges : pandas.core.frame.DataFrame
        The edges dataframe.
    tail : str
        The column name in the edges dataframe for the tail vertex index.
    head : str
        The column name in the edges dataframe for the head vertex index.
    data : str
        The column name in the edges dataframe for the int edge attribute.
    vertex_count : int
        The vertex count in the given network edges.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """

    rs_indptr = np.zeros(
        vertex_count + 1, dtype=np.uint32
    )  # make sure it is filled with zeros
    edge_count = len(edges)
    rs_indices = np.empty(edge_count, dtype=np.uint32)
    rs_data = np.empty(edge_count, dtype=np.uint32)

    _coo_tocsc_uint32(
        edges[tail].values.astype(np.uint32),
        edges[head].values.astype(np.uint32),
        edges[data].values.astype(np.uint32),
        rs_indptr,
        rs_indices,
        rs_data,
    )

    return rs_indptr, rs_indices, rs_data



cpdef void compute_SF_in(
    cnp.uint32_t[::1] csc_indptr,
    cnp.uint32_t[::1] csc_edge_idx,
    DATATYPE_t[::1] c_a_vec,
    DATATYPE_t[::1] f_a_vec,
    cnp.uint32_t[::1] tail_indices,
    cnp.uint32_t[::1] head_indices,
    cnp.uint32_t[::1] demand_indices,
    DATATYPE_t[::1] demand_values,
    DATATYPE_t[::1] v_a_vec,
    DATATYPE_t[::1] u_i_vec,
    DATATYPE_t[::1] f_i_vec,
    DATATYPE_t[::1] u_j_c_a_vec,
    DATATYPE_t[::1] v_i_vec,
    int vertex_count,
    int dest_vert_index,
):

    cdef:
        int edge_count = tail_indices.shape[0]
        DATATYPE_t u_r, v_a_new, v_i, u_i
        size_t i, h_a_count
        cnp.uint32_t vert_idx 

    # initialization
    for i in range(<size_t>vertex_count):
        u_i_vec[i] = DATATYPE_INF_PY
        f_i_vec[i] = 0.0
        u_j_c_a_vec[i] = DATATYPE_INF_PY
        v_i_vec[i] = 0.0
    u_i_vec[<size_t>dest_vert_index] = 0.0

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
    # v_i_vec = np.zeros(vertex_count, dtype=DATATYPE_PY)  # vertex volume

    u_r = DATATYPE_INF_PY
    for i, vert_idx in enumerate(demand_indices):

        v_i_vec[<size_t>vert_idx] = demand_values[i]
        u_i = u_i_vec[<size_t>vert_idx]

        if u_i < u_r:
            u_r = u_i

    # if the destination can be reached from any of the origins
    if u_r < DATATYPE_INF_PY:

        # make sure f_i values are not zero
        for i in range(<size_t>vertex_count):
            if f_i_vec[i] < MIN_FREQ_PY:
                f_i_vec[i] = MIN_FREQ_PY

        h_a_count = 0
        for i in range(<size_t>edge_count):
            u_j_c_a_vec[i] *= -1.0
            h_a_count += <size_t>h_a_vec[i]

        # sort the links with descreasing order of u_j + c_a
        masked_a = np.ma.array(u_j_c_a_vec, mask=~h_a_vec)
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void _SF_in_first_pass_full(
    cnp.uint32_t[::1] csc_indptr, 
    cnp.uint32_t[::1] csc_edge_idx,
    DATATYPE_t[::1] c_a_vec,
    DATATYPE_t[::1] f_a_vec,
    cnp.uint32_t[::1] tail_indices,
    DATATYPE_t[::1] u_i_vec,
    DATATYPE_t[::1] f_i_vec,
    DATATYPE_t[::1] u_j_c_a_vec,
    cnp.uint8_t[::1] h_a_vec,
    int dest_vert_index,
) nogil:
    """All vertices are visited."""

    cdef:
        int edge_count = tail_indices.shape[0]
        PriorityQueue pqueue
        ElementState edge_state
        size_t i, edge_idx, tail_vert_idx
        DATATYPE_t u_j_c_a, u_i, f_i, beta, u_i_new, f_a

    # initialization of the heap elements 
    # all nodes have INFINITY key and NOT_IN_HEAP state
    init_heap(&pqueue, <size_t>edge_count)

    # only the incoming edges of the target vertex are inserted into the 
    # priority queue
    for i in range(<size_t>csc_indptr[<size_t>dest_vert_index], 
        <size_t>csc_indptr[<size_t>(dest_vert_index + 1)]):
        edge_idx = csc_edge_idx[i]
        insert(&pqueue, edge_idx, c_a_vec[edge_idx])
        u_j_c_a_vec[edge_idx] = c_a_vec[edge_idx]

    # first pass
    while pqueue.size > 0:

        edge_idx = extract_min(&pqueue)
        u_j_c_a = pqueue.Elements[edge_idx].key
        tail_vert_idx = <size_t>tail_indices[edge_idx]
        u_i = u_i_vec[tail_vert_idx]

        if u_i >= u_j_c_a:

            f_i = f_i_vec[tail_vert_idx]

            # compute the beta coefficient
            if (u_i < DATATYPE_INF) | (f_i > 0.0):

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

                if edge_state == NOT_IN_HEAP:

                    insert(&pqueue, edge_idx, u_j_c_a)
                    u_j_c_a_vec[edge_idx] = u_j_c_a 

                elif (pqueue.Elements[edge_idx].key > u_j_c_a):

                    decrease_key(&pqueue, edge_idx, u_j_c_a)
                    u_j_c_a_vec[edge_idx] = u_j_c_a

    free_heap(&pqueue)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void _SF_in_second_pass(
    cnp.uint32_t[::1] edge_indices,
    cnp.uint32_t[::1] tail_indices,
    cnp.uint32_t[::1] head_indices,
    DATATYPE_t[::1] v_i_vec,
    DATATYPE_t[::1] v_a_vec,
    DATATYPE_t[::1] f_i_vec,
    DATATYPE_t[::1] f_a_vec,
    size_t h_a_count
) nogil:

    cdef:
        size_t i, edge_idx, vert_idx
        DATATYPE_t v_i, f_i, f_a, v_a_new

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