"""
An implementation of Spiess and Florian's hyperpath generating algorithm.

reference: Spiess, H. and Florian, M. (1989). Optimal strategies: A new
assignment model for transit networks. Transportation Research Part B 23(2),
83-102.
"""

cimport cython
import numpy as np
cimport numpy as cnp

from cython.parallel import parallel, prange, threadid
from libc.stdlib cimport malloc, calloc, free

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
MIN_FREQ_PY = MIN_FREQ

# a very small time interval
cdef DATATYPE_t A_VERY_SMALL_TIME_INTERVAL
A_VERY_SMALL_TIME_INTERVAL = 1.0e+08 * MIN_FREQ
A_VERY_SMALL_TIME_INTERVAL_PY = A_VERY_SMALL_TIME_INTERVAL

cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size, int(*compar)(const_void *, const_void *)) noexcept nogil

cdef struct IndexedElement:
    size_t index
    DATATYPE_t value

cdef int _compare(const_void *a, const_void *b) noexcept:
    cdef DATATYPE_t v = (<IndexedElement*> a).value-(<IndexedElement*> b).value
    if v < 0:
        return -1
    else:
        return 1

include 'pq_4ary_heap.pyx'  # priority queue


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(False)
@cython.initializedcheck(False)
cdef void _coo_tocsc_uint32(
    cnp.uint32_t [::1] Ai,
    cnp.uint32_t [::1] Aj,
    cnp.uint32_t [::1] Ax,
    cnp.uint32_t [::1] Bp,
    cnp.uint32_t [::1] Bi,
    cnp.uint32_t [::1] Bx
) noexcept nogil:

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(False)
@cython.initializedcheck(False)
cdef void argsort(cnp.float64_t *data, cnp.uint32_t *order, size_t n) noexcept nogil:
    """
    Wrapper of the C function qsort
    source: https://github.com/jcrudy/cython-argsort/tree/master/cyargsort
    """
    cdef size_t i

    # Allocate index tracking array.
    cdef IndexedElement *order_struct = <IndexedElement *> malloc(n * sizeof(IndexedElement))

    # Copy data into index tracking array.
    for i in range(n):
        order_struct[i].index = i
        order_struct[i].value = data[i]

    # Sort index tracking array.
    qsort(<void *> order_struct, n, sizeof(IndexedElement), _compare)

    # Copy indices from index tracking array to output array.
    for i in range(n):
        order[i] = <cnp.uint32_t>order_struct[i].index

    # Free index tracking array.
    free(order_struct)


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


# Both boundscheck(False) and initializedcheck(False) are required for this function to operate,
# without them the threads appear to enter a deadlock due to the shared edge_volume array. However
# the indexing on that array and its slices should never collide.
#
# Optionally return the u_i_vec, however this is only possible when using a single destination.
# The caller is responsible for managing that memory.
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(False)
@cython.initializedcheck(False)
cdef void compute_SF_in_parallel(
    cnp.uint32_t[::1] indptr_view,  # column csc format
    cnp.uint32_t[::1] edge_idx_view,
    cnp.float64_t[::1] trav_time_view,
    cnp.float64_t[::1] freq_view,
    cnp.uint32_t[::1] tail_view,
    cnp.uint32_t[::1] head_view,
    cnp.uint32_t[:] d_vert_ids_view,  # destination vertices
    # in public_transport.run case, same as d_vert_ids_view in .assign are the unique destination values
    cnp.uint32_t[:] destination_vertex_indices_view,
    cnp.uint32_t[::1] o_vert_ids_view,  # origin vertices
    cnp.float64_t[::1] demand_vls_view,  # volume
    cnp.float64_t[::1] edge_volume_view,
    size_t vertex_count,
    size_t edge_count,
    int num_threads,
    cnp.float64_t[:, ::1] skim_col_view,
    cnp.float64_t[::1] u_i_vec_view,
    cnp.float64_t[:, :, ::1] skim_matrix,
    cnp.uint32_t[::1] centroids,
    cnp.uint32_t[::1] centroids_idx_pos,
    bint skimming,
    bint is_travel_time,
    size_t n_skim_cols,
) noexcept nogil:
    # Thread local variables are prefixed by "thread", anything else should be considered shared and thus read only
    cdef:
        cnp.uint32_t *thread_demand_origins
        cnp.float64_t *thread_demand_values
        cnp.float64_t *thread_edge_volume
        size_t demand_size

        cnp.float64_t *thread_u_i_vec
        cnp.float64_t *thread_f_i_vec
        cnp.float64_t *thread_u_j_c_a_vec
        cnp.float64_t *thread_v_i_vec
        cnp.uint8_t *thread_h_a_vec
        cnp.uint32_t *thread_edge_indices

        cnp.float64_t *thread_skim_i_vec
        cnp.float64_t *thread_skim_j_vec

        # This is a shared buffer, all threads will write into separate slices depending on their threadid.
        # When writing all threads must increment!
        cnp.float64_t *edge_volume = <cnp.float64_t *> calloc(num_threads, sizeof(cnp.float64_t) * edge_count)

        cnp.float64_t *u_i_vec_out = <cnp.float64_t *> malloc(num_threads * sizeof(cnp.float64_t) * vertex_count)

        cnp.float64_t *skim_i_vec_out = <cnp.float64_t *> calloc(
            num_threads, sizeof(cnp.float64_t) * vertex_count * n_skim_cols)

        int i  # openmp on windows requires iterator variable have signed type
        size_t k, j, destination_vertex_index

    with parallel(num_threads=min(num_threads, centroids.shape[0] if skimming else d_vert_ids_view.shape[0])):
        thread_demand_origins = <cnp.uint32_t  *> malloc(sizeof(cnp.uint32_t)  * d_vert_ids_view.shape[0])
        thread_demand_values  = <cnp.float64_t *> malloc(sizeof(cnp.float64_t) * d_vert_ids_view.shape[0])
        # Here we take out thread local slice of the shared buffer, each thread is assigned a unique id so
        # we can safely read and write without collisions.
        thread_edge_volume  = &edge_volume[threadid() * edge_count]
        thread_u_i_vec = &u_i_vec_out[threadid() * vertex_count]

        thread_skim_i_vec = &skim_i_vec_out[threadid() * vertex_count * n_skim_cols]

        thread_f_i_vec      = <cnp.float64_t *> malloc(sizeof(cnp.float64_t) * vertex_count)
        thread_u_j_c_a_vec  = <cnp.float64_t *> malloc(sizeof(cnp.float64_t) * edge_count)
        thread_v_i_vec      = <cnp.float64_t *> malloc(sizeof(cnp.float64_t) * vertex_count)
        thread_h_a_vec      = <cnp.uint8_t   *> malloc(sizeof(cnp.uint8_t)   * edge_count)
        thread_edge_indices = <cnp.uint32_t  *> malloc(sizeof(cnp.uint32_t)  * edge_count)

        thread_skim_j_vec = <cnp.float64_t *> calloc(edge_count, sizeof(cnp.float64_t) * n_skim_cols)

        for i in prange(destination_vertex_indices_view.shape[0]):
            destination_vertex_index = destination_vertex_indices_view[i]

            demand_size = 0
            for j in range(d_vert_ids_view.shape[0]):
                if d_vert_ids_view[j] == destination_vertex_index:
                    thread_demand_origins[demand_size] = o_vert_ids_view[j]
                    thread_demand_values[demand_size] = demand_vls_view[j]
                    # demand_size += 1 is not allowed as cython believes this is a reduction
                    demand_size = demand_size + 1

            # S&F
            compute_SF_in(
                indptr_view,
                edge_idx_view,
                trav_time_view,
                freq_view,
                tail_view,
                head_view,
                thread_demand_origins,
                thread_demand_values,
                demand_size,
                thread_edge_volume,
                thread_u_i_vec,
                thread_f_i_vec,
                thread_u_j_c_a_vec,
                thread_v_i_vec,
                thread_h_a_vec,
                thread_edge_indices,
                vertex_count,
                destination_vertex_index,
                skim_col_view,
                thread_skim_i_vec,
                thread_skim_j_vec,
                n_skim_cols,
                skim_matrix,
                centroids,
                centroids_idx_pos,
                skimming,
                is_travel_time
            )

        free(thread_demand_origins)
        free(thread_demand_values)
        free(thread_f_i_vec)
        free(thread_u_j_c_a_vec)
        free(thread_v_i_vec)
        free(thread_h_a_vec)
        free(thread_edge_indices)
        free(thread_skim_j_vec)

    # Accumulate results into output buffer. This could parallelised over the output indexes but
    # the lose of spacial locality may not be worth it.
    for i in range(num_threads):
        for j in range(edge_count):
            edge_volume_view[j] += edge_volume[i * edge_count + j]

    if d_vert_ids_view.shape[0] == 1:
        for k in range(vertex_count):
            u_i_vec_view[k] = u_i_vec_out[k]

    free(u_i_vec_out)
    free(skim_i_vec_out)
    free(edge_volume)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(False)
@cython.initializedcheck(False)
cdef void compute_SF_in(
    cnp.uint32_t[::1] csc_indptr,
    cnp.uint32_t[::1] csc_edge_idx,
    cnp.float64_t[::1] c_a_vec,  # travel time
    cnp.float64_t[::1] f_a_vec,  # freq
    cnp.uint32_t[::1] tail_indices,
    cnp.uint32_t[::1] head_indices,
    cnp.uint32_t *demand_indices,
    cnp.float64_t *demand_values,
    size_t demand_size,
    cnp.float64_t *v_a_vec,
    cnp.float64_t *u_i_vec,
    cnp.float64_t *f_i_vec,
    cnp.float64_t *u_j_c_a_vec,
    cnp.float64_t *v_i_vec,
    cnp.uint8_t *h_a_vec,
    cnp.uint32_t *edge_indices,
    size_t vertex_count,
    int dest_vert_index,
    cnp.float64_t[:, ::1] skim_col_vec,
    cnp.float64_t *skim_i_vec,
    cnp.float64_t *skim_j_vec,
    size_t n_skim_cols,
    cnp.float64_t[:, :, ::1] skim_matrix,
    cnp.uint32_t[::1] centroids,
    cnp.uint32_t[::1] centroids_idx_pos,
    bint skimming,
    bint is_travel_time,
) noexcept nogil:

    cdef:
        size_t edge_count = <size_t>tail_indices.shape[0]
        DATATYPE_t u_r, u_i
        size_t i, j, h_a_count
        cnp.uint32_t vert_idx
        int cent_dest

    # initialization
    for i in range(vertex_count):
        u_i_vec[i] = DATATYPE_INF
        f_i_vec[i] = 0.0
        v_i_vec[i] = 0.0
    u_i_vec[<size_t>dest_vert_index] = 0.0

    for i in range(vertex_count * n_skim_cols):
        skim_i_vec[i] = 0.0

    for i in range(edge_count):
        # v_a_vec[i] = 0.0
        u_j_c_a_vec[i] = DATATYPE_INF
        h_a_vec[i] = 0

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
        dest_vert_index,
        skim_col_vec,
        skim_i_vec,
        skim_j_vec,
        n_skim_cols,
        vertex_count
    )

    # store skimming results
    cent_dest = centroids_idx_pos[dest_vert_index]

    if skimming:
        if is_travel_time:
            for i in range(centroids.shape[0]):
                skim_matrix[cent_dest][i][0] = u_i_vec[centroids[i]]

            if n_skim_cols > 0:
                for j in range(<size_t>n_skim_cols):
                    for i in range(centroids.shape[0]):
                        skim_matrix[cent_dest][i][j+1] = skim_i_vec[centroids[i] + j * vertex_count]

        else:
            for j in range(<size_t>n_skim_cols):
                for i in range(centroids.shape[0]):
                    skim_matrix[cent_dest][i][j] = skim_i_vec[centroids[i] + j * vertex_count]

    # second pass #
    # ----------- #

    # demand is loaded into all the origin vertices
    # also we compute the min travel time from all the origin vertices
    u_r = DATATYPE_INF
    for i in range(demand_size):
        vert_idx = demand_indices[i]  # origin vertex id
        v_i_vec[<size_t>vert_idx] = demand_values[i]
        u_i = u_i_vec[<size_t>vert_idx]
        if u_i < u_r:
            u_r = u_i

    # if the destination can be reached from any of the origins
    if u_r < DATATYPE_INF:

        # make sure f_i values are not zero
        for i in range(<size_t>vertex_count):
            if f_i_vec[i] < MIN_FREQ:
                f_i_vec[i] = MIN_FREQ

        h_a_count = 0
        for i in range(<size_t>edge_count):
            u_j_c_a_vec[i] *= -1.0
            h_a_count += <size_t>h_a_vec[i]

        # Sort the links with decreasing order of u_j + c_a.
        # Because the sort function sorts in increasing order, we sort a
        # transformed array, multiplied by -1, and set the items
        # corresponding to edges that are not in the hyperpath to a
        # positive value (they will be at the end of the sorted array).
        # The items corresponding to edges that are in the hyperpath have a
        # negative transformed value.
        for i in range(<size_t>edge_count):
            if h_a_vec[i] == 0:  # if the edge is not on the hyperpath
                u_j_c_a_vec[i] = 1.0

        argsort(u_j_c_a_vec, edge_indices, edge_count)

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
    cnp.float64_t[::1] c_a_vec,  # travel time
    cnp.float64_t[::1] f_a_vec,  # freq
    cnp.uint32_t[::1] tail_indices,
    cnp.float64_t *u_i_vec,
    cnp.float64_t *f_i_vec,
    cnp.float64_t *u_j_c_a_vec,
    cnp.uint8_t *h_a_vec,
    int dest_vert_index,
    cnp.float64_t[:, ::1] skim_col_vec,
    cnp.float64_t *skim_i_vec,
    cnp.float64_t *skim_j_vec,
    size_t n_skim_cols,
    size_t vertex_count,
) noexcept nogil:
    """All vertices are visited."""

    cdef:
        int edge_count = tail_indices.shape[0]
        PriorityQueue pqueue
        ElementState edge_state
        size_t i, j, edge_idx, tail_vert_idx
        DATATYPE_t u_j_c_a, u_i, f_i, beta, u_i_new, f_a
        DATATYPE_t skim_i, skim_j, skim_i_new, beta_skim

        cnp.float64_t *skim_i_new_vec = <cnp.float64_t *> malloc(n_skim_cols * sizeof(cnp.float64_t))
        cnp.float64_t *skim_j_new_vec = <cnp.float64_t *> malloc(n_skim_cols * sizeof(cnp.float64_t))

    # initialization of the heap elements
    # all nodes have INFINITY key and NOT_IN_HEAP state
    init_heap(&pqueue, <size_t>edge_count)

    # only the incoming edges of the target vertex are inserted into the
    # priority queue
    for i in range(<size_t>csc_indptr[<size_t>dest_vert_index], <size_t>csc_indptr[<size_t>(dest_vert_index + 1)]):
        edge_idx = csc_edge_idx[i]
        insert(&pqueue, edge_idx, c_a_vec[edge_idx])
        u_j_c_a_vec[edge_idx] = c_a_vec[edge_idx]
        for j in range(<size_t>n_skim_cols):
            skim_j_vec[edge_idx + j * edge_count] = skim_col_vec[edge_idx][j]

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

            for j in range(<size_t>n_skim_cols):

                skim_i = skim_i_vec[tail_vert_idx + j * vertex_count]
                skim_j = skim_j_vec[edge_idx + j * edge_count]

                if (u_i < DATATYPE_INF) | (f_i > 0.0):
                    beta_skim = f_i * skim_i
                else:
                    beta_skim = 0.0

                skim_i_new = (beta_skim + f_a * skim_j) / (f_i + f_a)

                skim_i_vec[tail_vert_idx  + j * vertex_count] = skim_i_new
                skim_i_new_vec[j] = skim_i_new

            # update f_i
            f_i_vec[tail_vert_idx] = f_i + f_a

            # add the edge to hyperpath
            h_a_vec[edge_idx] = 1

        else:

            u_i_new = u_i
            for j in range(<size_t>n_skim_cols):
                skim_i_new = skim_i_vec[tail_vert_idx + j * vertex_count]
                skim_i_new_vec[j] = skim_i_new

        # loop on incoming edges
        for i in range(<size_t>csc_indptr[tail_vert_idx], <size_t>csc_indptr[tail_vert_idx + 1]):

            edge_idx = csc_edge_idx[i]
            edge_state = pqueue.Elements[edge_idx].state

            if edge_state != SCANNED:

                # u_j of current edge = u_i of outgoing edge
                u_j_c_a = u_i_new + c_a_vec[edge_idx]

                for j in range(<size_t>n_skim_cols):
                    skim_j_new_vec[j] = skim_i_new_vec[j] + skim_col_vec[edge_idx][j]

                if edge_state == NOT_IN_HEAP:

                    insert(&pqueue, edge_idx, u_j_c_a)
                    u_j_c_a_vec[edge_idx] = u_j_c_a
                    for j in range(<size_t>n_skim_cols):
                        skim_j_vec[edge_idx + j * edge_count] = skim_j_new_vec[j]

                elif (pqueue.Elements[edge_idx].key > u_j_c_a):

                    decrease_key(&pqueue, edge_idx, u_j_c_a)
                    u_j_c_a_vec[edge_idx] = u_j_c_a
                    for j in range(<size_t>n_skim_cols):
                        skim_j_vec[edge_idx + j * edge_count] = skim_j_new_vec[j]

    free_heap(&pqueue)
    free(skim_i_new_vec)
    free(skim_j_new_vec)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void _SF_in_second_pass(
    cnp.uint32_t *edge_indices,
    cnp.uint32_t[::1] tail_indices,
    cnp.uint32_t[::1] head_indices,
    cnp.float64_t *v_i_vec,
    cnp.float64_t *v_a_vec,
    cnp.float64_t *f_i_vec,
    cnp.float64_t[::1] f_a_vec,
    size_t h_a_count
) noexcept nogil:

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
        v_a_vec[edge_idx] += v_a_new
        v_i_vec[<size_t>head_indices[edge_idx]] += v_a_new
