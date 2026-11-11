cimport cython


# Dense-index kernels for validated mapping buffers. Outputs must be distinct
# from inputs and large enough; callers bind the Python-owned arrays under the GIL.
cdef void expand_link_indices(
    const Py_ssize_t[::1] graph_links,
    const Py_ssize_t[::1] offsets,
    const Py_ssize_t[::1] expansion,
    Py_ssize_t[::1] output,
) noexcept nogil

cdef void aggregate_link_costs(
    const Py_ssize_t[::1] offsets,
    const Py_ssize_t[::1] expansion,
    const cython.floating[:, :] costs,
    cython.floating[:, :] output,
) noexcept nogil

cdef void project_link_loads(
    const Py_ssize_t[::1] network_link_to_graph,
    const cython.floating[:, ::1] loads,
    cython.floating[:, ::1] output,
) noexcept nogil
