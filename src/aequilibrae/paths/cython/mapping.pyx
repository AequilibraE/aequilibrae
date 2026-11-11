"""Internal kernels for the standalone Python-owned network/graph mappings.

Python entry points here expect buffers already validated by paths.mapping.
Cython callers may cimport the allocation-free kernels from mapping.pxd.
"""

cimport cython
import numpy as np

from libc.string cimport memcpy, memset


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void expand_link_indices(
    const Py_ssize_t[::1] graph_links,
    const Py_ssize_t[::1] offsets,
    const Py_ssize_t[::1] expansion,
    Py_ssize_t[::1] output,
) noexcept nogil:
    """Concatenate expansion segments into caller-owned network-index storage."""
    cdef Py_ssize_t i, link, position = 0
    for link in graph_links:
        for i in range(offsets[link], offsets[link + 1]):
            output[position] = expansion[i]
            position += 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void aggregate_link_costs(
    const Py_ssize_t[::1] offsets,
    const Py_ssize_t[::1] expansion,
    const cython.floating[:, :] costs,
    cython.floating[:, :] output,
) noexcept nogil:
    """Gather each expansion's costs, with no hashing or shared output writes."""
    cdef Py_ssize_t link, field, position
    cdef cython.floating total
    for link in range(offsets.shape[0] - 1):
        for field in range(costs.shape[1]):
            total = 0
            for position in range(offsets[link], offsets[link + 1]):
                total += costs[expansion[position], field]
            output[link, field] = total


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void project_link_loads(
    const Py_ssize_t[::1] network_link_to_graph,
    const cython.floating[:, ::1] loads,
    cython.floating[:, ::1] output,
) noexcept nogil:
    """Copy floating-point rows without doing arithmetic."""
    cdef Py_ssize_t row, link, width = loads.shape[1] * sizeof(cython.floating)
    if width == 0:
        return
    for row in range(network_link_to_graph.shape[0]):
        link = network_link_to_graph[row]
        if link >= 0:
            memcpy(&output[row, 0], &loads[link, 0], width)
        else:
            memset(&output[row, 0], 0, width)


def expand_links(
    const Py_ssize_t[::1] graph_links,
    const Py_ssize_t[::1] offsets,
    const Py_ssize_t[::1] expansion,
):
    """Count, allocate once, then fill the concatenated directed-link indices."""
    cdef Py_ssize_t link
    # Python integer accumulation avoids overflow before allocating the output.
    count = 0
    for link in graph_links:
        count += offsets[link + 1] - offsets[link]
    result = np.empty(count, dtype=np.intp)
    cdef Py_ssize_t[::1] output = result
    with nogil:
        expand_link_indices(graph_links, offsets, expansion, output)
    return result


def aggregate_costs(
    const Py_ssize_t[::1] offsets,
    const Py_ssize_t[::1] expansion,
    const cython.floating[:, :] costs,
    cython.floating[:, :] output,
):
    """Dispatch float32/float64 buffers to the allocation-free aggregation kernel."""
    with nogil:
        aggregate_link_costs(offsets, expansion, costs, output)


def project_loads(
    const Py_ssize_t[::1] network_link_to_graph,
    const cython.floating[:, ::1] loads,
    cython.floating[:, ::1] output,
):
    """Copy complete floating-point rows into a directed-network load array."""
    with nogil:
        project_link_loads(network_link_to_graph, loads, output)
