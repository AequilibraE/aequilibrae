import operator

import numpy as np

from libc.stddef cimport size_t
from libcpp cimport bool as cpp_bool

from aequilibrae.paths.cython.pq_heap_types cimport FourAryHeap
from aequilibrae.paths.cython.graph_context cimport NodeBasedContext
from aequilibrae.paths.cython.search_results cimport SearchResults


def choose_destinations(destinations, node_count):
    """Validate destination indices or a mask and return unique node indices."""
    if destinations is None:
        return range(node_count)

    candidate = np.asarray(destinations)
    if candidate.ndim == 1 and candidate.dtype.kind == "b":
        if candidate.size != node_count:
            raise ValueError("a destination mask must have one entry per node")
        indices = np.flatnonzero(candidate).tolist()
    else:
        try:
            indices = [operator.index(destinations)]
        except TypeError:
            try:
                indices = [operator.index(value) for value in destinations]
            except TypeError:
                raise TypeError(
                    "destinations must be a node index, an iterable of node indices, or a boolean mask"
                ) from None

    if any(index < 0 or index >= node_count for index in indices):
        raise ValueError("destination is outside the context's node range")
    return sorted(set(indices))


def dijkstra(RoutingContext context, origin, destinations, SearchResults results = None):
    """Find shortest paths from one origin, reusing results when supplied.

    Destinations may be a node index, an iterable or a boolean mask; None means
    all nodes. Empty targets disable early exit. Return the filled SearchResults.
    """
    cdef size_t origin_index, node, node_count
    cdef cpp_bool *mask
    if context is None:
        raise TypeError("context must not be None")
    node_count = context.node_count
    origin = operator.index(origin)
    if not 0 <= origin < node_count:
        raise ValueError("origin is outside the context's node range")
    origin_index = origin

    destination_indices = choose_destinations(destinations, node_count)

    if results is None:
        results = context.make_results()
    elif results.context is not context:
        raise ValueError("different context")

    # Fill the destination mask without making its public views writable.
    mask = <cpp_bool *>results.cpp.destination_mask
    node = 0
    while node < node_count:
        mask[node] = 0
        node += 1
    for destination in destination_indices:
        mask[<size_t>destination] = 1
    results.cpp.destination_count = len(destination_indices)

    # Cython does not dispatch overloaded free functions, so specialize here.
    with nogil:
        if RoutingContext is NodeBasedContext:
            cpp_dijkstra[FourAryHeap](context.view(), origin_index, results.cpp)
        else:
            cpp_turn_dijkstra[FourAryHeap](context.view(), origin_index, results.cpp)

    return results
