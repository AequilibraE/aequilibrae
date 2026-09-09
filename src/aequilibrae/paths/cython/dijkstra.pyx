import operator

import numpy as np

from libc.stddef cimport size_t

from aequilibrae.paths.cython.pq_heap_types cimport FourAryHeap
from aequilibrae.paths.cython.graph_context cimport NodeBasedContext
from aequilibrae.paths.cython.search_results cimport SearchResults


def _normalise_destinations(destinations, node_count):
    """Return unique destination node indices after validating the public input."""
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
    """Populate reusable state-tree results for one source and one or more targets.

    ``destinations`` accepts a node index, an iterable of node indices, or a
    boolean mask with one entry per physical node. ``None`` requests all nodes.
    Duplicate indices are ignored. The search stops after every requested node
    settles, or after exhausting the reachable state space. An empty iterable
    or all-false mask disables early exit and therefore settles every reachable
    state.

    NodeBasedContext searches physical-node states. TurnBasedContext searches
    incoming-link states with sparse turn costs. If ``results`` is omitted it is
    allocated before releasing the GIL; otherwise its existing arrays are reused.
    Use the ``SearchResults.*_to(destination)`` methods to inspect individual
    paths from a multi-target search. The singular path properties remain
    available for single-target calls.
    """
    cdef size_t origin_index, node, node_count
    if context is None:
        raise TypeError("context must not be None")
    node_count = context.node_count
    origin = operator.index(origin)
    if not 0 <= origin < node_count:
        raise ValueError("origin is outside the context's node range")
    origin_index = origin

    destination_indices = _normalise_destinations(destinations, node_count)

    if results is None:
        results = context.make_results()
    elif results.context is not context:
        raise ValueError("different context")

    # The NumPy allocation is externally read-only; write through the private
    # C++ pointer while holding the GIL without changing its public flags.
    node = 0
    while node < node_count:
        results.cpp.destination_mask[node] = 0
        node += 1
    for destination in destination_indices:
        results.cpp.destination_mask[<size_t>destination] = 1

    # Cython does not dispatch overloaded free functions, so specialize here.
    with nogil:
        if RoutingContext is NodeBasedContext:
            cpp_dijkstra[FourAryHeap](context.cpp, origin_index, results.cpp)
        else:
            cpp_turn_dijkstra[FourAryHeap](context.cpp, origin_index, results.cpp)

    return results
