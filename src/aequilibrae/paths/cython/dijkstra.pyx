"""One internal search interface for both routing modes."""

from aequilibrae.paths.cython.basic_path_finding cimport (
    FOUR_ARY_HEAP, PAIRING_HEAP, STD_PRIORITY_QUEUE, HeapType,
)
from aequilibrae.paths.cython.basic_path_finding import HEAP_MAP, available_heaps
from aequilibrae.paths.cython.pq_heap_types cimport FourAryHeap, PairingHeap, StdPriorityQueueAdapter


cdef HeapType routing_heap_from_name(object heap) except *:
    try:
        return <HeapType>HEAP_MAP[heap]
    except (KeyError, TypeError):
        raise ValueError(f"heap must be one of {available_heaps()}")


cdef void run_dijkstra(
    RoutingContext context,
    const CppSearchQuery &query,
    const CppMutableSearchResults &results,
    HeapType heap_type,
) noexcept nogil:
    if RoutingContext is NodeBasedContext:
        if heap_type == PAIRING_HEAP:
            cpp_dijkstra[PairingHeap](context.view(), query, results)
        elif heap_type == STD_PRIORITY_QUEUE:
            cpp_dijkstra[StdPriorityQueueAdapter](context.view(), query, results)
        else:
            cpp_dijkstra[FourAryHeap](context.view(), query, results)
    else:
        if heap_type == PAIRING_HEAP:
            cpp_turn_dijkstra[PairingHeap](context.view(), query, results)
        elif heap_type == STD_PRIORITY_QUEUE:
            cpp_turn_dijkstra[StdPriorityQueueAdapter](context.view(), query, results)
        else:
            cpp_turn_dijkstra[FourAryHeap](context.view(), query, results)


def dijkstra(RoutingContext context, SearchQuery query not None, SearchResults results not None, heap="4ary"):
    """Replace results using the bound graph costs and query.

    Results are caller-owned and may be reused with any matching dimensions.
    Queries and contexts supply all inputs; neither is retained by results.
    This call allocates no result buffers. The selected routing heap is local
    to each search; heap must be "4ary", "pairing", or "std".
    """
    if context is None:
        raise TypeError("context must not be None")

    if query.node_count != context.node_count:
        raise ValueError("query node_count does not match context")

    if (results.node_count != context.node_count or
            results.state_count != context.state_count or
            results.link_count != context.link_count):
        raise ValueError("results dimensions do not match context")

    cdef HeapType heap_type = routing_heap_from_name(heap)
    cdef CppSearchQuery cpp_query = query.view()
    cdef CppMutableSearchResults cpp_results = results.view()

    with nogil:
        run_dijkstra(context, cpp_query, cpp_results, heap_type)

    return results
