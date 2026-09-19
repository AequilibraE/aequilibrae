"""One internal search interface for both routing modes."""

from aequilibrae.paths.cython.pq_heap_types cimport FourAryHeap


def dijkstra(RoutingContext context, SearchQuery query not None, SearchResults results not None):
    """Replace results using the bound graph costs and query.

    Results are caller-owned and may be reused with any matching dimensions.
    Queries and contexts supply all inputs; neither is retained by results.
    This call allocates no result buffers. The routing heap is still local to
    each search and will be addressed separately.
    """
    if context is None:
        raise TypeError("context must not be None")

    if query.node_count != context.node_count:
        raise ValueError("query node_count does not match context")

    if (results.node_count != context.node_count or
            results.state_count != context.state_count or
            results.link_count != context.link_count):
        raise ValueError("results dimensions do not match context")

    with nogil:
        if RoutingContext is NodeBasedContext:
            cpp_dijkstra[FourAryHeap](context.view(), query.view(), results.view())
        else:
            cpp_turn_dijkstra[FourAryHeap](context.view(), query.view(), results.view())

    return results
