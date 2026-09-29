"""One internal search interface for both routing modes."""

from aequilibrae.paths.cython.workspaces cimport CPP_FOUR_ARY, CPP_PAIRING, CPP_STD


HEAP_MAP = {"4ary": CPP_FOUR_ARY, "pairing": CPP_PAIRING, "std": CPP_STD}


def available_heaps() -> list:
    """Return the priority queue implementations."""
    return list(HEAP_MAP.keys())


def validate_routing_heap(heap: str):
    if heap not in HEAP_MAP:
        raise ValueError(f"heap must be one of {available_heaps()}")


def dijkstra(
    RoutingContext context,
    SearchQuery query not None,
    SearchResults results not None,
    SearchWorkspace workspace not None,
):
    """Replace results using the bound graph costs and query.

    Results are caller-owned and may be reused with any matching dimensions.
    Queries and contexts supply all inputs; neither is retained by results.
    This call allocates no result buffers. The workspace owns one fixed heap
    and can be reused for sequential searches with matching dimensions.
    """
    if context is None:
        raise TypeError("context must not be None")

    if query.node_count != context.node_count:
        raise ValueError("query node_count does not match context")

    if (results.node_count != context.node_count or
            results.state_count != context.state_count or
            results.link_count != context.link_count):
        raise ValueError("results dimensions do not match context")

    if workspace.node_count != context.node_count or workspace.state_count != context.state_count:
        raise ValueError("workspace dimensions do not match context")

    cdef CppSearchQuery cpp_query = query.view()
    cdef CppMutableSearchResults cpp_results = results.view()
    cdef CppSearchWorkspace cpp_workspace = workspace.view()
    with nogil:
        cpp_dijkstra(context.view(), cpp_query, cpp_results, cpp_workspace)

    return results
