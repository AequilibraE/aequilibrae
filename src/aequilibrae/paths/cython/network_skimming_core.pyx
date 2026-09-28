# cython: language_level=3
"""Parallel centroid skimming with the shared routing and skimming kernels."""

import operator

import numpy as np
cimport cython
from cython.parallel cimport prange, threadid
from libc.stddef cimport size_t
from libcpp.vector cimport vector

from aequilibrae.paths.cython.context cimport (
    NodeBasedContext, TurnBasedContext, SkimmingContext, CppSkimmingContext,
)
from aequilibrae.paths.cython.dijkstra cimport (
    HeapType, RoutingContext, routing_heap_from_name, run_dijkstra,
)
from aequilibrae.paths.cython.outputs cimport SkimmingOutputs, CppSkimmingOutputsView
from aequilibrae.paths.cython.queries cimport CppSearchQuery
from aequilibrae.paths.cython.search_results cimport SearchResults, CppMutableSearchResults
from aequilibrae.paths.cython.skimming cimport cpp_skimming
from aequilibrae.paths.cython.workspaces cimport SkimmingWorkspace, CppSkimmingWorkspace


cdef void skim_origin(
    RoutingContext routing,
    size_t row,
    size_t node,
    const CppSkimmingContext[double] &fields,
    const CppMutableSearchResults &search,
    const CppSkimmingWorkspace[double] &scratch,
    const CppSkimmingOutputsView[double] &output,
    HeapType heap_type,
) noexcept nogil:
    cdef CppSearchQuery query
    query.node_count = search.node_count
    query.origin = node
    query.target_mask = NULL
    query.target_count = 0

    run_dijkstra(routing, query, search, heap_type)
    cpp_skimming[double](search.read_view(), fields, scratch, output.origin(row))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void skim_origins(
    RoutingContext routing,
    const size_t[::1] rows,
    const size_t[::1] nodes,
    const CppSkimmingContext[double] &fields,
    CppMutableSearchResults *workers,
    CppSkimmingWorkspace[double] *scratch,
    const CppSkimmingOutputsView[double] &output,
    int cores,
    HeapType heap_type,
) noexcept nogil:
    cdef Py_ssize_t index

    for index in prange(rows.shape[0], num_threads=cores, schedule="guided"):
        skim_origin(
            routing, rows[index], nodes[index], fields, workers[threadid()],
            scratch[threadid()], output, heap_type,
        )


def skimming_parallel(
    routing,
    SkimmingContext fields not None,
    SkimmingOutputs output not None,
    origins,
    cores,
    heap="4ary",
):
    """Replace selected centroid rows using worker-local searches and scratch.

    ``origins`` contains (output row, local node) pairs. The routing context,
    skim fields and output must all use the same compact graph node/link order.
    """
    cores = operator.index(cores)
    if not 1 <= cores <= np.iinfo(np.int32).max:
        raise ValueError("cores must be positive and fit an OpenMP thread count")

    cdef int thread_count = cores
    cdef HeapType heap_type = routing_heap_from_name(heap)
    cdef vector[CppMutableSearchResults] workers
    cdef vector[CppSkimmingWorkspace[double]] scratch
    cdef CppSkimmingContext[double] skim_view
    cdef CppSkimmingOutputsView[double] output_view
    cdef SearchResults search
    cdef SkimmingWorkspace workspace
    cdef bint node_based = isinstance(routing, NodeBasedContext)

    if not isinstance(routing, (NodeBasedContext, TurnBasedContext)):
        raise TypeError("routing must be a node or turn routing context")

    if fields.link_count != routing.link_count or output.field_names != fields.field_names:
        raise ValueError("skim fields must match the routing links and output names")

    if output.destination_count > routing.node_count:
        raise ValueError("skim destinations exceed routing nodes")

    indices = np.asarray(origins)
    if indices.ndim != 2 or indices.shape[1] != 2 or indices.dtype.kind not in "iu":
        raise ValueError("origins must contain output row and local node indices")

    if np.any(indices < 0) or np.any(indices[:, 0] >= output.origin_count) or np.any(indices[:, 1] >= routing.node_count):
        raise ValueError("origin rows or nodes are out of range")

    if np.unique(indices[:, 0]).size != indices.shape[0]:
        raise ValueError("origin rows must be unique")

    if indices.shape[0] == 0:
        output.reset()
        return output

    thread_count = min(thread_count, indices.shape[0])
    indices = np.asarray(indices, dtype=np.uintp, order="C")
    cdef const size_t[::1] rows = np.ascontiguousarray(indices[:, 0])
    cdef const size_t[::1] nodes = np.ascontiguousarray(indices[:, 1])

    # Retain the owners while C++ views borrow their buffers in the OpenMP loop.
    owners = []
    for _ in range(thread_count):
        search = SearchResults(routing.node_count, routing.state_count, routing.link_count)
        workspace = SkimmingWorkspace(routing.state_count, fields.additive_field_count)
        owners.append((search, workspace))
        workers.push_back(search.view())
        scratch.push_back(workspace.view())

    skim_view = fields.view()
    output_view = output.view()

    with nogil:
        output_view.reset()
        if node_based:
            skim_origins(
                <NodeBasedContext>routing, rows, nodes, skim_view, workers.data(), scratch.data(),
                output_view, thread_count, heap_type,
            )
        else:
            skim_origins(
                <TurnBasedContext>routing, rows, nodes, skim_view, workers.data(), scratch.data(),
                output_view, thread_count, heap_type,
            )
    return output
