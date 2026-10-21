"""Operations on search paths, with explicit query, scratch and output owners."""

import numpy as np
cimport cython

from libcpp.vector cimport vector
from aequilibrae.utils.cython.array_allocations import readonly_view


cdef class LoadingQuery:
    """Borrow demand [destinations, classes] for one loading operation.

    Values may change between calls, but shape and allocation stay fixed. No
    copying, packing or writeability changes are made. The caller chooses the
    correct origin row; this object does not retain a search or routing context.
    """

    def __cinit__(self):
        self.demand_buffer = None

    def __init__(self, demand):
        if self.demand_buffer is not None:
            raise RuntimeError("LoadingQuery cannot be reinitialised")
        if demand is None:
            raise TypeError("demand must be a contiguous float64 buffer")

        cdef const double[:, ::1] values = demand
        if not np.asarray(values).flags.aligned:
            raise ValueError("demand must be aligned")

        self.demand_buffer = values
        self.destination_count = values.shape[0]
        self.class_count = values.shape[1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef DenseDemandRow view(self) noexcept nogil:
        cdef DenseDemandRow query
        query.destination_count = self.destination_count
        query.field_count = self.class_count

        if self.destination_count and self.class_count:
            query.data = &self.demand_buffer[0, 0]
        return query

    @property
    def demand(self):
        return readonly_view(self.demand_buffer)


def network_loading(SearchResults results not None, LoadingQuery query not None,
                    LoadingWorkspace workspace not None, LoadingOutputs output not None):
    """Accumulate this origin's demand without changing paths or allocating buffers.

    Rows represent physical nodes 0..destination_count-1. Only finalised,
    non-intrazonal paths are loaded. Output is not cleared: reset it once before
    processing the origins of an iteration. Scratch is replaced on every call.
    """
    if query.destination_count > results.node_count:
        raise ValueError("demand destination_count exceeds results node_count")
    if workspace.state_count != results.state_count:
        raise ValueError("workspace state_count does not match results")
    if output.link_count != results.link_count:
        raise ValueError("output link_count does not match results")
    if workspace.class_count != query.class_count or output.class_count != query.class_count:
        raise ValueError("query, workspace and output class_count must match")

    with nogil:
        cpp_network_loading(results.read_view(), query.view(), workspace.view(), output.view())
    return output


def sum_weighted_turn_costs(SearchResults results not None, LoadingQuery query not None):
    """Return demand-weighted turn cost over finalised, non-intrazonal paths.

    Missing paths are ignored. Negative and nonfinite demand follows ordinary
    floating-point multiplication and addition. No scratch or output is written.
    """
    cdef double total
    if query.destination_count > results.node_count:
        raise ValueError("demand destination_count exceeds results node_count")
    with nogil:
        total = cpp_sum_weighted_turn_costs(results.read_view(), query.view())
    return total


def reduce_loading_outputs(workers, LoadingOutputs output not None):
    """Replace output with the sum of completed workers, leaving them unchanged."""
    cdef LoadingOutputs worker
    cdef vector[CppLoadingOutputs[double]] views
    # Keep owners alive even if workers is a generator. Only the small view
    # table is built here; the numeric output was allocated by its caller.
    owners = tuple(workers)
    for worker in owners:
        if worker is None:
            raise TypeError("workers must contain LoadingOutputs")
        if worker is output:
            raise ValueError("reduction output must be distinct from its workers")
        if worker.link_count != output.link_count or worker.class_count != output.class_count:
            raise ValueError("worker and output dimensions must match")
        views.push_back(worker.view())

    with nogil:
        cpp_reduce_loading_outputs[double](views.data(), views.size(), output.view())

    return output
