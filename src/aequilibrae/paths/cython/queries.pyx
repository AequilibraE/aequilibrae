"""Inputs that vary between routing searches and loading operations."""

import operator
import numpy as np
cimport cython

from aequilibrae.utils.cython.array_allocations cimport const_array_pointer
from aequilibrae.utils.cython.array_allocations import readonly_view


cdef class SearchQuery:
    """Bind an origin and an optional borrowed boolean target mask.

    None requests a full search. A mask must contain at least one target and
    stay unchanged while this query is used: its count is computed only once.
    The origin can change between searches without replacing the mask. Separate
    workers need separate queries if they change origins independently.
    """

    def __cinit__(self):
        self.target_mask_buffer = None

    def __init__(self, node_count, origin, target_mask=None):
        if self.node_count:
            raise RuntimeError("SearchQuery cannot be reinitialized")

        node_count = operator.index(node_count)
        if node_count < 1:
            raise ValueError("node_count must be positive")

        self.node_count = node_count
        self.origin = origin

        if target_mask is not None:
            self.target_mask_buffer = target_mask
            if <size_t>self.target_mask_buffer.shape[0] != self.node_count:
                raise ValueError("target_mask must have one entry per node")

            self.target_count = np.count_nonzero(np.asarray(self.target_mask_buffer))
            if self.target_count == 0:
                raise ValueError("target_mask must contain a target; use None for a full search")

    @property
    def origin(self):
        return self.origin_index

    @origin.setter
    def origin(self, value):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("origin must be a node index, not a boolean")

        value = operator.index(value)
        if not 0 <= value < self.node_count:
            raise ValueError("origin is outside the query's node range")

        self.origin_index = value

    @property
    def target_mask(self):
        return None if self.target_mask_buffer is None else readonly_view(self.target_mask_buffer)

    cdef CppSearchQuery view(self) noexcept nogil:
        cdef CppSearchQuery query
        query.node_count = self.node_count
        query.origin = self.origin_index
        query.target_count = self.target_count

        if self.target_mask_buffer is not None:
            query.target_mask = const_array_pointer(self.target_mask_buffer)
        return query


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
            raise RuntimeError("LoadingQuery cannot be reinitialized")
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
    cdef CppLoadingQuery[double] view(self) noexcept nogil:
        cdef CppLoadingQuery[double] query
        query.destination_count = self.destination_count
        query.class_count = self.class_count

        if self.destination_count and self.class_count:
            query.demand = &self.demand_buffer[0, 0]
        return query

    @property
    def demand(self):
        return readonly_view(self.demand_buffer)
