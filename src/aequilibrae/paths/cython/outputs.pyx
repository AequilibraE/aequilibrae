"""Buffers produced by downstream operations on routing results."""

import operator

import numpy as np
cimport cython

from aequilibrae.utils.cython.array_allocations cimport array
from aequilibrae.utils.cython.array_allocations import readonly_view


cdef class LoadingOutputs:
    """Own fixed-size link loads, without demand, scratch or a graph reference.

    Loads start at zero. Loading accumulates into them; reset explicitly before
    another iteration. Read-only views keep storage alive and reflect reuse.
    """

    def __cinit__(self):
        self.link_loads_buffer = None

    def __init__(self, link_count, class_count):
        if self.link_loads_buffer is not None:
            raise RuntimeError("LoadingOutputs cannot be reinitialized")

        link_count, class_count = map(operator.index, (link_count, class_count))
        if link_count < 0 or class_count < 0:
            raise ValueError("link_count and class_count must be nonnegative")

        self.link_count = link_count
        self.class_count = class_count
        self.link_loads_buffer = array[double]((link_count, class_count), True, 0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef CppLoadingOutputs[double] view(self) noexcept nogil:
        cdef CppLoadingOutputs[double] output
        output.link_count = self.link_count
        output.class_count = self.class_count

        if self.link_count and self.class_count:
            output.link_loads = &self.link_loads_buffer[0, 0]

        return output

    def reset(self):
        """Clear this iteration's accumulation without replacing its allocation."""
        with nogil:
            self.view().reset()

    @property
    def link_loads(self):
        return readonly_view(self.link_loads_buffer)


def _validate_skim_names(field_names):
    """Names identify output matrices, so ambiguity must be rejected at setup."""
    if isinstance(field_names, str):
        raise TypeError("field_names must be a sequence of names, not a string")

    names = tuple(field_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("skim field names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("skim field names must be unique")

    return names


cdef class SkimmingOutputs:
    """Own one ordered skim array, without inputs or scratch references.

    Storage is [origin rows, fields, destinations]. Each skimming call replaces
    one contiguous origin block. Reset explicitly to clear all blocks to infinity.
    Named matrices are strided views of this storage and keep it alive after the
    wrapper is deleted.

    The layout of [origin rows, fields, destinations] allows contiguous origin and
    field slices. Given the skimming happens as for each origin, for each field,
    skim all destinations, this makes sense, although odd.
    """

    def __init__(self, origin_count, destination_count, field_names):
        if self.field_names is not None:
            raise RuntimeError("SkimmingOutputs cannot be reinitialized")

        origin_count = operator.index(origin_count)
        destination_count = operator.index(destination_count)
        if origin_count < 0 or destination_count < 0:
            raise ValueError("origin_count and destination_count must be nonnegative")

        names = _validate_skim_names(field_names)
        self.origin_count = origin_count
        self.destination_count = destination_count
        self.field_count = len(names)

        # Each origin gets one contiguous block of fields and destinations.
        # Infinity also marks origin rows that have not been written yet.
        self.skims_buffer = array[double](
            (origin_count, self.field_count, destination_count), True, np.inf
        )
        self.field_names = names

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef CppSkimmingOutputsView[double] view(self) noexcept nogil:
        cdef CppSkimmingOutputsView[double] output
        output.origin_count = self.origin_count
        output.destination_count = self.destination_count
        output.field_count = self.field_count

        # Empty buffers have no first element to take an address from.
        if self.origin_count and self.destination_count and self.field_count:
            output.data = &self.skims_buffer[0, 0, 0]

        return output

    def reset(self):
        """Clear all rows without replacing their allocation."""
        with nogil:
            self.view().reset()

    @property
    def skims(self):
        """Read-only, contiguous [origin rows, fields, destinations] array."""
        return readonly_view(self.skims_buffer)

    @property
    def matrices(self):
        """Named [origin rows, destinations] views, which may be non-contiguous."""
        skims = self.skims

        # Slicing the read-only array keeps both its storage and read-only
        # guarantee. No matrix data is copied when building this dictionary.
        return {name: skims[:, field, :] for field, name in enumerate(self.field_names)}
