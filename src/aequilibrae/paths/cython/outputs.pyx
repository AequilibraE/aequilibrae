"""Buffers produced by downstream operations on routing results."""

import operator
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
