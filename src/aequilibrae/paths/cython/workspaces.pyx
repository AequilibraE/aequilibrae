"""Independent, fixed-size operation scratch, with an optional AoN grouping."""

import operator
import numpy as np
cimport cython

from aequilibrae.utils.cython.array_allocations cimport array, array_pointer
from aequilibrae.utils.cython.array_allocations import readonly_view


cdef class LoadingWorkspace:
    """Demand cascade scratch for one worker.

    Loading replaces this scratch on every call. It does not hold link output,
    demand or a search reference. Retained views show the last cascade.
    """

    def __init__(self, state_count, class_count):
        if self.state_count:
            raise RuntimeError("LoadingWorkspace cannot be reinitialized")

        state_count, class_count = map(operator.index, (state_count, class_count))
        if state_count < 1 or class_count < 0:
            raise ValueError("state_count must be positive and class_count nonnegative")

        self.state_count = state_count
        self.class_count = class_count
        self.state_loads_buffer = array[double]((state_count, class_count), True, 0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef CppLoadingWorkspace[double] view(self) noexcept nogil:
        cdef CppLoadingWorkspace[double] workspace
        workspace.state_count = self.state_count
        workspace.class_count = self.class_count
        if self.class_count:
            workspace.state_loads = &self.state_loads_buffer[0, 0]
        return workspace

    @property
    def state_loads(self):
        return readonly_view(self.state_loads_buffer)


cdef class SkimmingWorkspace:
    """State sums for additive fields, independent of inputs and output."""

    def __init__(self, state_count, field_count):
        if self.state_count:
            raise RuntimeError("SkimmingWorkspace cannot be reinitialized")

        state_count, field_count = map(operator.index, (state_count, field_count))
        if state_count < 1 or field_count < 0:
            raise ValueError("state_count must be positive and field_count nonnegative")

        self.state_count = state_count
        self.field_count = field_count
        self.state_skims_buffer = array[double]((state_count, field_count), True, np.inf)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef CppSkimmingWorkspace[double] view(self) noexcept nogil:
        cdef CppSkimmingWorkspace[double] workspace
        workspace.state_count = self.state_count
        workspace.field_count = self.field_count

        if self.field_count:
            workspace.state_skims = &self.state_skims_buffer[0, 0]
        return workspace

    @property
    def state_skims(self):
        return readonly_view(self.state_skims_buffer)


cdef class SelectLinkWorkspace:
    """One path-membership flag per state, reused across selected sets.

    Loading scratch is supplied separately so regular and selected loading
    can share a cascade allocation without sharing unrelated skim scratch.
    """

    def __init__(self, state_count):
        if self.state_count:
            raise RuntimeError("SelectLinkWorkspace cannot be reinitialized")

        state_count = operator.index(state_count)
        if state_count < 1:
            raise ValueError("state_count must be positive")

        self.state_count = state_count
        self.selected_paths_buffer = array[cpp_bool](state_count, True, False)

    cdef CppSelectLinkWorkspace view(self) noexcept nogil:
        cdef CppSelectLinkWorkspace workspace
        workspace.state_count = self.state_count
        workspace.selected_paths = array_pointer(self.selected_paths_buffer)

        return workspace

    @property
    def selected_paths(self):
        return readonly_view(self.selected_paths_buffer)


cdef class AoNWorkspace:
    """Allocate the small workspaces needed by one assignment worker.

    Omit a width to leave that operation unallocated. Components are usable
    independently and keep their own storage alive if the group is deleted.
    Dimensions are fixed; kernels never prepare or resize these buffers.
    """

    def __init__(self, state_count, *, class_count=None, field_count=None, select_links=False):
        if self.state_count:
            raise RuntimeError("AoNWorkspace cannot be reinitialized")

        state_count = operator.index(state_count)
        if state_count < 1:
            raise ValueError("state_count must be positive")
        self.state_count = state_count

        if class_count is not None:
            self.loading = LoadingWorkspace(state_count, class_count)
        if field_count is not None:
            self.skimming = SkimmingWorkspace(state_count, field_count)
        if select_links:
            self.select_link = SelectLinkWorkspace(state_count)

    cdef CppAoNWorkspace[double] view(self) noexcept nogil:
        cdef CppAoNWorkspace[double] workspace

        if self.loading is not None:
            workspace.loading = self.loading.view()
        if self.skimming is not None:
            workspace.skimming = self.skimming.view()
        if self.select_link is not None:
            workspace.select_link = self.select_link.view()

        return workspace
