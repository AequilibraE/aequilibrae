import operator
cimport cython

import numpy as np

from aequilibrae.utils.cython.array_allocations cimport array, array_pointer
from aequilibrae.utils.cython.array_allocations import readonly_view


cdef class AoNWorkspace:
    """Own downstream scratch separately from paths and graph inputs.

    Prepare the required buffers before borrowing a view so the origin loop
    can reuse their allocations.
    """

    def __cinit__(self):
        self.state_skims_buffer = None
        self.state_loads_buffer = None
        self.selected_paths_buffer = None

    def __init__(self, state_count, field_count=0):
        if self.state_count:
            raise RuntimeError("AoNWorkspace cannot be reinitialized")

        state_count = operator.index(state_count)
        if state_count < 1:
            raise ValueError("state_count must be positive")

        self.state_count = state_count
        self.prepare_skims(field_count)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef CppAoNWorkspace[double] view(self) noexcept nogil:
        cdef CppAoNWorkspace[double] workspace
        workspace.state_count = self.state_count
        workspace.skim_field_count = self.skim_field_count
        workspace.loading_class_count = self.loading_class_count

        if self.skim_field_count:
            workspace.state_skims = &self.state_skims_buffer[0, 0]

        if self.loading_class_count:
            workspace.state_loads = &self.state_loads_buffer[0, 0]

        if self.selected_paths_buffer is not None:
            workspace.selected_paths = array_pointer(self.selected_paths_buffer)

        return workspace

    cpdef prepare_skims(self, object field_count):
        """Allocate before the origin loop; reuse an allocation of the same width."""
        field_count = operator.index(field_count)
        if field_count < 0:
            raise ValueError("field_count must be nonnegative")
        if self.state_skims_buffer is not None and field_count == self.skim_field_count:
            return
        self.state_skims_buffer = array[double]((self.state_count, field_count), True, np.inf)
        self.skim_field_count = field_count

    cpdef prepare_loading(self, object class_count):
        class_count = operator.index(class_count)
        if class_count < 0:
            raise ValueError("class_count must be nonnegative")

        if self.state_loads_buffer is not None and class_count == self.loading_class_count:
            return

        self.state_loads_buffer = array[double]((self.state_count, class_count), True, 0)
        self.loading_class_count = class_count

    cpdef prepare_select_links(self):
        if self.selected_paths_buffer is None:
            self.selected_paths_buffer = array[cpp_bool](self.state_count, True, False)

    @property
    def selected_paths(self):
        return None if self.selected_paths_buffer is None else readonly_view(self.selected_paths_buffer)

    @property
    def state_loads(self):
        return None if self.state_loads_buffer is None else readonly_view(self.state_loads_buffer)

    @property
    def state_skims(self):
        return readonly_view(self.state_skims_buffer)
