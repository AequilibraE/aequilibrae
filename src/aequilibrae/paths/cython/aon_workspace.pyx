import operator

import numpy as np

from aequilibrae.paths.cython.graph_context cimport GraphContext
from aequilibrae.utils.cython.array_allocations cimport array
from aequilibrae.utils.cython.array_allocations import readonly_view


cdef class AoNWorkspace:
    """Reuse one worker's scratch buffers for skimming and demand loading.

    Resize under the GIL and use from one worker at a time. Read-only views
    keep old buffers alive after resizing.
    """

    def __init__(self, context: GraphContext, field_count: int = 0):
        if not isinstance(context, GraphContext):
            raise TypeError("context must be a GraphContext")
        if context.node_count == 0:
            raise ValueError("context must be initialised")

        self.context = context
        self.cpp.state_count = context.state_count

        self.state_skims_buffer = None
        self.state_loads_buffer = None
        self.prepare_skims(field_count)

    cpdef prepare_skims(self, object field_count):
        """Allocate skim scratch if the field count changed; requires the GIL."""
        field_count = operator.index(field_count)

        if field_count < 0:
            raise ValueError("field_count must be nonnegative")
        if self.state_skims_buffer is not None and field_count == self.cpp.skim_field_count:
            return

        self.state_skims_buffer = array[double]((self.cpp.state_count, field_count), True, np.inf)
        self.cpp.skim_field_count = field_count
        self.cpp.state_skims = &self.state_skims_buffer[0, 0] if field_count else NULL

    cpdef prepare_loading(self, object class_count):
        """Allocate demand scratch if the class count changed; requires the GIL."""
        class_count = operator.index(class_count)

        if class_count < 0:
            raise ValueError("class_count must be nonnegative")
        if self.state_loads_buffer is not None and class_count == self.cpp.loading_class_count:
            return

        self.state_loads_buffer = array[double]((self.cpp.state_count, class_count), True, 0)
        self.cpp.loading_class_count = class_count
        self.cpp.state_loads = &self.state_loads_buffer[0, 0] if class_count else NULL

    @property
    def loading_class_count(self):
        """Number of demand classes supported by the loading scratch."""
        return self.cpp.loading_class_count

    @property
    def state_loads(self):
        """Read-only demand totals by search state, or None before loading setup."""
        return None if self.state_loads_buffer is None else readonly_view(self.state_loads_buffer)

    @property
    def state_count(self):
        """Number of search states supported by this workspace."""
        return self.cpp.state_count

    @property
    def skim_field_count(self):
        """Number of fields supported by the skim scratch."""
        return self.cpp.skim_field_count

    @property
    def state_skims(self):
        """Read-only field sums by search state from the last field skim."""
        return readonly_view(self.state_skims_buffer)
