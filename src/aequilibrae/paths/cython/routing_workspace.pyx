import operator

import numpy as np

from aequilibrae.paths.cython.graph_context cimport GraphContext


cdef class RoutingWorkspace:
    """Per-results scratch storage, owned by NumPy and borrowed by C++.

    Allocate/resize under the GIL, then use from one worker at a time. Retained
    state_skims views pin their allocation; preparing a different field count
    replaces the buffer, while repeated skims of the same width overwrite it.
    """

    def __init__(self, context, field_count=0):
        if not isinstance(context, GraphContext):
            raise TypeError("context must be a GraphContext")
        if context.node_count == 0:
            raise ValueError("context must be initialised")
        self.context = context
        self.cpp.state_count = context.state_count
        self._state_skims = None
        self.prepare_skims(field_count)

    cpdef prepare_skims(self, object field_count):
        """Ensure packed [state_count, field_count] scratch space (requires GIL)."""
        cdef double[::1] flat
        field_count = operator.index(field_count)
        if field_count < 0:
            raise ValueError("field_count must be nonnegative")
        if self._state_skims is not None and field_count == self.cpp.skim_field_count:
            return
        array = np.full((self.cpp.state_count, field_count), np.inf, dtype=np.float64)
        flat = array.reshape(-1)
        self.cpp.skim_field_count = field_count
        self.cpp.state_skims = &flat[0] if flat.shape[0] else NULL
        self._state_skims = array
        array.flags.writeable = False

    @property
    def state_count(self):
        return self.cpp.state_count

    @property
    def skim_field_count(self):
        return self.cpp.skim_field_count

    @property
    def state_skims(self):
        """Read-only scratch view from the last field skim, NOT the last search.

        Rows are search states; root is zero and unfinalized states are infinite.
        New searches and cost-only skims do not update these scratch values.
        """
        return self._state_skims.view()
