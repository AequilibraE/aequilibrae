import operator

import numpy as np

from aequilibrae.paths.cython.graph_context cimport GraphContext
from aequilibrae.utils.cython.array_allocations cimport array as cython_array
from aequilibrae.utils.cython.array_allocations import readonly_view


cdef class SkimmingContext:
    """Keep link fields and reusable origin-destination skim buffers.

    Fields are aligned, contiguous float64 vectors, retained without copying
    and marked read-only. Do not change them through other views. Centroids are
    the first centroid_count nodes, or all nodes if None. Outputs start at
    infinity; workers may share them only when writing different origin rows.
    """

    def __init__(
        self,
        context,
        fields,
        centroid_count=None,
        *,
        include_costs=False,
        include_turn_costs=False
    ):
        cdef const double[::1] field_view
        cdef vector[const double *] pointers

        if self.context is not None:
            raise RuntimeError("SkimmingContext cannot be reinitialized")
        if not isinstance(context, GraphContext):
            raise TypeError("context must be a GraphContext")
        if context.node_count == 0:
            raise ValueError("context must be initialised")
        if centroid_count is None:
            centroid_count = context.node_count
        else:
            centroid_count = operator.index(centroid_count)
        if not 0 <= centroid_count <= context.node_count:
            raise ValueError("centroid_count must be between 0 and context.node_count")

        arrays = []
        for array in fields:
            if not isinstance(array, np.ndarray):
                raise TypeError("skim fields must be NumPy arrays")

            field_view = array  # Check dtype, dimensions and contiguity without copying.

            if field_view.shape[0] != context.link_count:
                raise ValueError("each skim field must have context.link_count entries")
            if not array.flags.aligned:
                raise ValueError("skim fields must be aligned")

            arrays.append(array)
            pointers.push_back(&field_view[0] if field_view.shape[0] else NULL)

        width = len(arrays)
        self.od_costs_buffer = None
        self.od_turn_costs_buffer = None
        self.od_skims_buffer = cython_array[double]((centroid_count, centroid_count, width), True, np.inf)

        if include_costs:
            self.od_costs_buffer = cython_array[double]((centroid_count, centroid_count), True, np.inf)

        if include_turn_costs:
            self.od_turn_costs_buffer = cython_array[double]((centroid_count, centroid_count), True, np.inf)

        # Mark inputs read-only only after validation and output allocation succeed.
        for array in arrays:
            array.flags.writeable = False

        self.link_fields = tuple(arrays)
        self.field_pointers.swap(pointers)
        self.centroid_count = centroid_count
        self.field_count = width
        self.context = context

    @property
    def fields(self):
        """Read-only views of the prepared link fields, in input order."""
        return tuple(readonly_view(array) for array in self.link_fields)

    @property
    def od_skims(self):
        """Read-only link sums by origin, destination and field."""
        return readonly_view(self.od_skims_buffer)

    @property
    def od_costs(self):
        """Routing costs including penalties, or None if not requested."""
        return None if self.od_costs_buffer is None else readonly_view(self.od_costs_buffer)

    @property
    def od_turn_costs(self):
        """Turn costs alone, or None if not requested."""
        return None if self.od_turn_costs_buffer is None else readonly_view(self.od_turn_costs_buffer)
