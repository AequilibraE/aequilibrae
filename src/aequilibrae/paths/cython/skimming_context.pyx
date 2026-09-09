import operator

import numpy as np

from aequilibrae.paths.cython.graph_context cimport GraphContext


cdef class SkimmingContext:
    """Prepared link fields and reusable centroid OD arrays.

    Inputs must be contiguous, aligned float64 NumPy vectors in context.costs
    order. They are retained without copying and marked read-only. Cython reads
    them through const memoryviews and C++ pointers. Do not change their data
    through other aliases or make them writable again while this object is used.

    Centroids are the first centroid_count nodes; None uses every context node.
    od_skims has shape (centroid_count, centroid_count, field_count). Optional
    od_costs and od_turn_costs each have shape (centroid_count, centroid_count).
    All outputs start at infinity and are exposed as read-only, zero-copy views.

    results.skim_fields(self) writes the current origin's row. Repeated calls
    overwrite that row, not the whole matrix. Finish all origins before reading
    a complete iteration's output. Use .copy() to keep an earlier iteration.

    Workers may share this object if they write different origin rows and use
    separate SearchResults/workspaces. Do not read a row while it is being
    written, reinitialize the context, or force any buffers writable/resized.
    """

    def __init__(self, context, fields, centroid_count=None, *,
                 include_costs=False, include_turn_costs=False):
        cdef const double[::1] field_view
        cdef double[::1] output_view
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
        od_skims = np.full((centroid_count, centroid_count, width), np.inf, dtype=np.float64)
        od_costs = np.full((centroid_count, centroid_count), np.inf) if include_costs else None
        od_turn_costs = np.full((centroid_count, centroid_count), np.inf) if include_turn_costs else None

        # Keep private output pointers so kernels can write into read-only NumPy buffers.
        output_view = od_skims.reshape(-1)
        self._od_skims_ptr = &output_view[0] if output_view.shape[0] else NULL
        if od_costs is not None:
            output_view = od_costs.reshape(-1)
            self._od_costs_ptr = &output_view[0] if output_view.shape[0] else NULL
            od_costs.flags.writeable = False
        if od_turn_costs is not None:
            output_view = od_turn_costs.reshape(-1)
            self._od_turn_costs_ptr = &output_view[0] if output_view.shape[0] else NULL
            od_turn_costs.flags.writeable = False
        od_skims.flags.writeable = False

        # Mark inputs read-only only after validation and output allocation succeed.
        for array in arrays:
            array.flags.writeable = False
        self._fields = tuple(arrays)
        self._field_pointers.swap(pointers)
        self._od_skims = od_skims
        self._od_costs = od_costs
        self._od_turn_costs = od_turn_costs
        self.centroid_count = centroid_count
        self.field_count = width
        self.context = context

    @property
    def fields(self):
        """Read-only views of the prepared link fields, in input order."""
        return tuple(array.view() for array in self._fields)

    @property
    def od_skims(self):
        """Read-only (origin, destination, field) output; reused each iteration."""
        return self._od_skims.view()

    @property
    def od_costs(self):
        """Routing costs including penalties, or None if not requested."""
        return None if self._od_costs is None else self._od_costs.view()

    @property
    def od_turn_costs(self):
        """Only cumulative turn penalties, or None if not requested."""
        return None if self._od_turn_costs is None else self._od_turn_costs.view()
