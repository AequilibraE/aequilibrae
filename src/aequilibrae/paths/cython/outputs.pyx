"""Buffers produced by downstream operations on routing results."""

import operator

import numpy as np
cimport cython
from libc.math cimport isfinite
from libcpp.algorithm cimport copy_n

from aequilibrae.paths.cython.parallel_numpy cimport (
    assign_link_loads,
    linear_combination,
    triple_linear_combination,
)
from aequilibrae.utils.cython.array_allocations cimport array
from aequilibrae.utils.cython.array_allocations import readonly_view


cdef void _validate_loading_source(LoadingOutputs output, LoadingOutputs source) except *:
    if source.link_count != output.link_count or source.class_count != output.class_count:
        raise ValueError("source and output loading dimensions must match")


cdef void _validate_loading_cores(int cores) except *:
    if cores < 1:
        raise ValueError("cores must be positive")


cdef class LoadingOutputs:
    """Own fixed-size link loads, without demand, scratch or a graph reference.

    Loads start at zero. Loading accumulates into them; reset explicitly before
    another iteration. Read-only views keep storage alive and reflect reuse.

    Copy and blend methods replace this object's contents and return it. Sources
    must have matching dimensions, link/column ordering and units; only the
    dimensions can be checked here. No PCE conversion or summing is performed.
    The destination may also be a source. Callers must prevent concurrent reads
    or writes of the destination while an operation is running.

    Two-way weights must be finite and in [0, 1]. All operations require positive
    cores; a negative threading_threshold disables threading. Load values follow
    the numeric helpers' ordinary floating-point arithmetic, including NaN/inf.
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
        """Clear this iteration's buffers."""
        with nogil:
            self.view().reset()

    def copy_from(self, LoadingOutputs source not None):
        """Copy the data from source to self."""
        _validate_loading_source(self, source)

        cdef size_t count = self.link_count * self.class_count
        if count:
            copy_n(&source.link_loads_buffer[0, 0], count, &self.link_loads_buffer[0, 0])

        return self

    def copy_from_compact(
        self,
        LoadingOutputs source not None,
        const long long[::1] crosswalk,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """Copy the data from source to self projected through crosswalk
        """
        _validate_loading_cores(cores)

        if crosswalk.shape[0] != self.link_count:
            raise ValueError("crosswalk must have shape (link_count,)")
        elif self.class_count != source.class_count:
            raise ValueError("source and output must have the same number of classes")

        if self.link_count and self.class_count and source.link_count:
            with nogil:
                assign_link_loads[double](
                    self.link_loads_buffer,
                    source.link_loads_buffer,
                    crosswalk,
                    cores,
                    threading_threshold,
                )

        return self

    def blend_cfw(
        self,
        LoadingOutputs aon not None,
        LoadingOutputs previous_direction not None,
        double conjugate_weight,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """Replace loads with a conjugate Frank-Wolfe direction.

        self = conjugate_weight * previous_direction + (1 - conjugate_weight) * aon
        """

        # blend_result gives its weight to its first source. Rotating the CFW
        # sources makes that source the previous direction
        return self.blend_result(
            previous_direction,
            aon,
            conjugate_weight,
            cores=cores,
            threading_threshold=threading_threshold,
        )

    def blend_bfw(
        self,
        LoadingOutputs aon not None,
        LoadingOutputs previous_direction not None,
        LoadingOutputs older_direction not None,
        weights,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """Replace loads with a bi-conjugate Frank-Wolfe direction.

        self = weights[0] * aon + weights[1] * previous_direction + weights[2] * older_direction

        Weights must be three finite values in [0, 1], summing to one.
        """
        cdef const double[::1] coefficients

        _validate_loading_source(self, aon)
        _validate_loading_source(self, previous_direction)
        _validate_loading_source(self, older_direction)
        _validate_loading_cores(cores)

        values = np.array(weights, dtype=np.float64, order="C", copy=True)
        if values.shape != (3,):
            raise ValueError("BFW weights must contain exactly three values")
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError("BFW weights must be finite and between zero and one")
        if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("BFW weights must sum to one")
        coefficients = values

        with nogil:
            triple_linear_combination[double](
                self.link_loads_buffer,
                aon.link_loads_buffer,
                previous_direction.link_loads_buffer,
                older_direction.link_loads_buffer,
                coefficients,
                cores,
                threading_threshold,
            )
        return self

    def blend_result(
        self,
        LoadingOutputs direction not None,
        LoadingOutputs previous_result not None,
        double stepsize,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """Replace loads with the next accepted result (or a trial result).

        self = stepsize * direction + (1 - stepsize) * previous_result

        Pass self as previous_result to update the accepted result in place.
        """
        _validate_loading_source(self, direction)
        _validate_loading_source(self, previous_result)
        _validate_loading_cores(cores)
        if not isfinite(stepsize) or stepsize < 0.0 or stepsize > 1.0:
            raise ValueError("blend weight must be finite and between zero and one")

        with nogil:
            linear_combination[double](
                self.link_loads_buffer,
                direction.link_loads_buffer,
                previous_result.link_loads_buffer,
                stepsize,
                cores,
                threading_threshold,
            )
        return self

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


def _validate_selection_names(set_names):
    """Match names as well as sizes so sets cannot silently exchange outputs."""
    if isinstance(set_names, str):
        raise TypeError("set_names must be a sequence of names, not a string")
    names = tuple(set_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("selection names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("selection names must be unique")
    return names


cdef class SelectLinkLoadingOutputs:
    """Own [sets, links, classes] accumulators, without inputs or scratch.

    Loading adds to existing values. Reset once before processing the origins
    of an iteration. Each worker needs its own accumulator.
    """

    def __init__(self, link_count, class_count, set_names):
        if self.set_names is not None:
            raise RuntimeError("SelectLinkLoadingOutputs cannot be reinitialized")

        link_count, class_count = map(operator.index, (link_count, class_count))
        if link_count < 0 or class_count < 0:
            raise ValueError("link_count and class_count must be nonnegative")

        names = _validate_selection_names(set_names)
        self.link_count = link_count
        self.class_count = class_count
        self.set_count = len(names)
        self.link_loads_buffer = array[double]((self.set_count, link_count, class_count), True, 0)
        self.set_names = names

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef CppSelectLinkLoadingOutputsView[double] view(self) noexcept nogil:
        cdef CppSelectLinkLoadingOutputsView[double] output
        output.set_count = self.set_count
        output.link_count = self.link_count
        output.class_count = self.class_count

        if self.set_count and self.link_count and self.class_count:
            output.data = &self.link_loads_buffer[0, 0, 0]

        return output

    def reset(self):
        with nogil:
            self.view().reset()

    @property
    def link_loads(self):
        return readonly_view(self.link_loads_buffer)

    @property
    def loads(self):
        """Named, read-only [links, classes] views without copying data."""
        values = self.link_loads
        return {name: values[index] for index, name in enumerate(self.set_names)}


cdef class SelectLinkODOutputs:
    """Own matching demand in [origins, sets, destinations, classes] order.

    A call replaces one origin block. Workers may share this output only when
    writing different origin rows. Reset clears all rows, including skipped ones.
    """

    def __init__(self, origin_count, destination_count, class_count, set_names):
        if self.set_names is not None:
            raise RuntimeError("SelectLinkODOutputs cannot be reinitialized")

        origin_count, destination_count, class_count = map(
            operator.index, (origin_count, destination_count, class_count)
        )
        if origin_count < 0 or destination_count < 0 or class_count < 0:
            raise ValueError("origin_count, destination_count and class_count must be nonnegative")

        names = _validate_selection_names(set_names)
        self.origin_count = origin_count
        self.destination_count = destination_count
        self.class_count = class_count
        self.set_count = len(names)
        self.demand_buffer = array[double](
            (origin_count, self.set_count, destination_count, class_count), True, 0
        )
        self.set_names = names

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef CppSelectLinkODOutputsView[double] view(self) noexcept nogil:
        cdef CppSelectLinkODOutputsView[double] output
        output.origin_count = self.origin_count
        output.set_count = self.set_count
        output.destination_count = self.destination_count
        output.class_count = self.class_count

        if self.origin_count and self.set_count and self.destination_count and self.class_count:
            output.data = &self.demand_buffer[0, 0, 0, 0]

        return output

    def reset(self):
        with nogil:
            self.view().reset()

    @property
    def demand(self):
        return readonly_view(self.demand_buffer)

    @property
    def matrices(self):
        """Named [origins, destinations, classes] views; these may be strided."""
        values = self.demand
        return {name: values[:, index] for index, name in enumerate(self.set_names)}


cdef class SelectLinkOutputs:
    """Allocate optional loading and OD components without coupling their use.

    Each component can be passed to the operation on its own and can outlive
    this group. Disabling both leaves no numeric output allocations.
    """

    def __init__(self, link_count, destination_count, class_count, set_names, *,
                 origin_count=1, link_loads=True, od=True):
        if self.initialized:
            raise RuntimeError("SelectLinkOutputs cannot be reinitialized")

        link_count, destination_count, class_count, origin_count = map(
            operator.index, (link_count, destination_count, class_count, origin_count)
        )
        if min(link_count, destination_count, class_count, origin_count) < 0:
            raise ValueError("output dimensions must be nonnegative")

        names = _validate_selection_names(set_names)
        if link_loads:
            self.loading = SelectLinkLoadingOutputs(link_count, class_count, names)
        if od:
            self.od = SelectLinkODOutputs(origin_count, destination_count, class_count, names)

        self.initialized = True

    def reset(self):
        """Clear only the components allocated by this group."""
        if self.loading is not None:
            self.loading.reset()

        if self.od is not None:
            self.od.reset()


cdef class AoNOutputs:
    """Allocate the independent outputs needed by an assignment iteration.

    Components own their buffers and can outlive this group. Access arrays and
    names through the components; the group keeps no second copy of their shape.
    The demand-weighted turn total is a scalar, independent of skimming.
    """

    def __init__(
        self,
        links,
        zones,
        classes,
        *,
        skim_names=(),
        select_link_names=(),
        select_link_loads=True,
        select_link_od=True,
    ):
        if self.loading is not None:
            raise RuntimeError("AoNOutputs cannot be reinitialized")

        links, zones, classes = map(operator.index, (links, zones, classes))
        if min(links, zones, classes) < 0:
            raise ValueError("output dimensions must be nonnegative")

        skim_names = _validate_skim_names(skim_names)
        select_link_names = _validate_selection_names(select_link_names)

        self.loading = LoadingOutputs(links, classes)

        if skim_names:
            self.skimming = SkimmingOutputs(zones, zones, skim_names)

        if select_link_names and (select_link_loads or select_link_od):
            self.select_link = SelectLinkOutputs(
                links,
                zones,
                classes,
                select_link_names,
                origin_count=zones,
                link_loads=select_link_loads,
                od=select_link_od,
            )

    cdef CppAoNOutputsView view(self) noexcept nogil:
        cdef CppAoNOutputsView output

        output.loading = self.loading.view()

        if self.skimming is not None:
            output.skimming = self.skimming.view()

        if self.select_link is not None:
            if self.select_link.loading is not None:
                output.selected_loading = self.select_link.loading.view()
            if self.select_link.od is not None:
                output.selected_od = self.select_link.od.view()

        return output

    def reset(self):
        """Clear all components and the scalar without replacing any storage."""
        with nogil:
            self.view().reset()

        self.turn_cost_total = 0
