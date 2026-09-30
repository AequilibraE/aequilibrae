"""
Output buffers for routing related results.
"""

import logging
import operator

import numpy as np
cimport cython
from libc.math cimport isfinite
from libcpp.algorithm cimport copy_n

from aequilibrae.paths.cython.parallel_numpy cimport (
    project_link_loads,
    linear_combination,
    linear_combination_skims,
    triple_linear_combination,
    triple_linear_combination_skims,
)
from aequilibrae.utils.cython.array_allocations cimport array
from aequilibrae.utils.cython.array_allocations import readonly_view


logger = logging.getLogger(__name__)


cdef void _validate_loading_source(LoadingOutputs output, LoadingOutputs source) except *:
    if source.link_count != output.link_count or source.class_count != output.class_count:
        raise ValueError("source and output loading dimensions must match")


cdef void _validate_cores(int cores) except *:
    if cores < 1:
        raise ValueError("cores must be positive")


cdef void _validate_blend(double weight, int cores) except *:
    _validate_cores(cores)
    if not isfinite(weight) or weight < 0.0 or weight > 1.0:
        raise ValueError("blend weight must be finite and between zero and one")


cdef object _bfw_weights(weights, int cores):
    _validate_cores(cores)

    # A private copy gives every component in a grouped blend the same weights.
    values = np.array(weights, dtype=np.float64, order="C", copy=True)
    if values.shape != (3,):
        raise ValueError("BFW weights must contain exactly three values")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("BFW weights must be finite and between zero and one")
    if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("BFW weights must sum to one")
    return values


cdef void _warn_zero_weight_skim_blend(bint zero_weight) except *:
    if zero_weight:
        # Skims use infinity for unreachable and unwritten entries. A zero-weight
        # blend can yield NaN through floating-point arithmetic, so it logs a warning.
        logger.warning("zero weight blend: infinite skim values may produce NaN")


cdef void _validate_skimming_source(SkimmingOutputs output, SkimmingOutputs source) except *:
    if output.origin_count != source.origin_count or output.destination_count != source.destination_count:
        raise ValueError("source and output skimming dimensions must match")
    if output.field_names != source.field_names:
        raise ValueError("source and output skim field names and order must match")


cdef void _validate_selected_loading_source(
    SelectLinkLoadingOutputs output, SelectLinkLoadingOutputs source
) except *:
    if output.link_count != source.link_count or output.class_count != source.class_count:
        raise ValueError("source and output select-link loading dimensions must match")
    if output.set_names != source.set_names:
        raise ValueError("source and output selection names and order must match")


cdef void _validate_selected_od_source(SelectLinkODOutputs output, SelectLinkODOutputs source) except *:
    if (
        output.origin_count != source.origin_count
        or output.destination_count != source.destination_count
        or output.class_count != source.class_count
    ):
        raise ValueError("source and output select-link OD dimensions must match")
    if output.set_names != source.set_names:
        raise ValueError("source and output selection names and order must match")


cdef void _validate_selected_source(SelectLinkOutputs output, SelectLinkOutputs source) except *:
    if (output.loading is None) != (source.loading is None) or (output.od is None) != (source.od is None):
        raise ValueError("source and output select-link components must match")
    if output.loading is not None:
        _validate_selected_loading_source(output.loading, source.loading)
    if output.od is not None:
        _validate_selected_od_source(output.od, source.od)


cdef void _validate_aon_source(AoNOutputs output, AoNOutputs source) except *:
    if (
        (output.skimming is None) != (source.skimming is None)
        or (output.select_link is None) != (source.select_link is None)
    ):
        raise ValueError("source and output AoN components must match")
    _validate_loading_source(output.loading, source.loading)
    if output.skimming is not None:
        _validate_skimming_source(output.skimming, source.skimming)
    if output.select_link is not None:
        _validate_selected_source(output.select_link, source.select_link)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, :, ::1] _selected_od_blend_view(SelectLinkODOutputs output):
    """Fold adjacent origin and set axes without copying. The caller excludes empty buffers."""
    cdef Py_ssize_t rows = output.origin_count * output.set_count
    cdef Py_ssize_t destinations = output.destination_count
    cdef Py_ssize_t classes = output.class_count

    # Contiguous storage lets one 3D view distribute work across every origin
    # and selection set rather than loop over the 4th axis.
    return <double[:rows, :destinations, :classes]> &output.demand_buffer[0, 0, 0, 0]


cdef class LoadingOutputs:
    """Holds the link-by-class loads produced during an assignment.

    Routing adds demand to this table as it loads links. The table starts at
    zero and can be cleared for the next iteration, copied into another output,
    or combined with other compatible load tables. Its ``link_loads`` property
    provides a read-only view of the current values.
    """

    def __cinit__(self):
        self.link_loads_buffer = None

    def __init__(self, link_count, class_count):
        if self.link_loads_buffer is not None:
            raise RuntimeError("LoadingOutputs cannot be reinitialised")

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
        if count and source is not self:
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
        """Copy the data from source to self projected through crosswalk."""
        _validate_cores(cores)

        if crosswalk.shape[0] != self.link_count:
            raise ValueError("crosswalk must have shape (link_count,)")
        elif self.class_count != source.class_count:
            raise ValueError("source and output must have the same number of classes")

        # The source link count is the sentinel for a removed network link.
        indices = np.asarray(crosswalk)
        if np.any(indices < 0) or np.any(indices > source.link_count):
            raise ValueError("crosswalk contains an invalid compact link")
        if source is self:
            raise ValueError("projection requires distinct source and output storage")

        if self.link_count and self.class_count:
            with nogil:
                project_link_loads[double](
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

        # CFW gives the conjugate weight to the previous direction.
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
        coefficients = _bfw_weights(weights, cores)

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
        _validate_blend(stepsize, cores)

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
    """Holds the skim values calculated between origins and destinations.

    The values are arranged by origin, skim field, and destination, with one
    named matrix for each requested field. Routing fills the values for each
    origin, and resetting the output marks every entry as infinity until it is
    calculated again. The ``skims`` and ``matrices`` properties provide
    read-only views of the current results.
    """

    def __init__(self, origin_count, destination_count, field_names):
        if self.field_names is not None:
            raise RuntimeError("SkimmingOutputs cannot be reinitialised")

        origin_count = operator.index(origin_count)
        destination_count = operator.index(destination_count)
        if origin_count < 0 or destination_count < 0:
            raise ValueError("origin_count and destination_count must be nonnegative")

        names = _validate_skim_names(field_names)
        self.origin_count = origin_count
        self.destination_count = destination_count
        self.field_count = len(names)

        self.skims_buffer = array[double](
            (origin_count, self.field_count, destination_count), True, np.inf
        )
        self.field_names = names

    @classmethod
    def from_matrices(cls, dict matrices):
        """Create a SkimmingOutputs from a dictionary of matrices."""
        if not matrices:
            raise ValueError("Provide at least one skim matrix")

        arrays = []
        shape = None
        for matrix in matrices.values():
            matrix = np.asarray(matrix)
            if matrix.ndim != 2:
                raise ValueError(f"Skim matrices must be two-dimensional, got {matrix.ndim}")

            if shape is None:
                shape = matrix.shape

            if matrix.shape != shape:
                raise ValueError(f"Skim matrices must have the same shape, expected {shape}, got {matrix.shape}")

            arrays.append(matrix)

        cdef SkimmingOutputs output = cls(*shape, tuple(matrices.keys()))
        values = np.asarray(output.skims_buffer)

        for field, matrix in enumerate(arrays):
            np.copyto(values[:, field, :], matrix)

        return output

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef CppSkimmingOutputsView[double] view(self) noexcept nogil:
        cdef CppSkimmingOutputsView[double] output
        output.origin_count = self.origin_count
        output.destination_count = self.destination_count
        output.field_count = self.field_count

        # Can't obtain a pointer if there's no values
        if self.origin_count and self.destination_count and self.field_count:
            output.data = &self.skims_buffer[0, 0, 0]

        return output

    def reset(self):
        """Clear outputs."""
        with nogil:
            self.view().reset()

    def copy_from(self, SkimmingOutputs source not None):
        """Copy skims with matching dimensions and ordered field names."""
        _validate_skimming_source(self, source)

        cdef size_t count = self.origin_count * self.field_count * self.destination_count
        if count and source is not self:
            with nogil:
                copy_n(&source.skims_buffer[0, 0, 0], count, &self.skims_buffer[0, 0, 0])

        return self

    def blend_cfw(
        self,
        SkimmingOutputs aon not None,
        SkimmingOutputs previous_direction not None,
        double conjugate_weight,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """self = conjugate_weight * previous_direction + (1 - conjugate_weight) * aon."""
        return self.blend_result(
            previous_direction, aon, conjugate_weight,
            cores=cores, threading_threshold=threading_threshold,
        )

    def blend_bfw(
        self,
        SkimmingOutputs aon not None,
        SkimmingOutputs previous_direction not None,
        SkimmingOutputs older_direction not None,
        weights,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """self = weights[0] * aon + weights[1] * previous_direction + weights[2] * older_direction.

        Weights must be finite, nonnegative and sum to one. Zero value weights will produce NaNs
        for inaccessible zones.
        """
        _validate_skimming_source(self, aon)
        _validate_skimming_source(self, previous_direction)
        _validate_skimming_source(self, older_direction)
        cdef const double[::1] coefficients = _bfw_weights(weights, cores)
        _warn_zero_weight_skim_blend(
            coefficients[0] == 0.0 or coefficients[1] == 0.0 or coefficients[2] == 0.0
        )

        with nogil:
            triple_linear_combination_skims[double](
                self.skims_buffer, aon.skims_buffer,
                previous_direction.skims_buffer, older_direction.skims_buffer,
                coefficients, cores, threading_threshold,
            )
        return self

    def blend_result(
        self,
        SkimmingOutputs direction not None,
        SkimmingOutputs previous_result not None,
        double stepsize,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """self = stepsize * direction + (1 - stepsize) * previous_result.

        The weight must be finite and in [0, 1]. Zero value weights will produce NaNs
        for inaccessible zones.
        """
        _validate_skimming_source(self, direction)
        _validate_skimming_source(self, previous_result)
        _validate_blend(stepsize, cores)
        _warn_zero_weight_skim_blend(stepsize == 0.0 or stepsize == 1.0)

        with nogil:
            linear_combination_skims[double](
                self.skims_buffer, direction.skims_buffer, previous_result.skims_buffer,
                stepsize, cores, threading_threshold,
            )
        return self

    @property
    def skims(self):
        """Read-only, contiguous [origin rows, fields, destinations] array."""
        return readonly_view(self.skims_buffer)

    @property
    def matrices(self):
        """Named [origin rows, destinations] views, which may be non-contiguous."""
        skims = self.skims

        # These slices preserve the array's read-only storage for each matrix.
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
    """Holds selected-link loads for each named selection set, link, and class.

    As routes are loaded, this table accumulates the demand associated with each
    selected-link set. Reset it at the start of an assignment iteration to begin
    a fresh set of totals. The ``loads`` property presents the current values by
    selection name.
    """

    def __init__(self, link_count, class_count, set_names):
        if self.set_names is not None:
            raise RuntimeError("SelectLinkLoadingOutputs cannot be reinitialised")

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

    def copy_from(self, SelectLinkLoadingOutputs source not None):
        """Copy selected loads with matching dimensions and ordered set names."""
        _validate_selected_loading_source(self, source)

        cdef size_t count = self.set_count * self.link_count * self.class_count
        if count and source is not self:
            with nogil:
                copy_n(&source.link_loads_buffer[0, 0, 0], count, &self.link_loads_buffer[0, 0, 0])
        return self

    def blend_cfw(
        self,
        SelectLinkLoadingOutputs aon not None,
        SelectLinkLoadingOutputs previous_direction not None,
        double conjugate_weight,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """self = conjugate_weight * previous_direction + (1 - conjugate_weight) * aon."""
        return self.blend_result(
            previous_direction, aon, conjugate_weight,
            cores=cores, threading_threshold=threading_threshold,
        )

    def blend_bfw(
        self,
        SelectLinkLoadingOutputs aon not None,
        SelectLinkLoadingOutputs previous_direction not None,
        SelectLinkLoadingOutputs older_direction not None,
        weights,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """self = weights[0] * aon + weights[1] * previous_direction + weights[2] * older_direction.

        Weights must be finite, nonnegative and sum to one.
        """
        _validate_selected_loading_source(self, aon)
        _validate_selected_loading_source(self, previous_direction)
        _validate_selected_loading_source(self, older_direction)
        cdef const double[::1] coefficients = _bfw_weights(weights, cores)

        with nogil:
            triple_linear_combination_skims[double](
                self.link_loads_buffer, aon.link_loads_buffer,
                previous_direction.link_loads_buffer, older_direction.link_loads_buffer,
                coefficients, cores, threading_threshold,
            )
        return self

    def blend_result(
        self,
        SelectLinkLoadingOutputs direction not None,
        SelectLinkLoadingOutputs previous_result not None,
        double stepsize,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """self = stepsize * direction + (1 - stepsize) * previous_result.

        The weight must be finite and in [0, 1].
        """
        _validate_selected_loading_source(self, direction)
        _validate_selected_loading_source(self, previous_result)
        _validate_blend(stepsize, cores)

        with nogil:
            linear_combination_skims[double](
                self.link_loads_buffer, direction.link_loads_buffer, previous_result.link_loads_buffer,
                stepsize, cores, threading_threshold,
            )
        return self

    @property
    def loads(self):
        """Named, read-only [links, classes] read only views."""
        values = self.link_loads
        return {name: values[index] for index, name in enumerate(self.set_names)}


cdef class SelectLinkODOutputs:
    """Holds selected-link demand by origin, selection set, destination, and class.

    Each named selection set receives a demand matrix showing the trips whose
    routes meet that selection. Routing writes the results one origin at a time,
    and ``matrices`` provides the current matrix for each selection name.
    """

    def __init__(self, origin_count, destination_count, class_count, set_names):
        if self.set_names is not None:
            raise RuntimeError("SelectLinkODOutputs cannot be reinitialised")

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

    def copy_from(self, SelectLinkODOutputs source not None):
        """Copy selected demand with matching dimensions and ordered set names."""
        _validate_selected_od_source(self, source)

        cdef size_t count = self.origin_count * self.set_count * self.destination_count * self.class_count
        if count and source is not self:
            with nogil:
                copy_n(&source.demand_buffer[0, 0, 0, 0], count, &self.demand_buffer[0, 0, 0, 0])
        return self

    def blend_cfw(
        self,
        SelectLinkODOutputs aon not None,
        SelectLinkODOutputs previous_direction not None,
        double conjugate_weight,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """self = conjugate_weight * previous_direction + (1 - conjugate_weight) * aon."""
        return self.blend_result(
            previous_direction, aon, conjugate_weight,
            cores=cores, threading_threshold=threading_threshold,
        )

    def blend_bfw(
        self,
        SelectLinkODOutputs aon not None,
        SelectLinkODOutputs previous_direction not None,
        SelectLinkODOutputs older_direction not None,
        weights,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """self = weights[0] * aon + weights[1] * previous_direction + weights[2] * older_direction.

        Weights must be finite, nonnegative and sum to one.
        """
        _validate_selected_od_source(self, aon)
        _validate_selected_od_source(self, previous_direction)
        _validate_selected_od_source(self, older_direction)
        cdef const double[::1] coefficients = _bfw_weights(weights, cores)
        cdef double[:, :, ::1] target, first, second, third

        if self.origin_count and self.set_count and self.destination_count and self.class_count:
            target = _selected_od_blend_view(self)
            first = _selected_od_blend_view(aon)
            second = _selected_od_blend_view(previous_direction)
            third = _selected_od_blend_view(older_direction)
            with nogil:
                triple_linear_combination_skims[double](
                    target, first, second, third, coefficients, cores, threading_threshold,
                )
        return self

    def blend_result(
        self,
        SelectLinkODOutputs direction not None,
        SelectLinkODOutputs previous_result not None,
        double stepsize,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """self = stepsize * direction + (1 - stepsize) * previous_result.

        The weight must be finite and in [0, 1].
        """
        _validate_selected_od_source(self, direction)
        _validate_selected_od_source(self, previous_result)
        _validate_blend(stepsize, cores)
        cdef double[:, :, ::1] target, first, second

        if self.origin_count and self.set_count and self.destination_count and self.class_count:
            target = _selected_od_blend_view(self)
            first = _selected_od_blend_view(direction)
            second = _selected_od_blend_view(previous_result)
            with nogil:
                linear_combination_skims[double](
                    target, first, second, stepsize, cores, threading_threshold,
                )
        return self

    @property
    def demand(self):
        return readonly_view(self.demand_buffer)

    @property
    def matrices(self):
        """Named [origins, destinations, classes] views. These may be strided."""
        values = self.demand
        return {name: values[:, index] for index, name in enumerate(self.set_names)}


cdef class SelectLinkOutputs:
    """Groups the selected-link load and origin-destination outputs for an assignment.

    It creates the requested result tables for a shared set of selection names
    and lets them be reset, copied, or blended together. The loading component
    records selected-link volumes, while the OD component records selected-trip
    demand.
    """

    def __init__(self, link_count, destination_count, class_count, set_names, *,
                 origin_count=1, link_loads=True, od=True):
        if self.initialised:
            raise RuntimeError("SelectLinkOutputs cannot be reinitialised")

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

        self.initialised = True

    def reset(self):
        """Clear only the components allocated by this group."""
        if self.loading is not None:
            self.loading.reset()

        if self.od is not None:
            self.od.reset()

    def copy_from(self, SelectLinkOutputs source not None):
        """Copy matching enabled components, checking the entire group first."""
        _validate_selected_source(self, source)
        if self.loading is not None:
            self.loading.copy_from(source.loading)
        if self.od is not None:
            self.od.copy_from(source.od)
        return self

    def blend_cfw(
        self,
        SelectLinkOutputs aon not None,
        SelectLinkOutputs previous_direction not None,
        double conjugate_weight,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """self = conjugate_weight * previous_direction + (1 - conjugate_weight) * aon."""
        _validate_selected_source(self, aon)
        _validate_selected_source(self, previous_direction)
        _validate_blend(conjugate_weight, cores)

        if self.loading is not None:
            self.loading.blend_cfw(
                aon.loading, previous_direction.loading, conjugate_weight,
                cores=cores, threading_threshold=threading_threshold,
            )
        if self.od is not None:
            self.od.blend_cfw(
                aon.od, previous_direction.od, conjugate_weight,
                cores=cores, threading_threshold=threading_threshold,
            )
        return self

    def blend_bfw(
        self,
        SelectLinkOutputs aon not None,
        SelectLinkOutputs previous_direction not None,
        SelectLinkOutputs older_direction not None,
        weights,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """Blend each component with three finite, nonnegative weights summing to one.

        self = weights[0] * aon + weights[1] * previous_direction + weights[2] * older_direction
        """
        _validate_selected_source(self, aon)
        _validate_selected_source(self, previous_direction)
        _validate_selected_source(self, older_direction)
        coefficients = _bfw_weights(weights, cores)

        if self.loading is not None:
            self.loading.blend_bfw(
                aon.loading, previous_direction.loading, older_direction.loading, coefficients,
                cores=cores, threading_threshold=threading_threshold,
            )
        if self.od is not None:
            self.od.blend_bfw(
                aon.od, previous_direction.od, older_direction.od, coefficients,
                cores=cores, threading_threshold=threading_threshold,
            )
        return self

    def blend_result(
        self,
        SelectLinkOutputs direction not None,
        SelectLinkOutputs previous_result not None,
        double stepsize,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """self = stepsize * direction + (1 - stepsize) * previous_result.

        The weight must be finite and in [0, 1].
        """
        _validate_selected_source(self, direction)
        _validate_selected_source(self, previous_result)
        _validate_blend(stepsize, cores)

        if self.loading is not None:
            self.loading.blend_result(
                direction.loading, previous_result.loading, stepsize,
                cores=cores, threading_threshold=threading_threshold,
            )
        if self.od is not None:
            self.od.blend_result(
                direction.od, previous_result.od, stepsize,
                cores=cores, threading_threshold=threading_threshold,
            )
        return self


cdef class AoNOutputs:
    """Collects the results produced by an all-or-nothing assignment.

    It always contains link loads and may also contain skim matrices and
    selected-link results. The object keeps these results together while an
    assignment is reset, copied, or blended, and stores the demand-weighted
    total turn cost alongside them.
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
            raise RuntimeError("AoNOutputs cannot be reinitialised")

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
        """Clear output."""
        with nogil:
            self.view().reset()

        self.turn_cost_total = 0
        self.unassigned_demand = 0

    def copy_from(self, AoNOutputs source not None):
        """Copy matching components and the turn total."""
        _validate_aon_source(self, source)

        self.loading.copy_from(source.loading)

        if self.skimming is not None:
            self.skimming.copy_from(source.skimming)

        if self.select_link is not None:
            self.select_link.copy_from(source.select_link)

        self.turn_cost_total = source.turn_cost_total
        self.unassigned_demand = source.unassigned_demand

        return self

    def blend_cfw(
        self,
        AoNOutputs aon not None,
        AoNOutputs previous_direction not None,
        double conjugate_weight,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """self = conjugate_weight * previous_direction + (1 - conjugate_weight) * aon.

        Turn cost uses the same weights as every array component.
        """
        _validate_aon_source(self, aon)
        _validate_aon_source(self, previous_direction)
        _validate_blend(conjugate_weight, cores)

        self.loading.blend_cfw(
            aon.loading, previous_direction.loading, conjugate_weight,
            cores=cores, threading_threshold=threading_threshold,
        )
        if self.skimming is not None:
            self.skimming.blend_cfw(
                aon.skimming, previous_direction.skimming, conjugate_weight,
                cores=cores, threading_threshold=threading_threshold,
            )
        if self.select_link is not None:
            self.select_link.blend_cfw(
                aon.select_link, previous_direction.select_link, conjugate_weight,
                cores=cores, threading_threshold=threading_threshold,
            )
        self.turn_cost_total = (
            conjugate_weight * previous_direction.turn_cost_total
            + (1.0 - conjugate_weight) * aon.turn_cost_total
        )
        self.unassigned_demand = (
            conjugate_weight * previous_direction.unassigned_demand
            + (1.0 - conjugate_weight) * aon.unassigned_demand
        )
        return self

    def blend_bfw(
        self,
        AoNOutputs aon not None,
        AoNOutputs previous_direction not None,
        AoNOutputs older_direction not None,
        weights,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """Blend components and turn cost with finite, nonnegative weights summing to one.

        self = weights[0] * aon + weights[1] * previous_direction + weights[2] * older_direction
        """
        _validate_aon_source(self, aon)
        _validate_aon_source(self, previous_direction)
        _validate_aon_source(self, older_direction)
        coefficients = _bfw_weights(weights, cores)

        self.loading.blend_bfw(
            aon.loading, previous_direction.loading, older_direction.loading, coefficients,
            cores=cores, threading_threshold=threading_threshold,
        )
        if self.skimming is not None:
            self.skimming.blend_bfw(
                aon.skimming, previous_direction.skimming, older_direction.skimming, coefficients,
                cores=cores, threading_threshold=threading_threshold,
            )
        if self.select_link is not None:
            self.select_link.blend_bfw(
                aon.select_link, previous_direction.select_link, older_direction.select_link, coefficients,
                cores=cores, threading_threshold=threading_threshold,
            )
        self.turn_cost_total = (
            coefficients[0] * aon.turn_cost_total
            + coefficients[1] * previous_direction.turn_cost_total
            + coefficients[2] * older_direction.turn_cost_total
        )
        self.unassigned_demand = (
            coefficients[0] * aon.unassigned_demand
            + coefficients[1] * previous_direction.unassigned_demand
            + coefficients[2] * older_direction.unassigned_demand
        )
        return self

    def blend_result(
        self,
        AoNOutputs direction not None,
        AoNOutputs previous_result not None,
        double stepsize,
        *,
        int cores=1,
        Py_ssize_t threading_threshold=10000,
    ):
        """self = stepsize * direction + (1 - stepsize) * previous_result.

        Applies to every component and the turn total. The weight must be
        finite and in [0, 1]. Pass self as previous_result for an in-place update.
        """
        _validate_aon_source(self, direction)
        _validate_aon_source(self, previous_result)
        _validate_blend(stepsize, cores)

        self.loading.blend_result(
            direction.loading, previous_result.loading, stepsize,
            cores=cores, threading_threshold=threading_threshold,
        )
        if self.skimming is not None:
            self.skimming.blend_result(
                direction.skimming, previous_result.skimming, stepsize,
                cores=cores, threading_threshold=threading_threshold,
            )
        if self.select_link is not None:
            self.select_link.blend_result(
                direction.select_link, previous_result.select_link, stepsize,
                cores=cores, threading_threshold=threading_threshold,
            )
        self.turn_cost_total = stepsize * direction.turn_cost_total + (1.0 - stepsize) * previous_result.turn_cost_total
        self.unassigned_demand = (
            stepsize * direction.unassigned_demand + (1.0 - stepsize) * previous_result.unassigned_demand
        )
        return self
