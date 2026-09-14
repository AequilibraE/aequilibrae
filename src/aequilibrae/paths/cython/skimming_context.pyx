"""Named skim inputs, independent of routing contexts and output storage."""

import operator
from collections.abc import Mapping

import numpy as np

from aequilibrae.paths.cython.outputs import SkimmingOutputs, _validate_skim_names
from aequilibrae.utils.cython.array_allocations cimport const_array_pointer
from aequilibrae.utils.cython.array_allocations import readonly_view


cdef class SkimmingContext:
    """Retain field meanings and borrow additive link buffers without copying.

    Output order is link_fields, link_fields_with_turn_costs, cost_name, then
    turn_cost_name. Mappings preserve their insertion order; names must be
    unique across all four groups. Label fields need no supplied array.

    Link buffers must be aligned, contiguous float64 vectors in local link
    order. Values may change between calls, but not during a call. Binding
    does not change the caller's writeability flags or follow later routing
    cost rebindings. No graph, result, output or workspace is retained.
    """

    def __init__(
        self,
        link_count,
        *,
        link_fields=None,
        link_fields_with_turn_costs=None,
        cost_name=None,
        turn_cost_name=None,
    ):
        cdef const double[::1] values
        cdef vector[const double *] pointers
        cdef size_t plain_field_count = 0
        cdef CppSkimmingContext[double] configuration

        if self.field_names is not None:
            raise RuntimeError("SkimmingContext cannot be reinitialized")

        link_count = operator.index(link_count)
        if link_count < 0:
            raise ValueError("link_count must be nonnegative")

        names = []
        buffers = []

        # Keep the two additive groups together. Each projection can then read
        # a range of fields without checking individual field types.
        for group_index, group in enumerate((link_fields, link_fields_with_turn_costs)):
            if group is None:
                continue

            if not isinstance(group, Mapping):
                raise TypeError("link fields must be mappings of names to buffers")

            for name, buffer in group.items():
                # Require a usable buffer rather than silently copying a list
                # or converting its dtype. The typed view checks the layout.
                data = np.asarray(memoryview(buffer))
                values = data

                if values.shape[0] != link_count:
                    raise ValueError("each skim field must have link_count entries")
                if not data.flags.aligned:
                    raise ValueError("skim fields must be aligned")

                names.append(name)
                buffers.append(values)
                pointers.push_back(const_array_pointer(values))

            if group_index == 0:
                plain_field_count = len(buffers)

        # Label fields name outputs only; their values come from the search.
        for name in (cost_name, turn_cost_name):
            if name is not None:
                names.append(name)

        names = _validate_skim_names(names)

        self.link_count = link_count
        self.field_count = len(names)
        self.additive_field_count = len(buffers)

        # The pointer table cannot keep buffers alive on its own. Retain their
        # memoryviews too, and never resize the table while callers use it.
        self.field_buffers = tuple(buffers)
        self.field_pointers.swap(pointers)

        configuration.link_count = self.link_count
        configuration.field_count = self.field_count
        configuration.additive_field_count = self.additive_field_count
        configuration.link_fields = self.field_pointers.data()

        # Scratch and output use the same field order for the additive groups.
        configuration.plain_field_count = plain_field_count
        configuration.turn_field_offset = plain_field_count
        configuration.turn_field_count = self.additive_field_count - plain_field_count

        # Work out label positions once, not while processing each origin.
        # Each label contributes either one matrix or none.
        configuration.cost_field_count = cost_name is not None
        configuration.turn_cost_field_count = turn_cost_name is not None
        configuration.cost_field_index = self.additive_field_count
        configuration.turn_cost_field_index = (
            self.additive_field_count + configuration.cost_field_count
        )

        self.configuration = configuration
        self.field_names = names

    cdef CppSkimmingContext[double] view(self) noexcept nogil:
        """Borrow the prepared layout; the caller must keep this owner alive."""
        return self.configuration

    @property
    def fields(self):
        """Read-only views of the supplied link buffers, keyed by field name."""
        return {
            name: readonly_view(buffer)
            for name, buffer in zip(self.field_names, self.field_buffers)
        }

    def make_outputs(self, destination_count, *, origin_count=1):
        """Allocate matching outputs; one row suffices for a one-shot search.

        The output copies names and dimensions, not a reference to this owner.
        Assignment can request more origin rows without changing field inputs.
        """
        return SkimmingOutputs(origin_count, destination_count, self.field_names)
