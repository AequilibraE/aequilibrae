"""Routing, skimming and select-link input contexts."""

import operator
from collections.abc import Mapping

import numpy as np
cimport cython

from libc.stddef cimport size_t

from aequilibrae.paths.cython.outputs import (
    SelectLinkOutputs, SkimmingOutputs, _validate_selection_names, _validate_skim_names,
)
from aequilibrae.utils.cython.array_allocations cimport array, const_array_pointer
from aequilibrae.utils.cython.array_allocations import readonly_view


def validate_index_array(value, name):
    """Copy node or link indices, rejecting negative and non-integer values."""
    index_array = np.asarray(value)
    if index_array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    if index_array.size:
        if index_array.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integers")

        if np.any(index_array < 0) or np.any(index_array > np.iinfo(np.uintp).max):
            raise ValueError(f"{name} contains an invalid index")

    return np.array(index_array, dtype=np.uintp, order="C", copy=True)


def validate_offsets(value, name, expected_end, end_description, expected_size=None):
    """Copy row offsets and check their length, order and endpoints."""
    offsets = validate_index_array(value, name)

    if expected_size is None:
        if offsets.size < 2:
            raise ValueError(f"{name} must describe at least one node")
    elif offsets.size != expected_size:
        raise ValueError(f"{name} must have {expected_size} entries")

    if offsets[0] != 0 or offsets[-1] != expected_end:
        raise ValueError(f"{name} must start at zero and end at {end_description}")

    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"{name} must be non-decreasing")
    return offsets


cdef class GraphContext:
    """Keep fixed topology and borrow the current routing costs.

    Use a concrete node or turn context. Costs must be contiguous float64
    buffers; binding them never copies or changes their writeability flags.
    """

    def __cinit__(self):
        self.node_offsets = None

    def __init__(self, fs, heads, costs, *, blocked_centroid_count=0):
        if type(self) is GraphContext:
            raise TypeError("GraphContext is an abstract base class")

        if self.node_offsets is not None:
            raise RuntimeError("routing contexts cannot be reinitialized")

        heads_array = validate_index_array(heads, "heads")
        fs_array = validate_offsets(fs, "fs", heads_array.size, "the number of links")
        if np.any(heads_array >= fs_array.size - 1):
            raise ValueError("heads contains an out-of-range node")

        blocked_centroid_count = operator.index(blocked_centroid_count)
        if not 0 <= blocked_centroid_count <= fs_array.size - 1:
            raise ValueError("blocked_centroid_count must be between zero and node_count")

        self.node_offsets = fs_array
        self.heads_buffer = heads_array
        self.blocked_centroid_count = blocked_centroid_count
        self.update_costs(costs)

    cpdef update_costs(self, object costs):
        """Bind a checked objective without copying it; failed checks keep the old one."""
        cdef const double[::1] values

        if costs is None:
            raise TypeError("costs must be a contiguous float64 buffer")

        values = costs
        if values.shape[0] != self.heads_buffer.shape[0]:
            raise ValueError("costs must have one value per link")

        data = np.asarray(values)
        if not data.flags.aligned:
            raise ValueError("costs must be aligned")
        if np.any(np.isnan(data)) or np.any(data < 0):
            raise ValueError("costs must be nonnegative and must not contain NaN")

        self.costs_buffer = values

    def with_costs(self, costs):
        """Share topology with a new context that can bind its own objective."""
        cdef GraphContext other = type(self).__new__(type(self))
        cdef TurnBasedContext source_turns, other_turns
        other.node_offsets = self.node_offsets
        other.heads_buffer = self.heads_buffer
        other.blocked_centroid_count = self.blocked_centroid_count
        other.update_costs(costs)

        if isinstance(self, TurnBasedContext):
            source_turns = <TurnBasedContext>self
            other_turns = <TurnBasedContext>other
            other_turns.tails_buffer = source_turns.tails_buffer
            other_turns.turn_offsets = source_turns.turn_offsets
            other_turns.turn_links = source_turns.turn_links
            other_turns.turn_penalties_buffer = source_turns.turn_penalties_buffer
            other_turns.uturns_allowed = source_turns.uturns_allowed
        return other

    cdef CppNodeBasedContext graph_view(self) noexcept nogil:
        """Build a C++ graph view that borrows this object's buffers."""
        cdef CppNodeBasedContext graph
        graph.fs = const_array_pointer(self.node_offsets)
        graph.heads = const_array_pointer(self.heads_buffer)
        graph.costs = const_array_pointer[double](self.costs_buffer)
        graph.blocked_centroid_count = self.blocked_centroid_count
        graph.node_count = self.node_offsets.shape[0] - 1
        graph.link_count = self.heads_buffer.shape[0]
        return graph

    @property
    def node_count(self):
        """Number of physical nodes."""
        return self.node_offsets.shape[0] - 1

    @property
    def link_count(self):
        """Number of directed links."""
        return self.heads_buffer.shape[0]

    @property
    def state_count(self):
        """Number of search states; one per node for node-based routing."""
        return self.node_count

    @property
    def fs(self):
        """Read-only offsets marking each node's outgoing links."""
        return readonly_view(self.node_offsets)

    @property
    def heads(self):
        """Read-only destination node for each directed link."""
        return readonly_view(self.heads_buffer)

    @property
    def costs(self):
        """Read-only cost for each directed link."""
        return readonly_view(self.costs_buffer)


cdef class NodeBasedContext(GraphContext):
    """Graph context whose search states are nodes."""

    cdef CppNodeBasedContext view(self) noexcept nogil:
        """Borrow the graph buffers for a node-based search."""
        return self.graph_view()


cdef class TurnBasedContext(GraphContext):
    """Graph context with sparse directed-link turn penalties.

    ``turn_fs`` has ``link_count + 1`` entries. Explicit turns are grouped by
    their incoming link; missing turns cost zero. Search states are incoming
    links plus a virtual root.
    """

    def __init__(
            self,
            fs,
            heads,
            costs,
            turn_fs=None,
            turn_to_links=None,
            turn_penalties=None,
            *,
            allow_uturns=True,
            blocked_centroid_count=0
    ):
        super().__init__(fs, heads, costs, blocked_centroid_count=blocked_centroid_count)

        n = self.node_count
        m = self.link_count

        if turn_fs is None and turn_to_links is None and turn_penalties is None:
            turn_fs = np.zeros(m + 1, dtype=np.uintp)
            turn_to_links = []
            turn_penalties = []
        elif turn_fs is None or turn_to_links is None or turn_penalties is None:
            raise ValueError("turn_fs, turn_to_links and turn_penalties must be supplied together")

        to_array = validate_index_array(turn_to_links, "turn_to_links")
        turn_fs_array = validate_offsets(turn_fs, "turn_fs", to_array.size, "the number of explicit turns", m + 1)
        turn_penalties = np.asarray(turn_penalties, dtype=np.float64, order="C")

        if turn_penalties.ndim != 1 or turn_penalties.size != to_array.size:
            raise ValueError("turn_penalties must be one-dimensional with one value per explicit turn")
        if np.any(to_array >= m):
            raise ValueError("turn_to_links contains an out-of-range link")
        if np.any(np.isnan(turn_penalties)) or np.any(turn_penalties < 0):
            raise ValueError("turn_penalties must be nonnegative and must not contain NaN")

        tails_array = np.repeat(np.arange(n, dtype=np.uintp), np.diff(self.node_offsets).astype(np.intp))
        from_links = np.repeat(np.arange(m, dtype=np.uintp), np.diff(turn_fs_array).astype(np.intp))
        same_row = from_links[1:] == from_links[:-1]

        if np.any(same_row & (to_array[1:] <= to_array[:-1])):
            raise ValueError("turn_to_links must be strictly increasing within each row")
        if np.any(np.asarray(self.heads_buffer)[from_links] != tails_array[to_array]):
            raise ValueError("explicit turns must connect consecutive links")

        self.tails_buffer = tails_array
        self.turn_offsets = turn_fs_array
        self.turn_links = to_array
        self.turn_penalties_buffer = array[double](turn_penalties.size, False, 0)

        np.copyto(np.asarray(self.turn_penalties_buffer), turn_penalties)
        self.uturns_allowed = bool(allow_uturns)

    cdef CppTurnBasedContext view(self) noexcept nogil:
        """Borrow the graph and turn buffers for a turn-based search."""
        cdef CppTurnBasedContext turns
        turns.graph = self.graph_view()
        turns.tails = const_array_pointer(self.tails_buffer)
        turns.turn_fs = const_array_pointer(self.turn_offsets)
        turns.turn_to_links = const_array_pointer(self.turn_links)
        turns.turn_penalties = const_array_pointer[double](self.turn_penalties_buffer)
        turns.allow_uturns = self.uturns_allowed
        return turns

    @property
    def state_count(self):
        """One search state per incoming link, plus the root."""
        return self.link_count + 1

    @property
    def tails(self):
        """Read-only starting node for each directed link."""
        return readonly_view(self.tails_buffer)

    @property
    def turn_fs(self):
        """Read-only offsets grouping turns by incoming link."""
        return readonly_view(self.turn_offsets)

    @property
    def turn_to_links(self):
        """Read-only outgoing link for each explicit turn."""
        return readonly_view(self.turn_links)

    @property
    def turn_penalties(self):
        """Read-only penalty for each explicit turn."""
        return readonly_view(self.turn_penalties_buffer)

    @property
    def allow_uturns(self):
        """Whether paths may turn back to the previous node."""
        return self.uturns_allowed


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


cdef class SelectLinkContext:
    """Copy named local link sets into owned membership masks.

    A path matches a set when it uses any member. Duplicate indices have no
    extra effect and empty sets match nothing. Names preserve mapping order.
    Masks are fixed after construction and may be shared across workers.
    """

    def __init__(self, link_count, selections):
        cdef cpp_bool[:, ::1] masks
        cdef size_t row, link_index
        if self.set_names is not None:
            raise RuntimeError("SelectLinkContext cannot be reinitialized")

        link_count = operator.index(link_count)
        if link_count < 0:
            raise ValueError("link_count must be nonnegative")
        if not isinstance(selections, Mapping):
            raise TypeError("selections must be a mapping of names to local link indices")

        names = _validate_selection_names(selections)
        masks = array[cpp_bool]((len(names), link_count), True, False)

        for row, name in enumerate(names):
            for member in selections[name]:
                if isinstance(member, (bool, np.bool_)):
                    raise TypeError("selected link indices must be integers, not booleans")

                link = operator.index(member)
                if not 0 <= link < link_count:
                    raise ValueError("selected link index must be in [0, link_count)")

                link_index = link
                masks[row, link_index] = True

        # Do not retain the index lists: no operation needs them after setup.
        self.link_count = link_count
        self.set_count = len(names)
        self.masks_buffer = masks
        self.set_names = names

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef CppSelectLinkContext view(self) noexcept nogil:
        cdef CppSelectLinkContext context
        context.link_count = self.link_count
        context.set_count = self.set_count

        if self.set_count and self.link_count:
            context.masks = &self.masks_buffer[0, 0]

        return context

    @property
    def masks(self):
        """Read-only [sets, links] view which keeps mask storage alive."""
        return readonly_view(self.masks_buffer)

    def make_outputs(self, destination_count, class_count, *, origin_count=1, link_loads=True, od=True):
        """Allocate either or both outputs; one origin row suffices for one shot."""
        return SelectLinkOutputs(
            self.link_count,
            destination_count,
            class_count,
            self.set_names,
            origin_count=origin_count,
            link_loads=link_loads,
            od=od,
        )
