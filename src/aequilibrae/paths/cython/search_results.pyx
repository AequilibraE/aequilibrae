import operator

cimport cython
import numpy as np

from libc.stddef cimport size_t
from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector

from aequilibrae.utils.cython.array_allocations cimport array as cython_array, array_pointer
from aequilibrae.utils.cython.array_allocations import readonly_view
from aequilibrae.paths.cython.graph_context cimport GraphContext


# Match the existing C++ kernel exactly, without reinterpreting signed arrays.
cdef size_t INVALID = <size_t>-1

cdef size_t validate_destination_count(object destination_count, size_t node_count) except *:
    """Validate the number of destination rows, defaulting to all nodes."""
    if destination_count is None:
        return node_count
    destination_count = operator.index(destination_count)
    if not 0 <= destination_count <= node_count:
        raise ValueError("destination_count must be between 0 and context.node_count")
    return destination_count


def prepare_output_array(out, node_count, field_count):
    """Allocate an output array if absent, otherwise check its shape and layout."""
    if out is None:
        return np.empty((node_count, field_count), dtype=np.float64)
    if not isinstance(out, np.ndarray):
        raise TypeError("out must be a NumPy array")
    if out.dtype != np.dtype(np.float64):
        raise TypeError("out must have dtype float64")
    if out.shape != (node_count, field_count):
        raise ValueError(f"out must have shape ({node_count}, {field_count})")
    if not out.flags.c_contiguous or not out.flags.aligned:
        raise ValueError("out must be aligned and C-contiguous")
    if not out.flags.writeable:
        raise ValueError("out must be writable")
    return out


cdef class SearchResults:
    """Store a search tree, path costs and reusable loading/skimming scratch.

    Arrays use search-state indices; terminal_states maps nodes to final states.
    Only reached_first[:settled_count] contains settled states. Read-only views
    keep their buffers alive and change when results are reused. Copy views to
    keep a snapshot, and use separate results for concurrent searches.
    """

    def __init__(self, context):
        if not isinstance(context, GraphContext):
            raise TypeError("context must be a GraphContext")
        n = context.node_count
        states = context.state_count
        if n == 0:
            raise ValueError("context must be initialised")

        self.context = context  # Strong reference pins the graph snapshot.
        self.node_count = n
        self.workspace = AoNWorkspace(context)
        self.prepared_skims = None

        self.predecessors_buffer = cython_array[size_t](states, True, INVALID)
        self.connectors_buffer = cython_array[size_t](states, True, INVALID)
        self.settled_states = cython_array[size_t](states, True, INVALID)
        self.terminal_states_buffer = cython_array[size_t](n, True, INVALID)

        self.distances_buffer = cython_array[double](states, True, np.inf)
        self.turn_costs_buffer = cython_array[double](states, True, np.inf)

        self.destination_mask_buffer = cython_array[cpp_bool](n, True, False)

        self.cpp.predecessors = array_pointer[size_t](self.predecessors_buffer)
        self.cpp.connectors = array_pointer[size_t](self.connectors_buffer)
        self.cpp.reached_first = array_pointer[size_t](self.settled_states)
        self.cpp.terminal_states = array_pointer[size_t](self.terminal_states_buffer)

        self.cpp.distances = array_pointer[double](self.distances_buffer)
        self.cpp.turn_costs = array_pointer[double](self.turn_costs_buffer)

        self.cpp.destination_mask = array_pointer[cpp_bool](self.destination_mask_buffer)
        self.cpp.root = INVALID
        self.cpp.origin = INVALID
        self.cpp.destination_count = 0
        self.cpp.reached_destination_count = 0
        self.cpp.settled_count = 0


    cdef void network_loading_nogil(self, const double *demand,
                                   size_t destination_count, size_t class_count,
                                   double *link_loads) noexcept nogil:
        """Add demand to link loads using prepared scratch, without input checks."""
        cpp_network_loading[double](self.cpp, destination_count, demand, class_count,
                                    self.workspace.cpp, link_loads)

    def network_loading(self, demand, link_loads):
        """Add demand along settled paths and return the supplied link loads.

        Accepts aligned, contiguous float64 demand [destinations, classes] and
        writable loads [links, classes], with no overlap. Skips unreachable and
        same-origin trips. Loads are not reset; use separate buffers per worker.
        """
        cdef const double[::1] demand_view
        cdef double[::1] loads_view
        cdef size_t rows, classes

        if not isinstance(demand, np.ndarray):
            raise TypeError("demand must be a NumPy array")
        if demand.dtype != np.dtype(np.float64):
            raise TypeError("demand must have dtype float64")
        if demand.ndim != 2 or demand.shape[0] > self.node_count:
            raise ValueError("demand must have shape (D, classes), D <= context.node_count")
        if not demand.flags.c_contiguous or not demand.flags.aligned:
            raise ValueError("demand must be aligned and C-contiguous")

        rows, classes = demand.shape
        # Unlike skimming, loading always requires a caller-owned accumulator.
        if link_loads is None:
            raise TypeError("link_loads must be a NumPy array")

        prepare_output_array(link_loads, self.context.link_count, classes)
        if np.shares_memory(demand, link_loads):
            raise ValueError("demand and link_loads must not overlap")

        self.workspace.prepare_loading(classes)
        scratch = self.workspace.state_loads_buffer

        if np.shares_memory(demand, scratch) or np.shares_memory(link_loads, scratch):
            raise ValueError("loading inputs and output must not overlap workspace scratch")

        demand_view = demand.reshape(-1)
        loads_view = link_loads.reshape(-1)

        cdef const double *input_ptr = &demand_view[0] if demand_view.shape[0] else NULL
        cdef double *output_ptr = &loads_view[0] if loads_view.shape[0] else NULL
        with nogil:
            self.network_loading_nogil(input_ptr, rows, classes, output_ptr)

        return link_loads

    cdef void skim_fields_nogil(
        self,
        const double *const *fields,
        size_t field_count,
        size_t destination_count,
        double *output
    ) noexcept nogil:
        """Sum link fields using prepared scratch, without input checks."""
        cpp_skim_fields[double](
            self.cpp, destination_count, fields, field_count, self.workspace.cpp, output
        )

    cdef void skim_costs_nogil(self, size_t destination_count, double *output) noexcept nogil:
        """Copy settled path costs to an output buffer without input checks."""
        cpp_skim_costs[double](self.cpp, destination_count, output)

    cdef void skim_turn_costs_nogil(self, size_t destination_count, double *output) noexcept nogil:
        """Copy settled turn costs to an output buffer without input checks."""
        cpp_skim_turn_costs[double](self.cpp, destination_count, output)

    cpdef prepare_skims(self, SkimmingContext fields):
        """Prepare skim scratch and check that it does not overlap the fields.

        Reuses scratch when the fields and width are unchanged. Requires the GIL.
        """
        if fields is None:
            raise TypeError("fields must be SkimmingContext")
        if fields.context is not self.context:
            raise ValueError("different context")
        if fields is self.prepared_skims and fields.field_count == self.workspace.cpp.skim_field_count:
            return

        self.workspace.prepare_skims(fields.field_count)
        scratch = self.workspace.state_skims_buffer

        for array in fields.link_fields:
            if np.shares_memory(array, scratch):
                raise ValueError("skim inputs must not overlap workspace scratch")

        # Retained fields cannot overlap a future, independently allocated scratch
        # buffer. Only a field change or a required resize needs preparation.
        self.prepared_skims = fields

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void skim_prepared_nogil(self, SkimmingContext fields) noexcept nogil:
        """Write this origin's prepared skim rows without further checks."""
        cdef size_t zones = fields.centroid_count
        cdef size_t origin = self.cpp.origin

        # Each worker borrows only its origin's row. Inputs/pointer tables are shared.
        if fields.field_count:
            self.skim_fields_nogil(fields.field_pointers.data(), fields.field_count, zones,
                                   &fields.od_skims_buffer[origin, 0, 0])

        if fields.od_costs_buffer is not None:
            self.skim_costs_nogil(zones, &fields.od_costs_buffer[origin, 0])

        if fields.od_turn_costs_buffer is not None:
            self.skim_turn_costs_nogil(zones, &fields.od_turn_costs_buffer[origin, 0])

    cdef object skim_prepared(self, SkimmingContext fields):
        """Check prepared fields, write this origin's skims and return the row."""
        if fields.context is not self.context:
            raise ValueError("different context")
        if self.cpp.settled_count == 0:
            raise ValueError("run a search before writing an OD row")
        if self.cpp.origin >= fields.centroid_count:
            raise ValueError("origin is outside the prepared centroid range")
        self.prepare_skims(fields)
        with nogil:
            self.skim_prepared_nogil(fields)
        return readonly_view(fields.od_skims_buffer[self.cpp.origin])

    def skim_fields(self, fields, out=None, *, destination_count=None):
        """Sum link fields along settled paths, without adding turn penalties.

        Pass float64 link vectors to fill out [destinations, fields], or a
        SkimmingContext to write its current origin row. Prepared fields supply
        their own output and destination count. Unreachable values are infinity;
        the searched origin is zero. This does not extend a partial search.
        """
        cdef const double[::1] field_view
        cdef double[::1] output_view
        cdef vector[const double *] pointers
        cdef size_t field_count, rows

        if isinstance(fields, SkimmingContext):
            if out is not None or destination_count is not None:
                raise ValueError("SkimmingContext already supplies the output and centroid count")
            return self.skim_prepared(fields)

        rows = validate_destination_count(destination_count, self.node_count)
        # Keep every allocation alive through the nogil call (also for generators).
        arrays = [np.asarray(field) for field in fields]
        field_count = len(arrays)

        pointers.reserve(field_count)
        for array in arrays:
            field_view = array  # Enforces dtype, dimensionality and contiguity.

            if field_view.shape[0] != self.context.link_count:
                raise ValueError("each skim field must have context.link_count entries")

            if not array.flags.aligned:
                raise ValueError("skim fields must be aligned")

            pointers.push_back(&field_view[0] if field_view.shape[0] else NULL)

        out = prepare_output_array(out, rows, field_count)
        output_view = out.reshape(-1)
        self.workspace.prepare_skims(field_count)
        scratch = self.workspace.state_skims

        if np.shares_memory(out, scratch) or any(np.shares_memory(array, scratch) for array in arrays):
            raise ValueError("skim inputs and output must not overlap workspace scratch")

        cdef double *output = &output_view[0] if output_view.shape[0] else NULL
        with nogil:
            self.skim_fields_nogil(pointers.data(), field_count, rows, output)

        return out

    def skim_costs(self, out=None, *, destination_count=None):
        """Write settled path costs, including turn penalties, to [destinations, 1].

        Defaults to all nodes and a new output array. Unreachable values are infinity.
        """
        cdef double[::1] output_view
        cdef size_t rows = validate_destination_count(destination_count, self.node_count)

        out = prepare_output_array(out, rows, 1)
        output_view = out.reshape(-1)

        cdef double *output = &output_view[0] if output_view.shape[0] else NULL
        with nogil:
            self.skim_costs_nogil(rows, output)

        return out

    def skim_turn_costs(self, out=None, *, destination_count=None):
        """Write turn costs alone, using the same output rules as skim_costs.

        Settled node-based paths have zero turn cost; unreachable values are infinity.
        """
        cdef double[::1] output_view
        cdef size_t rows = validate_destination_count(destination_count, self.node_count)
        out = prepare_output_array(out, rows, 1)
        output_view = out.reshape(-1)
        cdef double *output = &output_view[0] if output_view.shape[0] else NULL
        with nogil:
            self.skim_turn_costs_nogil(rows, output)
        return out

    @property
    def sentinel(self):
        """Index value used when no state or link is available."""
        return INVALID

    @property
    def predecessors(self):
        """Read-only parent state for each search state."""
        return readonly_view(self.predecessors_buffer)

    @property
    def connectors(self):
        """Read-only link used to reach each search state."""
        return readonly_view(self.connectors_buffer)

    @property
    def reached_first(self):
        """Read-only settlement order; only the first settled_count entries are valid."""
        return readonly_view(self.settled_states)

    @property
    def settled_count(self):
        """Number of states settled by the search."""
        return self.cpp.settled_count

    @property
    def state_count(self):
        """Number of search states supported by these buffers."""
        return self.predecessors_buffer.size

    @property
    def distances(self):
        """Read-only path cost for each state, including turn penalties."""
        return readonly_view(self.distances_buffer)

    @property
    def turn_costs(self):
        """Read-only turn cost along the path to each state."""
        return readonly_view(self.turn_costs_buffer)

    @property
    def terminal_states(self):
        """Read-only final state chosen for each physical node."""
        return readonly_view(self.terminal_states_buffer)

    @property
    def destination_mask(self):
        """Read-only boolean mask of physical nodes requested by the last search."""
        return readonly_view(self.destination_mask_buffer)

    @property
    def destinations(self):
        """Requested physical-node indices as a newly allocated array."""
        return np.flatnonzero(self.destination_mask_buffer).astype(np.uintp, copy=False)

    @property
    def destination_count(self):
        """Number of requested destinations."""
        return self.cpp.destination_count

    @property
    def reached_destination_count(self):
        """Number of requested destinations reached by the search."""
        return self.cpp.reached_destination_count

    @property
    def root(self):
        """Root search state, or None before a search."""
        return None if self.cpp.root == INVALID else self.cpp.root

    @property
    def origin(self):
        """Origin node, or None before a search."""
        return None if self.cpp.origin == INVALID else self.cpp.origin

    @property
    def destination(self):
        """The requested node for a single-target search, otherwise ``None``."""
        destinations = self.destinations
        return int(destinations[0]) if destinations.size == 1 else None

    @property
    def all_destinations_reached(self):
        """Whether every requested destination was reached."""
        return (self.cpp.settled_count > 0 and
                self.cpp.reached_destination_count == self.cpp.destination_count)

    @property
    def reachable(self):
        """Backward-compatible alias for ``all_destinations_reached``."""
        return self.all_destinations_reached

    def reachable_to(self, destination):
        """Whether a finalised path to ``destination`` is available."""
        cdef size_t destination_index = self.validate_destination(destination)
        return (self.cpp.settled_count > 0 and
                self.cpp.terminal_states[destination_index] != INVALID)

    def path_cost_to(self, destination):
        """Routing cost to one node, or infinity if it was not finalised."""
        cdef size_t destination_index = self.validate_destination(destination)
        cdef size_t terminal = self.cpp.terminal_states[destination_index]
        return np.inf if terminal == INVALID else self.cpp.distances[terminal]

    def path_turn_cost_to(self, destination):
        """Cumulative turn cost to one node, or infinity if not finalised."""
        cdef size_t destination_index = self.validate_destination(destination)
        cdef size_t terminal = self.cpp.terminal_states[destination_index]
        return np.inf if terminal == INVALID else self.cpp.turn_costs[terminal]

    def path_nodes_to(self, destination):
        """Reconstruct the finalised path to one physical node."""
        cdef size_t destination_index = self.validate_destination(destination)
        if self.cpp.terminal_states[destination_index] == INVALID:
            return np.empty(0, dtype=np.uintp)
        links = self.path_links_to(destination_index)
        return np.concatenate((np.array([self.cpp.origin], dtype=np.uintp),
                               self.context.heads[links]))

    def path_links_to(self, destination):
        """Reconstruct local link indices to one physical node."""
        cdef size_t destination_index = self.validate_destination(destination)
        cdef size_t state = self.cpp.terminal_states[destination_index]
        if state == INVALID:
            return np.empty(0, dtype=np.uintp)
        links = []
        while state != self.cpp.root:
            links.append(self.cpp.connectors[state])
            state = self.cpp.predecessors[state]
        return np.array(links[::-1], dtype=np.uintp)

    def validate_destination(self, destination):
        """Return a node index, rejecting values outside the graph."""
        destination = operator.index(destination)
        if not 0 <= destination < self.context.node_count:
            raise ValueError("destination is outside the context's node range")
        return destination

    def single_destination(self):
        """Return the sole destination, or None; reject multi-target searches."""
        if self.cpp.destination_count == 0:
            return None
        if self.cpp.destination_count != 1:
            raise ValueError("use the *_to(destination) methods for multi-target results")
        return self.destination

    @property
    def path_cost(self):
        """Path cost for the sole destination, or infinity if unreachable."""
        destination = self.single_destination()
        return np.inf if destination is None else self.path_cost_to(destination)

    @property
    def path_turn_cost(self):
        """Turn cost for the sole destination, or infinity if unreachable."""
        destination = self.single_destination()
        return np.inf if destination is None else self.path_turn_cost_to(destination)

    @property
    def path_nodes(self):
        """Nodes on the path to the sole destination, or an empty array."""
        destination = self.single_destination()
        return np.empty(0, dtype=np.uintp) if destination is None else self.path_nodes_to(destination)

    @property
    def path_links(self):
        """Links on the path to the sole destination, or an empty array."""
        destination = self.single_destination()
        return np.empty(0, dtype=np.uintp) if destination is None else self.path_links_to(destination)
