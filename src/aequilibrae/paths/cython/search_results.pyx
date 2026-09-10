import operator

import numpy as np

from libc.stddef cimport size_t
from libcpp.vector cimport vector

from aequilibrae.paths.cython.graph_context cimport GraphContext


# Match the existing C++ kernel exactly, without reinterpreting signed arrays.
cdef size_t INVALID = <size_t>-1


ctypedef unsigned char uchar

ctypedef fused ResultArrayElement:
    size_t
    uchar
    double


cdef ResultArrayElement *_array_pointer(ResultArrayElement[::1] array) except+:
    """Return a typed pointer after Cython validates a contiguous memoryview."""
    return &array[0] if array.shape[0] else NULL


cdef size_t _skim_destination_count(object destination_count, size_t node_count) except *:
    if destination_count is None:
        return node_count
    destination_count = operator.index(destination_count)
    if not 0 <= destination_count <= node_count:
        raise ValueError("destination_count must be between 0 and context.node_count")
    return destination_count


def _skim_output(out, node_count, field_count):
    """Validate without copying: kernels only accept packed row-major buffers."""
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
    """One search's results, allocated with the GIL and populated without it.

    Array properties are read-only, zero-copy views. A retained view keeps its
    NumPy allocation alive even after this wrapper is deleted. Reusing results
    overwrites those same buffers: use array.copy() for a historical snapshot.

    Do not read retained views while a search is writing this object, or force
    their underlying allocations writable/resized. Concurrent searches require
    separate results objects; they may share the same context.

    Predecessors, connectors, reached_first, distances and turn_costs are indexed
    by search state, not necessarily physical node. terminal_states maps physical
    nodes to selected final states, while destination_mask records the requested
    physical nodes. The root has no predecessor or connector.
    Only reached_first[:settled_count] belongs to the finalized search tree.
    Unfinalized states have sentinel predecessors/connectors and infinite costs.
    """

    def __init__(self, context):
        if not isinstance(context, GraphContext):
            raise TypeError("context must be a GraphContext")
        n = context.node_count
        states = context.state_count
        if n == 0:
            raise ValueError("context must be initialised")

        predecessors = np.full(states, INVALID, dtype=np.uintp)
        connectors = np.full(states, INVALID, dtype=np.uintp)
        reached_first = np.full(states, INVALID, dtype=np.uintp)
        destination_mask = np.zeros(n, dtype=np.uint8)
        distances = np.full(states, np.inf, dtype=np.float64)
        turn_costs = np.full(states, np.inf, dtype=np.float64)
        terminal_states = np.full(n, INVALID, dtype=np.uintp)

        self.context = context  # Strong reference pins the graph snapshot.
        self._node_count = n
        self.workspace = AoNWorkspace(context)
        self._prepared_skims = None
        self._prepared_workspace = None
        self._predecessors = predecessors
        self._connectors = connectors
        self._reached_first = reached_first
        self._destination_mask = destination_mask
        self._distances = distances
        self._turn_costs = turn_costs
        self._terminal_states = terminal_states

        self.cpp.predecessors = _array_pointer[size_t](self._predecessors)
        self.cpp.connectors = _array_pointer[size_t](self._connectors)
        self.cpp.reached_first = _array_pointer[size_t](self._reached_first)
        self.cpp.destination_mask = _array_pointer[uchar](self._destination_mask)
        self.cpp.distances = _array_pointer[double](self._distances)
        self.cpp.turn_costs = _array_pointer[double](self._turn_costs)
        self.cpp.terminal_states = _array_pointer[size_t](self._terminal_states)
        self.cpp.root = INVALID
        self.cpp.origin = INVALID
        self.cpp.destination_count = 0
        self.cpp.reached_destination_count = 0
        self.cpp.settled_count = 0

        for array in (
            predecessors,
            connectors,
            reached_first,
            destination_mask,
            distances,
            turn_costs,
            terminal_states,
        ):
            array.flags.writeable = False

    cdef void network_loading_nogil(self, const double *demand,
                                   size_t destination_count, size_t class_count,
                                   double *link_loads) noexcept nogil:
        cpp_network_loading[double](self.cpp, destination_count, demand, class_count,
                                    self.workspace.cpp, link_loads)

    def network_loading(self, demand, link_loads):
        """Accumulate one origin's demand into caller-owned link loads.

        demand must be aligned, C-contiguous float64 [D, classes], with rows
        corresponding to nodes 0..D-1 (D <= context.node_count). Read-only inputs
        are accepted. link_loads must be writable, aligned, C-contiguous float64
        [context.link_count, classes]. No input copies or output allocations.

        Unreachable/unfinalized destinations and intrazonal demand are ignored;
        a pre-search result loads nothing. This does not extend a partial search.
        NaN/inf demand propagates through ordinary addition on its path.
        Scratch is reset per call; link_loads is NOT reset. Returns link_loads.

        For parallel AoN, give each worker separate results and a disjoint slice
        of a zeroed [threads, links, classes] array. Accumulate origins, then use
        array.sum(axis=0) after all workers finish. Clear it for a new iteration.
        Do not concurrently mutate inputs, search, skim, load, or resize this
        workspace; do not force internal search/context/scratch arrays writable.
        """
        cdef const double[::1] demand_view
        cdef double[::1] loads_view
        cdef size_t rows, classes
        if not isinstance(demand, np.ndarray):
            raise TypeError("demand must be a NumPy array")
        if demand.dtype != np.dtype(np.float64):
            raise TypeError("demand must have dtype float64")
        if demand.ndim != 2 or demand.shape[0] > self._node_count:
            raise ValueError("demand must have shape (D, classes), D <= context.node_count")
        if not demand.flags.c_contiguous or not demand.flags.aligned:
            raise ValueError("demand must be aligned and C-contiguous")
        rows, classes = demand.shape
        # Unlike skimming, loading always requires a caller-owned accumulator.
        if link_loads is None:
            raise TypeError("link_loads must be a NumPy array")
        _skim_output(link_loads, self.context.link_count, classes)
        if np.shares_memory(demand, link_loads):
            raise ValueError("demand and link_loads must not overlap")
        self.workspace.prepare_loading(classes)
        scratch = self.workspace._state_loads
        if np.shares_memory(demand, scratch) or np.shares_memory(link_loads, scratch):
            raise ValueError("loading inputs and output must not overlap workspace scratch")
        demand_view = demand.reshape(-1)
        loads_view = link_loads.reshape(-1)
        cdef const double *input_ptr = &demand_view[0] if demand_view.shape[0] else NULL
        cdef double *output_ptr = &loads_view[0] if loads_view.shape[0] else NULL
        with nogil:
            self.network_loading_nogil(input_ptr, rows, classes, output_ptr)
        return link_loads

    cdef void skim_fields_nogil(self, const double *const *fields,
                               size_t field_count, size_t destination_count,
                               double *output) noexcept nogil:
        cpp_skim_fields[double](self.cpp, destination_count, fields, field_count,
                                self.workspace.cpp, output)

    cdef void skim_costs_nogil(self, size_t destination_count, double *output) noexcept nogil:
        cpp_skim_costs[double](self.cpp, destination_count, output)

    cdef void skim_turn_costs_nogil(self, size_t destination_count, double *output) noexcept nogil:
        cpp_skim_turn_costs[double](self.cpp, destination_count, output)

    cpdef prepare_skims(self, SkimmingContext fields):
        """Prepare this worker's scratch for a SkimmingContext (requires GIL).

        Repeated calls with the same fields and scratch allocation do no work.
        skim_fields calls this automatically; Cython callers can call it once
        before a loop of searches and skim_prepared_nogil calls.
        """
        if fields is None:
            raise TypeError("fields must be SkimmingContext")
        if fields.context is not self.context:
            raise ValueError("different context")
        if fields is self._prepared_skims and self.workspace._state_skims is self._prepared_workspace:
            return
        self.workspace.prepare_skims(fields.field_count)
        scratch = self.workspace._state_skims
        for array in fields._fields:
            if np.shares_memory(array, scratch):
                raise ValueError("skim inputs must not overlap workspace scratch")
        # Cache this check. It is only repeated if fields or scratch allocation change.
        self._prepared_skims = fields
        self._prepared_workspace = scratch

    cdef void skim_prepared_nogil(self, SkimmingContext fields) noexcept nogil:
        cdef size_t z = fields.centroid_count
        cdef size_t row_offset = self.cpp.origin * z
        # Each worker writes only its origin's row. Inputs and pointer tables are shared.
        if fields.field_count:
            self.skim_fields_nogil(fields._field_pointers.data(), fields.field_count, z,
                                   fields._od_skims_ptr + row_offset * fields.field_count)
        if fields._od_costs_ptr != NULL:
            self.skim_costs_nogil(z, fields._od_costs_ptr + row_offset)
        if fields._od_turn_costs_ptr != NULL:
            self.skim_turn_costs_nogil(z, fields._od_turn_costs_ptr + row_offset)

    cdef object _skim_prepared(self, SkimmingContext fields):
        if fields.context is not self.context:
            raise ValueError("different context")
        if self.cpp.settled_count == 0:
            raise ValueError("run a search before writing an OD row")
        if self.cpp.origin >= fields.centroid_count:
            raise ValueError("origin is outside the prepared centroid range")
        self.prepare_skims(fields)
        with nogil:
            self.skim_prepared_nogil(fields)
        return fields._od_skims[self.cpp.origin]

    def skim_fields(self, fields, out=None, *, destination_count=None):
        """Sum arbitrary link fields along each finalized node's selected path.

        Pass a SkimmingContext to use prepared inputs and write the current
        origin's row into its OD arrays, including any requested costs and turn
        penalties. Returns a read-only view of that origin's od_skims row. The
        prepared object supplies the output and centroid count; do not pass out
        or destination_count. It must use the same context, and a search must
        have run from an origin in its centroid range. No field validation or
        pointer-table allocation is repeated on this path.

        Alternatively, ``fields`` is a sequence of contiguous float64 vectors, each with
        context.link_count entries in context.costs order. Read-only inputs are
        accepted. No penalties are added to these fields, including a field
        containing context.costs; use skim_costs for costs WITH turn penalties.
        Negative/NaN/infinite attributes use ordinary floating-point addition.

        Returns (and optionally writes into) a C-contiguous float64 array shaped
        (destination_count, len(fields)). Rows are nodes 0 to destination_count-1.
        None selects all context nodes. The count must be between zero and
        context.node_count; it is independent of the search's destination mask.
        The origin is zero if included; unfinalized/unreachable nodes are infinite.
        Before any search, all entries are infinite.

        For centroids at nodes 0 to z-1, search with destinations=range(z), then
        pass destination_count=z and out=cube[origin] for a (z, z, F) OD cube.
        All settled states are still summed, including intermediate network nodes.
        Partial searches only skim finalized paths; this method does not search.

        The reusable workspace retains state-level sums in workspace.state_skims.
        Do not concurrently search, skim, resize workspace, or mutate inputs.
        """
        cdef const double[::1] field_view
        cdef double[::1] output_view
        cdef vector[const double *] pointers
        cdef size_t field_count
        cdef size_t rows
        if isinstance(fields, SkimmingContext):
            if out is not None or destination_count is not None:
                raise ValueError("SkimmingContext already supplies the output and centroid count")
            return self._skim_prepared(fields)
        rows = _skim_destination_count(destination_count, self._node_count)
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
        out = _skim_output(out, rows, field_count)
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
        """Copy routing costs INCLUDING turn penalties into (destination_count, 1).

        None selects all context nodes. Otherwise output covers the first
        destination_count nodes, with the same count rules as skim_fields.
        ``out`` optionally supplies a writable C-contiguous float64 array.
        For a centroid OD matrix, pass destination_count=z and
        out=matrix[origin].reshape(z, 1) for each origin.
        Origin costs are zero if included; unfinalized/unreachable costs are infinite.
        No scratch space is needed; the stored distance labels are copied.
        """
        cdef double[::1] output_view
        cdef size_t rows = _skim_destination_count(destination_count, self._node_count)
        out = _skim_output(out, rows, 1)
        output_view = out.reshape(-1)
        cdef double *output = &output_view[0] if output_view.shape[0] else NULL
        with nogil:
            self.skim_costs_nogil(rows, output)
        return out

    def skim_turn_costs(self, out=None, *, destination_count=None):
        """Copy ONLY cumulative turn penalties into (destination_count, 1).

        Same output and destination_count rules as skim_costs.
        Settled node-based paths yield zero; unfinalized/unreachable paths yield
        infinity, including before a search. This is a component of skim_costs,
        not an additional cost to add to it.
        """
        cdef double[::1] output_view
        cdef size_t rows = _skim_destination_count(destination_count, self._node_count)
        out = _skim_output(out, rows, 1)
        output_view = out.reshape(-1)
        cdef double *output = &output_view[0] if output_view.shape[0] else NULL
        with nogil:
            self.skim_turn_costs_nogil(rows, output)
        return out

    @property
    def sentinel(self):
        return INVALID

    @property
    def predecessors(self):
        return self._predecessors.view()

    @property
    def connectors(self):
        return self._connectors.view()

    @property
    def reached_first(self):
        return self._reached_first.view()

    @property
    def settled_count(self):
        return self.cpp.settled_count

    @property
    def state_count(self):
        return self._predecessors.size

    @property
    def distances(self):
        return self._distances.view()

    @property
    def turn_costs(self):
        return self._turn_costs.view()

    @property
    def terminal_states(self):
        return self._terminal_states.view()

    @property
    def destination_mask(self):
        """Read-only boolean mask of physical nodes requested by the last search."""
        return self._destination_mask.view(np.bool_)

    @property
    def destinations(self):
        """Requested physical-node indices as a newly allocated array."""
        return np.flatnonzero(self._destination_mask).astype(np.uintp, copy=False)

    @property
    def destination_count(self):
        return self.cpp.destination_count

    @property
    def reached_destination_count(self):
        return self.cpp.reached_destination_count

    @property
    def root(self):
        return None if self.cpp.root == INVALID else self.cpp.root

    @property
    def origin(self):
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
        """Whether a finalized path to ``destination`` is available."""
        cdef size_t destination_index = self._destination_index(destination)
        return (self.cpp.settled_count > 0 and
                self.cpp.terminal_states[destination_index] != INVALID)

    def path_cost_to(self, destination):
        """Routing cost to one node, or infinity if it was not finalized."""
        cdef size_t destination_index = self._destination_index(destination)
        cdef size_t terminal = self.cpp.terminal_states[destination_index]
        return np.inf if terminal == INVALID else self.cpp.distances[terminal]

    def path_turn_cost_to(self, destination):
        """Cumulative turn cost to one node, or infinity if not finalized."""
        cdef size_t destination_index = self._destination_index(destination)
        cdef size_t terminal = self.cpp.terminal_states[destination_index]
        return np.inf if terminal == INVALID else self.cpp.turn_costs[terminal]

    def path_nodes_to(self, destination):
        """Reconstruct the finalized path to one physical node."""
        cdef size_t destination_index = self._destination_index(destination)
        if self.cpp.terminal_states[destination_index] == INVALID:
            return np.empty(0, dtype=np.uintp)
        links = self.path_links_to(destination_index)
        return np.concatenate((np.array([self.cpp.origin], dtype=np.uintp),
                               self.context.heads[links]))

    def path_links_to(self, destination):
        """Reconstruct local link indices to one physical node."""
        cdef size_t destination_index = self._destination_index(destination)
        cdef size_t state = self.cpp.terminal_states[destination_index]
        if state == INVALID:
            return np.empty(0, dtype=np.uintp)
        links = []
        while state != self.cpp.root:
            links.append(self.cpp.connectors[state])
            state = self.cpp.predecessors[state]
        return np.array(links[::-1], dtype=np.uintp)

    def _destination_index(self, destination):
        destination = operator.index(destination)
        if not 0 <= destination < self.context.node_count:
            raise ValueError("destination is outside the context's node range")
        return destination

    def _single_destination(self):
        if self.cpp.destination_count == 0:
            return None
        if self.cpp.destination_count != 1:
            raise ValueError("use the *_to(destination) methods for multi-target results")
        return self.destination

    @property
    def path_cost(self):
        """Single-target compatibility view; use path_cost_to for one-to-many."""
        destination = self._single_destination()
        return np.inf if destination is None else self.path_cost_to(destination)

    @property
    def path_turn_cost(self):
        """Single-target compatibility view; use path_turn_cost_to for one-to-many."""
        destination = self._single_destination()
        return np.inf if destination is None else self.path_turn_cost_to(destination)

    @property
    def path_nodes(self):
        """Single-target compatibility view; use path_nodes_to for one-to-many."""
        destination = self._single_destination()
        return np.empty(0, dtype=np.uintp) if destination is None else self.path_nodes_to(destination)

    @property
    def path_links(self):
        """Single-target compatibility view; use path_links_to for one-to-many."""
        destination = self._single_destination()
        return np.empty(0, dtype=np.uintp) if destination is None else self.path_links_to(destination)
