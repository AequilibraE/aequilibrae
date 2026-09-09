import operator

import numpy as np

from libc.stddef cimport size_t

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
