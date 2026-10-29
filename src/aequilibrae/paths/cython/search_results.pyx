"""Internal storage for a finalized path tree, independent of its graph."""

import operator
import numpy as np

from aequilibrae.utils.cython.array_allocations cimport array, array_pointer
from aequilibrae.utils.cython.array_allocations import readonly_view


cdef size_t INVALID = <size_t>-1


cdef class SearchResults:
    """Keep one search's paths and labels, with no graph or operation scratch.

    Allocate once per worker, or once for an individual search. Reusing results
    overwrites retained views; copy an array to keep its previous values. Link
    indices remain in the routing context's local order. Converting those links
    to physical nodes or external IDs belongs to callers with the graph data.
    """

    def __cinit__(self):
        self.predecessors_buffer = None

    def __init__(self, node_count, state_count, link_count):
        if self.predecessors_buffer is not None:
            raise RuntimeError("SearchResults cannot be reinitialized")

        node_count, state_count, link_count = map(operator.index, (node_count, state_count, link_count))
        if node_count < 1 or state_count < 1 or link_count < 0:
            raise ValueError("node_count and state_count must be positive; link_count must be nonnegative")

        self.node_count = node_count
        self.state_count = state_count
        self.link_count = link_count
        self.predecessors_buffer = array[size_t](state_count, True, INVALID)
        self.connectors_buffer = array[size_t](state_count, True, INVALID)
        self.settlement_order_buffer = array[size_t](state_count, True, INVALID)
        self.terminal_states_buffer = array[size_t](node_count, True, INVALID)
        self.distances_buffer = array[double](state_count, True, np.inf)
        self.turn_costs_buffer = array[double](state_count, True, np.inf)

    cdef CppMutableSearchResults view(self) noexcept nogil:
        """Borrow writable buffers and the owner's metadata for a search."""
        cdef CppMutableSearchResults result
        result.node_count = self.node_count
        result.state_count = self.state_count
        result.link_count = self.link_count
        result.predecessors = array_pointer(self.predecessors_buffer)
        result.connectors = array_pointer(self.connectors_buffer)
        result.settlement_order = array_pointer(self.settlement_order_buffer)
        result.terminal_states = array_pointer(self.terminal_states_buffer)
        result.distances = array_pointer(self.distances_buffer)
        result.turn_costs = array_pointer(self.turn_costs_buffer)
        result.metadata = &self.metadata
        return result

    cdef CppSearchResults read_view(self) noexcept nogil:
        """Downstream operations may read paths, but must not change them."""
        return self.view().read_view()

    def reset(self):
        """Clear labels and search metadata in place.

        :Returns:
            ``None``. Existing array views remain valid and contain sentinel or
            infinity values.
        """
        with nogil:
            self.view().reset()

    @property
    def sentinel(self):
        return INVALID

    @property
    def predecessors(self):
        return readonly_view(self.predecessors_buffer)

    @property
    def connectors(self):
        return readonly_view(self.connectors_buffer)

    @property
    def settlement_order(self):
        """Only the first settled_count entries describe finalized states."""
        return readonly_view(self.settlement_order_buffer)

    @property
    def terminal_states(self):
        return readonly_view(self.terminal_states_buffer)

    @property
    def distances(self):
        """Routing objective by state, including turn costs."""
        return readonly_view(self.distances_buffer)

    @property
    def turn_costs(self):
        """Turn-cost component by state; already included in distances."""
        return readonly_view(self.turn_costs_buffer)

    @property
    def origin(self):
        return None if self.metadata.origin == INVALID else self.metadata.origin

    @property
    def root(self):
        return None if self.metadata.root == INVALID else self.metadata.root

    @property
    def settled_count(self):
        return self.metadata.settled_count

    @property
    def target_count(self):
        return self.metadata.target_count

    @property
    def reached_target_count(self):
        return self.metadata.reached_target_count

    @property
    def all_targets_reached(self):
        """A full search has no targets; inspect exhausted for its completion."""
        return (self.metadata.settled_count > 0 and
                self.metadata.reached_target_count == self.metadata.target_count)

    @property
    def exhausted(self):
        """True when no reachable states remain unexplored.

        False before a search or when the search stopped after reaching its
        targets. Missing terminals need not be unreachable in a partial search.
        """
        return self.metadata.exhausted

    cdef size_t validate_destination(self, object destination) except *:
        if isinstance(destination, (bool, np.bool_)):
            raise TypeError("destination must be a node index, not a boolean")

        destination = operator.index(destination)
        if not 0 <= destination < self.node_count:
            raise ValueError("destination is outside the results' node range")

        return destination

    def reachable_to(self, destination):
        """Return whether a finalized path is available to a node.

        :Arguments:
            **destination** (:obj:`int`): Local node index.

        :Returns:
            ``bool``: Whether the node has a finalized terminal state.

        :Raises:
            **TypeError**: If ``destination`` is not an integer.
            **ValueError**: If ``destination`` is outside the node range.
        """
        cdef size_t node = self.validate_destination(destination)

        return self.terminal_states_buffer[node] != INVALID

    def path_cost_to(self, destination):
        """Return the routing objective for a destination.

        :Arguments:
            **destination** (:obj:`int`): Local node index.

        :Returns:
            ``float``: Finalised routing cost, or infinity when unavailable.
        """
        cdef size_t node = self.validate_destination(destination)
        cdef size_t terminal = self.terminal_states_buffer[node]

        return np.inf if terminal == INVALID else self.distances_buffer[terminal]

    def path_turn_cost_to(self, destination):
        """Return the accumulated turn-cost component for a destination.

        :Arguments:
            **destination** (:obj:`int`): Local node index.

        :Returns:
            ``float``: Finalised turn cost, or infinity when unavailable.
        """
        cdef size_t node = self.validate_destination(destination)
        cdef size_t terminal = self.terminal_states_buffer[node]

        return np.inf if terminal == INVALID else self.turn_costs_buffer[terminal]

    def path_states_to(self, destination):
        """Return the chosen arrival-state path in path order.

        The returned array includes the root and terminal state. Follow these
        states when turn history matters; an intermediate node's cheapest
        terminal state may belong to another arrival history.

        :Arguments:
            **destination** (:obj:`int`): Local node index.

        :Returns:
            :obj:`numpy.ndarray`: A copied ``uintp`` state path. An
            unfinalized destination returns an empty array; an origin path
            contains only the root.
        """
        cdef size_t node = self.validate_destination(destination)
        cdef size_t state = self.terminal_states_buffer[node]

        if state == INVALID:
            return np.empty(0, dtype=np.uintp)

        states = []
        while state != self.metadata.root:
            states.append(state)
            state = self.predecessors_buffer[state]
        states.append(self.metadata.root)

        return np.array(states[::-1], dtype=np.uintp)

    def path_links_to(self, destination):
        """Return the local directed links for a destination path.

        :Arguments:
            **destination** (:obj:`int`): Local node index.

        :Returns:
            :obj:`numpy.ndarray`: A copied ``uintp`` link path in traversal
            order. An unavailable or intrazonal path returns an empty array.
        """
        states = self.path_states_to(destination)
        return np.asarray(self.connectors_buffer)[states[1:]]
