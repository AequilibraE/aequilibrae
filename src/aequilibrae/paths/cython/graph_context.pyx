import numpy as np

from libc.stddef cimport size_t

from aequilibrae.utils.cython.array_allocations cimport array, const_array_pointer
from aequilibrae.utils.cython.array_allocations import readonly_view


def validate_index_array(value, name):
    """Copy node or link indices, rejecting negative and non-integer values."""
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    if array.size:
        if array.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integers")

        if np.any(array < 0) or np.any(array > np.iinfo(np.uintp).max):
            raise ValueError(f"{name} contains an invalid index")

    return np.array(array, dtype=np.uintp, order="C", copy=True)


def validate_offsets(value, name, expected_end, end_description, expected_size=None):
    """Copy row offsets and check their length, order and endpoints."""
    array = validate_index_array(value, name)

    if expected_size is None:
        if array.size < 2:
            raise ValueError(f"{name} must describe at least one node")
    elif array.size != expected_size:
        raise ValueError(f"{name} must have {expected_size} entries")

    if array[0] != 0 or array[-1] != expected_end:
        raise ValueError(f"{name} must start at zero and end at {end_description}")

    if np.any(array[1:] < array[:-1]):
        raise ValueError(f"{name} must be non-decreasing")
    return array


cdef class GraphContext:
    """Keep a fixed copy of graph links and costs for routing.

    Use NodeBasedContext or TurnBasedContext, not this base class directly.
    """

    def __init__(self, fs, heads, costs):
        if type(self) is GraphContext:
            raise TypeError("GraphContext is an abstract base class")

        heads_array = validate_index_array(heads, "heads")
        fs_array = validate_offsets(fs, "fs", heads_array.size, "the number of links")
        costs = np.asarray(costs, dtype=np.float64, order="C")

        if costs.ndim != 1 or costs.size != heads_array.size:
            raise ValueError("costs must be one-dimensional with one value per link")
        if np.any(heads_array >= fs_array.size - 1):
            raise ValueError("heads contains an out-of-range node")
        if np.any(np.isnan(costs)) or np.any(costs < 0):
            raise ValueError("costs must be nonnegative and must not contain NaN")

        self.node_offsets = fs_array
        self.heads_buffer = heads_array
        self.costs_buffer = array[double](costs.size, False, 0)
        np.copyto(np.asarray(self.costs_buffer), costs)
        self.link_ids_buffer = np.arange(heads_array.size, dtype=np.uintp)

    cdef CppNodeBasedContext graph_view(self) noexcept nogil:
        """Build a C++ graph view that borrows this object's buffers."""
        cdef CppNodeBasedContext graph
        graph.fs = const_array_pointer(self.node_offsets)
        graph.heads = const_array_pointer(self.heads_buffer)
        graph.costs = const_array_pointer[double](self.costs_buffer)
        graph.link_ids = const_array_pointer(self.link_ids_buffer)
        graph.node_count = self.node_offsets.shape[0] - 1
        graph.link_count = self.heads_buffer.shape[0]
        return graph

    def make_results(self):
        """Allocate search results for this graph."""
        from aequilibrae.paths.cython.search_results import SearchResults

        return SearchResults(self)

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

    @property
    def link_ids(self):
        """Read-only link indices in routing order."""
        return readonly_view(self.link_ids_buffer)


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
            allow_uturns=True
    ):
        super().__init__(fs, heads, costs)

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
