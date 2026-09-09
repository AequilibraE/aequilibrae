import numpy as np

from libc.stddef cimport size_t


ctypedef fused _ArrayElement:
    size_t
    double


cdef const _ArrayElement *_array_pointer(const _ArrayElement[::1] array) noexcept:
    """Return a typed pointer after Cython validates a contiguous memoryview."""
    return &array[0] if array.shape[0] else NULL


def validate_index_array(value, name):
    """Avoid truncation and signed-index wraparound."""
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size:
        if array.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integers")
        if np.any(array < 0) or np.any(array > np.iinfo(np.uintp).max):
            raise ValueError(f"{name} contains an invalid index")
    return np.array(array, dtype=np.uintp, order="C", copy=True)


def validate_fs(value, name, expected_end, end_description, expected_size=None):
    """Validate a forward-star offset array and return an owned uintp copy."""
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
    """Immutable, NumPy-owned forward-star graph snapshot.

    This common base owns and validates the graph arrays shared by all routing
    contexts. It is not directly instantiable; use :class:`NodeBasedContext` or
    :class:`TurnBasedContext`.
    """

    def __init__(self, fs, heads, costs):
        if type(self) is GraphContext:
            raise TypeError("GraphContext is an abstract base class")
        self._initialize_graph(fs, heads, costs)

    cdef void _initialize_graph(self, object fs, object heads, object costs) except *:
        heads_array = validate_index_array(heads, "heads")
        fs_array = validate_fs(fs, "fs", heads_array.size, "the number of links")
        costs_array = np.array(costs, dtype=np.float64, order="C", copy=True)

        if costs_array.ndim != 1 or costs_array.size != heads_array.size:
            raise ValueError("costs must be one-dimensional with one value per link")
        if np.any(heads_array >= fs_array.size - 1):
            raise ValueError("heads contains an out-of-range node")
        if np.any(np.isnan(costs_array)) or np.any(costs_array < 0):
            raise ValueError("costs must be nonnegative and must not contain NaN")

        ids_array = np.arange(heads_array.size, dtype=np.uintp)
        for array in (fs_array, heads_array, costs_array, ids_array):
            array.flags.writeable = False

        self._fs = fs_array
        self._heads = heads_array
        self._costs = costs_array
        self._link_ids = ids_array

    cdef void _initialize_cpp_graph(self, CppNodeBasedContext *cpp) except *:
        """Point a C++ graph context at this object's retained graph arrays."""
        cpp.fs = _array_pointer[size_t](self._fs)
        cpp.heads = _array_pointer[size_t](self._heads)
        cpp.costs = _array_pointer[double](self._costs)
        cpp.link_ids = _array_pointer[size_t](self._link_ids)
        cpp.node_count = self._fs.size - 1
        cpp.link_count = self._heads.size

    def make_results(self):
        from aequilibrae.paths.cython.search_results import SearchResults

        return SearchResults(self)

    @property
    def node_count(self):
        return self._fs.size - 1

    @property
    def link_count(self):
        return self._heads.size

    @property
    def state_count(self):
        return self.node_count

    @property
    def fs(self):
        return self._fs.view()

    @property
    def heads(self):
        return self._heads.view()

    @property
    def costs(self):
        return self._costs.view()

    @property
    def link_ids(self):
        return self._link_ids.view()


cdef class NodeBasedContext(GraphContext):
    """Graph context whose search states are nodes."""

    def __init__(self, fs, heads, costs):
        self._initialize_graph(fs, heads, costs)
        self._initialize_cpp_graph(&self.cpp)


cdef class TurnBasedContext(GraphContext):
    """Graph context with sparse directed-link turn penalties.

    ``turn_fs`` has ``link_count + 1`` entries. Explicit turns are grouped by
    their incoming link; missing turns cost zero. Search states are incoming
    links plus a virtual root.
    """

    def __init__(self, fs, heads, costs, turn_fs=None, turn_to_links=None,
                 turn_penalties=None, *, allow_uturns=True):
        self._initialize_graph(fs, heads, costs)
        n = self.node_count
        m = self.link_count
        if turn_fs is None and turn_to_links is None and turn_penalties is None:
            turn_fs = np.zeros(m + 1, dtype=np.uintp)
            turn_to_links = []
            turn_penalties = []
        elif turn_fs is None or turn_to_links is None or turn_penalties is None:
            raise ValueError("turn_fs, turn_to_links and turn_penalties must be supplied together")

        to_array = validate_index_array(turn_to_links, "turn_to_links")
        turn_fs_array = validate_fs(
            turn_fs, "turn_fs", to_array.size, "the number of explicit turns", m + 1
        )
        penalties_array = np.array(turn_penalties, dtype=np.float64, order="C", copy=True)
        if penalties_array.ndim != 1 or penalties_array.size != to_array.size:
            raise ValueError("turn_penalties must be one-dimensional with one value per explicit turn")
        if np.any(to_array >= m):
            raise ValueError("turn_to_links contains an out-of-range link")
        if np.any(np.isnan(penalties_array)) or np.any(penalties_array < 0):
            raise ValueError("turn_penalties must be nonnegative and must not contain NaN")

        tails_array = np.repeat(np.arange(n, dtype=np.uintp), np.diff(self._fs).astype(np.intp))
        from_links = np.repeat(np.arange(m, dtype=np.uintp), np.diff(turn_fs_array).astype(np.intp))
        same_row = from_links[1:] == from_links[:-1]
        if np.any(same_row & (to_array[1:] <= to_array[:-1])):
            raise ValueError("turn_to_links must be strictly increasing within each row")
        if np.any(self._heads[from_links] != tails_array[to_array]):
            raise ValueError("explicit turns must connect consecutive links")

        for array in (tails_array, turn_fs_array, to_array, penalties_array):
            array.flags.writeable = False
        self._tails = tails_array
        self._turn_fs = turn_fs_array
        self._turn_to_links = to_array
        self._turn_penalties = penalties_array

        self._initialize_cpp_graph(&self.cpp.graph)
        self.cpp.tails = _array_pointer[size_t](self._tails)
        self.cpp.turn_fs = _array_pointer[size_t](self._turn_fs)
        self.cpp.turn_to_links = _array_pointer[size_t](self._turn_to_links)
        self.cpp.turn_penalties = _array_pointer[double](self._turn_penalties)
        self.cpp.allow_uturns = bool(allow_uturns)

    @property
    def state_count(self):
        return self.link_count + 1

    @property
    def tails(self):
        return self._tails.view()

    @property
    def turn_fs(self):
        return self._turn_fs.view()

    @property
    def turn_to_links(self):
        return self._turn_to_links.view()

    @property
    def turn_penalties(self):
        return self._turn_penalties.view()

    @property
    def allow_uturns(self):
        return self.cpp.allow_uturns
