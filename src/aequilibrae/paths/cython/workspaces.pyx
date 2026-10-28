import operator
import numpy as np
cimport cython
from aequilibrae.matrix.cython.sparse_matrix_builder cimport SparseMatrixBuilder

from aequilibrae.utils.cython.array_allocations cimport array, array_pointer
from aequilibrae.utils.cython.array_allocations import readonly_view


cdef class SearchWorkspace:
    """Heap for Dijkstra."""

    def __cinit__(self):
        self.heap_storage = NULL

    def __init__(self, node_count, state_count, heap="4ary"):
        cdef CppSearchHeap heap_type
        if self.heap_storage != NULL:
            raise RuntimeError("SearchWorkspace cannot be reinitialised")

        node_count, state_count = map(operator.index, (node_count, state_count))
        if node_count < 1 or state_count < 1:
            raise ValueError("node_count and state_count must be positive")

        if heap == "4ary":
            heap_type = CPP_FOUR_ARY
        elif heap == "pairing":
            heap_type = CPP_PAIRING
        elif heap == "std":
            heap_type = CPP_STD
        else:
            raise ValueError("heap must be one of ['4ary', 'pairing', 'std']")

        self.heap_storage = new CppSearchHeapStorage[double](state_count, heap_type)
        self.node_count = node_count
        self.state_count = state_count
        self.heap = heap

    def __dealloc__(self):
        if self.heap_storage != NULL:
            del self.heap_storage

    cdef CppSearchWorkspace[double] view(self) noexcept nogil:
        cdef CppSearchWorkspace[double] workspace
        workspace.heap = self.heap_storage
        return workspace


cdef class AStarWorkspace(SearchWorkspace):
    """Heap and cost and estimate scratch buffers for A*."""

    def __init__(self, node_count, state_count, heap="4ary"):
        super().__init__(node_count, state_count, heap)
        self.costs_buffer = array[double](self.state_count, True, np.inf)
        self.estimates_buffer = array[double](self.node_count, True, -1.0)

    cdef CppAStarWorkspace[double] a_star_view(self) noexcept nogil:
        cdef CppAStarWorkspace[double] workspace
        workspace.search = self.view()
        workspace.costs = array_pointer(self.costs_buffer)
        workspace.estimates = array_pointer(self.estimates_buffer)
        return workspace


cdef class LoadingWorkspace:
    """Link loading scratch buffers."""

    def __init__(self, state_count, class_count):
        if self.state_count:
            raise RuntimeError("LoadingWorkspace cannot be reinitialised")

        state_count, class_count = map(operator.index, (state_count, class_count))
        if state_count < 1 or class_count < 0:
            raise ValueError("state_count must be positive and class_count nonnegative")

        self.state_count = state_count
        self.class_count = class_count
        self.state_loads_buffer = array[double]((state_count, class_count), True, 0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef CppLoadingWorkspace[double] view(self) noexcept nogil:
        cdef CppLoadingWorkspace[double] workspace
        workspace.state_count = self.state_count
        workspace.class_count = self.class_count
        if self.class_count:
            workspace.state_loads = &self.state_loads_buffer[0, 0]
        return workspace

    @property
    def state_loads(self):
        return readonly_view(self.state_loads_buffer)


cdef class SkimmingWorkspace:
    """Skimming scratch buffers."""

    def __init__(self, state_count, field_count):
        if self.state_count:
            raise RuntimeError("SkimmingWorkspace cannot be reinitialised")

        state_count, field_count = map(operator.index, (state_count, field_count))
        if state_count < 1 or field_count < 0:
            raise ValueError("state_count must be positive and field_count nonnegative")

        self.state_count = state_count
        self.field_count = field_count
        self.state_skims_buffer = array[double]((state_count, field_count), True, np.inf)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef CppSkimmingWorkspace[double] view(self) noexcept nogil:
        cdef CppSkimmingWorkspace[double] workspace
        workspace.state_count = self.state_count
        workspace.field_count = self.field_count

        if self.field_count:
            workspace.state_skims = &self.state_skims_buffer[0, 0]
        return workspace

    @property
    def state_skims(self):
        return readonly_view(self.state_skims_buffer)


cdef class SelectLinkWorkspace:
    """Select link scratch buffers."""

    def __init__(self, state_count):
        if self.state_count:
            raise RuntimeError("SelectLinkWorkspace cannot be reinitialised")

        state_count = operator.index(state_count)
        if state_count < 1:
            raise ValueError("state_count must be positive")

        self.state_count = state_count
        self.selected_paths_buffer = array[cpp_bool](state_count, True, False)

    cdef CppSelectLinkWorkspace view(self) noexcept nogil:
        cdef CppSelectLinkWorkspace workspace
        workspace.state_count = self.state_count
        workspace.selected_paths = array_pointer(self.selected_paths_buffer)

        return workspace

    @property
    def selected_paths(self):
        return readonly_view(self.selected_paths_buffer)


cdef class SelectLinkODWorkspace:
    """COO builder per set for a single worker."""

    def __init__(self, origin_count, destination_count, field_names, set_names):
        if self.builders is not None:
            raise RuntimeError("SelectLinkODWorkspace cannot be reinitialised")

        self.builders = tuple(
            SparseMatrixBuilder(destination_count, field_names, origin_count=origin_count)
            for _ in set_names
        )

        cdef SparseMatrixBuilder builder
        for builder in self.builders:
            self.builder_views.push_back(builder.view[double]())

    cdef CppSelectLinkODOutputsView[double] view(self) noexcept nogil:
        cdef CppSelectLinkODOutputsView[double] result
        result.set_count = self.builder_views.size()
        result.builders = self.builder_views.data()
        return result


cdef class AoNWorkspace:
    """All scratch buffers required by an assignment worker."""

    def __init__(self, node_count, state_count, *, heap="4ary", class_count=None, field_count=None, select_links=False,
                 od_zone_count=None, demand_names=(), set_names=()):
        if self.state_count:
            raise RuntimeError("AoNWorkspace cannot be reinitialised")

        node_count, state_count = map(operator.index, (node_count, state_count))
        if node_count < 1 or state_count < 1:
            raise ValueError("node_count and state_count must be positive")
        self.node_count = node_count
        self.state_count = state_count
        self.search = SearchWorkspace(node_count, state_count, heap=heap)

        if class_count is not None:
            self.loading = LoadingWorkspace(state_count, class_count)
        if field_count is not None:
            self.skimming = SkimmingWorkspace(state_count, field_count)
        if select_links:
            self.select_link = SelectLinkWorkspace(state_count)
        if od_zone_count is not None:
            self.selected_od = SelectLinkODWorkspace(od_zone_count, od_zone_count, demand_names, set_names)

    cdef CppAoNWorkspace[double] view(self) noexcept nogil:
        cdef CppAoNWorkspace[double] workspace

        workspace.search = self.search.view()
        if self.loading is not None:
            workspace.loading = self.loading.view()
        if self.skimming is not None:
            workspace.skimming = self.skimming.view()
        if self.select_link is not None:
            workspace.select_link = self.select_link.view()
        if self.selected_od is not None:
            workspace.selected_od = self.selected_od.view()

        return workspace
