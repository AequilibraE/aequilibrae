from libc.stddef cimport size_t

from aequilibrae.paths.cython.context cimport CppSkimmingContext, CppSelectLinkContext
from aequilibrae.paths.cython.demand cimport DenseDemandRow, SparseDemandRow
from aequilibrae.paths.cython.outputs cimport CppLoadingOutputs, CppSelectLinkLoadingOutputsView
from aequilibrae.paths.cython.search_query cimport CppSearchQuery
from aequilibrae.paths.cython.search_results cimport CppMutableSearchResults
from aequilibrae.paths.cython.workspaces cimport CppAoNWorkspace


cdef extern from "aon.hpp" namespace "aequilibrae::paths::cpp::routing" nogil:
    cdef cppclass CppAoNOrigin "aequilibrae::paths::cpp::routing::AoNOrigin"[DemandRow]:
        CppAoNOrigin() noexcept

        CppSearchQuery search
        DemandRow loading

    cdef cppclass CppAoNInputs "aequilibrae::paths::cpp::routing::AoNInputs"[T]:
        CppAoNInputs() noexcept

        CppSkimmingContext[T] skimming
        CppSelectLinkContext selection

    cdef cppclass CppAoNWorkerView "aequilibrae::paths::cpp::routing::AoNWorkerView"[T]:
        CppAoNWorkerView() noexcept

        CppMutableSearchResults[T] search
        CppAoNWorkspace[T] workspace
        CppLoadingOutputs[T] loading
        CppSelectLinkLoadingOutputsView[T] selected_loading

        T turn_cost_total
        T unassigned_demand
        void reset() noexcept


ctypedef fused AoNOriginView:
    CppAoNOrigin[DenseDemandRow]
    CppAoNOrigin[SparseDemandRow]
