from libc.stddef cimport size_t
from aequilibrae.paths.cython.context cimport CppSkimmingContext, CppSelectLinkContext
from aequilibrae.paths.cython.queries cimport CppSearchQuery
from aequilibrae.matrix.cython.dense_matrix cimport CppDenseMatrixOriginView
from aequilibrae.matrix.cython.sparse_matrix cimport CppSparseMatrixOriginView
from aequilibrae.paths.cython.search_results cimport CppMutableSearchResults
from aequilibrae.paths.cython.workspaces cimport CppAoNWorkspace
from aequilibrae.paths.cython.outputs cimport (
    CppLoadingOutputs, CppSelectLinkLoadingOutputsView,
)


cdef extern from "aon.hpp" namespace "aequilibrae::paths::cpp::routing" nogil:
    cdef cppclass CppAoNOrigin "aequilibrae::paths::cpp::routing::AoNOrigin"[DemandRow]:
        CppAoNOrigin() noexcept
        CppSearchQuery search
        DemandRow loading

    cdef cppclass CppAoNInputs "aequilibrae::paths::cpp::routing::AoNInputs":
        CppAoNInputs() noexcept
        CppSkimmingContext[double] skimming
        CppSelectLinkContext selection

    cdef cppclass CppAoNWorkerView "aequilibrae::paths::cpp::routing::AoNWorkerView":
        CppAoNWorkerView() noexcept
        CppMutableSearchResults search
        CppAoNWorkspace[double] workspace
        CppLoadingOutputs[double] loading
        CppSelectLinkLoadingOutputsView[double] selected_loading
        double turn_cost_total
        double unassigned_demand
        void reset() noexcept


ctypedef fused AoNOriginView:
    CppAoNOrigin[CppDenseMatrixOriginView[const double]]
    CppAoNOrigin[CppSparseMatrixOriginView[double]]
