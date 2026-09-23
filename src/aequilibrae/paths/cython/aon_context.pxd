from libc.stddef cimport size_t
from aequilibrae.paths.cython.context cimport CppSkimmingContext, CppSelectLinkContext
from aequilibrae.paths.cython.queries cimport CppSearchQuery, CppLoadingQuery
from aequilibrae.paths.cython.search_results cimport CppMutableSearchResults
from aequilibrae.paths.cython.workspaces cimport CppAoNWorkspace
from aequilibrae.paths.cython.outputs cimport (
    CppLoadingOutputs, CppSelectLinkLoadingOutputsView,
)


cdef extern from "aon.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    cdef cppclass CppAoNOrigin "aequilibrae::paths::cpp::mvp::AoNOrigin":
        CppAoNOrigin() noexcept
        CppSearchQuery search
        CppLoadingQuery[double] loading

    cdef cppclass CppAoNInputs "aequilibrae::paths::cpp::mvp::AoNInputs":
        CppAoNInputs() noexcept
        CppSkimmingContext[double] skimming
        CppSelectLinkContext selection
        const CppAoNOrigin *origins
        size_t origin_count

    cdef cppclass CppAoNWorkerView "aequilibrae::paths::cpp::mvp::AoNWorkerView":
        CppAoNWorkerView() noexcept
        CppMutableSearchResults search
        CppAoNWorkspace[double] workspace
        CppLoadingOutputs[double] loading
        CppSelectLinkLoadingOutputsView[double] selected_loading
        double turn_cost_total
        double unassigned_demand
        void reset() noexcept
