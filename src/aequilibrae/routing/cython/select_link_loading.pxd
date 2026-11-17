from libc.stddef cimport size_t
from aequilibrae.routing.cython.search_results cimport SearchResults, CppSearchResults
from aequilibrae.routing.cython.context cimport SelectLinkContext, CppSelectLinkContext
from aequilibrae.routing.cython.outputs cimport (
    SelectLinkLoadingOutputs, SelectLinkODOutputs, CppSelectLinkLoadingOutputsView,
    CppSelectLinkODOriginView, CppSelectLinkODOutputsView,
)
from aequilibrae.routing.cython.workspaces cimport (
    LoadingWorkspace, SelectLinkWorkspace, CppLoadingWorkspace, CppSelectLinkWorkspace,
)


cdef extern from "select_link_loading.hpp" namespace "aequilibrae::paths::cpp::routing" nogil:
    void cpp_select_link_loading "aequilibrae::paths::cpp::routing::select_link_loading"[T, DemandRow](
        const CppSearchResults[double] &results,
        const DemandRow &query,
        const CppSelectLinkContext &context,
        const CppSelectLinkWorkspace &selection,
        const CppLoadingWorkspace[T] &loading,
        const CppSelectLinkLoadingOutputsView[T] &loads,
        const CppSelectLinkODOriginView[T] &od,
    ) noexcept

    void cpp_reduce_select_link_loading "aequilibrae::paths::cpp::routing::reduce_select_link_loading_outputs"[T](
        const CppSelectLinkLoadingOutputsView[T] *workers,
        size_t worker_count,
        const CppSelectLinkLoadingOutputsView[T] &output,
        const CppSelectLinkODOutputsView[T] *od_workers,
        size_t od_worker_count,
        const CppSelectLinkODOutputsView[T] &od_output,
    ) except +
