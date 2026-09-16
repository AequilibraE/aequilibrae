from libc.stddef cimport size_t
from aequilibrae.paths.cython.search_results cimport SearchResults, CppSearchResults
from aequilibrae.paths.cython.queries cimport LoadingQuery, CppLoadingQuery
from aequilibrae.paths.cython.context cimport SelectLinkContext, CppSelectLinkContext
from aequilibrae.paths.cython.outputs cimport (
    SelectLinkLoadingOutputs, SelectLinkODOutputs,
    CppSelectLinkLoadingOutputsView, CppSelectLinkODOriginView,
)
from aequilibrae.paths.cython.workspaces cimport (
    LoadingWorkspace, SelectLinkWorkspace, CppLoadingWorkspace, CppSelectLinkWorkspace,
)


cdef extern from "select_link_loading.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    void cpp_select_link_loading "aequilibrae::paths::cpp::mvp::select_link_loading"[T](
        const CppSearchResults &results,
        const CppLoadingQuery[T] &query,
        const CppSelectLinkContext &context,
        const CppSelectLinkWorkspace &selection,
        const CppLoadingWorkspace[T] &loading,
        const CppSelectLinkLoadingOutputsView[T] &loads,
        const CppSelectLinkODOriginView[T] &od,
    ) noexcept

    void cpp_reduce_select_link_loading_outputs "aequilibrae::paths::cpp::mvp::reduce_select_link_loading_outputs"[T](
        const CppSelectLinkLoadingOutputsView[T] *workers,
        size_t worker_count,
        const CppSelectLinkLoadingOutputsView[T] &output,
    ) noexcept
