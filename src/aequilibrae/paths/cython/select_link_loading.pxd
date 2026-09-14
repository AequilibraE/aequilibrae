from libcpp cimport bool as cpp_bool
from aequilibrae.paths.cython.search_results cimport CppSearchResults
from aequilibrae.paths.cython.queries cimport CppLoadingQuery
from aequilibrae.paths.cython.outputs cimport CppLoadingOutputs
from aequilibrae.paths.cython.workspaces cimport CppLoadingWorkspace, CppSelectLinkWorkspace


cdef extern from "select_link_loading.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    void cpp_select_link_loading "aequilibrae::paths::cpp::mvp::select_link_loading"[T](
        const CppSearchResults &results, const CppLoadingQuery[T] &query,
        const cpp_bool *selected_links, CppSelectLinkWorkspace selection,
        CppLoadingWorkspace[T] loading, T *od, CppLoadingOutputs[T] output) noexcept
