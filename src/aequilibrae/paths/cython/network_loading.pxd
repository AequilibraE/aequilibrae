from libc.stddef cimport size_t
from aequilibrae.paths.cython.search_results cimport SearchResults, CppSearchResults
from aequilibrae.paths.cython.queries cimport LoadingQuery, CppLoadingQuery
from aequilibrae.paths.cython.workspaces cimport LoadingWorkspace, CppLoadingWorkspace
from aequilibrae.paths.cython.outputs cimport LoadingOutputs, CppLoadingOutputs


cdef extern from "network_loading.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    void cpp_network_loading "aequilibrae::paths::cpp::mvp::network_loading"[T](
        const CppSearchResults &results, const CppLoadingQuery[T] &query,
        CppLoadingWorkspace[T] workspace, CppLoadingOutputs[T] output) noexcept

    void cpp_reduce_loading_outputs "aequilibrae::paths::cpp::mvp::reduce_loading_outputs"[T](
        const CppLoadingOutputs[T] *workers, size_t worker_count,
        CppLoadingOutputs[T] output) noexcept
