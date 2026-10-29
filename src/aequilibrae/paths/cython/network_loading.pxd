from libc.stddef cimport size_t
from aequilibrae.paths.cython.search_results cimport SearchResults, CppSearchResults
from aequilibrae.paths.cython.queries cimport LoadingQuery, CppLoadingQuery
from aequilibrae.paths.cython.workspaces cimport LoadingWorkspace, CppLoadingWorkspace
from aequilibrae.paths.cython.outputs cimport LoadingOutputs, CppLoadingOutputs


cdef extern from "network_loading.hpp" namespace "aequilibrae::paths::cpp::routing" nogil:
    void cpp_network_loading "aequilibrae::paths::cpp::routing::network_loading"[T](
        const CppSearchResults &results,
        const CppLoadingQuery[T] &query,
        const CppLoadingWorkspace[T] &workspace,
        const CppLoadingOutputs[T] &output,
    ) noexcept

    T cpp_sum_weighted_turn_costs "aequilibrae::paths::cpp::routing::sum_weighted_turn_costs"[T](
        const CppSearchResults &results, const CppLoadingQuery[T] &query) noexcept

    T cpp_sum_unassigned_demand "aequilibrae::paths::cpp::routing::sum_unassigned_demand"[T](
        const CppSearchResults &results, const CppLoadingQuery[T] &query) noexcept

    void cpp_reduce_loading_outputs "aequilibrae::paths::cpp::routing::reduce_loading_outputs"[T](
        const CppLoadingOutputs[T] *workers,
        size_t worker_count,
        const CppLoadingOutputs[T] &output,
    ) noexcept
