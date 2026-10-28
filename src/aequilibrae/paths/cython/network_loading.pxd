from libc.stddef cimport size_t
from aequilibrae.paths.cython.search_results cimport SearchResults, CppSearchResults
from aequilibrae.paths.cython.demand cimport DemandRowView, DenseDemandRow, SparseDemandRow
from aequilibrae.paths.cython.workspaces cimport LoadingWorkspace, CppLoadingWorkspace
from aequilibrae.paths.cython.outputs cimport LoadingOutputs, CppLoadingOutputs


cdef extern from "network_loading.hpp" namespace "aequilibrae::paths::cpp::routing" nogil:
    void cpp_network_loading "aequilibrae::paths::cpp::routing::network_loading"[T, DemandRow](
        const CppSearchResults[double] &results,
        const DemandRow &query,
        const CppLoadingWorkspace[T] &workspace,
        const CppLoadingOutputs[T] &output,
    ) noexcept

    T _cpp_sum_weighted_turn_costs "aequilibrae::paths::cpp::routing::sum_weighted_turn_costs"[T, DemandRow](
        const CppSearchResults[double] &results, const DemandRow &query) noexcept

    T _cpp_sum_unassigned_demand "aequilibrae::paths::cpp::routing::sum_unassigned_demand"[T, DemandRow](
        const CppSearchResults[double] &results, const DemandRow &query) noexcept

    void cpp_reduce_loading_outputs "aequilibrae::paths::cpp::routing::reduce_loading_outputs"[T](
        const CppLoadingOutputs[T] *workers,
        size_t worker_count,
        const CppLoadingOutputs[T] &output,
    ) noexcept


# Cython cannot infer a template type used only for the return value, or pass a
# fused type as an explicit C++ template argument so we explicitly do the fused
# type dispatch. These are compile time conditionals.
cdef inline double cpp_sum_weighted_turn_costs(
    const CppSearchResults[double] &results, const DemandRowView &query,
) noexcept nogil:
    if DemandRowView is SparseDemandRow:
        return _cpp_sum_weighted_turn_costs[double, SparseDemandRow](results, query)
    else:
        return _cpp_sum_weighted_turn_costs[double, DenseDemandRow](results, query)


cdef inline double cpp_sum_unassigned_demand(
    const CppSearchResults[double] &results, const DemandRowView &query,
) noexcept nogil:
    if DemandRowView is SparseDemandRow:
        return _cpp_sum_unassigned_demand[double, SparseDemandRow](results, query)
    else:
        return _cpp_sum_unassigned_demand[double, DenseDemandRow](results, query)
