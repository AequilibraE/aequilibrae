from libc.stddef cimport size_t
from libcpp cimport bool as cpp_bool
from aequilibrae.paths.cython.search_results cimport CppSearchResults
from aequilibrae.paths.cython.aon_workspace cimport CppAoNWorkspace


cdef extern from "skimming.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    void cpp_skim_fields "aequilibrae::paths::cpp::mvp::skim_fields"[T](
        const CppSearchResults &results, size_t destination_count,
        const T *const *fields, size_t field_count,
        CppAoNWorkspace[T] &workspace, T *output) noexcept

    void cpp_skim_costs "aequilibrae::paths::cpp::mvp::skim_costs"[T](
        const CppSearchResults &results, size_t destination_count, T *output) noexcept

    void cpp_skim_turn_costs "aequilibrae::paths::cpp::mvp::skim_turn_costs"[T](
        const CppSearchResults &results, size_t destination_count, T *output) noexcept

    T cpp_sum_weighted_turn_costs "aequilibrae::paths::cpp::mvp::sum_weighted_turn_costs"[T](
        const CppSearchResults &search, size_t zones, const T *demand,
        size_t classes, const cpp_bool *penalty_fields, size_t fields, T *skims) noexcept
