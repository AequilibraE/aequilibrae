from libc.stddef cimport size_t
from libcpp cimport bool as cpp_bool
from aequilibrae.paths.cython.aon_workspace cimport AoNWorkspace, CppAoNWorkspace
from aequilibrae.paths.cython.skimming_context cimport SkimmingContext

cdef extern from "search_results.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    cdef cppclass CppSearchResults "aequilibrae::paths::cpp::mvp::SearchResults":
        CppSearchResults() noexcept
        size_t *predecessors
        size_t *connectors
        size_t *reached_first
        # Borrowed immutable search input: node_count bytes, with a matching
        # precomputed destination_count. Both are preserved by Dijkstra.
        const cpp_bool *destination_mask
        double *distances
        double *turn_costs
        size_t *terminal_states
        size_t root
        size_t origin
        size_t destination_count
        size_t reached_destination_count
        size_t settled_count


cdef extern from "skimming.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    void cpp_skim_fields "aequilibrae::paths::cpp::mvp::skim_fields"[T](
        const CppSearchResults &results, size_t destination_count,
        const T *const *fields, size_t field_count,
        CppAoNWorkspace[T] &workspace, T *output) noexcept

    void cpp_skim_costs "aequilibrae::paths::cpp::mvp::skim_costs"[T](
        const CppSearchResults &results, size_t destination_count, T *output) noexcept

    void cpp_skim_turn_costs "aequilibrae::paths::cpp::mvp::skim_turn_costs"[T](
        const CppSearchResults &results, size_t destination_count, T *output) noexcept

    cdef T cpp_sum_weighted_turn_costs"aequilibrae::paths::cpp::mvp::sum_weighted_turn_costs"[T](
        const CppSearchResults &search,
        size_t zones,
        const T *demand,
        size_t classes,
        const cpp_bool *penalty_fields,
        size_t fields,
        T *skims
    ) noexcept


cdef extern from "network_loading.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    void cpp_network_loading "aequilibrae::paths::cpp::mvp::network_loading"[T](
        const CppSearchResults &results, size_t destination_count,
        const T *demand, size_t class_count,
        CppAoNWorkspace[T] &workspace, T *link_loads) noexcept


# This call skips Python checks so prepared workers can use it in their loop.
# The caller must prepare loading/path flags and check buffer sizes first.
cdef extern from "select_link_loading.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    void cpp_select_link_loading "aequilibrae::paths::cpp::mvp::select_link_loading"[T](
        const CppSearchResults &results, size_t destination_count,
        const T *demand, size_t class_count, const cpp_bool *selected_links,
        CppAoNWorkspace[T] &workspace, T *od, T *link_loads) noexcept


cdef class SearchResults:
    cdef CppSearchResults cpp
    cdef size_t node_count
    cdef readonly AoNWorkspace workspace
    # Prepare workspace.prepare_loading(class_count) under the GIL first.
    # demand: packed [destination_count, class_count], count <= node_count.
    # link_loads: caller-owned packed [link_count, class_count], accumulated into.
    # All buffers must remain alive, with exclusive access to results/workspace
    # and output. Scratch, demand and output must not overlap. No allocations.
    cdef void network_loading_nogil(self, const double *demand,
                                   size_t destination_count, size_t class_count,
                                   double *link_loads) noexcept nogil
    cdef SkimmingContext prepared_skims
    cpdef prepare_skims(self, SkimmingContext fields)
    # Unchecked allocation-free entry points. Call workspace.prepare_skims(F)
    # under the GIL first for skim_fields_nogil. Inputs: F pointers to L doubles.
    # Output: packed [destination_count, F], or [destination_count, 1] for costs.
    # Rows are the first destination_count nodes; require 0 <= count <= node_count.
    # This output count is independent of cpp.destination_count and its mask.
    # Empty outputs may use NULL. Scratch still covers all search states.
    # Caller owns buffer lifetimes, sizes, non-overlap with scratch, and locking.
    cdef void skim_fields_nogil(self, const double *const *fields,
                               size_t field_count, size_t destination_count,
                               double *output) noexcept nogil
    cdef void skim_costs_nogil(self, size_t destination_count, double *output) noexcept nogil
    cdef void skim_turn_costs_nogil(self, size_t destination_count, double *output) noexcept nogil
    # Prepared input/output entry point. Requires the same context, a completed
    # search with origin < centroid_count, and workspace prepared for field_count.
    # Concurrent workers must use separate results and write different OD rows.
    cdef void skim_prepared_nogil(self, SkimmingContext fields) noexcept nogil
    cdef object skim_prepared(self, SkimmingContext fields)
    cdef readonly object context

    cdef size_t[::1] predecessors_buffer
    cdef size_t[::1] connectors_buffer
    cdef size_t[::1] settled_states
    cdef size_t[::1] terminal_states_buffer

    cdef double[::1] distances_buffer
    cdef double[::1] turn_costs_buffer

    cdef cpp_bool[::1] destination_mask_buffer
