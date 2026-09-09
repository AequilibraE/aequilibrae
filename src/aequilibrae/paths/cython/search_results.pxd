from libc.stddef cimport size_t
from aequilibrae.paths.cython.routing_workspace cimport RoutingWorkspace, CppRoutingWorkspace
from aequilibrae.paths.cython.skimming_context cimport SkimmingContext

cdef extern from "search_results.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    cdef cppclass CppSearchResults "aequilibrae::paths::cpp::mvp::SearchResults":
        CppSearchResults() noexcept
        size_t *predecessors
        size_t *connectors
        size_t *reached_first
        unsigned char *destination_mask
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
        CppRoutingWorkspace[T] &workspace, T *output) noexcept
    void cpp_skim_costs "aequilibrae::paths::cpp::mvp::skim_costs"[T](
        const CppSearchResults &results, size_t destination_count, T *output) noexcept
    void cpp_skim_turn_costs "aequilibrae::paths::cpp::mvp::skim_turn_costs"[T](
        const CppSearchResults &results, size_t destination_count, T *output) noexcept


cdef class SearchResults:
    cdef CppSearchResults cpp
    cdef size_t _node_count
    cdef readonly RoutingWorkspace workspace
    cdef SkimmingContext _prepared_skims
    cdef object _prepared_workspace
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
    cdef object _skim_prepared(self, SkimmingContext fields)
    cdef readonly object context
    cdef object _predecessors
    cdef object _connectors
    cdef object _reached_first
    cdef object _destination_mask
    cdef object _distances
    cdef object _turn_costs
    cdef object _terminal_states
