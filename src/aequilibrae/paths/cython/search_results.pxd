from libc.stddef cimport size_t
from libcpp cimport bool as cpp_bool


cdef extern from "search_results.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    cdef cppclass CppSearchMetadata "aequilibrae::paths::cpp::mvp::SearchMetadata":
        CppSearchMetadata() noexcept
        size_t origin
        size_t root
        size_t settled_count
        size_t target_count
        size_t reached_target_count
        cpp_bool exhausted

    cdef cppclass CppSearchResults "aequilibrae::paths::cpp::mvp::SearchResults":
        CppSearchResults() noexcept
        size_t node_count
        size_t state_count
        size_t link_count
        const size_t *predecessors
        const size_t *connectors
        const size_t *settlement_order
        const size_t *terminal_states
        const double *distances
        const double *turn_costs
        const CppSearchMetadata *metadata

    cdef cppclass CppMutableSearchResults "aequilibrae::paths::cpp::mvp::MutableSearchResults":
        CppMutableSearchResults() noexcept
        size_t node_count
        size_t state_count
        size_t link_count
        size_t *predecessors
        size_t *connectors
        size_t *settlement_order
        size_t *terminal_states
        double *distances
        double *turn_costs
        CppSearchMetadata *metadata
        CppSearchResults read_view() noexcept


cdef class SearchResults:
    cdef readonly size_t node_count, state_count, link_count
    cdef CppSearchMetadata metadata
    cdef size_t[::1] predecessors_buffer, connectors_buffer
    cdef size_t[::1] settlement_order_buffer, terminal_states_buffer
    cdef double[::1] distances_buffer, turn_costs_buffer
    cdef CppMutableSearchResults view(self) noexcept nogil
    cdef CppSearchResults read_view(self) noexcept nogil
    cdef size_t validate_destination(self, object destination) except *
