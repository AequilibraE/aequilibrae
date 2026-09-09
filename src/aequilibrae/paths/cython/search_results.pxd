from libc.stddef cimport size_t

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


cdef class SearchResults:
    cdef CppSearchResults cpp
    cdef readonly object context
    cdef object _predecessors
    cdef object _connectors
    cdef object _reached_first
    cdef object _destination_mask
    cdef object _distances
    cdef object _turn_costs
    cdef object _terminal_states
