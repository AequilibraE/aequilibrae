from libc.stddef cimport size_t
from libcpp cimport bool as cpp_bool


cdef extern from "queries.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    cdef cppclass CppSearchQuery "aequilibrae::paths::cpp::mvp::SearchQuery":
        CppSearchQuery() noexcept
        size_t node_count
        size_t origin
        const cpp_bool *target_mask
        size_t target_count

    cdef cppclass CppLoadingQuery "aequilibrae::paths::cpp::mvp::LoadingQuery"[T]:
        CppLoadingQuery() noexcept
        size_t destination_count
        size_t class_count
        const T *demand


cdef class SearchQuery:
    cdef readonly size_t node_count, target_count
    cdef size_t origin_index
    cdef const cpp_bool[::1] target_mask_buffer
    cdef CppSearchQuery view(self) noexcept nogil


cdef class LoadingQuery:
    cdef readonly size_t destination_count, class_count
    cdef const double[:, ::1] demand_buffer
    cdef CppLoadingQuery[double] view(self) noexcept nogil
