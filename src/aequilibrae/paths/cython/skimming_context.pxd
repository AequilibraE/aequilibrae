from libc.stddef cimport size_t
from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector


cdef extern from "skimming_context.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    cdef cppclass CppSkimmingContext "aequilibrae::paths::cpp::mvp::SkimmingContext"[T]:
        CppSkimmingContext() noexcept
        size_t link_count
        size_t field_count
        size_t additive_field_count
        const T *const *link_fields
        size_t plain_field_count
        size_t turn_field_offset
        size_t turn_field_count
        size_t cost_field_index
        size_t turn_cost_field_index
        size_t cost_field_count
        size_t turn_cost_field_count

        cpp_bool needs_state_sums() noexcept
        cpp_bool has_link_fields() noexcept
        cpp_bool has_link_fields_with_turn_costs() noexcept
        cpp_bool has_cost_field() noexcept
        cpp_bool has_turn_cost_field() noexcept


cdef class SkimmingContext:
    cdef readonly size_t link_count, field_count, additive_field_count
    cdef readonly tuple field_names
    cdef CppSkimmingContext[double] configuration
    cdef tuple field_buffers
    cdef vector[const double *] field_pointers
    cdef CppSkimmingContext[double] view(self) noexcept nogil
