from libc.stddef cimport size_t
from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector


cdef extern from "context.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    cdef cppclass CppNodeBasedContext "aequilibrae::paths::cpp::mvp::NodeBasedContext":
        CppNodeBasedContext() noexcept
        size_t node_count
        size_t link_count
        const size_t *fs
        const size_t *heads
        const double *costs
        size_t blocked_centroid_count

    cdef cppclass CppTurnBasedContext "aequilibrae::paths::cpp::mvp::TurnBasedContext":
        CppTurnBasedContext() noexcept
        CppNodeBasedContext graph
        const size_t *tails
        const size_t *turn_fs
        const size_t *turn_to_links
        const double *turn_penalties
        cpp_bool allow_uturns

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

    cdef cppclass CppSelectLinkContext "aequilibrae::paths::cpp::mvp::SelectLinkContext":
        CppSelectLinkContext() noexcept
        size_t link_count
        size_t set_count
        const cpp_bool *masks
        const cpp_bool *selection(size_t index) noexcept


cdef class GraphContext:
    cdef const size_t[::1] node_offsets, heads_buffer
    cdef const double[::1] costs_buffer
    cdef readonly size_t blocked_centroid_count
    cpdef update_costs(self, object costs)
    cdef CppNodeBasedContext graph_view(self) noexcept nogil


cdef class NodeBasedContext(GraphContext):
    cdef CppNodeBasedContext view(self) noexcept nogil


cdef class TurnBasedContext(GraphContext):
    cdef const size_t[::1] tails_buffer, turn_offsets, turn_links
    cdef double[::1] turn_penalties_buffer
    cdef cpp_bool uturns_allowed
    cdef CppTurnBasedContext view(self) noexcept nogil


cdef class SkimmingContext:
    cdef readonly size_t link_count, field_count, additive_field_count
    cdef readonly tuple field_names
    cdef CppSkimmingContext[double] configuration
    cdef tuple field_buffers
    cdef vector[const double *] field_pointers
    cdef CppSkimmingContext[double] view(self) noexcept nogil


cdef class SelectLinkContext:
    cdef readonly size_t link_count, set_count
    cdef readonly tuple set_names
    cdef cpp_bool[:, ::1] masks_buffer
    cdef CppSelectLinkContext view(self) noexcept nogil
