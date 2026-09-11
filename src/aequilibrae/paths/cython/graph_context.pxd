from libc.stddef cimport size_t
from libcpp cimport bool as cpp_bool


cdef extern from "graph_context.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    cdef cppclass CppNodeBasedContext "aequilibrae::paths::cpp::mvp::NodeBasedContext":
        CppNodeBasedContext() noexcept
        size_t node_count
        size_t link_count
        const size_t *fs
        const size_t *heads
        const double *costs
        const size_t *link_ids

    cdef cppclass CppTurnBasedContext "aequilibrae::paths::cpp::mvp::TurnBasedContext":
        CppTurnBasedContext() noexcept
        CppNodeBasedContext graph
        const size_t *tails
        const size_t *turn_fs
        const size_t *turn_to_links
        const double *turn_penalties
        cpp_bool allow_uturns


cdef class GraphContext:
    cdef const size_t[::1] node_offsets, heads_buffer, link_ids_buffer
    cdef double[::1] costs_buffer
    cdef CppNodeBasedContext graph_view(self) noexcept nogil


cdef class NodeBasedContext(GraphContext):
    cdef CppNodeBasedContext view(self) noexcept nogil


cdef class TurnBasedContext(GraphContext):
    cdef const size_t[::1] tails_buffer, turn_offsets, turn_links
    cdef double[::1] turn_penalties_buffer
    cdef cpp_bool uturns_allowed
    cdef CppTurnBasedContext view(self) noexcept nogil
