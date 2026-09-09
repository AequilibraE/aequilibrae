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
    cdef object _fs
    cdef object _heads
    cdef object _costs
    cdef object _link_ids
    cdef void _initialize_graph(self, object fs, object heads, object costs) except *
    cdef void _initialize_cpp_graph(self, CppNodeBasedContext *cpp) except *


cdef class NodeBasedContext(GraphContext):
    cdef CppNodeBasedContext cpp


cdef class TurnBasedContext(GraphContext):
    cdef CppTurnBasedContext cpp
    cdef object _tails
    cdef object _turn_fs
    cdef object _turn_to_links
    cdef object _turn_penalties
