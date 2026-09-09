from libc.stddef cimport size_t


cdef extern from "routing_workspace.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    cdef cppclass CppRoutingWorkspace "aequilibrae::paths::cpp::mvp::RoutingWorkspace"[T]:
        CppRoutingWorkspace() noexcept
        size_t state_count
        size_t skim_field_count
        T *state_skims


cdef class RoutingWorkspace:
    cdef CppRoutingWorkspace[double] cpp
    cdef object _state_skims
    cdef readonly object context
    cpdef prepare_skims(self, object field_count)
