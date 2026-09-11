from libc.stddef cimport size_t


cdef extern from "aon_workspace.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    cdef cppclass CppAoNWorkspace "aequilibrae::paths::cpp::mvp::AoNWorkspace"[T]:
        CppAoNWorkspace() noexcept
        size_t state_count
        size_t skim_field_count
        T *state_skims
        size_t loading_class_count
        T *state_loads


cdef class AoNWorkspace:
    cdef CppAoNWorkspace[double] cpp
    cdef double[:, ::1] state_skims_buffer, state_loads_buffer
    cdef readonly object context
    cpdef prepare_skims(self, object field_count)
    cpdef prepare_loading(self, object class_count)
