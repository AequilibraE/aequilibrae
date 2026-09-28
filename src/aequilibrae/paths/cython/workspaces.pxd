from libc.stddef cimport size_t
from libcpp cimport bool as cpp_bool


cdef extern from "workspaces.hpp" namespace "aequilibrae::paths::cpp::routing" nogil:
    cdef cppclass CppLoadingWorkspace "aequilibrae::paths::cpp::routing::LoadingWorkspace"[T]:
        CppLoadingWorkspace() noexcept
        size_t state_count
        size_t class_count
        T *state_loads

    cdef cppclass CppSkimmingWorkspace "aequilibrae::paths::cpp::routing::SkimmingWorkspace"[T]:
        CppSkimmingWorkspace() noexcept
        size_t state_count
        size_t field_count
        T *state_skims

    cdef cppclass CppSelectLinkWorkspace "aequilibrae::paths::cpp::routing::SelectLinkWorkspace":
        CppSelectLinkWorkspace() noexcept
        size_t state_count
        cpp_bool *selected_paths

    cdef cppclass CppAoNWorkspace "aequilibrae::paths::cpp::routing::AoNWorkspace"[T]:
        CppAoNWorkspace() noexcept
        CppLoadingWorkspace[T] loading
        CppSkimmingWorkspace[T] skimming
        CppSelectLinkWorkspace select_link


cdef class LoadingWorkspace:
    cdef readonly size_t state_count, class_count
    cdef double[:, ::1] state_loads_buffer
    cdef CppLoadingWorkspace[double] view(self) noexcept nogil


cdef class SkimmingWorkspace:
    cdef readonly size_t state_count, field_count
    cdef double[:, ::1] state_skims_buffer
    cdef CppSkimmingWorkspace[double] view(self) noexcept nogil


cdef class SelectLinkWorkspace:
    cdef readonly size_t state_count
    cdef cpp_bool[::1] selected_paths_buffer
    cdef CppSelectLinkWorkspace view(self) noexcept nogil


cdef class AoNWorkspace:
    cdef readonly size_t state_count
    cdef readonly LoadingWorkspace loading
    cdef readonly SkimmingWorkspace skimming
    cdef readonly SelectLinkWorkspace select_link
    cdef CppAoNWorkspace[double] view(self) noexcept nogil
