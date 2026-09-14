from libc.stddef cimport size_t


cdef extern from "outputs.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    cdef cppclass CppLoadingOutputs "aequilibrae::paths::cpp::mvp::LoadingOutputs"[T]:
        CppLoadingOutputs() noexcept
        size_t link_count
        size_t class_count
        T *link_loads
        void reset() noexcept


cdef class LoadingOutputs:
    cdef readonly size_t link_count, class_count
    cdef double[:, ::1] link_loads_buffer
    cdef CppLoadingOutputs[double] view(self) noexcept nogil
