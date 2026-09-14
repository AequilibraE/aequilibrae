from libc.stddef cimport size_t


cdef extern from "outputs.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    cdef cppclass CppLoadingOutputs "aequilibrae::paths::cpp::mvp::LoadingOutputs"[T]:
        CppLoadingOutputs() noexcept
        size_t link_count
        size_t class_count
        T *link_loads
        void reset() noexcept

    cdef cppclass CppSkimmingOriginView "aequilibrae::paths::cpp::mvp::SkimmingOriginView"[T]:
        CppSkimmingOriginView() noexcept
        size_t field_count
        size_t destination_count
        T *data
        T *field_data(size_t index) noexcept
        CppSkimmingOriginView[T] subfields(size_t first, size_t count) noexcept

    cdef cppclass CppSkimmingOutputsView "aequilibrae::paths::cpp::mvp::SkimmingOutputsView"[T]:
        CppSkimmingOutputsView() noexcept
        size_t origin_count
        size_t field_count
        size_t destination_count
        T *data
        CppSkimmingOriginView[T] origin(size_t index) noexcept
        void reset() noexcept


cdef class LoadingOutputs:
    cdef readonly size_t link_count, class_count
    cdef double[:, ::1] link_loads_buffer
    cdef CppLoadingOutputs[double] view(self) noexcept nogil


cdef class SkimmingOutputs:
    cdef readonly size_t origin_count, destination_count, field_count
    cdef readonly tuple field_names
    cdef double[:, :, ::1] skims_buffer
    cdef CppSkimmingOutputsView[double] view(self) noexcept nogil
