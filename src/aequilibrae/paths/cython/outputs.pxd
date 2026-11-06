from libc.stddef cimport size_t
from libcpp.vector cimport vector
from aequilibrae.matrix.cython.sparse_matrix_builder cimport CppSparseMatrixBuilderView

from aequilibrae.matrix.cython.dense_matrix cimport DenseMatrix
from aequilibrae.matrix.cython.views cimport CppDenseMatrixOriginView


cdef extern from "outputs.hpp" namespace "aequilibrae::paths::cpp::routing" nogil:
    cdef cppclass CppLoadingOutputs "aequilibrae::paths::cpp::routing::LoadingOutputs"[T]:
        CppLoadingOutputs() noexcept

        size_t link_count
        size_t class_count
        T *link_loads

        void reset() noexcept

    cdef cppclass CppSkimmingOutputsView "aequilibrae::paths::cpp::routing::SkimmingOutputsView"[T]:
        CppSkimmingOutputsView() noexcept

        size_t origin_count
        size_t field_count
        size_t destination_count
        T *data

        CppDenseMatrixOriginView[T] origin(size_t index) noexcept
        void reset() noexcept

    cdef cppclass CppSelectLinkLoadingOutputsView "aequilibrae::paths::cpp::routing::SelectLinkLoadingOutputsView"[T]:
        CppSelectLinkLoadingOutputsView() noexcept

        size_t set_count
        size_t link_count
        size_t class_count

        T *data
        CppLoadingOutputs[T] selection(size_t index) noexcept
        void reset() noexcept

    cdef cppclass CppSelectLinkODOriginView "aequilibrae::paths::cpp::routing::SelectLinkODOriginView"[T]:
        CppSelectLinkODOriginView() noexcept

        size_t set_count
        size_t origin_index
        CppSparseMatrixBuilderView[T] *builders

        void append(size_t, size_t, const T *) noexcept

    cdef cppclass CppSelectLinkODOutputsView "aequilibrae::paths::cpp::routing::SelectLinkODOutputsView"[T]:
        CppSelectLinkODOutputsView() noexcept

        size_t set_count
        CppSparseMatrixBuilderView[T] *builders

        CppSelectLinkODOriginView[T] origin(size_t index) noexcept
        void reset() noexcept

    cdef cppclass CppAoNOutputsView "aequilibrae::paths::cpp::routing::AoNOutputsView"[T]:
        CppAoNOutputsView() noexcept

        CppLoadingOutputs[T] loading
        CppSkimmingOutputsView[T] skimming
        CppSelectLinkLoadingOutputsView[T] selected_loading
        CppSelectLinkODOutputsView[T] selected_od

        void reset() noexcept


# Existing Python output owners retain their float64 buffers.


cdef class LoadingOutputs:
    cdef readonly size_t link_count, class_count
    cdef double[:, ::1] link_loads_buffer
    cdef CppLoadingOutputs[double] view(self) noexcept nogil


cdef class SkimmingOutputs:
    cdef readonly size_t origin_count, destination_count, field_count
    cdef readonly tuple field_names
    cdef readonly DenseMatrix dense_matrix
    cdef double[:, :, ::1] skims_buffer
    cdef CppSkimmingOutputsView[double] view(self) noexcept nogil


cdef class SelectLinkLoadingOutputs:
    cdef readonly size_t set_count, link_count, class_count
    cdef readonly tuple set_names
    cdef double[:, :, ::1] link_loads_buffer
    cdef CppSelectLinkLoadingOutputsView[double] view(self) noexcept nogil


cdef class SelectLinkODOutputs:
    cdef readonly size_t origin_count, set_count, destination_count, class_count
    cdef readonly tuple set_names, field_names, builders
    cdef dict _matrices
    cdef vector[CppSparseMatrixBuilderView[double]] builder_views
    cdef CppSelectLinkODOutputsView[double] view(self) noexcept nogil


cdef class SelectLinkOutputs:
    cdef readonly SelectLinkLoadingOutputs loading
    cdef readonly SelectLinkODOutputs od
    cdef bint initialised


cdef class AoNOutputs:
    cdef readonly LoadingOutputs loading
    cdef readonly SkimmingOutputs skimming
    cdef readonly SelectLinkOutputs select_link
    cdef readonly double turn_cost_total, unassigned_demand
    cdef CppAoNOutputsView[double] view(self) noexcept nogil
