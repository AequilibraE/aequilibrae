from libc.stddef cimport size_t

cimport cython

from .matrix_base cimport _MatrixBase

cdef extern from "dense_matrix.hpp" namespace "aequilibrae::matrix::cpp" nogil:
    cdef cppclass CppDenseMatrixOriginView "aequilibrae::matrix::cpp::DenseMatrixOriginView"[T]:
        CppDenseMatrixOriginView() noexcept
        size_t destination_count
        size_t field_count
        T *data
        T *destination_data(size_t index) noexcept

    cdef cppclass CppDenseMatrix "aequilibrae::matrix::cpp::DenseMatrix"[T]:
        CppDenseMatrix() noexcept
        size_t origin_count
        size_t destination_count
        size_t field_count
        T *data
        CppDenseMatrixOriginView[T] origin(size_t index) noexcept


@cython.final
cdef class DenseMatrix(_MatrixBase):
    cdef CppDenseMatrix[cython.floating] view(self, cython.floating _type=*) except * nogil
