from libc.stddef cimport size_t

cimport cython


cdef extern from "dense_matrix.hpp" namespace "aequilibrae::matrix::cpp" nogil:
    cdef cppclass CppDenseMatrixOriginView "aequilibrae::matrix::cpp::DenseMatrixOriginView"[T]:
        CppDenseMatrixOriginView() noexcept
        size_t destination_count
        size_t field_count
        T *data
        T *destination_data(size_t index) noexcept

    cdef cppclass CppDenseMatrixView "aequilibrae::matrix::cpp::DenseMatrixView"[T]:
        CppDenseMatrixView() noexcept
        size_t origin_count
        size_t destination_count
        size_t field_count
        T *data
        CppDenseMatrixOriginView[T] origin(size_t index) noexcept


cdef class DenseMatrix:
    cdef readonly size_t origin_count, destination_count, field_count
    cdef readonly tuple field_names
    cdef readonly object dtype
    cdef object _values
    cdef void *_data
    cdef bint _float32

    cdef CppDenseMatrixView[cython.floating] view(self, cython.floating _type=*) except * nogil
