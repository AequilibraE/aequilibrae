from libc.stddef cimport size_t
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
cimport cython
from .matrix_base cimport _MatrixBase

cdef extern from "sparse_matrix_builder.hpp" namespace "aequilibrae::matrix::cpp" nogil:
    cdef cppclass COOStorage[T]:
        COOStorage() except +
        vector[size_t] rows
        vector[size_t] columns
        vector[T] values

    cdef cppclass CppSparseMatrixBuilderView "aequilibrae::matrix::cpp::SparseMatrixBuilderView"[T]:
        CppSparseMatrixBuilderView() noexcept
        size_t origin_count
        size_t destination_count
        size_t field_count
        COOStorage[T] *storage

        size_t entry_count() noexcept
        size_t row(size_t) noexcept
        size_t column(size_t) noexcept
        const T *values(size_t) noexcept
        void append(size_t, size_t, const T *) noexcept
        void append_scaled(size_t, size_t, const T *, T) noexcept
        void reserve_entries(size_t) except +
        void extend(const CppSparseMatrixBuilderView[T] &) except +
        void reset() noexcept


@cython.final
cdef class SparseMatrixBuilder(_MatrixBase):
    cdef unique_ptr[COOStorage[float]] storage32
    cdef unique_ptr[COOStorage[double]] storage64

    cdef CppSparseMatrixBuilderView[cython.floating] view(self, cython.floating _type=*) except * nogil
    cdef object _arrays(self, cython.floating _type=*)
