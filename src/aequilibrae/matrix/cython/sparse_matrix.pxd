from libc.stddef cimport size_t

cimport cython

from .matrix_base cimport _MatrixBase

cdef extern from "sparse_matrix.hpp" namespace "aequilibrae::matrix::cpp" nogil:
    cdef cppclass CppSparseMatrixOriginView "aequilibrae::matrix::cpp::SparseMatrixOriginView"[T]:
        CppSparseMatrixOriginView() noexcept
        size_t destination_count
        size_t field_count
        size_t stored_count
        const size_t *col_index
        T *data
        size_t entry_count() noexcept
        size_t destination(size_t entry) noexcept
        const T *values(size_t entry) noexcept

    cdef cppclass CppSparseMatrix "aequilibrae::matrix::cpp::SparseMatrix"[T]:
        CppSparseMatrix() noexcept  # Required for Cython usage

        # Matrix base
        size_t origin_count
        size_t destination_count
        size_t field_count
        T *data

        # Sparse extras
        size_t n_alloc
        size_t *row_index
        size_t *col_index
        CppSparseMatrixOriginView[T] origin(size_t index) noexcept


@cython.final
cdef class SparseMatrix(_MatrixBase):
    cdef readonly size_t[::1] row_index
    cdef readonly size_t[::1] col_index
    cdef readonly size_t n_alloc

    cdef set_index(self, const size_t[::1] row_index, const size_t[::1] col_index, size_t nnz, bint verify=?)
    cdef set_cores(self, dict cores)

    cdef CppSparseMatrix[cython.floating] view(self, cython.floating _type=*) except * nogil
