from libc.stddef cimport size_t

cimport cython

from aequilibrae.matrix.cython.matrix_base cimport _MatrixBase

from aequilibrae.matrix.cython.views cimport CppSparseMatrix


@cython.final
cdef class SparseMatrix(_MatrixBase):
    cdef readonly size_t[::1] row_index
    cdef readonly size_t[::1] col_index
    cdef readonly size_t n_alloc

    cdef set_index(self, const size_t[::1] row_index, const size_t[::1] col_index, size_t nnz, bint verify=?)
    cdef dict validate_cores(self, dict cores, size_t nnz)
    cdef set_cores(self, dict cores)

    cdef CppSparseMatrix[cython.floating] view(self, cython.floating _type=*) except * nogil
