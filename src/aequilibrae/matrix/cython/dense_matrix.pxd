cimport cython

from aequilibrae.matrix.cython.matrix_base cimport _MatrixBase
from aequilibrae.matrix.cython.views cimport CppDenseMatrix


@cython.final
cdef class DenseMatrix(_MatrixBase):
    cdef CppDenseMatrix[cython.floating] view(self, cython.floating _type=*) except * nogil
