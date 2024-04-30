from libcpp.vector cimport vector
from libcpp cimport nullptr
from cython.operator cimport dereference as d

import scipy.sparse
import numpy as np

cdef class Sparse:
    """
    A class to implement sparse matrix operations such as reading, writing, and indexing
    """

    def __cinit__(self):
        """C level init. For C memory allocation and initialisation. Called exactly once per object."""
        pass

    def __init__(self):
        """Python level init, may be called multiple times, for things that can't be done in __cinit__."""
        pass

    def __dealloc__(self):
        """
        C level deallocation. For freeing memory allocated by this object. *Must* have GIL, `self` may be in a
        partially deallocated state already.
        """
        pass


cdef class COO(Sparse):
    """
    A class to implement sparse matrix operations such as reading, writing, and indexing
    """

    def __cinit__(self):
        """C level init. For C memory allocation and initialisation. Called exactly once per object."""

        self.row = new vector[size_t]()
        self.col = new vector[size_t]()
        self.data = new vector[double]()

    def __init__(self, shape=None):
        """Python level init, may be called multiple times, for things that can't be done in __cinit__."""

        self.shape = shape

    def __dealloc__(self):
        """
        C level deallocation. For freeing memory allocated by this object. *Must* have GIL, `self` may be in a
        partially deallocated state already.
        """

        del self.row
        self.row = <vector[size_t] *>nullptr

        del self.col
        self.col = <vector[size_t] *>nullptr

        del self.data
        self.data = <vector[double] *>nullptr

    def to_scipy(self, shape=None):
        row = <size_t[:self.row.size()]>&d(self.row)[0]
        col = <size_t[:self.col.size()]>&d(self.col)[0]
        data = <double[:self.data.size()]>&d(self.data)[0]

        if shape is None:
            shape = self.shape

        return scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float64, shape=shape)

    cdef void append(COO self, size_t i, size_t j, double v) noexcept nogil:
        self.row.push_back(i)
        self.col.push_back(j)
        self.data.push_back(v)
