from libcpp.vector cimport vector
from libcpp cimport nullptr
from cython.operator cimport dereference as d

import scipy.sparse
import numpy as np
import openmatrix as omx

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

    def to_disk(self, path, name: str):
        f = omx.open_file(path, "a")
        try:
            f[name] = self.to_scipy().tocsr().toarray()
        finally:
            f.close()

    @classmethod
    def from_disk(cls, path, names=None, aeq=False):
        """
        Read a OMX file and return a dictionary of matrix names to a scipy.sparse matrix, or
        aequilibrae.matrix.sparse matrix.
        """
        f = omx.open_file(path, "r")
        res = {}
        try:
            for matrix in (f.list_matrices() if names is None else names):
                if aeq:
                    res[matrix] = cls.from_matrix(f[matrix])
                else:
                    res[matrix] = scipy.sparse.csr_matrix(f[matrix])
            return res
        finally:
            f.close()


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

    def to_scipy(self, shape=None, dtype=np.float64):
        """
        Create scipy.sparse.coo_matrix from this COO matrix.
        """
        row = <size_t[:self.row.size()]>&d(self.row)[0]
        col = <size_t[:self.col.size()]>&d(self.col)[0]
        data = <double[:self.data.size()]>&d(self.data)[0]

        if shape is None:
            shape = self.shape

        return scipy.sparse.coo_matrix((data, (row, col)), dtype=dtype, shape=shape)

    @classmethod
    def from_matrix(cls, m):
        """
        Create COO matrix from an dense or scipy-like matrix.
        """
        if not isinstance(m, scipy.sparse.coo_matrix):
            m = scipy.sparse.coo_matrix(m)

        self = <COO?>cls()

        cdef size_t[:] row = m.row.astype(np.uint64), col = m.row.astype(np.uint64)
        cdef double[:] data = m.data

        self.row.insert(self.row.end(), &row[0], &row[-1] + 1)
        self.col.insert(self.col.end(), &col[0], &col[-1] + 1)
        self.data.insert(self.data.end(), &data[0], &data[-1] + 1)

        return self

    cdef void append(COO self, size_t i, size_t j, double v) noexcept nogil:
        self.row.push_back(i)
        self.col.push_back(j)
        self.data.push_back(v)
