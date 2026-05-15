from libcpp.vector cimport vector
from libcpp.utility cimport move
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

    def __cinit__(self, shape=None, f64: bool = True):
        """C level init. For C memory allocation and initialisation. Called exactly once per object."""
        self.shape = shape
        self.f64 = f64

        self.row = make_unique[vector[size_t]]()
        self.col = make_unique[vector[size_t]]()

        if self.f64:
            self.f64_data = make_unique[vector[double]]()
        else:
            self.f32_data = make_unique[vector[float]]()

    def __init__(self, *args, **kwargs):
        pass

    def to_scipy(self, shape=None):
        """
        Create scipy.sparse.coo_matrix from this COO matrix.
        """
        cdef:
            size_t[:] row, col
            double[:] f64_data
            float[:] f32_data
            size_t length

        # We can't construct a 0x0 matrix from 3x 0-sized arrays but we can tell scipy to make one.
        if not d(self.row).size() or not d(self.col).size():
            return scipy.sparse.coo_matrix((0, 0))

        length = d(self.row).size()
        row = <size_t[:length]>d(self.row).data()

        length = d(self.col).size()
        col = <size_t[:length]>d(self.col).data()

        if shape is None:
            shape = self.shape

        if self.f64:
            length = d(self.f64_data).size()
            f64_data = <double[:length]>d(self.f64_data).data()
            return scipy.sparse.coo_matrix((f64_data, (row, col)), dtype=np.float64, shape=shape, copy=True)
        else:
            length = d(self.f32_data).size()
            f32_data = <float[:length]>d(self.f32_data).data()
            return scipy.sparse.coo_matrix((f32_data, (row, col)), dtype=np.float32, shape=shape, copy=True)

    @classmethod
    def from_matrix(cls, m):
        """
        Create COO matrix from an dense or scipy-like matrix.
        """
        if not isinstance(m, scipy.sparse.coo_matrix):
            m = scipy.sparse.coo_matrix(m, dtype=m.dtype)

        self = <COO?>cls(f64=m.data.dtype == "float64")

        cdef size_t[:] row = m.row.astype(np.uint64), col = m.col.astype(np.uint64)
        cdef double[:] f64_data
        cdef float[:] f32_data

        d(self.row).insert(d(self.row).begin(), &row[0], &row[-1] + 1)
        d(self.col).insert(d(self.col).begin(), &col[0], &col[-1] + 1)

        if self.f64:
            f64_data = m.data
            d(self.f64_data).insert(d(self.f64_data).begin(), &f64_data[0], &f64_data[-1] + 1)
        else:
            f32_data = m.data
            d(self.f32_data).insert(d(self.f32_data).begin(), &f32_data[0], &f32_data[-1] + 1)

        return self

    @staticmethod
    cdef void init_f64_struct(COO_f64_struct &struct) noexcept nogil:
        struct.row = make_unique[vector[size_t]]()
        struct.col = make_unique[vector[size_t]]()
        struct.f64_data = make_unique[vector[double]]()

    @staticmethod
    cdef void init_f32_struct(COO_f32_struct &struct) noexcept nogil:
        struct.row = make_unique[vector[size_t]]()
        struct.col = make_unique[vector[size_t]]()
        struct.f32_data = make_unique[vector[float]]()

    @staticmethod
    cdef object from_f64_struct(COO_f64_struct &struct):
        cdef COO self = COO.__new__(COO)
        self.row = move(struct.row)
        self.col = move(struct.col)
        self.f64 = True
        self.f64_data = move(struct.f64_data)

        return self

    @staticmethod
    cdef object from_f32_struct(COO_f32_struct &struct):
        cdef COO self = COO.__new__(COO)
        self.row = move(struct.row)
        self.col = move(struct.col)
        self.f64 = False
        self.f32_data = move(struct.f32_data)

        return self

    cdef void f64_append(COO self, size_t i, size_t j, double v) noexcept nogil:
        d(self.row).push_back(i)
        d(self.col).push_back(j)
        d(self.f64_data).push_back(v)

    cdef void f32_append(COO self, size_t i, size_t j, float v) noexcept nogil:
        d(self.row).push_back(i)
        d(self.col).push_back(j)
        d(self.f32_data).push_back(v)

    @staticmethod
    cdef void f64_struct_append(COO_f64_struct &struct, size_t i, size_t j, double v) noexcept nogil:
        d(struct.row).push_back(i)
        d(struct.col).push_back(j)
        d(struct.f64_data).push_back(v)

    @staticmethod
    cdef void f32_struct_append(COO_f32_struct &struct, size_t i, size_t j, float v) noexcept nogil:
        d(struct.row).push_back(i)
        d(struct.col).push_back(j)
        d(struct.f32_data).push_back(v)
