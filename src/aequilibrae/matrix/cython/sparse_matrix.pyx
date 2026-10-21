"""Packed sparse matrices for native code and shared Python results."""

from collections.abc import Mapping

import numpy as np
import scipy.sparse as sparse

from aequilibrae.matrix.aequilibrae_matrix import AequilibraeSparseMatrix

cimport cython
from libcpp cimport nullptr

from .matrix_base cimport _MatrixBase
from aequilibrae.utils.cython.array_allocations cimport array, array_pointer


@cython.final
cdef class SparseMatrix(_MatrixBase):
    """
    Hold sparse matrices in [origins, destinations, fields] order.

    See sparse_matrix.hpp for a detailed description.
    """

    def __init__(self, zone_count, field_names, dtype=np.float64, *, origin_count=None):
        super().__init__(zone_count=zone_count, field_names=field_names, dtype=dtype, origin_count=origin_count)

        self._float32 = self.dtype == np.dtype(np.float32)

        if self._float32:
            self._values = array[float]((0, self.field_count))
        else:
            self._values = array[double]((0, self.field_count))

        self._data = nullptr
        self.row_index = array[size_t](self.origin_count + 1, True, 0)
        self.col_index = array[size_t](0)
        self.n_alloc = 0

    @classmethod
    def from_csr(cls, shape: tuple[int, int], row_index, col_index, cores: dict, dtype=None):
        """Copy CSR indices and named value arrays into one packed buffer.

        The indices are shared by all fields. An omitted dtype requires
        matching float32 or float64 source dtypes. Supply a dtype to convert
        values. An empty field collection defaults to float64. Column indices
        in each row must be sorted and unique. Stored zeros are kept.
        """
        cores = {name: np.asarray(value) for name, value in cores.items()}
        if dtype is None:
            dtypes = {value.dtype for value in cores.values()}
            if len(dtypes) > 1:
                raise TypeError("source matrices have mixed dtypes; supply dtype explicitly")
            dtype = next(iter(dtypes), np.dtype(np.float64))

        origins, destinations = shape
        cdef SparseMatrix self = <SparseMatrix?>cls(
            zone_count=destinations, field_names=cores.keys(), origin_count=origins, dtype=dtype
        )
        for index in (row_index, col_index):
            index = np.asarray(index)
            if index.ndim != 1 or (index.size and index.dtype.kind not in "iu"):
                raise TypeError("CSR indices must be one-dimensional integer arrays")
            if index.size and index.min() < 0:
                raise ValueError("CSR indices must be nonnegative")

        cdef const size_t[::1] _row_index = np.ascontiguousarray(row_index, dtype=np.uintp)
        cdef const size_t[::1] _col_index = np.ascontiguousarray(col_index, dtype=np.uintp)
        self.set_index(_row_index, _col_index, len(_col_index))
        self.set_cores(cores)

        return self

    @classmethod
    def from_scipy(cls, mat, **kwargs):
        """Copy to canonical CSR, keeping stored zeros and leaving mat unchanged."""
        mat = mat.tocsr(copy=True)
        mat.check_format(full_check=True)
        mat.sum_duplicates()
        mat.sort_indices()
        return cls.from_csr(mat.shape, mat.indptr, mat.indices, {"matrix": mat.data}, **kwargs)

    def to_scipy(self):
        shape = (self.origin_count, self.destination_count)
        values = np.asarray(self._values)
        return {
            name: sparse.csr_matrix((values[:, i], self.col_index, self.row_index), shape=shape, copy=False)
            for i, name in enumerate(self.field_names)
        }

    cdef set_index(self, const size_t[::1] row_index, const size_t[::1] col_index, size_t nnz, bint verify = True):
        cdef size_t row_start = 0, row_end = 0, col = 0, row = 0, i = 0

        if len(row_index) != self.origin_count + 1:
            raise ValueError("row_index length must be origin_count + 1")
        if len(col_index) != nnz:
            raise ValueError("col_index length must match nnz")

        if verify:
            if row_index[0] != 0:
                raise ValueError("row_index must start at zero")
            if row_index[self.origin_count] != nnz:
                raise ValueError("row_index not properly terminated")

            for row in range(0, self.origin_count):
                row_start = row_index[row]
                row_end = row_index[row + 1]

                if not row_start <= row_end:
                    raise ValueError("row indices must be ascending")
                elif row_end > nnz:
                    raise ValueError(
                        f"a value greater than {nnz} must not appear in the index (found at index {row + 1})"
                    )

                for i in range(row_start, row_end):
                    col = col_index[i]

                    if col >= self.destination_count:
                        raise ValueError(
                            f"column value exceeds shape (found {col}, with {self.destination_count} columns)"
                        )
                    if i > row_start and col <= col_index[i - 1]:
                        raise ValueError("column indices in each row must be sorted and unique")

        self.n_alloc = nnz

        self.row_index = array[size_t](self.origin_count + 1, False)
        self.row_index[:] = row_index
        self.row_index[self.origin_count] = self.n_alloc

        self.col_index = array[size_t](nnz, False)
        self.col_index[:] = col_index

    cdef set_cores(self, dict cores):
        if len(cores) != self.field_count:
            raise ValueError(f"expected {self.field_count} cores, got {len(cores)}")

        for key, value in cores.items():
            if not np.can_cast(value.dtype, self.dtype, casting="same_kind"):
                raise TypeError(f"expected values that can be cast to {self.dtype}, got {value.dtype}")
            if value.ndim != 1:
                raise ValueError(f"expected 1D array, got {value.shape}")
            if len(value) != self.n_alloc:
                raise ValueError(f"mismatched length for {key}, expected {self.n_alloc}, got {len(value)}")

        shape = (self.n_alloc, self.field_count)

        # Can't conditionally declare variables in Cython
        cdef float[:, ::1] values32
        cdef double[:, ::1] values64
        self._data = nullptr

        if self._float32:
            self._values = array[float](shape)
            values32 = self._values
            if self.n_alloc and self.field_count:
                self._data = &values32[0, 0]
        else:
            self._values = array[double](shape)
            values64 = self._values
            if self.n_alloc and self.field_count:
                self._data = &values64[0, 0]

        values = np.asarray(self._values)
        for i, name in enumerate(self.field_names):
            np.copyto(values[:, i], cores[name], casting="same_kind")

    @classmethod
    def from_matrix(cls, matrix, dtype=None):
        """Copy all named matrices into one packed buffer, keeping their order.

        Use the union of stored positions and fill missing field values with
        zero. Source sparse formats are not changed. An omitted dtype requires
        matching float32 or float64 source dtypes. Supply a dtype to convert
        values. An empty matrix defaults to float64. The source is not changed.
        Its index and metadata are not retained.
        """
        if not isinstance(matrix, AequilibraeSparseMatrix):
            raise TypeError("matrix must be an AequilibraeSparseMatrix")

        names = tuple(matrix)
        if not names:
            return cls(
                matrix.shape[1], (), dtype=np.float64 if dtype is None else dtype, origin_count=matrix.shape[0]
            )
        return cls.from_dict({name: matrix[name] for name in names}, dtype=dtype)

    @classmethod
    def from_dict(cls, matrices, dtype=None):
        """Copy named two-dimensional arrays into one packed sparse buffer"""
        if not isinstance(matrices, Mapping):
            raise TypeError("matrices must be a mapping of names to arrays")
        if not matrices:
            raise ValueError("matrices must contain at least one array")

        names = tuple(matrices)
        sources = {}
        original_formats = {}
        shapes = set()
        dtypes = set()

        for name, matrix in matrices.items():
            if sparse.issparse(matrix):
                value = matrix.tocsr(copy=True)
                value.check_format(full_check=True)
                value.sum_duplicates()
                value.sort_indices()
                original_formats[name] = matrix.format
            else:
                dense = np.asarray(matrix)
                if dense.ndim != 2:
                    raise ValueError(f"matrix {name!r} must be two-dimensional, got {dense.shape}")
                value = sparse.csr_array(dense)
                original_formats[name] = None

            if len(value.shape) != 2:
                raise ValueError(f"matrix {name!r} must be two-dimensional, got {value.shape}")

            shapes.add(value.shape)
            dtypes.add(value.dtype)
            sources[name] = value

        if len(shapes) != 1:
            raise ValueError("all matrices must have the same shape")

        shape = next(iter(shapes))
        if dtype is None:
            if len(dtypes) > 1:
                raise TypeError("source matrices have mixed dtypes, supply dtype explicitly to convert them")
            dtype = next(iter(dtypes))

        pattern = sparse.csr_array(shape, dtype=bool)
        for name, value in sources.items():
            # Stored zeros are positions too, including duplicates that sum to zero.
            positions = sparse.csr_array(
                (np.ones(value.nnz, dtype=bool), value.indices, value.indptr), shape=shape
            )

            if original_formats[name] == "dia":
                # DIA drops stored zeros when converted to CSR, so mark its diagonals first.
                positions = matrices[name].astype(bool, copy=True)
                positions.data[:] = True
                positions = positions.tocsr()

            pattern = pattern.maximum(positions)

        rows, cols = pattern.nonzero()
        cores = {
            name: np.asarray(value[rows, cols]).reshape(-1) if pattern.nnz else np.empty(0, dtype=value.dtype)
            for name, value in sources.items()
        }
        return cls.from_csr(shape, pattern.indptr, pattern.indices, cores, dtype=dtype)

    def to_matrix(self, index, index_name="main_index"):
        """Return an AequilibraeSparseMatrix sharing this buffer without copying values.

        Only square buffers can be returned. Named CSR data arrays may be
        strided. The supplied index is copied, and normal assignment in the
        returned matrix keeps its usual copy rules. Shared value changes must
        keep the stored positions unchanged.
        """
        if self.origin_count != self.destination_count:
            raise ValueError("only square buffers can be returned as an AequilibraeSparseMatrix")

        if len(index) != self.origin_count:
            raise ValueError("index length must match zone_count")

        values = np.asarray(self._values)
        rows = np.asarray(self.row_index, dtype=np.intp)
        cols = np.asarray(self.col_index, dtype=np.intp)
        shape = (self.origin_count, self.destination_count)
        arrays = {
            name: sparse.csr_array((values[:, field], cols, rows), shape=shape, copy=False)
            for field, name in enumerate(self.field_names)
        }
        return AequilibraeSparseMatrix._from_shared_arrays(arrays, index, index_name)

    cdef CppSparseMatrix[cython.floating] view(self, cython.floating _type=0) except * nogil:
        """Borrow a typed view while the owner lives. Allows view[float/double]()."""
        # Cython needs the fused type in an argument to specialise the method.
        if (cython.floating is float and not self._float32) or (cython.floating is double and self._float32):
            with gil:
                raise TypeError("view type must match the SparseMatrix dtype")

        cdef CppSparseMatrix[cython.floating] output
        output.origin_count = self.origin_count
        output.destination_count = self.destination_count
        output.field_count = self.field_count
        output.data = <cython.floating *>self._data
        output.n_alloc = self.n_alloc

        output.row_index = array_pointer(self.row_index)
        output.col_index = array_pointer(self.col_index)
        return output
