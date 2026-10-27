"""Packed sparse matrices for native code and shared Python results."""

from collections.abc import Mapping

import numpy as np
import scipy.sparse as sparse

from aequilibrae.matrix.aequilibrae_matrix import AequilibraeMatrix, AequilibraeSparseMatrix

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
        """Copy CSR indices and named value arrays into one packed buffer"""
        cores = {name: np.asarray(value) for name, value in cores.items()}

        if dtype is None:
            dtypes = {value.dtype for value in cores.values()}
            if len(dtypes) > 1:
                raise TypeError("source matrices have mixed dtypes. Supply dtype explicitly")
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

        packed_cores = self.validate_cores(cores, len(_col_index))
        active = np.zeros(len(_col_index), dtype=np.bool_)
        for value in packed_cores.values():
            active |= value != 0

        if not np.all(active):
            active_rows = np.searchsorted(
                np.asarray(_row_index)[1:],
                np.flatnonzero(active),
                side="right",
            )
            filtered_row_index = np.zeros(origins + 1, dtype=np.uintp)
            filtered_row_index[1:] = np.cumsum(
                np.bincount(active_rows, minlength=origins),
                dtype=np.uintp,
            )
            filtered_col_index = np.asarray(_col_index)[active]
            packed_cores = {name: value[active] for name, value in packed_cores.items()}
            self.set_index(filtered_row_index, filtered_col_index, len(filtered_col_index))

        self.set_cores(packed_cores)
        return self

    @classmethod
    def from_scipy(cls, mat, **kwargs):
        """Copy to canonical CSR, removing explicit zeros and leaving mat unchanged."""
        return cls.from_dict({"matrix": mat}, **kwargs)

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

    cdef dict validate_cores(self, dict cores, size_t nnz):
        if len(cores) != self.field_count:
            raise ValueError(f"expected {self.field_count} cores, got {len(cores)}")

        validated = {}
        for key, value in cores.items():
            value = np.asarray(value)
            if not np.can_cast(value.dtype, self.dtype, casting="same_kind"):
                raise TypeError(f"expected values that can be cast to {self.dtype}, got {value.dtype}")
            if value.ndim != 1:
                raise ValueError(f"expected 1D array, got {value.shape}")
            if len(value) != nnz:
                raise ValueError(f"mismatched length for {key}, expected {nnz}, got {len(value)}")
            validated[key] = value.astype(self.dtype, copy=False)
        return validated

    cdef set_cores(self, dict cores):
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
    def from_matrix(cls, matrix, dtype=None, *, transpose=False):
        """Copy all named matrices into one packed buffer, keeping their order, optionally transposed."""
        if not isinstance(matrix, (AequilibraeMatrix, AequilibraeSparseMatrix)):
            raise TypeError("matrix must be an AequilibraeMatrix or AequilibraeSparseMatrix")

        sources = {}
        for name, value in matrix.items():
            if value.shape != matrix.shape:
                raise ValueError("matrix fields must match the zone index shape")
            sources[name] = value

        if not sources:
            return cls(
                zone_count=matrix.shape[1],
                field_names=(),
                dtype=np.float64 if dtype is None else dtype,
                origin_count=matrix.shape[0],
            )
        return cls.from_dict(sources, dtype=dtype, transpose=transpose)

    @classmethod
    def from_dict(cls, matrices, dtype=None, *, transpose=False):
        """Pack named dense or sparse matrices using their canonical shared COO positions, optionally transposed."""
        if not isinstance(matrices, Mapping):
            raise TypeError("matrices must be a mapping of names to arrays")
        if not matrices:
            raise ValueError("matrices must contain at least one array")

        # Check all inputs before converting or copying any of them.
        shape = None
        source_dtype = None
        inputs = {}
        for name, matrix in matrices.items():
            if not sparse.issparse(matrix):
                matrix = np.asarray(matrix)
            if matrix.ndim != 2:
                raise ValueError(f"matrix {name!r} must be two-dimensional, got {matrix.shape}")

            if shape is None:
                shape = matrix.shape
            elif matrix.shape != shape:
                raise ValueError(f"all matrices must have the same shape; expected {shape}, got {matrix.shape}")

            if source_dtype is None:
                source_dtype = matrix.dtype
            elif matrix.dtype != source_dtype and dtype is None:
                raise TypeError("source matrices have mixed dtypes, supply dtype explicitly to convert them")

            if (
                not np.issubdtype(matrix.dtype, np.number)
                or np.iscomplexobj(matrix)
                or np.issubdtype(matrix.dtype, np.bool_)
            ):
                raise TypeError("Matrix values must be real numbers")

            inputs[name] = matrix

        dtype = np.dtype(source_dtype if dtype is None else dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("dtype must be float32 or float64")

        if transpose:
            shape = shape[::-1]

        # Canonicalise each field with SciPy before constructing the union of
        # CSR indices.
        coo_matrices = {}
        for name, matrix in inputs.items():
            if transpose:
                matrix = matrix.T

            if sparse.issparse(matrix):
                source = matrix.tocoo(copy=False)
                # Cast values before summing duplicates to avoid source-dtype overflow.
                csr = sparse.coo_array(
                    (source.data.astype(dtype), (source.row, source.col)), shape=shape,
                ).tocsr()
            else:
                csr = sparse.csr_array(np.asarray(matrix, dtype=dtype))
            csr.sum_duplicates()
            csr.eliminate_zeros()
            csr.sort_indices()
            coo_matrices[name] = csr.tocoo(copy=False)

        # Join all stored coordinates, then collapse them into one shared pattern.
        rows = np.concatenate([matrix.row for matrix in coo_matrices.values()])
        cols = np.concatenate([matrix.col for matrix in coo_matrices.values()])
        coordinates, positions = np.unique(np.column_stack((rows, cols)), axis=0, return_inverse=True)
        row_index = np.zeros(shape[0] + 1, dtype=np.uintp)
        row_index[1:] = np.cumsum(np.bincount(coordinates[:, 0], minlength=shape[0]), dtype=np.uintp)
        col_index = coordinates[:, 1]

        # Read every class at the union positions.
        cores = {}
        offset = 0
        for name, matrix in coo_matrices.items():
            values = np.zeros(len(coordinates), dtype=dtype)
            count = matrix.nnz
            values[positions[offset:offset + count]] = matrix.data
            cores[name] = values
            offset += count

        return cls.from_csr(shape, row_index, col_index, cores, dtype=dtype)

    def to_matrix(self, index, index_name="main_index"):
        """Return an AequilibraeSparseMatrix sharing this buffer without copying values"""
        if self.origin_count != self.destination_count:
            raise ValueError("only square buffers can be returned as an AequilibraeSparseMatrix")

        if len(index) != self.origin_count:
            raise ValueError("index length must match zone_count")

        values = np.asarray(self._values)
        rows = np.asarray(self.row_index)
        cols = np.asarray(self.col_index)
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
