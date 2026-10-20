"""Packed dense matrices for native code and shared Python results."""

import operator

import numpy as np
import scipy.sparse as sparse

from aequilibrae.matrix.aequilibrae_matrix import AequilibraeMatrix, _validate_matrix_key

cimport cython
from libcpp cimport nullptr

from .matrix_base cimport _MatrixBase
from aequilibrae.utils.cython.array_allocations cimport array


@cython.final
cdef class SparseMatrix(_MatrixBase):
    """
    Hold sparse matrices in [origins, destinations, fields] order.

    See sparse_matrix.hpp for a detailed description.
    """

    def __init__(self, zone_count, field_names, dtype=np.float64, *, origin_count=None):
        super().__init__(zone_count=zone_count, field_names=field_names, dtype=dtype, origin_count=origin_count)

        cdef float[:, :, ::1] values32
        cdef double[:, :, ::1] values64
        self._float32 = dtype == np.dtype(np.float32)

        self._values = None
        self._data = nullptr
        self.row_index = array[size_t](self.origin_count + 1, True, 0)
        self.col_index = None

    @classmethod
    def from_csr(cls, shape: tuple[int, int], row_index, col_index, cores: dict, **kwargs):
        cdef:
            size_t[::1] _row_index = row_index.astype("uint64")
            size_t[::1] _col_index = col_index.astype("uint64")

        cdef SparseMatrix self = <SparseMatrix?>cls(zone_count=shape[0], field_names=cores.keys(), origin_count=shape[1])
        # cdef SparseMatrix[double, float] *matrix = <SparseMatrix[double, float]*?>self._sparse_mat

        # matrix.set_index(&_row_index[0], &_col_index[0] if len(col_index) else NULL, len(col_index))
        self.set_index(_row_index, _col_index, len(_col_index))
        self.set_cores(cores)

        return self

    @classmethod
    def from_scipy(cls, mat, **kwargs):
        return cls.from_csr(mat.shape, mat.indptr, mat.indices, {"matrix": mat.data})

    def to_scipy(self):
        shape = (self.origin_count, self.destination_count)
        return {
            name: sparse.csr_matrix((self._values[:, i], self.col_index, self.row_index), shape=shape)
            for i, name in enumerate(self.field_names)
        }


    # // Construct from an existing CSR index in pointer form (e.g. memory views)
    # // The length of row_index must be n_rows, the sentinel value at
    # // row_index[n_rows] is not copied, we set it ourselves.
    # void set_index(const size_t *const row_index, const size_t *const
    # col_index,
    #                size_t nnz, bool verify) {
    #   if (row_index == nullptr || col_index == nullptr) {
    #     return; // FIXME: should we throw something instead?
    #   } else if (nnz == 0) {
    #     throw std::length_error(
    #         "Cannot have a core with a row or column of length 0");
    #   }

    #   if (verify) {
    #     if (row_index[this->n_rows] != nnz) {
    #       throw std::logic_error("row_index not properly terminated.");
    #     }

    #     for (size_t row = 0; row < this->n_rows; row++) {
    #       size_t row_start = row_index[row];
    #       size_t row_end = row_index[row + 1];

    #       if (!(row_end >= row_start)) {
    #         throw std::logic_error("Row indices must be ascending");
    #       } else if (row_start > nnz) {
    #         throw std::logic_error("A value greater than the row_index sentinel
    #         "
    #                                "value must not appear in the index.");
    #       }

    #       for (size_t i = row_start; i < row_end; i++) {
    #         size_t col = col_index[i];

    #         if (col > this->n_cols) {
    #           throw std::logic_error(
    #               "Column value exceeds previously supplied bounds");
    #         }
    #       }
    #     }
    #   }

    cdef set_index(self, const size_t[::1] row_index, const size_t[::1] col_index, size_t nnz, bint verify = True):
        if nnz == 0:
            raise ValueError("cannot have a core with a row or column of length 0")

        cdef size_t row_start = 0, row_end = 0, col = 0, row = 0

        if verify:
            if row_index[self.origin_count] != nnz:
                raise ValueError("row_index not properly terminated")

            for row in range(0, self.origin_count):
                row_start = row_index[row]
                row_end = row_index[row + 1]

                if not row_start <= row_end:
                    raise ValueError("row indices must be ascending")
                elif row_start > nnz:
                    raise ValueError(f"a value greater than {nnz} must not appear in the index (found at index {row})")

                for i in range(row_start, row_end):
                    col = col_index[i]

                    if col > self.destination_count:
                        raise ValueError(f"column value exceeds shape (found {col}, with {self.destination_count} columns)")

        # //   // The number of cells we'll allocate is the length of the column index
        # //   this->n_alloc = nnz;
        # //   this->true_n_alloc = nnz;

        # //   // Copy the row index, setting the sentential value
        # //   memory::release(this->row_index);
        # //   access::rwp(this->row_index) = memory::acquire<size_t>(this->n_rows + 1);
        # //   std::memcpy(access::rwp(this->row_index), row_index,
        # //               this->n_rows * sizeof(size_t));
        # //   access::rwp(this->row_index)[this->n_rows] = this->n_alloc;

        # //   // Copy the col index
        # //   memory::release(this->col_index);
        # //   access::rwp(this->col_index) = memory::acquire<size_t>(this->n_alloc);
        # //   std::memcpy(access::rwp(this->col_index), col_index,
        # //               this->n_alloc * sizeof(size_t));
        # // }

        self.n_alloc = nnz
        self.row_index = row_index.copy()
        self.row_index[self.origin_count] = self.n_alloc
        self.col_index = col_index.copy()

    cdef set_cores(self, dict cores):
        if len(cores) != self.field_count:
            raise ValueError(f"expected {self.field_count} cores, got {len(cores)}")

        length = None
        for key, value in cores.items():
            if value.dtype != self.dtype:
                raise TypeError(f"expected {self.dtype}, got {value.dtype}")
            if value.ndim != 1:
                raise ValueError(f"expected 1D array, got {value.shape}")

            if length is None:
                length = len(value)

            if length is not None and length != len(value):
                raise ValueError(f"mismatched lengths for {key}, expected {length}, got {len(value)}")

        shape = (length, self.field_count)

        if self._float32:
            self._values = array[float](shape)
        else:
            self._values = array[double](shape)

        for i, value in enumerate(cores.values()):
            self._values[:, i] = value

    # @classmethod
    # def from_matrix(cls, matrix, dtype=None):
    #     """Copy all named matrices into one packed buffer, keeping their order.

    #     An omitted dtype requires matching float32 or float64 source dtypes.
    #     Supply a dtype to convert values. An empty matrix defaults to float64.
    #     The source is not changed. Its index and metadata are not retained.
    #     """
    #     if not isinstance(matrix, AequilibraeMatrix):
    #         raise TypeError("matrix must be an AequilibraeMatrix")

    #     names = tuple(matrix)
    #     if dtype is None:
    #         dtypes = {matrix[name].dtype for name in names}
    #         if len(dtypes) > 1:
    #             raise TypeError("source matrices have mixed dtypes; supply dtype explicitly")

    #         dtype = next(iter(dtypes), np.dtype(np.float64))

    #     cdef DenseMatrix output = cls(matrix.shape[0], names, dtype=dtype)
    #     values = np.asarray(output._values)
    #     for field, name in enumerate(names):
    #         np.copyto(values[:, :, field], matrix[name], casting="same_kind")

    #     return output

    # def to_matrix(self, index, index_name="main_index"):
    #     """Return an AequilibraeMatrix sharing this buffer without copying values.

    #     Only square buffers can be returned. Named arrays may be strided.
    #     The supplied index is copied, and normal assignment in the returned
    #     matrix keeps its usual copy rules.
    #     """
    #     if self.origin_count != self.destination_count:
    #         raise ValueError("only square buffers can be returned as an AequilibraeMatrix")

    #     if len(index) != self.origin_count:
    #         raise ValueError("index length must match zone_count")

    #     values = np.asarray(self._values)
    #     arrays = {name: values[:, :, field] for field, name in enumerate(self.field_names)}
    #     return AequilibraeMatrix._from_shared_arrays(arrays, index, index_name)

    cdef CppSparseMatrix[cython.floating] view(self, cython.floating _type=0) except * nogil:
        """Borrow a typed view. The default argument allows view[float/double]()."""
        # Cython needs the fused type in an argument to specialise the method.
        if (cython.floating is float and not self._float32) or (cython.floating is double and self._float32):
            with gil:
                raise TypeError("view type must match the SparseMatrix dtype")

        if self.col_index is None:
            raise RuntimeError("col_index is None, no values have been set")

        cdef CppSparseMatrix[cython.floating] output
        output.origin_count = self.origin_count
        output.destination_count = self.destination_count
        output.field_count = self.field_count
        output.data = <cython.floating *>self._data

        output.row_index = &self.row_index[0]
        output.col_index = &self.col_index[0]
        return output
