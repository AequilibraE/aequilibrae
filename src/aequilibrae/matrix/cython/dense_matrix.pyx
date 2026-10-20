"""Packed dense matrices for native code and shared Python results."""

import operator

import numpy as np

from aequilibrae.matrix.aequilibrae_matrix import AequilibraeMatrix, _validate_matrix_key

cimport cython
from .matrix_base cimport _MatrixBase
from aequilibrae.utils.cython.array_allocations cimport array


@cython.final
cdef class DenseMatrix(_MatrixBase):
    """Hold matrices in [origins, destinations, fields] order."""

    def __init__(self, zone_count, field_names, dtype=np.float64, fill_value=np.inf, *, origin_count=None):
        super().__init__(zone_count=zone_count, field_names=field_names, dtype=dtype, origin_count=origin_count)

        cdef float[:, :, ::1] values32
        cdef double[:, :, ::1] values64
        self._float32 = self.dtype == np.dtype(np.float32)

        shape = (self.origin_count, self.destination_count, self.field_count)

        if self._float32:
            values = array[float](shape, True, fill_value)
            values32 = values
            if self.origin_count and self.destination_count and self.field_names:
                self._data = &values32[0, 0, 0]
        else:
            values = array[double](shape, True, fill_value)
            values64 = values
            if self.origin_count and self.destination_count and self.field_names:
                self._data = &values64[0, 0, 0]

        self._values = values

    @classmethod
    def from_matrix(cls, matrix, dtype=None):
        """Copy all named matrices into one packed buffer, keeping their order.

        An omitted dtype requires matching float32 or float64 source dtypes.
        Supply a dtype to convert values. An empty matrix defaults to float64.
        The source is not changed. Its index and metadata are not retained.
        """
        if not isinstance(matrix, AequilibraeMatrix):
            raise TypeError("matrix must be an AequilibraeMatrix")

        names = tuple(matrix)
        if dtype is None:
            dtypes = {matrix[name].dtype for name in names}
            if len(dtypes) > 1:
                raise TypeError("source matrices have mixed dtypes; supply dtype explicitly")

            dtype = next(iter(dtypes), np.dtype(np.float64))

        cdef DenseMatrix output = cls(matrix.shape[0], names, dtype=dtype)
        values = np.asarray(output._values)
        for field, name in enumerate(names):
            np.copyto(values[:, :, field], matrix[name], casting="same_kind")

        return output

    def to_matrix(self, index, index_name="main_index"):
        """Return an AequilibraeMatrix sharing this buffer without copying values.

        Only square buffers can be returned. Named arrays may be strided.
        The supplied index is copied, and normal assignment in the returned
        matrix keeps its usual copy rules.
        """
        if self.origin_count != self.destination_count:
            raise ValueError("only square buffers can be returned as an AequilibraeMatrix")

        if len(index) != self.origin_count:
            raise ValueError("index length must match zone_count")

        values = np.asarray(self._values)
        arrays = {name: values[:, :, field] for field, name in enumerate(self.field_names)}
        return AequilibraeMatrix._from_shared_arrays(arrays, index, index_name)

    cdef CppDenseMatrix[cython.floating] view(self, cython.floating _type=0) except * nogil:
        """Borrow a typed view. The default argument allows view[float/double]()."""
        # Cython needs the fused type in an argument to specialise the method.
        if (cython.floating is float and not self._float32) or (cython.floating is double and self._float32):
            with gil:
                raise TypeError("view type must match the DenseMatrix dtype")

        cdef CppDenseMatrix[cython.floating] output
        output.origin_count = self.origin_count
        output.destination_count = self.destination_count
        output.field_count = self.field_count
        output.data = <cython.floating *>self._data
        return output
