"""Packed dense matrices for native code and shared Python results."""

import operator

import numpy as np

from aequilibrae.matrix.aequilibrae_matrix import AequilibraeMatrix, _validate_matrix_key
from aequilibrae.utils.cython.array_allocations cimport array


cdef class DenseMatrix:
    """Hold square matrices in [origins, destinations, fields] order.

    Zone IDs and metadata belong to the Python result, not this buffer.
    Only float32 and float64 are supported, with one dtype for the whole
    buffer. Native callers use view[float]() or view[double]() and must keep
    this object alive while using the view.

    to_matrix() shares writable, named slices of the buffer. The returned
    AequilibraeMatrix keeps the buffer alive after this object is discarded.
    Adding or replacing matrices in that result does not change this buffer.
    """

    def __init__(self, zone_count, field_names, dtype=np.float64, fill_value=np.inf):
        if self._values is not None:
            raise RuntimeError("DenseMatrix cannot be reinitialised")
        zone_count = operator.index(zone_count)
        if zone_count < 0:
            raise ValueError("zone_count must be nonnegative")
        if isinstance(field_names, str):
            raise TypeError("field_names must be a sequence of names, not a string")

        names = tuple(field_names)
        for name in names:
            _validate_matrix_key(name)
        if len(set(names)) != len(names):
            raise ValueError("field_names must be unique")

        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("dtype must be float32 or float64")

        shape = (zone_count, zone_count, len(names))
        cdef float[:, :, ::1] values32
        cdef double[:, :, ::1] values64
        self._float32 = dtype == np.dtype(np.float32)
        if self._float32:
            values = array[float](shape, True, fill_value)
            values32 = values
            if zone_count and names:
                self._data = &values32[0, 0, 0]
        else:
            values = array[double](shape, True, fill_value)
            values64 = values
            if zone_count and names:
                self._data = &values64[0, 0, 0]

        self.origin_count = self.destination_count = zone_count
        self.field_count = len(names)
        self.field_names = names
        self.dtype = dtype
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

        Named arrays may be strided. The supplied index is copied, and normal
        assignment in the returned matrix keeps its usual copy rules.
        """
        if len(index) != self.origin_count:
            raise ValueError("index length must match zone_count")
        values = np.asarray(self._values)
        arrays = {name: values[:, :, field] for field, name in enumerate(self.field_names)}
        return AequilibraeMatrix._from_shared_arrays(arrays, index, index_name)

    cdef CppDenseMatrixView[MatrixFloat] view(self, MatrixFloat _type=0) except * nogil:
        """Borrow a typed view. The default argument allows view[float/double]()."""
        # Cython needs the fused type in an argument to specialise the method.
        if (MatrixFloat is float and not self._float32) or (MatrixFloat is double and self._float32):
            with gil:
                raise TypeError("view type must match the DenseMatrix dtype")

        cdef CppDenseMatrixView[MatrixFloat] output
        output.origin_count = self.origin_count
        output.destination_count = self.destination_count
        output.field_count = self.field_count
        output.data = <MatrixFloat *>self._data
        return output
