"""Packed dense matrices for native code and shared Python results."""

from collections.abc import Mapping

import numpy as np

from aequilibrae.matrix.aequilibrae_matrix import AequilibraeMatrix

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
        """Copy all named matrices into one packed buffer, keeping their order."""
        if not isinstance(matrix, AequilibraeMatrix):
            raise TypeError("matrix must be an AequilibraeMatrix")

        for name, values in matrix.items():
            if (
                not np.issubdtype(values.dtype, np.number)
                or np.iscomplexobj(values)
                or np.issubdtype(values.dtype, np.bool_)
            ):
                raise TypeError("Matrix values must be real numbers")
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("Matrix values must be finite and nonnegative")

        if not matrix:
            return cls(matrix.shape[0], (), dtype=np.float64 if dtype is None else dtype)
        return cls.from_dict({name: matrix[name] for name in matrix}, dtype=dtype)

    @classmethod
    def from_dict(cls, matrices, dtype=None):
        """Copy named two-dimensional arrays into one packed buffer."""
        if not isinstance(matrices, Mapping):
            raise TypeError("matrices must be a mapping of names to arrays")

        if not matrices:
            raise ValueError("matrices must contain at least one array")

        names = tuple(matrices)
        sources = {name: np.asarray(matrices[name]) for name in names}
        shape = next(iter(sources.values())).shape

        for name, values in sources.items():
            if values.ndim != 2:
                raise ValueError(f"matrix {name!r} must be two-dimensional, got {values.shape}")
            if values.shape != shape:
                raise ValueError(
                    f"all matrices must have the same shape; {name!r} has shape {values.shape}, expected {shape}"
                )

        if dtype is None:
            dtypes = {values.dtype for values in sources.values()}
            if len(dtypes) > 1:
                raise TypeError("source matrices have mixed dtypes; supply dtype explicitly")

            dtype = next(iter(dtypes), np.dtype(np.float64))

        cdef DenseMatrix output = cls(shape[1], names, dtype=dtype, origin_count=shape[0])
        values = np.asarray(output._values)
        for field, name in enumerate(names):
            np.copyto(values[:, :, field], sources[name], casting="same_kind")

        return output

    def to_matrix(self, index, index_name="main_index"):
        """Return an AequilibraeMatrix sharing this buffer without copying values."""
        if self.origin_count != self.destination_count:
            raise ValueError("only square buffers can be returned as an AequilibraeMatrix")

        if len(index) != self.origin_count:
            raise ValueError("index length must match zone_count")

        values = np.asarray(self._values)
        arrays = {name: values[:, :, field] for field, name in enumerate(self.field_names)}
        return AequilibraeMatrix._from_shared_arrays(arrays, index, index_name)

    cdef CppDenseMatrix[cython.floating] view(self, cython.floating _type=0) except * nogil:
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
