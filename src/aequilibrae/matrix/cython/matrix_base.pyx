import operator

import numpy as np
import cython

from aequilibrae.matrix.aequilibrae_matrix import _validate_matrix_key

cdef class _MatrixBase:
    def __init__(self, zone_count, field_names, dtype=np.float64, *, origin_count=None):
        if self._values is not None:
            raise RuntimeError(f"{self.__class__.__name__} cannot be reinitialised")

        zone_count = operator.index(zone_count)
        if zone_count < 0:
            raise ValueError("zone_count must be nonnegative")

        origin_count = zone_count if origin_count is None else operator.index(origin_count)
        if origin_count < 0:
            raise ValueError("origin_count must be nonnegative")

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

        self.origin_count = origin_count
        self.destination_count = zone_count
        self.field_count = len(names)
        self.field_names = names
        self.dtype = dtype
