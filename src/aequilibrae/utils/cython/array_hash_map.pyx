cimport cython
import operator
import numpy as np

from libc.stdint cimport uint64_t
from libcpp.memory cimport unique_ptr
from aequilibrae.utils.cython.array_allocations cimport array, array_pointer, const_array_pointer


@cython.final
cdef class ArrayHashMap:
    """Initialise an immutable uint64 hash map from parallel arrays.

    :Arguments:
        **keys**, **values**: One-dimensional, contiguous uint64 buffers of equal
        length. Pairs are copied into C++ storage, keeping the last value for
        duplicate keys. Read-only buffers are accepted.

    Index with an integer to obtain a Python integer, or with a one-dimensional,
    contiguous uint64 buffer to obtain a new NumPy uint64 array in query order.
    Both forms raise ``KeyError`` for a missing key; bulk lookup reports the
    first missing key without returning partial results.
    """

    cdef unique_ptr[CppArrayHashMap[uint64_t, uint64_t]] _map

    def __cinit__(self, const uint64_t[::1] keys not None, const uint64_t[::1] values not None):
        if keys.shape[0] != values.shape[0]:
            raise ValueError("keys and values must have equal lengths")
        self._map.reset(new CppArrayHashMap[uint64_t, uint64_t](
            const_array_pointer(keys), const_array_pointer(values), keys.shape[0]
        ))

    def __getitem__(self, key):
        cdef uint64_t scalar_key, scalar_value
        cdef const uint64_t[::1] keys
        cdef uint64_t[::1] values
        cdef size_t missing, count

        if isinstance(key, (int, np.integer)):
            scalar_key = operator.index(key)
            if self._map.get().lookup(&scalar_key, &scalar_value, 1) != 1:
                raise KeyError(scalar_key)
            return scalar_value

        if key is None:
            raise ValueError("keys must be a one-dimensional, contiguous uint64 buffer")
        keys = key
        count = keys.shape[0]
        values = array[uint64_t](count, False)
        with nogil:
            missing = self._map.get().lookup(const_array_pointer(keys), array_pointer(values), count)
        if missing != count:
            raise KeyError(keys[missing])
        return np.asarray(values)
