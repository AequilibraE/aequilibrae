# cython: language_level=3
cimport cython
import operator
import numpy as np

from cython.view cimport array as cython_view_array
from libcpp.algorithm cimport fill_n


cpdef object array(shape, bint fill=True, ArrayElement fill_value=0):
    """Allocate a contiguous Cython buffer, optionally filling every element.

    Supports empty dimensions. Assign directly to a typed memoryview to retain
    the buffer, for example: array[double]((rows, columns), True, 0).
    """
    cdef cython_view_array buffer
    format_char = {
        "char": "c",
        "signed char": "b",
        "unsigned char": "B",
        "bool": "?",
        "short": "h",
        "unsigned short": "H",
        "int": "i",
        "unsigned int": "I",
        "long": "l",
        "unsigned long": "L",
        "long long": "q",
        "unsigned long long": "Q",
        "ssize_t": "q", # HACK: should be "n" but cython doesn't support it for some reason
        "size_t": "Q", # HACK: should be "N" but cython doesn't support it for some reason
        "float": "f",
        "double": "d",
    }[cython.typeof(fill_value)]

    if not isinstance(shape, tuple):
        shape = (shape,)
    shape = tuple([operator.index(dim) for dim in shape])
    if not shape or any([dim < 0 for dim in shape]):
        raise ValueError("shape must contain nonnegative dimensions")

    buffer = cython_view_array(
        shape=tuple([max(dim, 1) for dim in shape]),
        itemsize=sizeof(ArrayElement),
        format=format_char,
        mode="c",
        allocate_buffer=True,
    )

    if fill:
        fill_n(<ArrayElement*>buffer.data, buffer.len // buffer.itemsize, fill_value)

    if 0 in shape:
        return buffer[tuple([slice(0, dim) for dim in shape])]
    return buffer


def readonly_view(array):
    """Return a read-only NumPy view without copying or releasing its buffer.

    Python's memoryview supplies toreadonly(), which Cython views lack, and
    prevents NumPy from re-enabling writes.
    """
    return np.asarray(memoryview(array).toreadonly())
