cimport cython
from cython.view cimport array as cython_view_array
from libc.stdint cimport int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from libcpp cimport bool


# Preserve external typedefs in pointer return types. Mixing fixed-width and
# native integer types lets Cython resolve typedefs to incompatible C++ types.
ctypedef fused ArrayElement:
    int8_t
    int16_t
    int32_t
    int64_t
    uint8_t
    uint16_t
    uint32_t
    uint64_t
    bool
    ssize_t
    size_t
    float
    double


cpdef object array(object shape, bint fill=*, ArrayElement fill_value=*)


cdef inline ArrayElement *array_pointer(ArrayElement[::1] view) noexcept nogil:
    """Return a writable buffer pointer, or NULL for an empty view."""
    with cython.boundscheck(False), cython.wraparound(False):
        return &view[0] if view.shape[0] else NULL


cdef inline const ArrayElement *const_array_pointer(const ArrayElement[::1] view) noexcept nogil:
    """Return a read-only buffer pointer, or NULL for an empty view."""
    with cython.boundscheck(False), cython.wraparound(False):
        return &view[0] if view.shape[0] else NULL
