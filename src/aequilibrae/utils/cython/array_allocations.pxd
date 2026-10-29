cimport cython
from cython.view cimport array as cython_view_array
from libcpp cimport bool


ctypedef fused ArrayElement:
    char
    signed char
    unsigned char
    bool
    short
    unsigned short
    int
    unsigned int
    long
    unsigned long
    long long
    unsigned long long
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
