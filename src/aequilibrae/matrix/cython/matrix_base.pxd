from libc.stddef cimport size_t

cimport cython

cdef class _MatrixBase:
    cdef readonly tuple field_names
    cdef readonly object dtype
    cdef object _values
    cdef bint _float32

    cdef readonly size_t origin_count, destination_count, field_count
    cdef void *_data
