from libc.stddef cimport size_t
from libcpp.vector cimport vector


cdef class SkimmingContext:
    cdef readonly object context
    cdef readonly size_t centroid_count
    cdef readonly size_t field_count
    cdef object _fields
    cdef vector[const double *] _field_pointers
    cdef object _od_skims
    cdef object _od_costs
    cdef object _od_turn_costs
    cdef double *_od_skims_ptr
    cdef double *_od_costs_ptr
    cdef double *_od_turn_costs_ptr
