from libc.stddef cimport size_t
from libcpp.vector cimport vector


cdef class SkimmingContext:
    cdef readonly object context
    cdef readonly size_t centroid_count
    cdef readonly size_t field_count
    cdef object link_fields
    cdef vector[const double *] field_pointers
    cdef double[:, :, ::1] od_skims_buffer
    cdef double[:, ::1] od_costs_buffer, od_turn_costs_buffer
