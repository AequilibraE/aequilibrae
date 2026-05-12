from libcpp.atomic cimport atomic
from libc.stdint cimport uint64_t


cdef class Bar:
    cdef:
        public object msg

        object __signal

        atomic[uint64_t] __counter
        atomic[uint64_t] __total

        uint64_t __counter_old
        uint64_t __total_old

    cpdef inline void set_total(self, uint64_t total) noexcept nogil
    cpdef inline void set_counter(self, uint64_t value) noexcept nogil

    cpdef inline uint64_t get_total(self) noexcept nogil
    cpdef inline uint64_t get_counter(self) noexcept nogil

    cpdef inline void inc(Bar self) noexcept nogil
