from libcpp.atomic cimport atomic
from libc.stdint cimport uint64_t


cdef class AtomicSignal:
    cdef:
        public object msg
        public float interval

        int __total
        object __signal
        object __task
        object __stop

        atomic[uint64_t] __counter

    cpdef inline void inc(AtomicSignal self) noexcept nogil
