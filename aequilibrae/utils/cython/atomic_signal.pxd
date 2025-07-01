from libcpp.atomic cimport atomic
from libc.stdint cimport uint64_t

cdef extern from *:
    """
    #if defined(_WIN32) || defined(MS_WINDOWS) || defined(_MSC_VER)
      #include "stdlib.h"
      #define aeq_sleep(ms)  _sleep(ms)
    #else
      #include <time.h>
      void aeq_sleep(int ms) {             \
        struct timespec ts;                \
        ts.tv_sec = ms / 1000;             \
        ts.tv_nsec = (ms % 1000) * 1000000;\
        nanosleep(&ts, NULL);              \
      }
    #endif
    """
    void msleep "aeq_sleep"(int milliseconds) nogil

cdef class AtomicSignal:
    cdef:
        public object msg
        readonly int interval

        object __signal
        object __task
        object __stop
        object __resend

        atomic[uint64_t] __counter
        atomic[uint64_t] __total

    cpdef inline void set_total(self, uint64_t total) noexcept nogil
    cpdef inline void inc(AtomicSignal self) noexcept nogil
