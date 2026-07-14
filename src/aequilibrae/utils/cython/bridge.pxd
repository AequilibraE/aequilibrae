from libcpp cimport bool
from libcpp.atomic cimport atomic
from libcpp.deque cimport deque
from libcpp.string cimport string
from libcpp.mutex cimport mutex
from libcpp.utility cimport pair
from libc.stdint cimport uint8_t


# std::make_pair is not available in the Cython libcpp.utilities shim. We'll import it ourselves based on the C++11 til
# C++14 definition because C++14 makes this signature weird. See https://github.com/cython/cython/issues/2706
cdef extern from "<utility>" namespace "std" nogil:
    pair[T, U] make_pair[T, U](T&& t, U&& u)


cdef extern from "aeq_log.hpp":
    # We lie to the Cython compiler here, Cython thinks this is a C function but it's actually a macro. We give the
    # arguments types so that Cython can attempt to enforce them for us. This lets us have statically checked types for
    # the arguments. Admittedly the errors Cython raises aren't descriptive, and when call from Python the type check is
    # deferred till runtime. But importantly msg_exp isn't computed if the provided is below the level we set.

    cppclass AeqLogClosure:
        mutex _log_queue_mutex
        deque[pair[int, string]] _log_queue
        uint8_t c_level

        void _log(uint8_t level, string msg)

    string aeq_format_string(...) noexcept nogil
    void log"AEQ_LOG"(AeqLogClosure *, int lvl, string msg_exp) noexcept nogil

    cdef:
        uint8_t DEBUG"AEQ_LOG_DEBUG"
        uint8_t INFO"AEQ_LOG_INFO"
        uint8_t WARNING"AEQ_LOG_WARNING"
        uint8_t ERROR"AEQ_LOG_ERROR"
        uint8_t CRITICAL"AEQ_LOG_CRITICAL"


cdef class Bridge:
    cdef:
        public object task
        atomic[bool] _stop
        public object bars

        object __logger
        object __exception_queue

        AeqLogClosure *c

    cdef bool should_stop(Bridge self) noexcept nogil
    cpdef void stop(self) noexcept nogil


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
    void msleep "aeq_sleep"(int milliseconds) noexcept nogil



