from libcpp cimport bool
from libcpp.atomic cimport atomic
from libcpp.deque cimport deque
from libcpp.string cimport string
from libcpp.mutex cimport mutex
from libcpp.utility cimport pair


# std::make_pair is not available in the Cython libcpp.utilities shim. We'll import it ourselves based on the C++11 til
# C++14 definition because C++14 makes this signature weird. See https://github.com/cython/cython/issues/2706
cdef extern from "<utility>" namespace "std" nogil:
    pair[T, U] make_pair[T, U](T&& t, U&& u)

# We use a niche piece of syntax here to make Cython generate a header we can use in C++ for this class
# https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#name-specification-clause
cdef public class Bridge [object Bridge, type Bridge_t]:
    cdef:
        public object task
        atomic[bool] _stop
        public object bars

        object __logger
        int __level "c_level"
        object __exception_queue

        mutex __log_queue_mutex
        deque[pair[int, string]] __log_queue

    cdef bool should_stop(Bridge self) noexcept nogil
    cdef void _log(self, int level, string msg) noexcept nogil
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


cdef extern from "aeq_log.h":
    # We lie to the Cython compiler here, Cython thinks this is a C function but it's actually a macro. We give the
    # arguments types so that Cython can attempt to enforce them for us. This lets us have statically checked types for
    # the arguments. Admittedly the errors Cython raises aren't descriptive, and when call from Python the type check is
    # deferred till runtime. But importantly msg_exp isn't computed if the provided is below the level we set.
    void log"AEQ_LOG"(Bridge bridge, int lvl, string msg_exp) noexcept nogil

    string f "aeq_format_string"(...) noexcept nogil


cdef public void _c_to_python_log_bridge"aeq_c_to_python_log_bridge"(Bridge b, int level, string msg) noexcept nogil
cdef public:
    int DEBUG"AEQ_LOG_DEBUG"
    int INFO"AEQ_LOG_INFO"
    int WARNING"AEQ_LOG_WARNING"
    int ERROR"AEQ_LOG_ERROR"
    int CRITICAL"AEQ_LOG_CRITICAL"
