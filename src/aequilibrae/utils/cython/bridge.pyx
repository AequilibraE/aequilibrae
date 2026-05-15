# distutils: language=c++
cimport cython

from libcpp cimport bool
from libcpp.atomic cimport memory_order
from libcpp.utility cimport move
from libcpp.string cimport string
from libcpp.mutex cimport unique_lock, mutex, defer_lock

from cysignals.signals cimport sig_check

from aequilibrae.utils.cython.bar cimport Bar

import threading
import logging
import queue


@cython.final
cdef class Bridge:
    """
    Thread-safe bridge for logging and progress reporting from C/Cython nogil contexts back to Python.

    This class runs a background thread that monitors a thread-safe queue for log messages generated in concurrent C++
    or Cython code execution. It ensures that logs are dispatched back to the main Python thread.

    The bridge also handles refreshing progress bars and propagating exceptions back to the main Python thread.

    It is designed to be used as a context manager.

    :Arguments:
        **logger** (:obj:`logging.Logger`, optional): The Python logger instance to which messages will
            be dispatched. If None, the root logger is used.

    :Attributes:
        **task** (:obj:`threading.Thread`): The background thread handle running the monitoring loop.
        **bars** (:obj:`list`): A list of active :obj:`Bar` instances being monitored.
    """
    def __cinit__(self):
        self.c = new AeqLogClosure()

    def __dealloc__(self):
        del self.c
        self.c = NULL

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.task = None
        self._stop.store(False, memory_order.memory_order_relaxed)

        self.__logger = logger or logging.getLogger()
        self.c.c_level = self.__logger.level
        if logger is None:
            self.__logger.warn(
                "AequilibraE Bridge is using the root logger. To prevent broken progress bars, ensure either progress "
                "bars are disabled (set AEQ_SHOW_PROGRESS=FALSE), or all StreamHandlers utilise AequilibraEStreamHandler"
            )

        self.__exception_queue = queue.SimpleQueue()

        self.bars = []

    def start(self):
        """
        Starts the background monitoring thread.
        """
        self.__level = self.__logger.level
        self.task = threading.Thread(target=self.loop)
        self.task.start()

    cpdef void stop(self) noexcept nogil:
        self._stop.store(True, memory_order.memory_order_relaxed)

    def new_bar(self, *args, **kwargs):
        """
        Creates and registers a new progress bar.

        The arguments passed here are forwarded directly to the :obj:`Bar` constructor.

        :Returns:
            **bar** (:obj:`Bar`): The newly created progress bar instance attached to this bridge.
        """
        bar = Bar(*args, **kwargs)

        self.bars.append(bar)
        return bar

    cdef bool should_stop(Bridge self) noexcept nogil:
        return self._stop.load(memory_order.memory_order_relaxed)

    def loop(self):
        # Create the lock, we use the unique_lock wrapper because it is movable while a mutex isn't and Cython loves to
        # generate temporary assignments.
        cdef unique_lock[mutex] lock = unique_lock[mutex](self.c._log_queue_mutex, defer_lock)

        # The pattern of this loop should be
        #  - Check termination criteria

        #  - Enter a try block, on exit of this block we should unlock the log lock if we own a mutex and it is
        #    locked. We own a mutex at this point but it may not be locked if lock() throws.

        #  - Release the gil, sleep using the portable macro defined in the pxd file.

        #  - We then attempt to acquire the lock WITHOUT the GIL held. The Cython documentation strongly suggests
        #    against any blocking lock operations while holding the GIL. If this throws then the finally block is hit.

        #  - Once we have the log lock we acquire the GIL, no other logs can be added to the queue at this point.

        #  - Consume the log queue, the std::string is moved to a local variable, deconstructing the empty/previous
        #    string. It is then encoded to a Python UTF-8 string, this creates a copy of the string as Python object, we
        #    then pass this off to self.__logger. The new std::string is deconstructed on the next iteration or when the
        #    local variable goes out of scope.

        #  - Unlock the log lock, we now longer require it.

        #  - Check signals with sig_check. Signals are only automatically processed when the main thread is executing
        #    Python byte code. We manually check the signals, catch any exceptions, place the exception information into
        #    a threading.queue, then set the termination flag. This is to transfer the information to the main thread
        #    safely. The termination flag signals to consumers of this Bridge instance that they should gracefully
        #    terminate. In the most common case this should cause all threads within a prange/parallel block to exit
        #    upon the next iteration and join the main thread. Once joined, the main thread should join on self.task
        #    (via self.stop()). This will block the main thread until this thread has finished. The main thread should
        #    then check the exception queue, raising anything if required.

        #  - Update progress bars.

        try:
            while not self.should_stop():
                with nogil:
                     msleep(500)
                     lock.lock()

                self.__unsafe_consume_log_queue()

                lock.unlock()

                sig_check()

                for bar in self.bars:
                    bar.refresh()

        except KeyboardInterrupt as e:
            self.__exception_queue.put((type(e), e.args))
            self._stop.store(True, memory_order.memory_order_relaxed)
        finally:
            if lock.owns_lock():
                lock.unlock()

    def __unsafe_consume_log_queue(self):
        cdef int level = 0
        cdef string msg

        while not self.__log_queue.empty():
            tmp = self.__log_queue.front()

            level = tmp.first
            msg = move(tmp.second)

            self.__logger.log(level, msg.decode("UTF-8"))

            self.__log_queue.pop_front()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
        self.task.join()

        cdef unique_lock[mutex] lock = unique_lock[mutex](self.c._log_queue_mutex, defer_lock)
        try:
            with nogil:
                lock.lock()
            self.__unsafe_consume_log_queue()
        finally:
            if lock.owns_lock():
                lock.unlock()

        try:
            exception_type, args = self.__exception_queue.get_nowait()
            e = exception_type(*args)
            e.add_note(
                "This exception was caught by another thread and re-raised here. "
                "This is not the source of this exception."
            )

            raise e
        except queue.Empty:
            pass

