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

from tqdm.contrib.logging import logging_redirect_tqdm


from cython.parallel cimport parallel, threadid, prange


cdef:
    int DEBUG = logging.DEBUG
    int INFO = logging.INFO
    int WARNING = logging.WARNING
    int ERROR = logging.ERROR
    int CRITICAL = logging.CRITICAL


@cython.final
cdef public class Bridge [object Bridge, type Bridge_t]:
    def __init__(self, logger: logging.Logger):
        self.task = None
        self._stop.store(False, memory_order.memory_order_relaxed)

        self.__level = logger.level
        self.__logger = logger

        self.__exception_queue = queue.SimpleQueue()

        self.bars = []

    def start(self):
        self.__level = self.__logger.level
        self.task = threading.Thread(target=self.loop)
        self.task.start()

    cpdef void stop(self) noexcept nogil:
        self._stop.store(True, memory_order.memory_order_relaxed)

    def new_bar(self, *args, **kwargs):
        idx = len(self.bars)
        bar = Bar(*args, **kwargs)

        self.bars.append(bar)
        return idx, bar

    cdef bool should_stop(Bridge self) noexcept nogil:
        return self._stop.load(memory_order.memory_order_relaxed)

    def loop(self):
        # Create the lock, we use the unique_lock wrapper because it is movable while a mutex isn't and Cython loves to
        # generate temporary assignments.
        cdef unique_lock[mutex] lock = unique_lock[mutex](self.__log_queue_mutex, defer_lock)

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
            with logging_redirect_tqdm(loggers=[self.__logger]):
                while not self.should_stop():
                    with nogil:
                         msleep(100)
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


    cdef void _log(self, int level, string msg) noexcept nogil:
        cdef unique_lock[mutex] lock = unique_lock[mutex](self.__log_queue_mutex, defer_lock)

        # We should never try to acquire the log lock while holding the GIL. Cython has helper functions to make this
        # more cleaner but those are not yet released (15/10/2025)
        with nogil:
            lock.lock()
            try:
                self.__log_queue.emplace_back(level, msg)
            finally:
                lock.unlock()

    def __enter__(self):
        self.start()

    def __exit__(self, *_):
        self.stop()
        self.task.join()

        cdef unique_lock[mutex] lock = unique_lock[mutex](self.__log_queue_mutex, defer_lock)
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


# This function is given a static C name so that we can call it from the expanded macro. We don't call the Bridge.log
# function directly within the macro because it is hidden in the Cython VTable and has a mangled name. We could access
# it directly but it's considered an implemented detail. So we use this as a small wrapper to let Cython figure things
# out for us. This function is entirely transparent to the Cython and C++ compiler. Cython is able to figure out the C
# function that this calls and the C++ compiler should inline this entirely as there is nothing else in the generated
# functions body.
cdef public inline void _c_to_python_log_bridge(Bridge b, int level, string msg) noexcept nogil:
    b._log(level, msg)


def long_running(bridge: Bridge):
    cdef Bar bar
    idx, bar = bridge.new_bar(total=100)

    cdef int i = 0
    try:
        with nogil:
            for i in prange(100):
                if bridge.should_stop():
                    log(bridge, WARNING, f("We've been asked to stop"))
                    break

                # if i == 50:
                #     bridge.stop()
                #     bridge.log(WARNING, f("STOP!"))
                #     break

                msleep(threadid() * 100)
                log(bridge, WARNING, f("Hello from thread ", threadid(), "! i: ", i))
                cpp_function_that_logs(bridge)
                bar.inc()
            else:
                log(bridge, WARNING, f("Exited normally"))
                pass

    finally:
        bar.finish()
