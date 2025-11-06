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
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.task = None
        self._stop.store(False, memory_order.memory_order_relaxed)

        self.__logger = logger or logging.getLogger()
        self.__level = self.__logger.level
        if logger is None:
            self.__logger.warn(
                "AequilibraE Bridge is using the root logger. To prevent broken progress bars, ensure either progress "
                "bars are disabled (set AEQ_SHOW_PROGRESS=FALSE), or all StreamHandlers utilise AequilibraEStreamHandler"
            )

        self.__exception_queue = queue.SimpleQueue()

        self.bars = []

        # Previously we used a global cdef function that forwarded its arguments to "self._log". This was necessary
        # because "self._log" cannot be accessed directly from C without going through the Cython vtable. While we can
        # assign it a known C name, that only changes the C name within the vtable. The idea was that the global
        # function could call "bridge._log", with Cython handling the details. This worked well, but required the global
        # function to be cimported into other modules because Cython only initialises objects from another extension
        # module that are cimported. If the user selectively imports using "from ... cimport ...", not all attributes
        # are imported and initialised. This creates an unusual situation at the C level where everything is declared
        # (because bridge.pxd is inserted into the extension module with all its declarations) but not
        # initialised. Instead, we export the function via a function pointer on the Bridge object. C attributes do not
        # need to be obtained through the vtable because cdef classes must inherit and define all C attributes of their
        # parents. Additionally, we can use "self._log" directly instead of a wrapper function because Bridge is
        # final. We can then access this function pointer from the AEQ_LOG macro. All this assumes that AEQ_LOG calls
        # the class method correctly and nothing fancy is required.
        self.__log_wrapper_func = self._log

    def start(self):
        self.__level = self.__logger.level
        self.task = threading.Thread(target=self.loop)
        self.task.start()

    cpdef void stop(self) noexcept nogil:
        self._stop.store(True, memory_order.memory_order_relaxed)

    def new_bar(self, *args, **kwargs):
        bar = Bar(*args, **kwargs)

        self.bars.append(bar)
        return bar

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
        return self

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

