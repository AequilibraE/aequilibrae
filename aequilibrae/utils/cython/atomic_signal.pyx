from libcpp.atomic cimport atomic, memory_order
from libc.stdint cimport uint64_t
from posix.time cimport timespec, nanosleep
from cython.operator cimport preincrement
cimport cython

import math
import threading
import functools
from aequilibrae.utils.aeq_signal import SIGNAL


@cython.final
cdef class AtomicSignal:
    def __init__(self, interval: float, msg: str = None, total: int = 0):
        self.interval = interval
        self.msg = msg or "{}/{}"
        self.__total = total

        self.__signal = SIGNAL(object)
        self.__task = None
        self.__stop = threading.Event()

    @property
    def total(self):
        return self.__total

    @total.setter
    def total(self, val: int):
        self.__total = val
        if self.__signal.pbar is not None:
            self.__signal.pbar.total = val

    def start(self):
        self.__stop.clear()
        self.__counter.store(0, memory_order.memory_order_relaxed)

        self.__task = threading.Thread(target=self.__loop)
        self.__task.start()

    def stop(self):
        self.__stop.set()
        self.__task.join()

    def __loop(self):
        self.__signal.emit(["start", self.total, self.msg.format(0, self.total)])

        cdef timespec req, rem

        req.tv_sec = math.floor(self.interval)
        req.tv_nsec = <long>((self.interval - req.tv_sec) * 1_000_000_000)
        try:
            while not self.__stop.is_set():
                with nogil:
                    nanosleep(&req, &rem)

                val = self.__counter.load(memory_order.memory_order_relaxed)
                self.__signal.emit(["update", val, self.msg.format(val, self.total)])
        finally:
            self.__signal.emit(["finished"])

    cpdef inline void inc(AtomicSignal self) noexcept nogil:
        preincrement(self.__counter)

    @classmethod
    def progress_bar(cls, *_args, **_kwargs):
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                signal = cls(*_args, **_kwargs)
                signal.start()
                try:
                    return f(*args, **kwargs, signal=signal)
                finally:
                    signal.stop()
            return wrapper
        return decorator
