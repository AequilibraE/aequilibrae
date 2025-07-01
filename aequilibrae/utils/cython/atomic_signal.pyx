from libcpp.atomic cimport atomic, memory_order
from libc.stdint cimport uint64_t
from cython.operator cimport preincrement
cimport cython

import math
import threading
import functools
from aequilibrae.utils.aeq_signal import SIGNAL


@cython.final
cdef class AtomicSignal:
    def __init__(self, interval: float, msg: str = None, total: int = 0):
        self.interval = <int>(interval * 1_000)
        self.msg = msg or "{}/{}"

        self.__total.store(total, memory_order.memory_order_relaxed)
        self.__signal = SIGNAL(object)
        self.__task = None
        self.__stop = threading.Event()

    def start(self):
        self.__stop.clear()
        self.__counter.store(0, memory_order.memory_order_relaxed)

        self.__task = threading.Thread(target=self.__loop)
        self.__task.start()

    def stop(self):
        self.__stop.set()
        self.__task.join()

    cpdef inline void set_total(self, uint64_t total) noexcept nogil:
        self.__total.store(total, memory_order.memory_order_relaxed)

    def __loop(self):
        cdef:
            uint64_t val = 0
            uint64_t total = self.__total.load(memory_order.memory_order_relaxed)
            uint64_t _total = total

        self.__signal.emit(["start", total, self.msg.format(val, total)])

        try:
            while not self.__stop.is_set():
                with nogil:
                    msleep(self.interval)
                    val = self.__counter.load(memory_order.memory_order_relaxed)
                    total = self.__total.load(memory_order.memory_order_relaxed)

                if _total != total:
                    _total = total
                    self.__signal.emit(["start", total, self.msg.format(val, total)])
                    self.__signal.emit(["set_position", val])
                else:
                    self.__signal.emit(["update", val, self.msg.format(val, total)])
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

    def __enter__(self):
        self.start()

        return self

    def __exit__(self, *_):
        self.stop()
