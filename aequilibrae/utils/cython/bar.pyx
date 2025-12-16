cimport cython
from libcpp.atomic cimport memory_order
from libc.stdint cimport uint64_t

from aequilibrae.utils.aeq_signal import SIGNAL


@cython.final
cdef class Bar:
    def __init__(self, msg: str = None, total: int = 0):
        self.msg = msg or "{}/{}"

        self.set_total(total)
        self.set_counter(0)

        self.__total_old = self.__counter_old = 0

        self.__signal = SIGNAL(object)
        self.refresh()

    cpdef inline void set_total(self, uint64_t total) noexcept nogil:
        self.__total.store(total, memory_order.memory_order_relaxed)

    cpdef inline void set_counter(self, uint64_t value) noexcept nogil:
        self.__counter.store(value, memory_order.memory_order_relaxed)

    cpdef inline uint64_t get_total(self) noexcept nogil:
        return self.__total.load(memory_order.memory_order_relaxed)

    cpdef inline uint64_t get_counter(self) noexcept nogil:
        return self.__counter.load(memory_order.memory_order_relaxed)

    cpdef inline void inc(self) noexcept nogil:
        self.__counter.fetch_add(1, memory_order.memory_order_relaxed)

    def refresh(self):
        cdef:
            uint64_t counter = self.get_counter()
            uint64_t total = self.get_total()

        if total != self.__total_old:
            self.__signal.emit(["start", total, self.msg.format(counter, total)])

            # Set position doesn't do what I thought it does, for TQDM it sets the "position" of the bar within multiple
            # progress bars, it does not set the "position" of the bar itself
            # self.__signal.emit(["set_position", counter])
            self.__signal.emit(["update", counter, self.msg.format(counter, total)])
        elif counter != self.__counter_old:
            self.__signal.emit(["update", counter, self.msg.format(counter, total)])
        else:
            return

        self.__counter_old = counter
        self.__total_old = total

    def finish(self):
        self.__signal.emit(["finished"])
