cimport cython
from libcpp.atomic cimport memory_order
from libc.stdint cimport uint64_t

from aequilibrae.utils.aeq_signal import SIGNAL


@cython.final
cdef class Bar:
    """
    An atomic progress counter designed for use outside of the GIL.

    This class maintains atomic counters for progress tracking that can be updated from `nogil` or C++ contexts without
    acquiring locks. It emits signals via a purely Python-side mechanism (SIGNAL) when refreshed by the :obj:`Bridge`,
    allowing QGIS or console updates (like tqdm) to occur in the main thread.

    :Arguments:
        **msg** (:obj:`str`, optional): A format string for the progress message. Must contain
            placeholders for current counter and total (e.g., "{}/{}"). Defaults to "{}/{}".

        **total** (:obj:`int`, optional): The total number of iterations expected. Defaults to 0.

    :Attributes:
        **msg** (:obj:`str`): The format string used for status updates.
    """

    def __init__(self, msg: str = None, total: int = 0):
        self.msg = msg or "{}/{}"

        self.set_total(total)
        self.set_counter(0)

        self.__total_old = self.__counter_old = 0

        self.__signal = SIGNAL(object)
        self.refresh()

    cpdef inline void set_total(self, uint64_t total) noexcept nogil:
        """
        Atomically sets the total number of expected iterations.

        :Arguments:
            **total** (:obj:`int`): The new total value.
        """
        self.__total.store(total, memory_order.memory_order_relaxed)

    cpdef inline void set_counter(self, uint64_t value) noexcept nogil:
        """
        Atomically sets the current progress counter.

        :Arguments:
            **value** (:obj:`int`): The new counter value.
        """
        self.__counter.store(value, memory_order.memory_order_relaxed)

    cpdef inline uint64_t get_total(self) noexcept nogil:
        """
        Retrieves the current total value atomically.

        :Returns:
            **total** (:obj:`int`): The current total.
        """
        return self.__total.load(memory_order.memory_order_relaxed)

    cpdef inline uint64_t get_counter(self) noexcept nogil:
        """
        Retrieves the current progress counter atomically.

        :Returns:
            **counter** (:obj:`int`): The current progress value.
        """
        return self.__counter.load(memory_order.memory_order_relaxed)

    cpdef inline void inc(self) noexcept nogil:
        """
        Atomically increments the progress counter by 1.
        """
        self.__counter.fetch_add(1, memory_order.memory_order_relaxed)

    def refresh(self):
        """
        Checks atomic counters against cached values and emits update signals if changed.

        This method is intended to be called by the :obj:`Bridge` loop running in a thread with the GIL acquired.
        """
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
        """
        Emits a final 'finished' signal to close the progress bar.
        """
        self.__signal.emit(["finished"])
