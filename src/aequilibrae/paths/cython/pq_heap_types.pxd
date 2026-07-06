from libc.stddef cimport size_t

from aequilibrae.utils.cython.bridge cimport AeqLogClosure

cdef extern from "pq_heap_base.hpp" namespace "aequilibrae::paths::cpp" nogil:
    cdef enum ElementState:
        SCANNED
        NOT_IN_HEAP
        IN_HEAP

    cdef cppclass PriorityQueueBase:
        void attach_logger(AeqLogClosure *closure) noexcept
        void init_heap(size_t length) noexcept
        void alloc_heap(size_t length) noexcept
        void reset_heap() noexcept
        void free_heap() noexcept
        void insert(size_t element_idx, double key) noexcept
        void decrease_key(size_t element_idx, double key_new) noexcept
        double peek() noexcept
        bint is_empty() noexcept
        size_t extract_min() noexcept
        ElementState effective_state(size_t element_idx) noexcept
        double element_key(size_t element_idx) noexcept

cdef extern from "pq_4ary_heap.hpp" namespace "aequilibrae::paths::cpp" nogil:
    cdef cppclass FourAryHeap(PriorityQueueBase):
        FourAryHeap() noexcept

cdef extern from "pq_pairing_heap.hpp" namespace "aequilibrae::paths::cpp" nogil:
    cdef cppclass PairingHeap(PriorityQueueBase):
        PairingHeap() noexcept

cdef extern from "pq_std_priority_queue_adapter.hpp" namespace "aequilibrae::paths::cpp" nogil:
    cdef cppclass StdPriorityQueueAdapter(PriorityQueueBase):
        StdPriorityQueueAdapter() noexcept
