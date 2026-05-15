from aequilibrae.paths.cython.pq_heap_base cimport PriorityQueueBase

cdef extern from "pq_4ary_heap.hpp" namespace "aequilibrae::paths::cpp" nogil:
    cdef cppclass FourAryHeap(PriorityQueueBase[FourAryHeap]):
        FourAryHeap() noexcept

cdef extern from "pq_pairing_heap.hpp" namespace "aequilibrae::paths::cpp" nogil:
    cdef cppclass PairingHeap(PriorityQueueBase[PairingHeap]):
        PairingHeap() noexcept

cdef extern from "pq_std_priority_queue_adapter.hpp" namespace "aequilibrae::paths::cpp" nogil:
    cdef cppclass StdPriorityQueueAdapter(PriorityQueueBase[StdPriorityQueueAdapter]):
        StdPriorityQueueAdapter() noexcept

ctypedef fused HeapType:
    FourAryHeap
    PairingHeap
    StdPriorityQueueAdapter
