# cython: boundscheck=False, wraparound=False, embedsignature=False, cdivision=True, initializedcheck=False

include "parameters.pxi"
from libc.stddef cimport size_t

cdef extern from "../cpp/pq_4ary_heap.hpp" namespace "aequilibrae::paths::cpp" nogil:
    cdef enum ElementState:
        SCANNED
        NOT_IN_HEAP
        IN_HEAP

    cdef struct Element:
        DTYPE_t key
        ElementState state
        size_t node_idx

    cdef struct PriorityQueue:
        size_t length
        size_t size
        size_t* A
        Element* Elements
        DTYPE_t* keys

    void init_heap(PriorityQueue* pqueue, size_t length) noexcept
    void free_heap(PriorityQueue* pqueue) noexcept
    void insert(PriorityQueue* pqueue, size_t element_idx, DTYPE_t key) noexcept
    void decrease_key(PriorityQueue* pqueue, size_t element_idx, DTYPE_t key_new) noexcept
    DTYPE_t peek(PriorityQueue* pqueue) noexcept
    bint is_empty "aequilibrae::paths::cpp::is_empty"(PriorityQueue* pqueue) noexcept
    size_t extract_min(PriorityQueue* pqueue) noexcept
