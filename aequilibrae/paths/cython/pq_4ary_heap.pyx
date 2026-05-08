# cython: boundscheck=False, wraparound=False, embedsignature=False, cdivision=True, initializedcheck=False

include "parameters.pxi"
from libcpp cimport bool
from libc.stddef cimport size_t

cdef extern from "../cpp/pq_4ary_heap.hpp" namespace "aequilibrae::paths::cpp" nogil:
    cdef enum ElementState:
        SCANNED
        NOT_IN_HEAP
        IN_HEAP

    cdef struct Element:
        double key
        ElementState state
        size_t node_idx

    cdef struct PriorityQueue:
        size_t length
        size_t size
        size_t* A
        Element* Elements
        double* keys

    void init_heap(PriorityQueue* pqueue, size_t length) noexcept
    void free_heap(PriorityQueue* pqueue) noexcept
    void insert(PriorityQueue* pqueue, size_t element_idx, double key) noexcept
    void decrease_key(PriorityQueue* pqueue, size_t element_idx, double key_new) noexcept
    double peek(PriorityQueue* pqueue) noexcept
    bool is_empty(PriorityQueue* pqueue) noexcept
    size_t extract_min(PriorityQueue* pqueue) noexcept
