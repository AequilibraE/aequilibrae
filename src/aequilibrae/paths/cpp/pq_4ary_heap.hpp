#pragma once

#include <cassert>
#include <cstddef>
#include <limits>

namespace aequilibrae::paths::cpp {

enum ElementState {
    SCANNED,
    NOT_IN_HEAP,
    IN_HEAP,
};

struct Element {
    double key;
    ElementState state;
    std::size_t node_idx;
};

struct PriorityQueue {
    std::size_t length;
    std::size_t size;
    std::size_t* A;
    Element* Elements;
    double* keys;
};

inline constexpr std::size_t kArity = 4;
inline constexpr double kInfinity = std::numeric_limits<double>::infinity();

inline void initialize_element(PriorityQueue* pqueue, std::size_t element_idx) noexcept {
    assert(pqueue != nullptr);
    assert(element_idx < pqueue->length);

    pqueue->Elements[element_idx].key = kInfinity;
    pqueue->Elements[element_idx].state = NOT_IN_HEAP;
    pqueue->Elements[element_idx].node_idx = pqueue->length;
}

inline void exchange_nodes(PriorityQueue* pqueue, std::size_t node_i, std::size_t node_j) noexcept {
    assert(pqueue != nullptr);
    assert(node_i < pqueue->size);
    assert(node_j < pqueue->size);

    const std::size_t element_i = pqueue->A[node_i];
    const std::size_t element_j = pqueue->A[node_j];

    assert(element_i < pqueue->length);
    assert(element_j < pqueue->length);

    pqueue->A[node_i] = element_j;
    pqueue->A[node_j] = element_i;
    pqueue->Elements[element_j].node_idx = node_i;
    pqueue->Elements[element_i].node_idx = node_j;
}

inline void decrease_key_from_node_index(PriorityQueue* pqueue, std::size_t node_idx, double key_new) noexcept {
    assert(pqueue != nullptr);
    assert(node_idx < pqueue->size);

    std::size_t i = node_idx;
    const std::size_t element_idx = pqueue->A[i];
    assert(element_idx < pqueue->length);
    assert(pqueue->Elements[element_idx].state == IN_HEAP);
    assert(key_new <= pqueue->Elements[element_idx].key);

    pqueue->Elements[element_idx].key = key_new;
    while (i > 0) {
        const std::size_t parent = (i - 1) / kArity;
        const std::size_t parent_element = pqueue->A[parent];
        assert(parent_element < pqueue->length);
        if (pqueue->Elements[parent_element].key > key_new) {
            exchange_nodes(pqueue, i, parent);
            i = parent;
        } else {
            break;
        }
    }
}

inline void min_heapify(PriorityQueue* pqueue, std::size_t node_idx) noexcept {
    assert(pqueue != nullptr);
    if (pqueue->size == 0) {
        return;
    }
    assert(node_idx < pqueue->size);

    std::size_t i = node_idx;
    while (true) {
        std::size_t smallest = i;
        double min_key = pqueue->Elements[pqueue->A[smallest]].key;

        // Mirror the original Cython tie-breaking: scan children from highest
        // index to lowest. Two valid heaps that differ only in tie-break choice
        // can produce different (but equal-cost) Dijkstra trees; reverse scan
        // matches the historical reference outputs.
        const std::size_t first_child = kArity * i + 1;
        const std::size_t end_child = first_child + kArity < pqueue->size ? first_child + kArity : pqueue->size;
        for (std::size_t k = end_child; k > first_child; --k) {
            const std::size_t child = k - 1;
            const double child_key = pqueue->Elements[pqueue->A[child]].key;
            if (child_key < min_key) {
                smallest = child;
                min_key = child_key;
            }
        }

        if (smallest == i) {
            return;
        }

        exchange_nodes(pqueue, i, smallest);
        i = smallest;
    }
}

inline void init_heap(PriorityQueue* pqueue, std::size_t length) noexcept {
    assert(pqueue != nullptr);

    pqueue->length = length;
    pqueue->size = 0;
    pqueue->A = length > 0 ? new std::size_t[length] : nullptr;
    pqueue->Elements = length > 0 ? new Element[length] : nullptr;
    pqueue->keys = nullptr;

    for (std::size_t i = 0; i < length; ++i) {
        pqueue->A[i] = length;
        initialize_element(pqueue, i);
    }
}

inline void free_heap(PriorityQueue* pqueue) noexcept {
    assert(pqueue != nullptr);

    delete[] pqueue->A;
    delete[] pqueue->Elements;
    pqueue->A = nullptr;
    pqueue->Elements = nullptr;
    pqueue->keys = nullptr;
    pqueue->length = 0;
    pqueue->size = 0;
}

inline void insert(PriorityQueue* pqueue, std::size_t element_idx, double key) noexcept {
    assert(pqueue != nullptr);
    assert(element_idx < pqueue->length);
    assert(pqueue->size < pqueue->length);
    assert(pqueue->Elements[element_idx].state == NOT_IN_HEAP);
    assert(key < kInfinity);

    const std::size_t node_idx = pqueue->size;
    ++pqueue->size;
    pqueue->Elements[element_idx].state = IN_HEAP;
    pqueue->Elements[element_idx].node_idx = node_idx;
    pqueue->A[node_idx] = element_idx;
    decrease_key_from_node_index(pqueue, node_idx, key);
}

inline void decrease_key(PriorityQueue* pqueue, std::size_t element_idx, double key_new) noexcept {
    assert(pqueue != nullptr);
    assert(element_idx < pqueue->length);
    assert(pqueue->Elements[element_idx].state == IN_HEAP);

    decrease_key_from_node_index(pqueue, pqueue->Elements[element_idx].node_idx, key_new);
}

inline double peek(PriorityQueue* pqueue) noexcept {
    assert(pqueue != nullptr);
    assert(pqueue->size > 0);

    return pqueue->Elements[pqueue->A[0]].key;
}

inline bool is_empty(PriorityQueue* pqueue) noexcept {
    assert(pqueue != nullptr);
    return pqueue->size == 0;
}

inline std::size_t extract_min(PriorityQueue* pqueue) noexcept {
    assert(pqueue != nullptr);
    assert(pqueue->size > 0);

    const std::size_t element_idx = pqueue->A[0];
    const std::size_t node_idx = pqueue->size - 1;

    exchange_nodes(pqueue, 0, node_idx);

    pqueue->Elements[element_idx].state = SCANNED;
    pqueue->Elements[element_idx].node_idx = pqueue->length;
    pqueue->A[node_idx] = pqueue->length;
    --pqueue->size;

    if (pqueue->size > 0) {
        min_heapify(pqueue, 0);
    }

    return element_idx;
}

}  // namespace aequilibrae::paths::cpp
