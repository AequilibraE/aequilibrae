#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace aequilibrae::paths::cpp {

enum ElementState {
  SCANNED,
  NOT_IN_HEAP,
  IN_HEAP,
};

struct Element {
  double key;
  std::uint32_t epoch; // matches PriorityQueue.current_epoch when valid
  ElementState state;
  std::size_t node_idx;
};

struct PriorityQueue {
  std::size_t length;
  std::size_t size;
  std::size_t *A;
  Element *Elements;
  double *keys;
  std::uint32_t current_epoch;
};

inline constexpr std::size_t kArity = 4;
inline constexpr double kInfinity = std::numeric_limits<double>::infinity();

inline void initialize_element(PriorityQueue *pqueue,
                               std::size_t element_idx) noexcept {
  assert(pqueue != nullptr);
  assert(element_idx < pqueue->length);

  pqueue->Elements[element_idx].key = kInfinity;
  pqueue->Elements[element_idx].epoch = 0;
  pqueue->Elements[element_idx].state = NOT_IN_HEAP;
  pqueue->Elements[element_idx].node_idx = pqueue->length;
}

// Returns true if the element is currently considered "in the heap" given
// the current epoch. Elements whose epoch != current_epoch are treated as
// NOT_IN_HEAP regardless of their stored state field, which makes O(1)
// epoch-bump reset valid.
inline bool element_is_in_heap_4ary(const PriorityQueue *pqueue,
                                    std::size_t element_idx) noexcept {
  const Element &e = pqueue->Elements[element_idx];
  return e.epoch == pqueue->current_epoch && e.state == IN_HEAP;
}

inline bool element_is_scanned_4ary(const PriorityQueue *pqueue,
                                    std::size_t element_idx) noexcept {
  const Element &e = pqueue->Elements[element_idx];
  return e.epoch == pqueue->current_epoch && e.state == SCANNED;
}

inline double element_key_4ary(const PriorityQueue *pqueue,
                               std::size_t element_idx) noexcept {
  return pqueue->Elements[element_idx].key;
}

// Epoch-aware effective state read. Stale-epoch elements look like NOT_IN_HEAP
// regardless of whatever ElementState was left over from a previous run.
inline ElementState effective_state_4ary(const PriorityQueue *pqueue,
                                         std::size_t element_idx) noexcept {
  const Element &e = pqueue->Elements[element_idx];
  if (e.epoch != pqueue->current_epoch)
    return NOT_IN_HEAP;
  return e.state;
}

inline void exchange_nodes(PriorityQueue *pqueue, std::size_t node_i,
                           std::size_t node_j) noexcept {
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

inline void decrease_key_from_node_index(PriorityQueue *pqueue,
                                         std::size_t node_idx,
                                         double key_new) noexcept {
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

inline void min_heapify(PriorityQueue *pqueue, std::size_t node_idx) noexcept {
  assert(pqueue != nullptr);
  if (pqueue->size == 0) {
    return;
  }
  assert(node_idx < pqueue->size);

  std::size_t i = node_idx;
  while (true) {
    std::size_t smallest = i;
    double min_key = pqueue->Elements[pqueue->A[smallest]].key;

    const std::size_t first_child = kArity * i + 1;
    const std::size_t end_child = first_child + kArity < pqueue->size
                                      ? first_child + kArity
                                      : pqueue->size;
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

// One-time allocation. Use `reset_heap` between Dijkstra runs.
inline void alloc_heap(PriorityQueue *pqueue, std::size_t length) noexcept {
  assert(pqueue != nullptr);

  pqueue->length = length;
  pqueue->size = 0;
  pqueue->A = length > 0 ? new std::size_t[length] : nullptr;
  pqueue->Elements = length > 0 ? new Element[length] : nullptr;
  pqueue->keys = nullptr;
  pqueue->current_epoch = 1;

  for (std::size_t i = 0; i < length; ++i) {
    pqueue->A[i] = length;
    initialize_element(pqueue, i);
  }
}

// O(length) reset between SSSP runs. Resets only the `state` field (the
// only one read by `_path_finding_with_pq`'s inner loop). The epoch counter
// is also bumped so any caller that does still rely on epoch-aware reads
// remains correct.
//
// We could in principle make this O(1) via the epoch bump alone, but the
// inner loop pays for that with an extra epoch comparison per CSR edge,
// which measurably regresses Chicago skimming. Resetting the state field
// in bulk is cheap (a single linear pass over a small array) and lets the
// inner loop stay tight.
inline void reset_heap(PriorityQueue *pqueue) noexcept {
  assert(pqueue != nullptr);
  assert(pqueue->size == 0);
  pqueue->current_epoch += 1;
  if (pqueue->current_epoch == 0) {
    pqueue->current_epoch = 1;
  }
  Element *E = pqueue->Elements;
  for (std::size_t i = 0; i < pqueue->length; ++i) {
    E[i].state = NOT_IN_HEAP;
    E[i].epoch = 0;
  }
}

// Backwards-compatible name: the old API allocated AND reset on every call.
// Some call sites still want that semantics (early-exit `path_finding`
// callers that don't go through the per-thread pool); we keep `init_heap`
// as the convenience combined-allocate-and-reset and add `alloc_heap` /
// `reset_heap` for the pooled call site.
inline void init_heap(PriorityQueue *pqueue, std::size_t length) noexcept {
  alloc_heap(pqueue, length);
}

inline void free_heap(PriorityQueue *pqueue) noexcept {
  assert(pqueue != nullptr);

  delete[] pqueue->A;
  delete[] pqueue->Elements;
  pqueue->A = nullptr;
  pqueue->Elements = nullptr;
  pqueue->keys = nullptr;
  pqueue->length = 0;
  pqueue->size = 0;
  pqueue->current_epoch = 0;
}

inline void insert(PriorityQueue *pqueue, std::size_t element_idx,
                   double key) noexcept {
  assert(pqueue != nullptr);
  assert(element_idx < pqueue->length);
  assert(pqueue->size < pqueue->length);
  // The element may have stale state from a previous epoch; we overwrite
  // it unconditionally here.
  assert(key < kInfinity);

  const std::size_t node_idx = pqueue->size;
  ++pqueue->size;
  pqueue->Elements[element_idx].state = IN_HEAP;
  pqueue->Elements[element_idx].epoch = pqueue->current_epoch;
  pqueue->Elements[element_idx].node_idx = node_idx;
  pqueue->A[node_idx] = element_idx;
  decrease_key_from_node_index(pqueue, node_idx, key);
}

inline void decrease_key(PriorityQueue *pqueue, std::size_t element_idx,
                         double key_new) noexcept {
  assert(pqueue != nullptr);
  assert(element_idx < pqueue->length);
  assert(element_is_in_heap_4ary(pqueue, element_idx));

  decrease_key_from_node_index(pqueue, pqueue->Elements[element_idx].node_idx,
                               key_new);
}

inline double peek(PriorityQueue *pqueue) noexcept {
  assert(pqueue != nullptr);
  assert(pqueue->size > 0);

  return pqueue->Elements[pqueue->A[0]].key;
}

inline bool is_empty(PriorityQueue *pqueue) noexcept {
  assert(pqueue != nullptr);
  return pqueue->size == 0;
}

inline std::size_t extract_min(PriorityQueue *pqueue) noexcept {
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

} // namespace aequilibrae::paths::cpp
