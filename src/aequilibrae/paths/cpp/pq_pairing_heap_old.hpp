#pragma once

// Indexed (addressable) pairing heap, drop-in replacement for the 4-ary
// indexed binary heap in `pq_4ary_heap.hpp`.
//
// Optimised variant (v2). Differences from a textbook two-pass pairing heap:
//
//   * Pointer-free: every node lives in a single `Element[]` slab indexed
//     by `node_idx`. Child / sibling / prev links are `std::size_t` indices,
//     sentinel `kNullIdx = SIZE_MAX`. No allocation on the hot path.
//
//   * `Element` packed to 40 bytes (key + state + child + next + prev). The
//     unused `node_idx` field from the 4-ary layout is dropped.
//
//   * **Multipass meld** in `extract_min` (Stasko & Vitter 1987 variant):
//     a single left-to-right sweep that pairs siblings into a running
//     accumulator. Empirically faster than two-pass on small heaps with
//     few decrease_keys per node (typical for road-network Dijkstra) and
//     removes the auxiliary stack entirely.
//
//   * Two flavours of meld:
//       - `meld_roots(a, b)`: both inputs have prev=next=kNullIdx already.
//         Skips two writes vs. the general case. Used everywhere except
//         the inside of multipass meld.
//       - `meld_subtree(root, sub)`: `sub` is freshly detached so prev/next
//         may be stale; root is a true root. Used inside multipass.
//
//   * `decrease_key` uses a `prev == kNullIdx` test instead of comparing
//     against `pqueue->root` to short-circuit when the decreased element
//     is already the root. Saves one load per call.
//
//   * `extract_min` fast-paths the 0- and 1-child cases.
//
//   * `__builtin_prefetch` hint on the sibling walk in multipass meld so
//     the next sibling's cache line lands while we are working on the
//     current pair.
//
//   * Public API (init_heap, free_heap, insert, decrease_key, extract_min,
//     peek, is_empty) and `Element.{key, state}` field names are preserved
//     so call sites in `basic_path_finding.pyx` and `hyperpath.pyx` need
//     no changes. `SCANNED` semantics match the 4-ary heap exactly.

#include <cassert>
#include <cstddef>
#include <limits>

namespace aequilibrae::paths::cpp {

enum ElementState {
  SCANNED,
  NOT_IN_HEAP,
  IN_HEAP,
};

inline constexpr std::size_t kNullIdx = static_cast<std::size_t>(-1);
inline constexpr double kInfinity = std::numeric_limits<double>::infinity();

#if defined(_MSC_VER)
#define AEQ_ALWAYS_INLINE __forceinline
#define AEQ_PREFETCH(p) ((void)0)
#elif defined(__GNUC__) || defined(__clang__)
#define AEQ_ALWAYS_INLINE inline __attribute__((always_inline))
#define AEQ_PREFETCH(p) __builtin_prefetch((p), 0, 1)
#else
#define AEQ_ALWAYS_INLINE inline
#define AEQ_PREFETCH(p) ((void)0)
#endif

// 40-byte Element. `key` first so a key load co-locates with state and the
// hot child/sibling links. Invariants:
//   - if state == IN_HEAP and the element is the heap root,
//         prev == kNullIdx, next == kNullIdx.
//   - if state == IN_HEAP and the element is NOT the root,
//         prev != kNullIdx (points to previous sibling, or to parent if
//         this element is the first child of its parent).
//   - state ∈ {NOT_IN_HEAP, IN_HEAP, SCANNED}; `child`/`next`/`prev` are
//     undefined when state != IN_HEAP except for `node_idx` purposes.
struct Element {
  double key;
  ElementState state;
  std::size_t child;
  std::size_t next;
  std::size_t prev;
};

struct PriorityQueue {
  std::size_t length; // capacity (= number of nodes)
  std::size_t size;   // number of elements currently IN_HEAP
  std::size_t *A;     // unused; kept nullptr for API symmetry
  Element *Elements;  // node-indexed slab
  double *keys;       // unused; kept for API symmetry
  std::size_t root;   // index of the current min, or kNullIdx
};

AEQ_ALWAYS_INLINE void initialize_element(Element *E, std::size_t i) noexcept {
  E[i].key = kInfinity;
  E[i].state = NOT_IN_HEAP;
  E[i].child = kNullIdx;
  E[i].next = kNullIdx;
  E[i].prev = kNullIdx;
}

inline void init_heap(PriorityQueue *pqueue, std::size_t length) noexcept {
  assert(pqueue != nullptr);

  pqueue->length = length;
  pqueue->size = 0;
  pqueue->A = nullptr; // multipass meld needs no scratch stack
  pqueue->Elements = length > 0 ? new Element[length] : nullptr;
  pqueue->keys = nullptr;
  pqueue->root = kNullIdx;

  Element *E = pqueue->Elements;
  for (std::size_t i = 0; i < length; ++i) {
    initialize_element(E, i);
  }
}

// Compatibility shims so the same path_finding code compiles under the
// pairing-heap backend. The pairing heap does not implement epoch-reset
// pooling; alloc_heap / reset_heap are provided as direct equivalents to
// init_heap and a no-op respectively.
inline void alloc_heap(PriorityQueue *pqueue, std::size_t length) noexcept {
  init_heap(pqueue, length);
}

inline void reset_heap(PriorityQueue * /*pqueue*/) noexcept {
  // Pairing heap does not support epoch-reset; pooled path will not be
  // exercised in this build configuration.
}

inline ElementState effective_state_4ary(const PriorityQueue *pqueue,
                                         std::size_t element_idx) noexcept {
  return pqueue->Elements[element_idx].state;
}

inline double element_key_4ary(const PriorityQueue *pqueue,
                               std::size_t element_idx) noexcept {
  return pqueue->Elements[element_idx].key;
}

inline void free_heap(PriorityQueue *pqueue) noexcept {
  assert(pqueue != nullptr);

  delete[] pqueue->Elements;
  pqueue->A = nullptr;
  pqueue->Elements = nullptr;
  pqueue->keys = nullptr;
  pqueue->length = 0;
  pqueue->size = 0;
  pqueue->root = kNullIdx;
}

// Meld two roots `a` and `b` (either may be kNullIdx). Both inputs are
// assumed to satisfy the root invariant (prev == next == kNullIdx). The
// result is the new combined root, also satisfying the root invariant.
AEQ_ALWAYS_INLINE std::size_t meld_roots(Element *E, std::size_t a,
                                         std::size_t b) noexcept {
  if (a == kNullIdx)
    return b;
  if (b == kNullIdx)
    return a;

  std::size_t winner, loser;
  if (E[a].key <= E[b].key) {
    winner = a;
    loser = b;
  } else {
    winner = b;
    loser = a;
  }

  // Link `loser` as new first child of `winner`; old first child becomes
  // loser's next sibling.
  const std::size_t old_child = E[winner].child;
  E[loser].next = old_child;
  E[loser].prev = winner;
  if (old_child != kNullIdx) {
    E[old_child].prev = loser;
  }
  E[winner].child = loser;
  // `winner` came in as a root, so its prev/next are already kNullIdx.
  return winner;
}

// Meld a true root with a freshly-detached subtree whose prev/next have
// been zeroed by the caller. Identical to `meld_roots` once both inputs
// are normalised, kept distinct for clarity at call sites.
AEQ_ALWAYS_INLINE std::size_t meld_subtree(Element *E, std::size_t root,
                                           std::size_t sub) noexcept {
  return meld_roots(E, root, sub);
}

inline void insert(PriorityQueue *pqueue, std::size_t element_idx,
                   double key) noexcept {
  assert(pqueue != nullptr);
  assert(element_idx < pqueue->length);
  assert(pqueue->size < pqueue->length);
  assert(pqueue->Elements[element_idx].state == NOT_IN_HEAP);
  assert(key < kInfinity);

  Element *E = pqueue->Elements;
  Element &e = E[element_idx];
  e.key = key;
  e.state = IN_HEAP;
  e.child = kNullIdx;
  e.next = kNullIdx;
  e.prev = kNullIdx;

  pqueue->root = meld_roots(E, pqueue->root, element_idx);
  ++pqueue->size;
}

// Cut `element_idx` out of its parent's child list. After this call the
// element satisfies the root invariant (prev == next == kNullIdx) but its
// children are intact.
AEQ_ALWAYS_INLINE void detach(Element *E, std::size_t element_idx) noexcept {
  const std::size_t prev = E[element_idx].prev;
  const std::size_t next = E[element_idx].next;

  // `prev` is either the previous sibling (we are not first child of
  // parent) or the parent (we are first child). Disambiguate by
  // checking the parent's `child` link.
  if (E[prev].child == element_idx) {
    E[prev].child = next;
  } else {
    E[prev].next = next;
  }
  if (next != kNullIdx) {
    E[next].prev = prev;
  }
  E[element_idx].prev = kNullIdx;
  E[element_idx].next = kNullIdx;
}

inline void decrease_key(PriorityQueue *pqueue, std::size_t element_idx,
                         double key_new) noexcept {
  assert(pqueue != nullptr);
  assert(element_idx < pqueue->length);
  assert(pqueue->Elements[element_idx].state == IN_HEAP);
  assert(key_new <= pqueue->Elements[element_idx].key);

  Element *E = pqueue->Elements;
  E[element_idx].key = key_new;

  // Short-circuit when this element is already the root: by invariant
  // the root has prev == kNullIdx, so we avoid loading pqueue->root.
  if (E[element_idx].prev == kNullIdx) {
    return;
  }
  detach(E, element_idx);
  pqueue->root = meld_roots(E, pqueue->root, element_idx);
}

AEQ_ALWAYS_INLINE double peek(PriorityQueue *pqueue) noexcept {
  assert(pqueue != nullptr);
  assert(pqueue->size > 0);
  assert(pqueue->root != kNullIdx);
  return pqueue->Elements[pqueue->root].key;
}

AEQ_ALWAYS_INLINE bool is_empty(PriorityQueue *pqueue) noexcept {
  assert(pqueue != nullptr);
  return pqueue->size == 0;
}

inline std::size_t extract_min(PriorityQueue *pqueue) noexcept {
  assert(pqueue != nullptr);
  assert(pqueue->size > 0);
  assert(pqueue->root != kNullIdx);

  Element *E = pqueue->Elements;
  const std::size_t min_idx = pqueue->root;
  const std::size_t first = E[min_idx].child;

  // Fast paths.
  if (first == kNullIdx) {
    // 0 children: heap becomes empty.
    pqueue->root = kNullIdx;
  } else {
    const std::size_t second = E[first].next;
    if (second == kNullIdx) {
      // 1 child: the child is the new root.
      E[first].prev = kNullIdx;
      // E[first].next is already kNullIdx.
      pqueue->root = first;
    } else {
      // >= 2 children: multipass meld (Stasko & Vitter 1987).
      //
      // Walk siblings left-to-right, pairing each adjacent pair and
      // melding the result into a running accumulator. Equivalent to
      // a single left-to-right pass of the two-pass scheme but
      // without the stack: merging into the accumulator immediately
      // rather than after the full sibling walk.
      //
      // Strictly speaking this is "single-pass" / "multipass", which
      // has worse amortised bounds than two-pass but better practical
      // constants on the small fan-outs typical of road networks.

      // Detach the first sibling and seed the accumulator with the
      // first pair (or the lone first element if odd count).
      std::size_t a = first;
      std::size_t b = second;
      std::size_t c = E[b].next;
      E[a].prev = E[a].next = kNullIdx;
      E[b].prev = E[b].next = kNullIdx;
      std::size_t acc = meld_roots(E, a, b);

      while (c != kNullIdx) {
        AEQ_PREFETCH(&E[c]);
        a = c;
        b = E[a].next;
        E[a].prev = E[a].next = kNullIdx;
        if (b == kNullIdx) {
          // Lone tail sibling.
          acc = meld_subtree(E, acc, a);
          break;
        }
        c = E[b].next;
        E[b].prev = E[b].next = kNullIdx;
        // Pair (a, b), then meld into accumulator.
        const std::size_t paired = meld_roots(E, a, b);
        acc = meld_subtree(E, acc, paired);
      }
      pqueue->root = acc;
    }
  }

  // Finalise the extracted element.
  E[min_idx].state = SCANNED;
  E[min_idx].child = kNullIdx;
  E[min_idx].next = kNullIdx;
  E[min_idx].prev = kNullIdx;
  --pqueue->size;

  return min_idx;
}

} // namespace aequilibrae::paths::cpp
