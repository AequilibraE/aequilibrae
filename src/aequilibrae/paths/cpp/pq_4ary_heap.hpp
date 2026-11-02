#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>

#include "pq_heap_base.hpp"

namespace aequilibrae::paths::cpp {

template <typename T>
class FourAryHeap final : public PriorityQueueBase<FourAryHeap<T>, T> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

public:
  static constexpr const char *kName = "FourAryHeap";

  FourAryHeap() noexcept = default;

  ~FourAryHeap() { this->free_heap(); }

private:
  friend class PriorityQueueBase<FourAryHeap<T>, T>;

  void init_heap_impl(std::size_t length) { this->alloc_heap(length); }

  void alloc_heap_impl(std::size_t length) {
    // UINT32_MAX is reserved for SCANNED; valid positions are [0, length).
    if (length > kScanned ||
        length > std::numeric_limits<std::size_t>::max() / sizeof(Node) -
                     kLeadingSlots) {
      throw std::invalid_argument("FourAryHeap capacity exceeds its position range");
    }

    const std::size_t slots = length + kLeadingSlots;
    void *allocation =
        ::operator new[](slots * sizeof(Node), std::align_val_t{kCacheLineBytes});
    Element *elements = nullptr;
    try {
      elements = new Element[length];
    } catch (...) {
      ::operator delete[](allocation, std::align_val_t{kCacheLineBytes});
      throw;
    }

    // Placement array-new starts the trivial nodes' lifetimes without writing
    // their contents. The raw allocation is retained for matching deallocation.
    Node *nodes = ::new (allocation) Node[slots];
    this->free_heap();

    length_ = length;
    size_ = 0;
    heap_allocation_ = allocation;
    heap_ = nodes + kLeadingSlots;
    elements_ = elements;
    current_epoch_ = 1;

    for (std::size_t i = 0; i < length_; ++i) {
      initialize_element(i);
    }
  }

  // O(1) reset: bumping the epoch invalidates every element (effective_state
  // treats a stale epoch as NOT_IN_HEAP and insert overwrites the stale key).
  // Only on epoch wrap-around do we pay for a full reinitialisation.
  void reset_heap_impl() noexcept {
    size_ = 0;
    current_epoch_ += 1;

    if (current_epoch_ == 0) {
      current_epoch_ = 1;
      for (std::size_t i = 0; i < length_; ++i) {
        initialize_element(i);
      }
    }
  }

  void free_heap_impl() noexcept {
    ::operator delete[](heap_allocation_, std::align_val_t{kCacheLineBytes});
    delete[] elements_;

    heap_ = nullptr;
    heap_allocation_ = nullptr;
    elements_ = nullptr;
    length_ = 0;
    size_ = 0;
    current_epoch_ = 0;
  }

  void insert_impl(std::size_t element_idx, T key) noexcept {
    assert(element_idx < length_);
    assert(size_ < length_);
    // SCANNED elements may be re-inserted (A* with an inconsistent heuristic).
    assert(this->effective_state(element_idx) != IN_HEAP);
    assert(key < kInfinity<T>);

    elements_[element_idx].epoch = current_epoch_;
    // Overwrite a potentially stale key from a previous epoch.
    elements_[element_idx].key = key;
    sift_up(size_++, {key, element_idx});
  }

  void decrease_key_impl(std::size_t element_idx, T key_new) noexcept {
    assert(element_idx < length_);
    assert(element_is_in_heap(element_idx));
    assert(key_new <= elements_[element_idx].key);

    elements_[element_idx].key = key_new;
    sift_up(elements_[element_idx].node_idx, {key_new, element_idx});
  }

  T peek_impl() const noexcept {
    assert(size_ > 0);
    return heap_[0].key;
  }

  bool is_empty_impl() const noexcept { return size_ == 0; }

  std::size_t extract_min_impl() noexcept {
    assert(size_ > 0);

    const std::size_t element_idx = heap_[0].element_idx;
    elements_[element_idx].node_idx = kScanned;
    if (--size_ > 0) {
      sift_down(heap_[size_]);
    }
    return element_idx;
  }

  ElementState effective_state_impl(std::size_t element_idx) const noexcept {
    const Element &element = elements_[element_idx];
    if (element.epoch != current_epoch_) {
      return NOT_IN_HEAP;
    }
    return element.node_idx == kScanned ? SCANNED : IN_HEAP;
  }

  T element_key_impl(std::size_t element_idx) const noexcept {
    return elements_[element_idx].key;
  }

  template <class F> void for_each_pending_impl(F &&fn) const {
    for (std::size_t i = 0; i < size_; ++i) {
      fn(heap_[i].element_idx);
    }
  }

  struct alignas(16) Element {
    T key;
    std::uint32_t epoch;
    std::uint32_t node_idx;
  };
  static_assert(sizeof(Element) == 16);

  // The allocation is line-aligned and the logical root is offset by 3 slots,
  // so every full group of four 16-byte children occupies one cache line.
  struct alignas(16) Node {
    T key;
    std::size_t element_idx;
  };

  static constexpr std::size_t kArity = 4;
  static constexpr std::size_t kCacheLineBytes = 64;
  static_assert(kArity * sizeof(Node) == kCacheLineBytes);
  static constexpr std::size_t kLeadingSlots = 3;
  static constexpr std::uint32_t kScanned =
      std::numeric_limits<std::uint32_t>::max();

  std::size_t length_ = 0;
  std::size_t size_ = 0;
  Node *heap_ = nullptr;
  void *heap_allocation_ = nullptr;
  Element *elements_ = nullptr;
  std::uint32_t current_epoch_ = 0;

  void initialize_element(std::size_t element_idx) noexcept {
    elements_[element_idx].key = kInfinity<T>;
    elements_[element_idx].epoch = 0;
    elements_[element_idx].node_idx = kScanned;
  }

  bool element_is_in_heap(std::size_t element_idx) const noexcept {
    const Element &element = elements_[element_idx];
    return element.epoch == current_epoch_ && element.node_idx != kScanned;
  }

  void place(std::size_t node_idx, Node node) noexcept {
    heap_[node_idx] = node;
    elements_[node.element_idx].node_idx = static_cast<std::uint32_t>(node_idx);
  }

  // Sifts move a hole instead of swapping, so each level writes once.
  void sift_up(std::size_t current, Node node) noexcept {
    while (current > 0) {
      const std::size_t parent = (current - 1) / kArity;
      if (heap_[parent].key <= node.key) {
        break;
      }
      place(current, heap_[parent]);
      current = parent;
    }
    place(current, node);
  }

  void sift_down(Node node) noexcept {
    std::size_t current = 0;
    while (kArity * current + kArity < size_) {
      const std::size_t first_child = kArity * current + 1;
      // Strict comparisons choose the right child on ties, preserving the
      // descending child scan's extraction order. The replacement wins ties.
      const std::size_t left =
          heap_[first_child].key < heap_[first_child + 1].key
              ? first_child
              : first_child + 1;
      const std::size_t right =
          heap_[first_child + 2].key < heap_[first_child + 3].key
              ? first_child + 2
              : first_child + 3;
      const std::size_t smallest =
          heap_[left].key < heap_[right].key ? left : right;
      if (!(heap_[smallest].key < node.key)) {
        place(current, node);
        return;
      }
      place(current, heap_[smallest]);
      current = smallest;
    }

    const std::size_t first_child = kArity * current + 1;
    std::size_t smallest = current;
    T min_key = node.key;
    for (std::size_t child = size_; child-- > first_child;) {
      if (heap_[child].key < min_key) {
        smallest = child;
        min_key = heap_[child].key;
      }
    }
    if (smallest != current) {
      place(current, heap_[smallest]);
      current = smallest;
    }
    place(current, node);
  }
};

} // namespace aequilibrae::paths::cpp
