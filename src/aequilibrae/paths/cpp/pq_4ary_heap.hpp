#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "pq_heap_base.hpp"

namespace aequilibrae::paths::cpp {

class FourAryHeap final : public PriorityQueueBase<FourAryHeap> {
public:
  static constexpr const char *kName = "FourAryHeap";

  FourAryHeap() noexcept = default;

  ~FourAryHeap() { free_heap(); }

private:
  friend class PriorityQueueBase<FourAryHeap>;

  void init_heap_impl(std::size_t length) noexcept { alloc_heap(length); }

  void alloc_heap_impl(std::size_t length) noexcept {
    free_heap();

    length_ = length;
    size_ = 0;
    heap_ = new Node[length];
    elements_ = new Element[length];
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
    delete[] heap_;
    delete[] elements_;

    heap_ = nullptr;
    elements_ = nullptr;
    length_ = 0;
    size_ = 0;
    current_epoch_ = 0;
  }

  void insert_impl(std::size_t element_idx, double key) noexcept {
    assert(element_idx < length_);
    assert(size_ < length_);
    // SCANNED elements may be re-inserted (A* with an inconsistent heuristic).
    assert(effective_state(element_idx) != IN_HEAP);
    assert(key < kInfinity);

    elements_[element_idx].state = IN_HEAP;
    elements_[element_idx].epoch = current_epoch_;
    elements_[element_idx].key = key;
    sift_up(size_++, {key, element_idx});
  }

  void decrease_key_impl(std::size_t element_idx, double key_new) noexcept {
    assert(element_idx < length_);
    assert(element_is_in_heap(element_idx));
    assert(key_new <= elements_[element_idx].key);

    elements_[element_idx].key = key_new;
    sift_up(elements_[element_idx].node_idx, {key_new, element_idx});
  }

  double peek_impl() const noexcept {
    assert(size_ > 0);
    return heap_[0].key;
  }

  bool is_empty_impl() const noexcept { return size_ == 0; }

  std::size_t extract_min_impl() noexcept {
    assert(size_ > 0);

    const std::size_t element_idx = heap_[0].element_idx;
    elements_[element_idx].state = SCANNED;
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
    return element.state;
  }

  double element_key_impl(std::size_t element_idx) const noexcept {
    return elements_[element_idx].key;
  }

  struct Element {
    double key;
    std::uint32_t epoch;
    ElementState state;
    std::size_t node_idx;
  };

  // Keys live in the heap so a node's children share a cache line.
  struct Node {
    double key;
    std::size_t element_idx;
  };

  static constexpr std::size_t kArity = 4;

  std::size_t length_ = 0;
  std::size_t size_ = 0;
  Node *heap_ = nullptr;
  Element *elements_ = nullptr;
  std::uint32_t current_epoch_ = 0;

  void initialize_element(std::size_t element_idx) noexcept {
    elements_[element_idx].key = kInfinity;
    elements_[element_idx].epoch = 0;
    elements_[element_idx].state = NOT_IN_HEAP;
    elements_[element_idx].node_idx = length_;
  }

  bool element_is_in_heap(std::size_t element_idx) const noexcept {
    const Element &element = elements_[element_idx];
    return element.epoch == current_epoch_ && element.state == IN_HEAP;
  }

  void place(std::size_t node_idx, Node node) noexcept {
    heap_[node_idx] = node;
    elements_[node.element_idx].node_idx = node_idx;
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
    while (true) {
      std::size_t smallest = current;
      double min_key = node.key;

      const std::size_t first_child = kArity * current + 1;
      const std::size_t end_child =
          first_child + kArity < size_ ? first_child + kArity : size_;

      for (std::size_t child = end_child; child-- > first_child;) {
        if (heap_[child].key < min_key) {
          smallest = child;
          min_key = heap_[child].key;
        }
      }

      if (smallest == current) {
        break;
      }
      place(current, heap_[smallest]);
      current = smallest;
    }
    place(current, node);
  }
};

} // namespace aequilibrae::paths::cpp
