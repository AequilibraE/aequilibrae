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
    heap_ = new std::size_t[length];
    elements_ = new Element[length];
    current_epoch_ = 1;

    for (std::size_t i = 0; i < length_; ++i) {
      heap_[i] = length_;
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

    const std::size_t node_idx = size_;
    ++size_;
    elements_[element_idx].state = IN_HEAP;
    elements_[element_idx].epoch = current_epoch_;
    elements_[element_idx].node_idx = node_idx;
    // The element may carry a stale key from a previous epoch; make the
    // sift-up below a genuine decrease.
    elements_[element_idx].key = kInfinity;
    heap_[node_idx] = element_idx;
    decrease_key_from_node_index(node_idx, key);
  }

  void decrease_key_impl(std::size_t element_idx, double key_new) noexcept {
    assert(element_idx < length_);
    assert(element_is_in_heap(element_idx));

    decrease_key_from_node_index(elements_[element_idx].node_idx, key_new);
  }

  double peek_impl() const noexcept {
    assert(size_ > 0);
    return elements_[heap_[0]].key;
  }

  bool is_empty_impl() const noexcept { return size_ == 0; }

  std::size_t extract_min_impl() noexcept {
    assert(size_ > 0);

    const std::size_t element_idx = heap_[0];
    const std::size_t node_idx = size_ - 1;

    exchange_nodes(0, node_idx);

    elements_[element_idx].state = SCANNED;
    elements_[element_idx].node_idx = length_;
    heap_[node_idx] = length_;
    --size_;

    if (size_ > 0) {
      min_heapify(0);
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

  static constexpr std::size_t kArity = 4;

  std::size_t length_ = 0;
  std::size_t size_ = 0;
  std::size_t *heap_ = nullptr;
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

  void exchange_nodes(std::size_t node_i, std::size_t node_j) noexcept {
    assert(node_i < size_);
    assert(node_j < size_);

    const std::size_t element_i = heap_[node_i];
    const std::size_t element_j = heap_[node_j];

    assert(element_i < length_);
    assert(element_j < length_);

    heap_[node_i] = element_j;
    heap_[node_j] = element_i;
    elements_[element_j].node_idx = node_i;
    elements_[element_i].node_idx = node_j;
  }

  void decrease_key_from_node_index(std::size_t node_idx,
                                    double key_new) noexcept {
    assert(node_idx < size_);

    std::size_t current = node_idx;
    const std::size_t element_idx = heap_[current];
    assert(element_idx < length_);
    assert(elements_[element_idx].state == IN_HEAP);
    assert(key_new <= elements_[element_idx].key);

    elements_[element_idx].key = key_new;
    while (current > 0) {
      const std::size_t parent = (current - 1) / kArity;
      const std::size_t parent_element = heap_[parent];
      assert(parent_element < length_);
      if (elements_[parent_element].key > key_new) {
        exchange_nodes(current, parent);
        current = parent;
      } else {
        break;
      }
    }
  }

  void min_heapify(std::size_t node_idx) noexcept {
    if (size_ == 0) {
      return;
    }

    assert(node_idx < size_);

    std::size_t current = node_idx;
    while (true) {
      std::size_t smallest = current;
      double min_key = elements_[heap_[smallest]].key;

      const std::size_t first_child = kArity * current + 1;
      const std::size_t end_child =
          first_child + kArity < size_ ? first_child + kArity : size_;

      for (std::size_t candidate = end_child; candidate > first_child;
           --candidate) {
        const std::size_t child = candidate - 1;
        const double child_key = elements_[heap_[child]].key;
        if (child_key < min_key) {
          smallest = child;
          min_key = child_key;
        }
      }

      if (smallest == current) {
        return;
      }

      exchange_nodes(current, smallest);
      current = smallest;
    }
  }
};

} // namespace aequilibrae::paths::cpp
