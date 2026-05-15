#pragma once

#include <cassert>
#include <cstddef>

#include "pq_heap_base.hpp"

namespace aequilibrae::paths::cpp {

class PairingHeap final : public PriorityQueueBase<PairingHeap> {
public:
  PairingHeap() noexcept = default;

  ~PairingHeap() { free_heap(); }

private:
  friend class PriorityQueueBase<PairingHeap>;

  void init_heap_impl(std::size_t length) noexcept {
    free_heap();

    length_ = length;
    size_ = 0;
    elements_ = length > 0 ? new Element[length] : nullptr;
    root_ = kNullIdx;

    for (std::size_t i = 0; i < length_; ++i) {
      initialize_element(i);
    }
  }

  void alloc_heap_impl(std::size_t length) noexcept { init_heap(length); }

  void reset_heap_impl() noexcept {}

  void free_heap_impl() noexcept {
    delete[] elements_;

    elements_ = nullptr;
    length_ = 0;
    size_ = 0;
    root_ = kNullIdx;
  }

  void insert_impl(std::size_t element_idx, double key) noexcept {
    assert(element_idx < length_);
    assert(size_ < length_);
    assert(elements_[element_idx].state == NOT_IN_HEAP);
    assert(key < kInfinity);

    Element &element = elements_[element_idx];
    element.key = key;
    element.state = IN_HEAP;
    element.child = kNullIdx;
    element.next = kNullIdx;
    element.prev = kNullIdx;

    root_ = meld_roots(root_, element_idx);
    ++size_;
  }

  void decrease_key_impl(std::size_t element_idx, double key_new) noexcept {
    assert(element_idx < length_);
    assert(elements_[element_idx].state == IN_HEAP);
    assert(key_new <= elements_[element_idx].key);

    elements_[element_idx].key = key_new;
    if (elements_[element_idx].prev == kNullIdx) {
      return;
    }

    detach(element_idx);
    root_ = meld_roots(root_, element_idx);
  }

  double peek_impl() const noexcept {
    assert(size_ > 0);
    assert(root_ != kNullIdx);
    return elements_[root_].key;
  }

  bool is_empty_impl() const noexcept { return size_ == 0; }

  std::size_t extract_min_impl() noexcept {
    assert(size_ > 0);
    assert(root_ != kNullIdx);

    const std::size_t min_idx = root_;
    const std::size_t first = elements_[min_idx].child;

    if (first == kNullIdx) {
      root_ = kNullIdx;
    } else {
      const std::size_t second = elements_[first].next;
      if (second == kNullIdx) {
        elements_[first].prev = kNullIdx;
        root_ = first;
      } else {
        std::size_t first_node = first;
        std::size_t second_node = second;
        std::size_t next_pair = elements_[second_node].next;
        elements_[first_node].prev = kNullIdx;
        elements_[first_node].next = kNullIdx;
        elements_[second_node].prev = kNullIdx;
        elements_[second_node].next = kNullIdx;
        std::size_t accumulator = meld_roots(first_node, second_node);

        while (next_pair != kNullIdx) {
#if defined(__GNUC__) || defined(__clang__)
          __builtin_prefetch(&elements_[next_pair], 0, 1);
#endif
          first_node = next_pair;
          second_node = elements_[first_node].next;
          elements_[first_node].prev = kNullIdx;
          elements_[first_node].next = kNullIdx;
          if (second_node == kNullIdx) {
            accumulator = meld_subtree(accumulator, first_node);
            break;
          }

          next_pair = elements_[second_node].next;
          elements_[second_node].prev = kNullIdx;
          elements_[second_node].next = kNullIdx;
          const std::size_t paired = meld_roots(first_node, second_node);
          accumulator = meld_subtree(accumulator, paired);
        }

        root_ = accumulator;
      }
    }

    elements_[min_idx].state = SCANNED;
    elements_[min_idx].child = kNullIdx;
    elements_[min_idx].next = kNullIdx;
    elements_[min_idx].prev = kNullIdx;
    --size_;

    return min_idx;
  }

  ElementState effective_state_impl(std::size_t element_idx) const noexcept {
    return elements_[element_idx].state;
  }

  double element_key_impl(std::size_t element_idx) const noexcept {
    return elements_[element_idx].key;
  }

  struct Element {
    double key;
    ElementState state;
    std::size_t child;
    std::size_t next;
    std::size_t prev;
  };

  static constexpr std::size_t kNullIdx = static_cast<std::size_t>(-1);

  std::size_t length_ = 0;
  std::size_t size_ = 0;
  Element *elements_ = nullptr;
  std::size_t root_ = kNullIdx;

  void initialize_element(std::size_t element_idx) noexcept {
    elements_[element_idx].key = kInfinity;
    elements_[element_idx].state = NOT_IN_HEAP;
    elements_[element_idx].child = kNullIdx;
    elements_[element_idx].next = kNullIdx;
    elements_[element_idx].prev = kNullIdx;
  }

  std::size_t meld_roots(std::size_t first_root,
                         std::size_t second_root) noexcept {
    if (first_root == kNullIdx) {
      return second_root;
    }
    if (second_root == kNullIdx) {
      return first_root;
    }

    std::size_t winner = first_root;
    std::size_t loser = second_root;
    if (elements_[second_root].key < elements_[first_root].key) {
      winner = second_root;
      loser = first_root;
    }

    const std::size_t old_child = elements_[winner].child;
    elements_[loser].next = old_child;
    elements_[loser].prev = winner;
    if (old_child != kNullIdx) {
      elements_[old_child].prev = loser;
    }
    elements_[winner].child = loser;
    return winner;
  }

  std::size_t meld_subtree(std::size_t root, std::size_t sub) noexcept {
    return meld_roots(root, sub);
  }

  void detach(std::size_t element_idx) noexcept {
    const std::size_t prev = elements_[element_idx].prev;
    const std::size_t next = elements_[element_idx].next;

    if (elements_[prev].child == element_idx) {
      elements_[prev].child = next;
    } else {
      elements_[prev].next = next;
    }
    if (next != kNullIdx) {
      elements_[next].prev = prev;
    }
    elements_[element_idx].prev = kNullIdx;
    elements_[element_idx].next = kNullIdx;
  }
};

} // namespace aequilibrae::paths::cpp
