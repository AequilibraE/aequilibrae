#pragma once

#include <cassert>
#include <cstddef>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "pq_heap_base.hpp"

namespace aequilibrae::paths::cpp {

template <typename T>
class StdPriorityQueueAdapter final
    : public PriorityQueueBase<StdPriorityQueueAdapter<T>, T> {
public:
  static constexpr const char *kName = "StdPriorityQueueAdapter";

  StdPriorityQueueAdapter() noexcept = default;

  ~StdPriorityQueueAdapter() { this->free_heap(); }

private:
  friend class PriorityQueueBase<StdPriorityQueueAdapter<T>, T>;

  using Entry = std::pair<T, std::size_t>;

  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq_;
  std::vector<T> keys_;
  std::vector<T> extracted_keys_;
  std::vector<ElementState> states_;
  std::size_t size_ = 0;

  void init_heap_impl(std::size_t length) noexcept { this->alloc_heap(length); }

  void alloc_heap_impl(std::size_t length) noexcept {
    this->free_heap();
    keys_.assign(length, kInfinity<T>);
    extracted_keys_.assign(length, kInfinity<T>);
    states_.assign(length, ElementState::NOT_IN_HEAP);
    size_ = 0;
  }

  void reset_heap_impl() noexcept {
    while (!pq_.empty()) {
      pq_.pop();
    }
    for (std::size_t i = 0; i < states_.size(); ++i) {
      states_[i] = ElementState::NOT_IN_HEAP;
      keys_[i] = kInfinity<T>;
      extracted_keys_[i] = kInfinity<T>;
    }
    size_ = 0;
  }

  void free_heap_impl() noexcept {
    while (!pq_.empty()) {
      pq_.pop();
    }
    keys_.clear();
    extracted_keys_.clear();
    states_.clear();
    size_ = 0;
  }

  void insert_impl(std::size_t element_idx, T key) noexcept {
    assert(element_idx < states_.size());
    // SCANNED elements may be re-inserted (A* with an inconsistent heuristic).
    assert(states_[element_idx] != ElementState::IN_HEAP);
    assert(key < kInfinity<T>);
    keys_[element_idx] = key;
    states_[element_idx] = ElementState::IN_HEAP;
    pq_.push({key, element_idx});
    ++size_;
  }

  void decrease_key_impl(std::size_t element_idx, T key_new) noexcept {
    assert(states_[element_idx] == ElementState::IN_HEAP);
    assert(key_new <= keys_[element_idx]);
    keys_[element_idx] = key_new;
    pq_.push({key_new, element_idx});
  }

  [[nodiscard]] T peek_impl() const noexcept { return pq_.top().first; }

  [[nodiscard]] bool is_empty_impl() const noexcept { return size_ == 0; }

  [[nodiscard]] std::size_t extract_min_impl() noexcept {
    discard_stale_entries();
    const Entry top = pq_.top();
    pq_.pop();

    const std::size_t idx = top.second;
    const T key = top.first;
    states_[idx] = ElementState::SCANNED;
    extracted_keys_[idx] = key;
    --size_;
    return idx;
  }

  [[nodiscard]] ElementState
  effective_state_impl(std::size_t element_idx) const noexcept {
    return states_[element_idx];
  }

  [[nodiscard]] T element_key_impl(std::size_t element_idx) const noexcept {
    if (states_[element_idx] == ElementState::SCANNED) {
      return extracted_keys_[element_idx];
    }
    return keys_[element_idx];
  }

  void discard_stale_entries() noexcept {
    while (!pq_.empty()) {
      const Entry top = pq_.top();
      const std::size_t idx = top.second;
      if (states_[idx] != ElementState::IN_HEAP) {
        pq_.pop();
        continue;
      }
      if (top.first != keys_[idx]) {
        pq_.pop();
        continue;
      }
      break;
    }
  }
};

} // namespace aequilibrae::paths::cpp
