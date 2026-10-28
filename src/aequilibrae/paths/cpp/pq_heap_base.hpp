#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>

#include "aeq_log.hpp"

namespace aequilibrae::paths::cpp {

enum ElementState {
  SCANNED,
  NOT_IN_HEAP,
  IN_HEAP,
};

template <typename T>
inline constexpr T kInfinity = std::numeric_limits<T>::infinity();

template <class Derived, typename T> class PriorityQueueBase {
  static_assert(std::is_floating_point_v<T>);

public:
  using value_type = T;
  PriorityQueueBase(const PriorityQueueBase &) = delete;
  PriorityQueueBase &operator=(const PriorityQueueBase &) = delete;
  PriorityQueueBase(PriorityQueueBase &&) = delete;
  PriorityQueueBase &operator=(PriorityQueueBase &&) = delete;

  // Attach an AeqLogClosure (from a Bridge) so the heap can report its use.
  // nullptr (the default) disables logging entirely.
  void attach_logger(AeqLogClosure *closure) noexcept {
    log_closure_ = closure;
  }

  void init_heap(std::size_t length) noexcept {
    AEQ_LOG(log_closure_, AEQ_LOG_DEBUG,
            aeq_format_string(Derived::kName, ": init_heap(length = ", length,
                              ")"));
    derived().init_heap_impl(length);
  }

  void alloc_heap(std::size_t length) noexcept {
    derived().alloc_heap_impl(length);
  }

  void reset_heap() noexcept {
    AEQ_LOG(log_closure_, AEQ_LOG_DEBUG,
            aeq_format_string(Derived::kName, ": reset_heap()"));
    derived().reset_heap_impl();
  }

  void free_heap() noexcept { derived().free_heap_impl(); }

  void insert(std::size_t element_idx, T key) noexcept {
    derived().insert_impl(element_idx, key);
  }

  void decrease_key(std::size_t element_idx, T key_new) noexcept {
    derived().decrease_key_impl(element_idx, key_new);
  }

  [[nodiscard]] T peek() const noexcept { return derived().peek_impl(); }

  [[nodiscard]] bool is_empty() const noexcept {
    return derived().is_empty_impl();
  }

  [[nodiscard]] std::size_t extract_min() noexcept {
    return derived().extract_min_impl();
  }

  [[nodiscard]] ElementState
  effective_state(std::size_t element_idx) const noexcept {
    return derived().effective_state_impl(element_idx);
  }

  [[nodiscard]] T element_key(std::size_t element_idx) const noexcept {
    return derived().element_key_impl(element_idx);
  }

protected:
  PriorityQueueBase() = default;
  ~PriorityQueueBase() = default;

  AeqLogClosure *log_closure_ = nullptr;

private:
  Derived &derived() noexcept { return *static_cast<Derived *>(this); }

  const Derived &derived() const noexcept {
    return *static_cast<const Derived *>(this);
  }
};

} // namespace aequilibrae::paths::cpp
