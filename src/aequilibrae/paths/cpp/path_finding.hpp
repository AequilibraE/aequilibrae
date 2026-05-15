#pragma once
#include "aeq_log.hpp"
#include "pq_heap_base.hpp"
#include <limits>
#include <type_traits>

namespace aequilibrae::paths::cpp {

template <class Queue>
size_t dijkstra(const size_t origin, const size_t max_size, const double *costs,
                const size_t *csr, const size_t *fs, size_t *predecessors,
                AeqLogClosure *b) noexcept {
  static_assert(std::is_base_of<PriorityQueueBase<Queue>, Queue>::value,
                "Queue provided does derive from PriorityQueueBase");
  Queue queue{};
  size_t found{};

  AEQ_LOG(b, AEQ_LOG_CRITICAL, aeq_format_string("origin = ", origin));

  queue.init_heap(max_size);

  while (!queue.is_empty()) {
    const double tail_value = queue.peek();
    const size_t tail_vertex = queue.extract_min();
    found++;

    for (size_t idx = fs[tail_vertex]; idx < fs[tail_vertex + 1]; idx++) {
      const size_t head_vertex = csr[idx];
      const ElementState head_state = queue.effective_state(head_vertex);

      if (head_state != SCANNED) {
        const double head_value = tail_value + costs[idx];
        if (head_value == std::numeric_limits<double>::infinity()) {
          continue;
        } else if (head_state != NOT_IN_HEAP) {
          queue.insert(head_vertex, head_value);
          predecessors[head_vertex] = tail_vertex;
        } else if (queue.element_key(head_vertex) > head_value) {
          queue.decrease_key(head_vertex, head_value);
          predecessors[head_vertex] = tail_vertex;
        }
      }
    }
  }

  queue.free_heap();

  return found;
}

} // namespace aequilibrae::paths::cpp
