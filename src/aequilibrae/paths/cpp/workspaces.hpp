#pragma once

#include <cstddef>
#include <utility>
#include <variant>

#include "pq_4ary_heap.hpp"
#include "pq_pairing_heap.hpp"
#include "pq_std_priority_queue_adapter.hpp"

namespace aequilibrae::paths::cpp::routing {

enum class SearchHeap { FourAry, Pairing, Std };

class SearchHeapStorage {
public:
  SearchHeapStorage(std::size_t state_count, SearchHeap type) {
    switch (type) {
    case SearchHeap::FourAry:
      break; // The variant defaults to this alternative.
    case SearchHeap::Pairing:
      heap.emplace<cpp::PairingHeap>();
      break;
    case SearchHeap::Std:
      heap.emplace<cpp::StdPriorityQueueAdapter>();
      break;
    }
    std::visit([state_count](auto &queue) { queue.alloc_heap(state_count); },
               heap);
  }

  template <class F> void visit(F &&fn) {
    std::visit(std::forward<F>(fn), heap);
  }

private:
  std::variant<cpp::FourAryHeap, cpp::PairingHeap, cpp::StdPriorityQueueAdapter>
      heap;
};

// Scratch heap for Dijkstra.
struct SearchWorkspace {
  SearchHeapStorage *heap = nullptr;
};

// A* also  needs tentative costs and per-node heuristic estimates.
struct AStarWorkspace {
  SearchWorkspace search;
  double *costs = nullptr;     // [states]
  double *estimates = nullptr; // [nodes]
};

// Ordinary and selected loading can reuse this cascade scratch sequentially
// because each operation replaces every state total.
template <typename T> struct LoadingWorkspace {
  std::size_t state_count = 0;
  std::size_t class_count = 0;
  T *state_loads = nullptr; // [states, classes]
};

// Only additive fields need state sums. Direct cost projection needs no
// scratch.
template <typename T> struct SkimmingWorkspace {
  std::size_t state_count = 0;
  std::size_t field_count = 0;
  T *state_skims = nullptr; // [states, additive fields]
};

// Process sets sequentially so membership scratch does not grow with set count.
// Demand cascades use a separately supplied LoadingWorkspace.
struct SelectLinkWorkspace {
  std::size_t state_count = 0;
  bool *selected_paths = nullptr;
};

// Group worker scratch for the assignment driver. Each kernel accepts only
// the small workspace it needs; none depends on this aggregate.
template <typename T> struct AoNWorkspace {
  SearchWorkspace search;
  LoadingWorkspace<T> loading;
  SkimmingWorkspace<T> skimming;
  SelectLinkWorkspace select_link;
};

} // namespace aequilibrae::paths::cpp::routing
