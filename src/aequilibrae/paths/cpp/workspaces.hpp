#pragma once

#include <cstddef>
#include <utility>
#include <variant>

#include "outputs.hpp"
#include "pq_4ary_heap.hpp"
#include "pq_pairing_heap.hpp"
#include "pq_std_priority_queue_adapter.hpp"

namespace aequilibrae::paths::cpp::routing {

enum class SearchHeap { FourAry, Pairing, Std };

template <typename T> class SearchHeapStorage {
public:
  SearchHeapStorage(std::size_t state_count, SearchHeap type) {
    switch (type) {
    case SearchHeap::FourAry:
      break; // The variant defaults to this alternative.
    case SearchHeap::Pairing:
      heap.template emplace<cpp::PairingHeap<T>>();
      break;
    case SearchHeap::Std:
      heap.template emplace<cpp::StdPriorityQueueAdapter<T>>();
      break;
    }
    std::visit([state_count](auto &queue) { queue.alloc_heap(state_count); },
               heap);
  }

  template <class F> void visit(F &&fn) {
    std::visit(std::forward<F>(fn), heap);
  }

private:
  std::variant<cpp::FourAryHeap<T>, cpp::PairingHeap<T>,
               cpp::StdPriorityQueueAdapter<T>> heap;
};

// Scratch heap for Dijkstra.
template <typename T> struct SearchWorkspace {
  SearchHeapStorage<T> *heap = nullptr;
};

// A* also  needs tentative costs and per-node heuristic estimates.
template <typename T> struct AStarWorkspace {
  SearchWorkspace<T> search;
  T *costs = nullptr;     // [states]
  T *estimates = nullptr; // [nodes]
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
  SearchWorkspace<T> search;
  LoadingWorkspace<T> loading;
  SkimmingWorkspace<T> skimming;
  SelectLinkWorkspace select_link;
  SelectLinkODOutputsView<T> selected_od;
};

} // namespace aequilibrae::paths::cpp::routing
