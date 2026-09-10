#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "aon_workspace.hpp"
#include "search_results.hpp"

namespace aequilibrae::paths::cpp::mvp {

// Allocation-free, single-origin cascade loading of a finalized state tree.
//
// Preconditions:
// - demand is packed [destination_count, class_count], a single origin's
//   contiguous slice of an [origins, destinations, classes] matrix. Rows map
//   to physical nodes [0, destination_count), destination_count <= node_count.
// - workspace has [state_count, class_count] state_loads scratch prepared.
// - link_loads is a writable packed [link_count, class_count] accumulator in
//   context-local link order. It is NOT reset here: zero it once per iteration.
// - Input, scratch, output, and search/context buffers do not overlap, and all
//   allocations remain alive. Each worker exclusively owns its results,
//   workspace, and output slice; demand and graph inputs may be shared.
// - Zero classes permits null demand/scratch/output; zero destinations permits
//   null demand. An empty link array permits null output.
//
// Unreachable/unfinalized destinations and intrazonal demand are ignored. This
// does not extend a partial search. Demand at selected terminals is propagated
// through STATE parents, not physical-node terminals, so nonterminal arrival
// histories in turn routing work without a routing-mode branch or path walks.
// Ordinary floating-point addition is used (including for NaN/inf demand).
// Runtime O((state_count + destination_count) * class_count), scratch
// O(state_count * class_count). Reverse settlement order is child-before-parent
// even with zero-cost links/cycles. Scratch is reset on every call.
template <typename T>
void network_loading(const SearchResults &results, std::size_t destination_count,
                     const T *demand, std::size_t class_count,
                     AoNWorkspace<T> &workspace, T *link_loads) noexcept {
  static_assert(std::is_floating_point_v<T>,
                "Network loading requires a floating-point type");
  if (class_count == 0) {
    return;
  }
  std::fill_n(workspace.state_loads, workspace.state_count * class_count, T{0});

  for (std::size_t node = 0; node < destination_count; ++node) {
    const auto terminal = results.terminal_states[node];
    if (terminal == std::numeric_limits<std::size_t>::max() ||
        terminal == results.root) {
      continue;
    }
    T *row = workspace.state_loads + terminal * class_count;
    for (std::size_t cls = 0; cls < class_count; ++cls) {
      row[cls] += demand[node * class_count + cls];
    }
  }

  for (std::size_t i = results.settled_count; i > 0; --i) {
    const auto state = results.reached_first[i - 1];
    if (state == results.root) {
      continue;
    }
    const T *row = workspace.state_loads + state * class_count;
    T *parent = workspace.state_loads + results.predecessors[state] * class_count;
    T *link = link_loads + results.connectors[state] * class_count;
    for (std::size_t cls = 0; cls < class_count; ++cls) {
      link[cls] += row[cls];
      parent[cls] += row[cls];
    }
  }
}

} // namespace aequilibrae::paths::cpp::mvp
