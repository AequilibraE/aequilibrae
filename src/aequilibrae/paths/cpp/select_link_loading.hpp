#pragma once

#include <algorithm>
#include <cstddef>

#include "network_loading.hpp"

namespace aequilibrae::paths::cpp::mvp {

// Load trips whose chosen path uses any link in a set. Membership has one
// flag per link.
//
// OD [destinations, classes] is overwritten. output is accumulated into.
template <typename T>
void select_link_loading(const SearchResults &results,
                         const LoadingQuery<T> &query,
                         const bool *selected_links,
                         SelectLinkWorkspace selection,
                         LoadingWorkspace<T> loading, T *od,
                         LoadingOutputs<T> output) noexcept {
  const auto classes = query.class_count;
  if (classes == 0) {
    return;
  }
  std::fill_n(loading.state_loads, loading.state_count * classes, T{0});
  std::fill_n(selection.selected_paths, selection.state_count, false);
  if (query.destination_count) {
    std::fill_n(od, query.destination_count * classes, T{0});
  }

  // Reuse the parent's membership flag. Keep it per state: different arrivals
  // at the same physical node can have different selected-link histories.
  for (std::size_t i = 0; i < results.metadata->settled_count; ++i) {
    const auto state = results.settlement_order[i];
    if (state != results.metadata->root) {
      selection.selected_paths[state] =
          selection.selected_paths[results.predecessors[state]] ||
          selected_links[results.connectors[state]];
    }
  }

  // Seed each matched OD once, even if its path crosses several set members.
  for (std::size_t node = 0; node < query.destination_count; ++node) {
    const auto terminal = results.terminal_states[node];

    if (terminal == invalid_state || terminal == results.metadata->root ||
        !selection.selected_paths[terminal]) {
      continue;
    }

    for (std::size_t cls = 0; cls < classes; ++cls) {
      const T value = query.demand[node * classes + cls];
      od[node * classes + cls] = value;
      loading.state_loads[terminal * classes + cls] += value;
    }
  }

  // Do not filter this pass by membership: demand must also reach ancestors
  // before the selected link. Ordinary loading uses this same cascade.
  cascade_loads(results, loading, output);
}

} // namespace aequilibrae::paths::cpp::mvp
