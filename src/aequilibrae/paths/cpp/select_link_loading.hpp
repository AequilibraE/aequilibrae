#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "aon_workspace.hpp"
#include "search_results.hpp"

namespace aequilibrae::paths::cpp::mvp {

// Load trips whose chosen path uses any link in one set, for one origin.
// selected_links: bool [link_count], true for links in this set.
// demand: T [destination_count, class_count], one row per physical node.
// od: T [destination_count, class_count], replaced on each call.
// link_loads: T [link_count, class_count], added to, never cleared here.
// Every link on a matching path gets its demand, not just selected links.
// Trips back to the origin and paths not finished by the search are skipped.
// Call network_loading separately for loads from all trips.
//
// The caller must prepare loading and select-link scratch before this call.
// All buffers must have the sizes above, destination_count <= node_count, and
// the search must contain a valid state tree. Buffers must not share memory;
// each worker needs its own scratch and outputs. Empty destination/class/link
// axes allow null pointers for those arrays. Zero classes does no work.
// This does not run or extend a search, and does not allocate memory.
//
// Tree passes avoid walking shared parts of routes again for every destination.
// Time: O((states + destinations) * classes).
// Scratch: O(states * classes + states). Reuse it for one set at a time so its
// size does not grow with the number of sets.
template <typename T>
void select_link_loading(const SearchResults &results,
                         std::size_t destination_count, const T *demand,
                         std::size_t class_count, const bool *selected_links,
                         AoNWorkspace<T> &workspace, T *od,
                         T *link_loads) noexcept {
  static_assert(std::is_floating_point_v<T>);
  if (class_count == 0) {
    return;
  }
  // Scratch is shared across sets and origins. Clear it so old trips cannot
  // leak into this set; clear OD too because unmatched trips must become zero.
  std::fill_n(workspace.state_loads, workspace.state_count * class_count, T{0});
  std::fill_n(workspace.selected_paths, workspace.state_count, false);
  if (destination_count) {
    std::fill_n(od, destination_count * class_count, T{0});
  }
  // Parents come first, even with zero-cost links, so each path can reuse its
  // parent's flag. Keep flags per STATE: with turns, two arrivals at the same
  // node can have different paths and can match different sets.
  for (std::size_t i = 0; i < results.settled_count; ++i) {
    const auto state = results.reached_first[i];
    // The root has no incoming link. Its empty path must not match a set.
    if (state != results.root) {
      workspace.selected_paths[state] =
          workspace.selected_paths[results.predecessors[state]] ||
          selected_links[results.connectors[state]];
    }
  }
  // Add demand once at the chosen arrival, rather than at each selected link.
  // Otherwise a path crossing several members would count the same trip twice.
  for (std::size_t node = 0; node < destination_count; ++node) {
    const auto terminal = results.terminal_states[node];
    if (terminal == std::numeric_limits<std::size_t>::max() ||
        terminal == results.root || !workspace.selected_paths[terminal]) {
      continue;
    }
    for (std::size_t cls = 0; cls < class_count; ++cls) {
      const T value = demand[node * class_count + cls];
      od[node * class_count + cls] = value;
      workspace.state_loads[terminal * class_count + cls] += value;
    }
  }
  // Children come first here so each parent receives all its matching trips.
  // Do not skip states with false flags: links BEFORE the first selected link
  // must also receive demand from matching destinations farther down the tree.
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
