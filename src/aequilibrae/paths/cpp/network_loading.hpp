#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "outputs.hpp"
#include "queries.hpp"
#include "search_results.hpp"
#include "workspaces.hpp"

namespace aequilibrae::paths::cpp::mvp {

// Both ordinary and selected loading seed demand at terminal states. Parents
// receive demand from all their children before loading their own connector.
// Preserves turn arrival history.
template <typename T>
void cascade_loads(const SearchResults &results,
                   const LoadingWorkspace<T> &workspace,
                   const LoadingOutputs<T> &output) noexcept {
  const auto classes = workspace.class_count;
  if (classes == 0) {
    return;
  }

  for (std::size_t i = results.metadata->settled_count; i > 0; --i) {
    const auto state = results.settlement_order[i - 1];
    if (state == results.metadata->root) {
      continue;
    }

    const T *row = workspace.state_loads + state * classes;
    T *parent = workspace.state_loads + results.predecessors[state] * classes;
    T *link = output.link_loads + results.connectors[state] * classes;

    for (std::size_t cls = 0; cls < classes; ++cls) {
      link[cls] += row[cls];
      parent[cls] += row[cls];
    }
  }
}

// Load one origin. Scratch is replaced each call, while output accumulates
// across origins. Missing terminals and intrazonal demand are ignored.
template <typename T>
void network_loading(const SearchResults &results, const LoadingQuery<T> &query,
                     const LoadingWorkspace<T> &workspace,
                     const LoadingOutputs<T> &output) noexcept {
  static_assert(std::is_floating_point_v<T>);
  const auto classes = query.class_count;
  if (classes == 0) {
    return;
  }

  std::fill_n(workspace.state_loads, workspace.state_count * classes, T{0});

  for (std::size_t node = 0; node < query.destination_count; ++node) {
    const auto terminal = results.terminal_states[node];
    if (terminal == invalid_state || terminal == results.metadata->root) {
      continue;
    }

    T *row = workspace.state_loads + terminal * classes;
    for (std::size_t cls = 0; cls < classes; ++cls) {
      row[cls] += query.demand[node * classes + cls];
    }
  }
  cascade_loads(results, workspace, output);
}

template <typename T>
T sum_weighted_turn_costs(const SearchResults &results,
                          const LoadingQuery<T> &query) noexcept {
  T total = 0;
  for (std::size_t node = 0; node < query.destination_count; ++node) {
    const auto terminal = results.terminal_states[node];
    if (terminal == invalid_state || terminal == results.metadata->root) {
      continue;
    }
    for (std::size_t cls = 0; cls < query.class_count; ++cls) {
      total += query.demand[node * query.class_count + cls] *
               results.turn_costs[terminal];
    }
  }
  return total;
}

// Reduce a set of LoadingOutputs into a single LoadingOutputs.
template <typename T>
void reduce_loading_outputs(const LoadingOutputs<T> *workers,
                            std::size_t worker_count,
                            const LoadingOutputs<T> &output) noexcept {
  output.reset();
  const auto size = output.link_count * output.class_count;

  for (std::size_t worker = 0; worker < worker_count; ++worker) {
    for (std::size_t i = 0; i < size; ++i) {
      output.link_loads[i] += workers[worker].link_loads[i];
    }
  }
}

} // namespace aequilibrae::paths::cpp::mvp
