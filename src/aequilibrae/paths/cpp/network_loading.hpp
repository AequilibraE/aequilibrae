#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "outputs.hpp"
#include "search_results.hpp"
#include "workspaces.hpp"

namespace aequilibrae::paths::cpp::routing {

// Both ordinary and selected loading seed demand at terminal states. Parents
// receive demand from all their children before loading their own connector.
// Preserves turn arrival history.
template <typename T, typename Cost>
void cascade_loads(const SearchResults<Cost> &results,
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
template <typename T, typename DemandRow, typename Cost>
void network_loading(const SearchResults<Cost> &results, const DemandRow &query,
                     const LoadingWorkspace<T> &workspace,
                     const LoadingOutputs<T> &output) noexcept {
  static_assert(std::is_floating_point_v<T>);
  const auto classes = query.field_count;
  if (classes == 0) {
    return;
  }

  std::fill_n(workspace.state_loads, workspace.state_count * classes, T{0});

  for (std::size_t entry = 0; entry < query.entry_count(); ++entry) {
    const auto node = query.destination(entry);
    const auto terminal = results.terminal_states[node];
    if (terminal == invalid_state || terminal == results.metadata->root) {
      continue;
    }

    const T *demand = query.values(entry);
    T *row = workspace.state_loads + terminal * classes;
    for (std::size_t cls = 0; cls < classes; ++cls) {
      row[cls] += demand[cls];
    }
  }
  cascade_loads(results, workspace, output);
}

template <typename T, typename DemandRow, typename Cost>
T sum_weighted_turn_costs(const SearchResults<Cost> &results,
                          const DemandRow &query) noexcept {
  T total = 0;
  for (std::size_t entry = 0; entry < query.entry_count(); ++entry) {
    const auto node = query.destination(entry);
    const auto terminal = results.terminal_states[node];
    if (terminal == invalid_state || terminal == results.metadata->root) {
      continue;
    }
    const T *demand = query.values(entry);
    for (std::size_t cls = 0; cls < query.field_count; ++cls) {
      total += demand[cls] * results.turn_costs[terminal];
    }
  }
  return total;
}

// Assignment searches every destination with demand. Missing terminals here
// therefore mean unreachable demand, not targets omitted by an early stop.
template <typename T, typename DemandRow, typename Cost>
T sum_unassigned_demand(const SearchResults<Cost> &results,
                        const DemandRow &query) noexcept {
  T total = 0;
  for (std::size_t entry = 0; entry < query.entry_count(); ++entry) {
    const auto node = query.destination(entry);
    if (node == results.metadata->origin ||
        results.terminal_states[node] != invalid_state) {
      continue;
    }
    const T *demand = query.values(entry);
    for (std::size_t cls = 0; cls < query.field_count; ++cls) {
      total += demand[cls];
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

} // namespace aequilibrae::paths::cpp::routing
