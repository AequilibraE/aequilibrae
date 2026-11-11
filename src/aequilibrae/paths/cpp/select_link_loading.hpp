#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "context.hpp"
#include "network_loading.hpp"

namespace aequilibrae::paths::cpp::routing {

// A path matches once it has used any state in this set.
template <typename Cost>
inline void mark_selected_paths(const SearchResults<Cost> &results,
                                const bool *selected_links,
                                const SelectLinkWorkspace &workspace) noexcept {
  std::fill_n(workspace.selected_paths, workspace.state_count, false);
  for (std::size_t i = 0; i < results.metadata->settled_count; ++i) {
    const auto state = results.settlement_order[i];
    if (state != results.metadata->root) {
      workspace.selected_paths[state] =
          workspace.selected_paths[results.predecessors[state]] ||
          selected_links[results.connectors[state]];
    }
  }
}

template <typename Cost>
inline bool selected_terminal(const SearchResults<Cost> &results,
                              std::size_t terminal,
                              const SelectLinkWorkspace &selection) noexcept {
  return terminal != invalid_state && terminal != results.metadata->root &&
         selection.selected_paths[terminal];
}

template <typename T, typename DemandRow, typename Cost>
void write_selected_od(const SearchResults<Cost> &results,
                       const DemandRow &query,
                       const SelectLinkWorkspace &selection,
                       const SelectLinkODOriginView<T> &od,
                       std::size_t set) noexcept {
  if (query.field_count == 0) {
    return;
  }

  // Append only matched, nonzero demand
  for (std::size_t entry = 0; entry < query.entry_count(); ++entry) {
    const auto node = query.destination(entry);
    if (selected_terminal(results, results.terminal_states[node], selection)) {
      const auto *values = query.values(entry);
      if (std::any_of(values, values + query.field_count,
                      [](T value) { return value != T{0}; }))
        od.append(set, node, values);
    }
  }
}

// Seed each matched OD once, even if its path crosses several set members.
// The cascade loading then loads the whole path.
template <typename T, typename DemandRow, typename Cost>
void load_selected_paths(const SearchResults<Cost> &results,
                         const DemandRow &query,
                         const SelectLinkWorkspace &selection,
                         const LoadingWorkspace<T> &loading,
                         const LoadingOutputs<T> &output) noexcept {
  const auto classes = query.field_count;
  if (classes == 0) {
    return;
  }

  std::fill_n(loading.state_loads, loading.state_count * classes, T{0});

  for (std::size_t entry = 0; entry < query.entry_count(); ++entry) {
    const auto node = query.destination(entry);
    const auto terminal = results.terminal_states[node];
    if (!selected_terminal(results, terminal, selection)) {
      continue;
    }

    const T *demand = query.values(entry);
    for (std::size_t cls = 0; cls < classes; ++cls) {
      loading.state_loads[terminal * classes + cls] += demand[cls];
    }
  }

  cascade_loads(results, loading, output);
}

template <typename T, typename DemandRow, typename Cost>
void select_link_loading(const SearchResults<Cost> &results,
                         const DemandRow &query,
                         const SelectLinkContext &context,
                         const SelectLinkWorkspace &selection,
                         const LoadingWorkspace<T> &loading,
                         const SelectLinkLoadingOutputsView<T> &loads,
                         const SelectLinkODOriginView<T> &od) noexcept {
  if (loads.set_count == 0 && od.set_count == 0) {
    return;
  }
  for (std::size_t set = 0; set < context.set_count; ++set) {
    mark_selected_paths(results, context.selection(set), selection);

    if (od.set_count) {
      write_selected_od(results, query, selection, od, set);
    }

    if (loads.set_count) {
      load_selected_paths(results, query, selection, loading,
                          loads.selection(set));
    }
  }
}

// Link accumulators are summed and worker-local COO records are concatenated.
template <typename T>
void reduce_select_link_loading_outputs(
    const SelectLinkLoadingOutputsView<T> *workers, std::size_t worker_count,
    const SelectLinkLoadingOutputsView<T> &output,
    const SelectLinkODOutputsView<T> *od_workers, std::size_t od_worker_count,
    const SelectLinkODOutputsView<T> &od_output) {
  output.reset();
  od_output.reset();

  // Count once per set so consolidation reserves its final COO capacity before
  // copying any worker buffers. This avoids repeated vector growth in extend().
  for (std::size_t set = 0; set < od_output.set_count; ++set) {
    std::size_t total_entries = 0;
    for (std::size_t worker = 0; worker < od_worker_count; ++worker) {
      const auto worker_entries =
          od_workers[worker].builders[set].entry_count();
      if (worker_entries >
          std::numeric_limits<std::size_t>::max() - total_entries)
        throw std::length_error("select-link COO entry count overflow");

      total_entries += worker_entries;
    }
    od_output.builders[set].reserve_entries(total_entries);
  }

  for (std::size_t worker = 0; worker < od_worker_count; ++worker) {
    for (std::size_t set = 0; set < od_output.set_count; ++set) {
      od_output.builders[set].extend(od_workers[worker].builders[set]);
    }
  }

  const auto size = output.set_count * output.link_count * output.class_count;
  for (std::size_t worker = 0; worker < worker_count; ++worker) {
    for (std::size_t i = 0; i < size; ++i) {
      output.data[i] += workers[worker].data[i];
    }
  }
}

} // namespace aequilibrae::paths::cpp::routing
