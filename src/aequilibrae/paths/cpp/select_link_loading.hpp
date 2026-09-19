#pragma once

#include <algorithm>
#include <cstddef>

#include "context.hpp"
#include "network_loading.hpp"

namespace aequilibrae::paths::cpp::mvp {

// A path matches once it has used any state in this set.
inline void mark_selected_paths(const SearchResults &results,
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

inline bool selected_terminal(const SearchResults &results,
                              std::size_t terminal,
                              const SelectLinkWorkspace &selection) noexcept {
  return terminal != invalid_state && terminal != results.metadata->root &&
         selection.selected_paths[terminal];
}

template <typename T>
void write_selected_od(const SearchResults &results,
                       const LoadingQuery<T> &query,
                       const SelectLinkWorkspace &selection, T *od) noexcept {
  if (query.class_count == 0) {
    return;
  }

  for (std::size_t node = 0; node < query.destination_count; ++node) {
    T *row = od + node * query.class_count;

    if (selected_terminal(results, results.terminal_states[node], selection)) {
      std::copy_n(query.demand + node * query.class_count, query.class_count,
                  row);
    } else {
      std::fill_n(row, query.class_count, T{0});
    }
  }
}

// Seed each matched OD once, even if its path crosses several set members.
// The cascade loading then loads the whole path.
template <typename T>
void load_selected_paths(const SearchResults &results,
                         const LoadingQuery<T> &query,
                         const SelectLinkWorkspace &selection,
                         const LoadingWorkspace<T> &loading,
                         const LoadingOutputs<T> &output) noexcept {
  const auto classes = query.class_count;
  if (classes == 0) {
    return;
  }

  std::fill_n(loading.state_loads, loading.state_count * classes, T{0});

  for (std::size_t node = 0; node < query.destination_count; ++node) {
    const auto terminal = results.terminal_states[node];
    if (!selected_terminal(results, terminal, selection)) {
      continue;
    }

    for (std::size_t cls = 0; cls < classes; ++cls) {
      loading.state_loads[terminal * classes + cls] +=
          query.demand[node * classes + cls];
    }
  }

  cascade_loads(results, loading, output);
}

template <typename T>
void select_link_loading(const SearchResults &results,
                         const LoadingQuery<T> &query,
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
      write_selected_od(results, query, selection, od.selection_data(set));
    }

    if (loads.set_count) {
      load_selected_paths(results, query, selection, loading,
                          loads.selection(set));
    }
  }
}

// Only link accumulators are reduced. OD rows don't need reduction
template <typename T>
void reduce_select_link_loading_outputs(
    const SelectLinkLoadingOutputsView<T> *workers, std::size_t worker_count,
    const SelectLinkLoadingOutputsView<T> &output) noexcept {
  output.reset();

  const auto size = output.set_count * output.link_count * output.class_count;
  for (std::size_t worker = 0; worker < worker_count; ++worker) {
    for (std::size_t i = 0; i < size; ++i) {
      output.data[i] += workers[worker].data[i];
    }
  }
}

} // namespace aequilibrae::paths::cpp::mvp
