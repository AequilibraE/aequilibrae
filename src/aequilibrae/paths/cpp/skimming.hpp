#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "context.hpp"
#include "outputs.hpp"
#include "search_results.hpp"
#include "workspaces.hpp"

namespace aequilibrae::paths::cpp::routing {

// Sum all supplied link fields in one walk of the state tree.
template <typename T>
void sum_skim_fields(const SearchResults &results,
                     const SkimmingContext<T> &context,
                     const SkimmingWorkspace<T> &workspace) noexcept {
  const auto width = context.additive_field_count;
  const T infinity = std::numeric_limits<T>::infinity();

  std::fill_n(workspace.state_skims, workspace.state_count * width, infinity);

  for (std::size_t i = 0; i < results.metadata->settled_count; ++i) {
    const auto state = results.settlement_order[i];
    T *row = workspace.state_skims + state * width;

    if (state == results.metadata->root) {
      // The root has used no links and has no parent to read from.
      std::fill_n(row, width, T{0});
      continue;
    }

    const T *parent =
        workspace.state_skims + results.predecessors[state] * width;
    const auto link = results.connectors[state];

    // FIXME: Not sure this is the greatest way to do this, pretty
    // non-contiguous accesses here, but the link_fields are borrowed points
    // from the graph
    for (std::size_t field = 0; field < width; ++field) {
      row[field] = parent[field] + context.link_fields[field][link];
    }
  }
}

// Copy each node's link sums from its terminal state.
template <typename T>
void skim_fields(const SearchResults &results,
                 const SkimmingWorkspace<T> &workspace,
                 const SkimmingOriginView<T> &output) noexcept {
  const T infinity = std::numeric_limits<T>::infinity();

  for (std::size_t node = 0; node < output.destination_count; ++node) {
    T *row = output.destination_data(node);
    const auto terminal = results.terminal_states[node];

    // A missing terminal may be unreachable or not yet searched.
    if (terminal == invalid_state) {
      std::fill_n(row, workspace.field_count, infinity);
      continue;
    }

    const auto state_offset = terminal * workspace.field_count;
    std::copy_n(workspace.state_skims + state_offset, workspace.field_count, row);
  }
}

// Copy labels rather than summing the objective again. This needs no scratch
// and preserves the search's exact double values, including its turn costs.
template <typename T>
void project_skim_labels(const SearchResults &results, const double *labels,
                         const SkimmingOriginView<T> &output,
                         std::size_t field) noexcept {
  const T infinity = std::numeric_limits<T>::infinity();

  for (std::size_t node = 0; node < output.destination_count; ++node) {
    T *row = output.destination_data(node);
    const auto terminal = results.terminal_states[node];

    if (terminal == invalid_state) {
      row[field] = infinity;
      continue;
    }

    row[field] = static_cast<T>(labels[terminal]);
  }
}

template <typename T>
void skim_costs(const SearchResults &results,
                const SkimmingOriginView<T> &output,
                std::size_t field) noexcept {
  project_skim_labels(results, results.distances, output, field);
}

template <typename T>
void skim_turn_costs(const SearchResults &results,
                     const SkimmingOriginView<T> &output,
                     std::size_t field) noexcept {
  project_skim_labels(results, results.turn_costs, output, field);
}

// Overwrite one output row using finalised paths.
template <typename T>
void skimming(const SearchResults &results, const SkimmingContext<T> &context,
              const SkimmingWorkspace<T> &workspace,
              const SkimmingOriginView<T> &output) noexcept {
  static_assert(std::is_floating_point_v<T>);

  if (context.needs_state_sums()) {
    sum_skim_fields(results, context, workspace);
    // Copy the first additive fields for each destination.
    skim_fields(results, workspace, output);
  }

  if (context.has_cost_field()) {
    skim_costs(results, output, context.cost_field_index);
  }

  if (context.has_turn_cost_field()) {
    skim_turn_costs(results, output, context.turn_cost_field_index);
  }
}

} // namespace aequilibrae::paths::cpp::routing
