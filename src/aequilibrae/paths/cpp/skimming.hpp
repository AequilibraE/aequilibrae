#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "routing_workspace.hpp"
#include "search_results.hpp"

namespace aequilibrae::paths::cpp::mvp {

// Allocation-free, single-origin skimming of finalized SearchResults.
//
// Preconditions (validated by the Python wrapper):
// - Each of the field_count field pointers addresses link_count contiguous T
//   values, in exactly the context.costs local directed-link order.
// - Workspace dimensions are [search state_count, field_count].
// - Output has destination_count * field_count contiguous T entries, row-major.
//   Rows are nodes [0, destination_count), with destination_count <=
//   node_count. This count is independent of the search's destination mask.
// - Scratch does not overlap inputs, output, or search buffers. All buffers
//   remain alive and externally serialized for the duration of the call.
//
// Additional fields are additive link attributes, NOT routing weights: negative
// values are allowed and IEEE NaNs/infinities propagate by ordinary addition.
// No turn penalties are added, even if an input pointer is context.costs.
// Use skim_costs instead to obtain the routing objective including penalties.
//
// Accumulate over STATES, not physical-node terminals: the best path THROUGH
// a node need not use its cheapest arrival. Parent-before-child settlement
// order makes this O((state_count + destination_count) * field_count), with no
// path retracing. Root skims are zero; unfinalized state/node skims are
// infinity. A pre-search result produces all infinity. Zero fields permits null
// pointers. With zero destinations, output may be null; state sums are still
// computed.
template <typename T>
void skim_fields(const SearchResults &results, std::size_t destination_count,
                 const T *const *fields, std::size_t field_count,
                 RoutingWorkspace<T> &workspace, T *output) noexcept {
  static_assert(std::is_floating_point_v<T>,
                "Skims require floating-point infinity");
  if (field_count == 0) {
    return;
  }
  const T infinity = std::numeric_limits<T>::infinity();
  // Reset all states so unreached states do not keep values from the last skim.
  std::fill_n(workspace.state_skims, workspace.state_count * field_count,
              infinity);

  // Include all settled states: paths to centroids can pass through other
  // nodes. Parents come first, so their sums are ready when we process their
  // children.
  for (std::size_t i = 0; i < results.settled_count; ++i) {
    const auto state = results.reached_first[i];
    T *row = workspace.state_skims + state * field_count;
    if (state == results.root) {
      // No links have been used at the root, and it has no parent or connector.
      std::fill_n(row, field_count, T{0});
    } else {
      // Use the parent state. The cheapest path to its node may use a different
      // state.
      const T *parent =
          workspace.state_skims + results.predecessors[state] * field_count;
      const auto link = results.connectors[state];
      for (std::size_t field = 0; field < field_count; ++field) {
        row[field] = parent[field] + fields[field][link];
      }
    }
  }

  // Only write the requested first nodes. Each row uses its chosen arrival
  // state.
  for (std::size_t node = 0; node < destination_count; ++node) {
    const auto terminal = results.terminal_states[node];
    T *row = output + node * field_count;
    if (terminal == std::numeric_limits<std::size_t>::max()) {
      // No settled path to this node. Replace any old output values
      // with infinity.
      std::fill_n(row, field_count, infinity);
    } else {
      std::copy_n(workspace.state_skims + terminal * field_count, field_count,
                  row);
    }
  }
}

// The routing objective and cumulative penalties are already computed by the
// search. Copying those labels takes O(destination_count) time, no workspace,
// and no summation/round-off changes for double outputs. Output is packed [Z,
// 1], where 0 <= Z <= node_count. A zero count permits a null output pointer.
template <typename T>
void project_costs(const SearchResults &results, std::size_t destination_count,
                   const double *labels, T *output) noexcept {
  static_assert(std::is_floating_point_v<T>,
                "Skims require floating-point infinity");
  for (std::size_t node = 0; node < destination_count; ++node) {
    const auto terminal = results.terminal_states[node];
    // A node may be unreached because the search stopped early. Check before
    // indexing.
    output[node] = terminal == std::numeric_limits<std::size_t>::max()
                       ? std::numeric_limits<T>::infinity()
                       : static_cast<T>(labels[terminal]);
  }
}

template <typename T>
void skim_costs(const SearchResults &results, std::size_t destination_count,
                T *output) noexcept {
  // Distances include turn penalties. Summing link costs alone would leave them
  // out.
  project_costs(results, destination_count, results.distances, output);
}

template <typename T>
void skim_turn_costs(const SearchResults &results,
                     std::size_t destination_count, T *output) noexcept {
  // The search already summed the penalties, so copy them directly.
  project_costs(results, destination_count, results.turn_costs, output);
}

} // namespace aequilibrae::paths::cpp::mvp
