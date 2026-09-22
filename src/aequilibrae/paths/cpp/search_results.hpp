#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

namespace aequilibrae::paths::cpp::mvp {

inline constexpr std::size_t invalid_state =
    std::numeric_limits<std::size_t>::max();

// This record belongs to the Cython results owner. Views point to it so a
// search updates the owner's counts as well as its arrays.
struct SearchMetadata {
  std::size_t origin = invalid_state;
  std::size_t root = invalid_state;
  std::size_t settled_count = 0;
  std::size_t target_count = 0;
  std::size_t reached_target_count = 0;
  bool exhausted = false;
};

// Read-only view of the chosen state tree. A state is a node in node routing
// and an incoming link in turn routing. terminal_states chooses the arrival
// at which a path ENDS; predecessors preserves the arrivals used THROUGH nodes.
// This distinction lets consumers follow turn paths without knowing the mode.
//
// Only settlement_order[:metadata->settled_count] is valid. Parents precede
// children, even for zero-cost paths. The root has no parent or connector.
// Unsettled states have invalid indices and infinite labels; an invalid
// terminal means no finalized path is available, not necessarily unreachable.
// Distances include turn costs. Consumers must not add the two labels together.
struct SearchResults {
  std::size_t node_count = 0;
  std::size_t state_count = 0;
  std::size_t link_count = 0;
  const std::size_t *predecessors = nullptr;
  const std::size_t *connectors = nullptr;
  const std::size_t *settlement_order = nullptr;
  const std::size_t *terminal_states = nullptr;
  const double *distances = nullptr;
  const double *turn_costs = nullptr;
  const SearchMetadata *metadata = nullptr;
};

// Only routing needs writable search buffers. Neither view owns storage;
// the Cython owner must remain alive throughout every call using a view.
struct MutableSearchResults {
  std::size_t node_count = 0;
  std::size_t state_count = 0;
  std::size_t link_count = 0;
  std::size_t *predecessors = nullptr;
  std::size_t *connectors = nullptr;
  std::size_t *settlement_order = nullptr;
  std::size_t *terminal_states = nullptr;
  double *distances = nullptr;
  double *turn_costs = nullptr;
  SearchMetadata *metadata = nullptr;

  void reset() const noexcept {
    *metadata = SearchMetadata{};
    const auto infinity = std::numeric_limits<double>::infinity();
    std::fill_n(predecessors, state_count, invalid_state);
    std::fill_n(connectors, state_count, invalid_state);
    std::fill_n(settlement_order, state_count, invalid_state);
    std::fill_n(terminal_states, node_count, invalid_state);
    std::fill_n(distances, state_count, infinity);
    std::fill_n(turn_costs, state_count, infinity);
  }

  SearchResults read_view() const noexcept {
    return {node_count, state_count,      link_count,      predecessors,
            connectors, settlement_order, terminal_states, distances,
            turn_costs, metadata};
  }
};

} // namespace aequilibrae::paths::cpp::mvp
