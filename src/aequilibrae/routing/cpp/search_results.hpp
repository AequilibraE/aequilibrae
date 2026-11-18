#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

namespace aequilibrae::paths::cpp::routing {

inline constexpr std::size_t invalid_state =
    std::numeric_limits<std::size_t>::max();

// This record belongs to the Cython results owner. Views point to it so a
// search updates the owner's counts as well as its arrays.
struct SearchMetadata {
  std::size_t origin = invalid_state;
  std::size_t root = invalid_state;
  std::size_t settled_count = 0;
  std::size_t terminal_count = 0;
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
template <typename T> struct SearchResults {
  std::size_t node_count = 0;
  std::size_t state_count = 0;
  std::size_t link_count = 0;
  const std::size_t *predecessors = nullptr;
  const std::size_t *connectors = nullptr;
  const std::size_t *settlement_order = nullptr;
  const std::size_t *terminal_states = nullptr;
  const T *distances = nullptr;
  const T *turn_costs = nullptr;
  const SearchMetadata *metadata = nullptr;
};

// Only routing needs writable search buffers. Neither view owns storage;
// the Cython owner must remain alive throughout every call using a view.
template <typename T> struct MutableSearchResults {
  std::size_t node_count = 0;
  std::size_t state_count = 0;
  std::size_t link_count = 0;
  std::size_t *predecessors = nullptr;
  std::size_t *connectors = nullptr;
  std::size_t *settlement_order = nullptr;
  std::size_t *terminal_states = nullptr;
  std::size_t *terminal_nodes = nullptr;
  T *distances = nullptr;
  T *turn_costs = nullptr;
  SearchMetadata *metadata = nullptr;

  void reset() const noexcept {
    const auto infinity = std::numeric_limits<T>::infinity();
    const auto settled_count = metadata->settled_count;
    // Pending states are cleared before a search returns. Only its settled
    // states and finalized terminal nodes can therefore contain old labels.
    // Dense trees favour sequential writes over scattered state accesses.
    if (settled_count >= state_count - state_count / 2) {
      std::fill_n(predecessors, state_count, invalid_state);
      std::fill_n(connectors, state_count, invalid_state);
      std::fill_n(distances, state_count, infinity);
      std::fill_n(turn_costs, state_count, infinity);
    } else {
      for (std::size_t i = 0; i < settled_count; ++i) {
        const auto state = settlement_order[i];
        predecessors[state] = invalid_state;
        connectors[state] = invalid_state;
        distances[state] = infinity;
        turn_costs[state] = infinity;
      }
    }
    std::fill_n(settlement_order, settled_count, invalid_state);

    const auto terminal_count = metadata->terminal_count;
    if (terminal_count >= node_count - node_count / 2) {
      std::fill_n(terminal_states, node_count, invalid_state);
    } else {
      for (std::size_t i = 0; i < terminal_count; ++i) {
        terminal_states[terminal_nodes[i]] = invalid_state;
      }
    }
    *metadata = SearchMetadata{};
  }

  // Call only when the node receives its first finalized terminal.
  void set_terminal(std::size_t node, std::size_t state) const noexcept {
    terminal_nodes[metadata->terminal_count++] = node;
    terminal_states[node] = state;
  }

  SearchResults<T> read_view() const noexcept {
    return {node_count, state_count,      link_count,      predecessors,
            connectors, settlement_order, terminal_states, distances,
            turn_costs, metadata};
  }
};

} // namespace aequilibrae::paths::cpp::routing
