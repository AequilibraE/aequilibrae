#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "context.hpp"
#include "pq_heap_base.hpp"
#include "queries.hpp"
#include "search_results.hpp"

namespace aequilibrae::paths::cpp::mvp {

inline void reset_search(const SearchQuery &query, std::size_t root,
                         const MutableSearchResults &results) noexcept {
  auto &metadata = *results.metadata;
  metadata = SearchMetadata{};
  metadata.origin = query.origin;
  metadata.root = root;
  metadata.target_count = query.target_count;

  const auto infinity = std::numeric_limits<double>::infinity();
  std::fill_n(results.predecessors, results.state_count, invalid_state);
  std::fill_n(results.connectors, results.state_count, invalid_state);
  std::fill_n(results.settlement_order, results.state_count, invalid_state);
  std::fill_n(results.terminal_states, results.node_count, invalid_state);
  std::fill_n(results.distances, results.state_count, infinity);
  std::fill_n(results.turn_costs, results.state_count, infinity);
}

inline bool reached_last_target(const SearchQuery &query, std::size_t node,
                                SearchMetadata &metadata) noexcept {
  if (query.target_mask == nullptr || !query.target_mask[node]) {
    return false;
  }
  ++metadata.reached_target_count;
  return metadata.reached_target_count == query.target_count;
}

// Contexts and queries contain only inputs. The result view writes through to
// its Cython owner's arrays and metadata without changing the view itself.
template <class Queue>
void dijkstra(const NodeBasedContext &context, const SearchQuery &query,
              const MutableSearchResults &results) noexcept {
  static_assert(std::is_base_of_v<PriorityQueueBase<Queue>, Queue>);
  reset_search(query, query.origin, results);
  auto &metadata = *results.metadata;
  bool stopped_at_targets = false;

  Queue queue;
  queue.init_heap(context.node_count);
  queue.insert(query.origin, 0.0);

  while (!queue.is_empty()) {
    const auto state = queue.extract_min();
    const double cost = queue.element_key(state);
    results.settlement_order[metadata.settled_count++] = state;
    results.distances[state] = cost;
    results.turn_costs[state] = 0.0;
    results.terminal_states[state] = state;

    if (reached_last_target(query, state, metadata)) {
      stopped_at_targets = true;
      break;
    }
    // A centroid may be reached, but only the origin may supply outgoing links.
    if (state < context.blocked_centroid_count && state != query.origin) {
      continue;
    }

    for (auto link = context.fs[state]; link < context.fs[state + 1]; ++link) {
      const auto next = context.heads[link];
      const auto next_state = queue.effective_state(next);
      if (next_state == SCANNED) {
        continue;
      }
      const double next_cost = cost + context.costs[link];
      if (!std::isfinite(next_cost)) {
        continue;
      }
      if (next_state == NOT_IN_HEAP) {
        queue.insert(next, next_cost);
      } else if (next_cost < queue.element_key(next)) {
        queue.decrease_key(next, next_cost);
      } else {
        continue;
      }
      results.predecessors[next] = state;
      results.connectors[next] = link;
    }
  }

  // The last target's outgoing links were not explored, even if the heap is
  // empty. Only completing the loop proves the reachable state space exhausted.
  metadata.exhausted = !stopped_at_targets;
  for (std::size_t state = 0; state < results.state_count; ++state) {
    if (queue.effective_state(state) == IN_HEAP) {
      results.predecessors[state] = invalid_state;
      results.connectors[state] = invalid_state;
    }
  }
}

template <class Queue>
void dijkstra(const TurnBasedContext &context, const SearchQuery &query,
              const MutableSearchResults &results) noexcept {
  static_assert(std::is_base_of_v<PriorityQueueBase<Queue>, Queue>);
  const auto &graph = context.graph;
  // Link states preserve incoming-link history. One virtual root lets first
  // links leave the origin without paying a turn cost, including edgeless graphs.
  const auto root = graph.link_count;
  reset_search(query, root, results);
  auto &metadata = *results.metadata;
  bool stopped_at_targets = false;
  results.turn_costs[root] = 0.0;

  Queue queue;
  queue.init_heap(results.state_count);
  queue.insert(root, 0.0);

  while (!queue.is_empty()) {
    const auto state = queue.extract_min();
    const double cost = queue.element_key(state);
    const auto node = state == root ? query.origin : graph.heads[state];
    results.settlement_order[metadata.settled_count++] = state;
    results.distances[state] = cost;

    if (results.terminal_states[node] == invalid_state) {
      results.terminal_states[node] = state;
      if (reached_last_target(query, node, metadata)) {
        stopped_at_targets = true;
        break;
      }
    }
    if (node < graph.blocked_centroid_count && node != query.origin) {
      continue;
    }

    auto turn = state == root ? 0 : context.turn_fs[state];
    const auto turn_end = state == root ? 0 : context.turn_fs[state + 1];
    for (auto next = graph.fs[node]; next < graph.fs[node + 1]; ++next) {
      if (queue.effective_state(next) == SCANNED) {
        continue;
      }
      // Sorted turn rows can be merged with outgoing links without materializing
      // an expanded graph or looking up a turn in a Python mapping.
      while (turn < turn_end && context.turn_to_links[turn] < next) {
        ++turn;
      }
      const bool explicit_turn =
          turn < turn_end && context.turn_to_links[turn] == next;
      const double penalty = explicit_turn ? context.turn_penalties[turn] : 0.0;
      if (state != root && !explicit_turn && !context.allow_uturns &&
          graph.heads[next] == context.tails[state]) {
        continue;
      }
      const double next_cost = cost + graph.costs[next] + penalty;
      if (!std::isfinite(next_cost)) {
        continue;
      }
      if (queue.effective_state(next) == NOT_IN_HEAP) {
        queue.insert(next, next_cost);
      } else if (next_cost < queue.element_key(next)) {
        queue.decrease_key(next, next_cost);
      } else {
        continue;
      }
      results.predecessors[next] = state;
      results.connectors[next] = next;
      results.turn_costs[next] = results.turn_costs[state] + penalty;
    }
  }

  metadata.exhausted = !stopped_at_targets;
  // Early exit must not expose tentative paths or their turn labels.
  for (std::size_t state = 0; state < results.state_count; ++state) {
    if (queue.effective_state(state) == IN_HEAP) {
      results.predecessors[state] = invalid_state;
      results.connectors[state] = invalid_state;
      results.turn_costs[state] = std::numeric_limits<double>::infinity();
    }
  }
}

} // namespace aequilibrae::paths::cpp::mvp
