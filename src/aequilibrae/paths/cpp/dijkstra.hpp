#pragma once

#include <cmath>
#include <limits>
#include <type_traits>

#include "context.hpp"
#include "pq_heap_base.hpp"
#include "queries.hpp"
#include "search_results.hpp"
#include "workspaces.hpp"

namespace aequilibrae::paths::cpp::routing {

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
void dijkstra_with_queue(const NodeBasedContext &context,
                         const SearchQuery &query,
                         const MutableSearchResults &results,
                         Queue &queue) noexcept {
  static_assert(std::is_base_of_v<PriorityQueueBase<Queue>, Queue>);
  results.reset();
  auto &metadata = *results.metadata;
  metadata.origin = query.origin;
  metadata.root = query.origin;
  metadata.target_count = query.target_count;
  bool stopped_at_targets = false;

  queue.reset_heap();
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

inline void record_arrival(const MutableSearchResults &results,
                           std::size_t label, std::size_t predecessor,
                           std::size_t link, double turn_cost) noexcept {
  results.predecessors[label] = predecessor;
  results.connectors[label] = link;
  results.turn_costs[label] = turn_cost;
}

template <class Queue>
void offer_arrival(const MutableSearchResults &results, Queue &queue,
                   std::size_t label, double cost, std::size_t predecessor,
                   std::size_t link, double turn_cost) noexcept {
  const auto state = queue.effective_state(label);
  if (state == NOT_IN_HEAP) {
    queue.insert(label, cost);
  } else if (state == IN_HEAP && cost < queue.element_key(label)) {
    queue.decrease_key(label, cost);
  } else {
    return;
  }
  record_arrival(results, label, predecessor, link, turn_cost);
}

// The first label keeps a node's best arrival; the second keeps the best one
// from any other previous node, for the move back the first cannot take.
template <class Queue>
void offer_paired_arrival(const TurnBasedContext &context,
                          const MutableSearchResults &results, Queue &queue,
                          std::size_t first, std::size_t second, double cost,
                          std::size_t predecessor, std::size_t link,
                          double turn_cost) noexcept {
  const auto first_state = queue.effective_state(first);
  if (first_state == NOT_IN_HEAP) {
    queue.insert(first, cost);
    record_arrival(results, first, predecessor, link, turn_cost);
    return;
  }
  const bool new_previous = context.last_nodes[link] !=
                            context.last_nodes[results.connectors[first]];
  if (first_state == IN_HEAP && cost < queue.element_key(first)) {
    if (new_previous) {
      // The displaced arrival is the best from another previous node, so it
      // replaces the second label, which cannot be settled yet.
      const double displaced = queue.element_key(first);
      if (queue.effective_state(second) == NOT_IN_HEAP) {
        queue.insert(second, displaced);
      } else if (displaced < queue.element_key(second)) {
        queue.decrease_key(second, displaced);
      }
      record_arrival(results, second, results.predecessors[first],
                     results.connectors[first], results.turn_costs[first]);
    }
    queue.decrease_key(first, cost);
    record_arrival(results, first, predecessor, link, turn_cost);
  } else if (new_previous) {
    offer_arrival(results, queue, second, cost, predecessor, link, turn_cost);
  }
}

template <class Queue>
void dijkstra_with_queue(const TurnBasedContext &context,
                         const SearchQuery &query,
                         const MutableSearchResults &results,
                         Queue &queue) noexcept {
  static_assert(std::is_base_of_v<PriorityQueueBase<Queue>, Queue>);
  const auto &graph = context.graph;
  // One virtual root lets first links leave the origin without paying a turn
  // cost, including edgeless graphs.
  const auto root = graph.link_count;
  results.reset();
  auto &metadata = *results.metadata;
  metadata.origin = query.origin;
  metadata.root = root;
  metadata.target_count = query.target_count;
  bool stopped_at_targets = false;
  results.turn_costs[root] = 0.0;

  queue.reset_heap();
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

    // A label shares its turn rows and label slots with every link it stands
    // for, so it indexes them directly. A second label only adds the move
    // back to its first label's previous node.
    const bool second_label =
        state != root && context.second_labels[state] == state;
    const auto reversal =
        second_label
            ? context.last_nodes[results.connectors[context.state_labels[state]]]
            : invalid_state;
    const auto previous = state == root || context.allow_uturns
                              ? invalid_state
                              : context.last_nodes[results.connectors[state]];
    const double turn_cost = results.turn_costs[state];
    auto turn = state == root ? 0 : context.turn_fs[state];
    const auto turn_end = state == root ? 0 : context.turn_fs[state + 1];
    for (auto next = graph.fs[node]; next < graph.fs[node + 1]; ++next) {
      if (second_label && context.first_nodes[next] != reversal) {
        continue;
      }
      const auto next_label = context.state_labels[next];
      const auto next_second = context.second_labels[next];
      if (queue.effective_state(next_label) == SCANNED &&
          (next_second == invalid_state ||
           queue.effective_state(next_second) == SCANNED)) {
        continue;
      }
      // Sorted turn rows can be merged with outgoing links without
      // materializing an expanded graph or looking up a turn in a Python
      // mapping.
      while (turn < turn_end && context.turn_to_links[turn] < next) {
        ++turn;
      }
      const bool explicit_turn =
          turn < turn_end && context.turn_to_links[turn] == next;
      const double penalty = explicit_turn ? context.turn_penalties[turn] : 0.0;
      if (!explicit_turn && context.first_nodes[next] == previous) {
        continue;
      }
      const double next_cost = cost + graph.costs[next] + penalty;
      if (!std::isfinite(next_cost)) {
        continue;
      }
      if (next_second == invalid_state) {
        offer_arrival(results, queue, next_label, next_cost, state, next,
                      turn_cost + penalty);
      } else {
        offer_paired_arrival(context, results, queue, next_label, next_second,
                             next_cost, state, next, turn_cost + penalty);
      }
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

template <class Context>
void dijkstra(const Context &context, const SearchQuery &query,
              const MutableSearchResults &results,
              const SearchWorkspace &workspace) noexcept {
  workspace.heap->visit([&](auto &queue) {
    dijkstra_with_queue(context, query, results, queue);
  });
}

} // namespace aequilibrae::paths::cpp::routing
