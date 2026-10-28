#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "context.hpp"
#include "heuristics.hpp"
#include "pq_heap_base.hpp"
#include "search_query.hpp"
#include "search_results.hpp"
#include "workspaces.hpp"

namespace aequilibrae::paths::cpp::routing {

template <typename T>
inline const NodeBasedContext<T> &
a_star_graph(const NodeBasedContext<T> &context) noexcept {
  return context;
}

template <typename T>
inline const NodeBasedContext<T> &
a_star_graph(const TurnBasedContext<T> &context) noexcept {
  return context.graph;
}

template <class Queue, class RoutingContext, class HeuristicContext, typename T>
void a_star_with_queue(const RoutingContext &context, const SearchQuery &query,
                       std::size_t destination,
                       const HeuristicContext &heuristic,
                       const MutableSearchResults<T> &results,
                       const AStarWorkspace<T> &workspace,
                       Queue &queue) noexcept {
  static_assert(std::is_base_of_v<PriorityQueueBase<Queue, T>, Queue>);
  constexpr bool turn_based =
      std::is_same_v<RoutingContext, TurnBasedContext<T>>;
  const auto &graph = a_star_graph(context);
  const auto root = turn_based ? graph.link_count : query.origin;
  const auto infinity = std::numeric_limits<T>::infinity();
  results.reset();

  auto &metadata = *results.metadata;
  metadata.origin = query.origin;
  metadata.root = root;
  metadata.target_count = 1;
  bool stopped_at_target = false;
  results.turn_costs[root] = 0.0;

  // Heap keys include the heuristic, but reported distances contain only costs.
  auto *costs = workspace.costs;
  auto *estimates = workspace.estimates;
  std::fill_n(costs, results.state_count, infinity);
  std::fill_n(estimates, graph.node_count, T{-1});
  const auto priority = [&](std::size_t node, T cost) {
    if (estimates[node] < 0.0) {
      estimates[node] = heuristic(node, destination);
    }
    // Overflow in the priority must not make a finite-cost path unreachable.
    return std::min(cost + estimates[node], std::numeric_limits<T>::max());
  };

  costs[root] = 0.0;
  queue.reset_heap();
  queue.insert(root, priority(query.origin, 0.0));

  while (!queue.is_empty()) {
    const auto state = queue.extract_min();
    const T cost = costs[state];
    std::size_t node = state;

    if constexpr (turn_based) {
      node = state == root ? query.origin : graph.heads[state];
    }

    results.settlement_order[metadata.settled_count++] = state;
    results.distances[state] = cost;

    if (results.terminal_states[node] == invalid_state) {
      results.terminal_states[node] = state;
      if (node == destination) {
        metadata.reached_target_count = 1;
        stopped_at_target = true;
        break;
      }
    }

    // A centroid may be reached, but only the origin may supply outgoing links.
    if (node < graph.blocked_centroid_count && node != query.origin) {
      continue;
    }

    std::size_t turn = 0, turn_end = 0;
    if constexpr (turn_based) {
      // The virtual root leaves the origin without an incoming turn.
      if (state != root) {
        turn = context.turn_fs[state];
        turn_end = context.turn_fs[state + 1];
      }
    }

    for (auto link = graph.fs[node]; link < graph.fs[node + 1]; ++link) {
      const auto next_node = graph.heads[link];
      const auto next = turn_based ? link : next_node;
      const auto next_state = queue.effective_state(next);
      if (next_state == SCANNED) {
        continue;
      }

      T penalty = 0;
      if constexpr (turn_based) {
        // Merge sorted sparse turn rows with the outgoing links.
        while (turn < turn_end && context.turn_to_links[turn] < link) {
          ++turn;
        }
        const bool explicit_turn =
            turn < turn_end && context.turn_to_links[turn] == link;

        penalty = explicit_turn ? context.turn_penalties[turn] : T{0};
        if (state != root && !explicit_turn && !context.allow_uturns &&
            context.first_nodes[link] == context.last_nodes[state]) {
          continue;
        }
      }

      const T next_cost = cost + graph.costs[link] + penalty;
      if (!std::isfinite(next_cost) || next_cost >= costs[next]) {
        continue;
      }

      const T key = priority(next_node, next_cost);
      if (next_state == NOT_IN_HEAP) {
        queue.insert(next, key);
      } else {
        queue.decrease_key(next, key);
      }

      costs[next] = next_cost;
      results.predecessors[next] = state;
      results.connectors[next] = link;
      results.turn_costs[next] = results.turn_costs[state] + penalty;
    }
  }

  // Reaching the destination does not prove the reachable state space
  // exhausted.
  metadata.exhausted = !stopped_at_target;
  for (std::size_t state = 0; state < results.state_count; ++state) {
    if (queue.effective_state(state) == IN_HEAP) {
      results.predecessors[state] = invalid_state;
      results.connectors[state] = invalid_state;
      results.turn_costs[state] = infinity;
    }
  }
}

template <class RoutingContext, class HeuristicContext, typename T>
void a_star(const RoutingContext &context, const SearchQuery &query,
            std::size_t destination, const HeuristicContext &heuristic,
            const MutableSearchResults<T> &results,
            const AStarWorkspace<T> &workspace) noexcept {
  workspace.search.heap->visit([&](auto &queue) {
    a_star_with_queue(context, query, destination, heuristic, results,
                      workspace, queue);
  });
}

} // namespace aequilibrae::paths::cpp::routing
