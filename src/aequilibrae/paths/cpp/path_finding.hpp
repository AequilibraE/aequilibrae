#pragma once
#include "aeq_log.hpp"
#include "graph_context.hpp"
#include "pq_heap_base.hpp"
#include "search_results.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace aequilibrae::paths::cpp {

inline constexpr double EARTH_RADIUS_METERS = 6371000.0;
inline constexpr double PI = 3.14159265358979323846;
inline constexpr double DEG_TO_RAD = PI / 180.0;
inline constexpr size_t SENTINEL = std::numeric_limits<size_t>::max();

enum class Heuristic : int { HAVERSINE, EQUIRECTANGULAR };

using HeuristicFn = double (*)(double lat1, double lon1, double lat2,
                               double lon2, void *data) noexcept;

inline double haversine_heuristic(double lat1, double lon1, double lat2,
                                  double lon2, void *data) noexcept {
  double cos_lat1 = *static_cast<double *>(data);
  double dlat = lat2 - lat1;
  double dlon = lon2 - lon1;
  double sin_dlat = std::sin(dlat / 2.0);
  double sin_dlon = std::sin(dlon / 2.0);
  double a =
      sin_dlat * sin_dlat + cos_lat1 * std::cos(lat2) * sin_dlon * sin_dlon;
  return 2.0 * EARTH_RADIUS_METERS * std::asin(std::sqrt(a));
}

inline double equirectangular_heuristic(double lat1, double lon1, double lat2,
                                        double lon2, void *data) noexcept {
  (void)data;
  double x = (lon2 - lon1) * std::cos((lat1 + lat2) / 2.0);
  double y = (lat2 - lat1);
  return EARTH_RADIUS_METERS * std::sqrt(x * x + y * y);
}

template <class Queue>
size_t dijkstra(size_t origin, const size_t max_size, const double *costs,
                const size_t *csr, const size_t *fs, size_t *predecessors,
                const size_t *ids, size_t *connectors, size_t *reached_first,
                const unsigned char *destinations, long long destination_count,
                AeqLogClosure *c) noexcept {
  static_assert(std::is_base_of<PriorityQueueBase<Queue>, Queue>::value,
                "Queue provided does not derive from PriorityQueueBase");

  Queue queue{};
  queue.attach_logger(c);
  size_t found = 0;
  const bool early_exit_enabled = destination_count >= 0;
  long long remaining = destination_count;

  AEQ_LOG(c, AEQ_LOG_DEBUG,
          aeq_format_string("Running Dijkstra's with origin = ", origin));

  for (size_t i = 0; i < max_size; i++) {
    predecessors[i] = SENTINEL;
    connectors[i] = SENTINEL;
    reached_first[i] = SENTINEL;
  }

  queue.init_heap(max_size);
  queue.insert(origin, 0.0);

  while (!queue.is_empty()) {
    // Read the key after extraction: lazy-deletion queues may hold stale
    // entries whose peek() disagrees with the element extract_min() returns.
    const size_t tail_vertex = queue.extract_min();
    const double tail_value = queue.element_key(tail_vertex);
    reached_first[found] = tail_vertex;
    found++;

    if (early_exit_enabled) {
      if (remaining > 0 && destinations[tail_vertex]) {
        remaining--;
      }
      if (remaining == 0) {
        for (size_t idx = 0; idx < max_size; idx++) {
          if (queue.effective_state(idx) == IN_HEAP) {
            predecessors[idx] = SENTINEL;
            connectors[idx] = SENTINEL;
          }
        }
        break;
      }
    }

    for (size_t idx = fs[tail_vertex]; idx < fs[tail_vertex + 1]; idx++) {
      const size_t head_vertex = csr[idx];
      const ElementState head_state = queue.effective_state(head_vertex);

      if (head_state != SCANNED) {
        const double head_value = tail_value + costs[idx];
        if (head_value == std::numeric_limits<double>::infinity()) {
          continue;
        } else if (head_state == NOT_IN_HEAP) {
          queue.insert(head_vertex, head_value);
          predecessors[head_vertex] = tail_vertex;
          connectors[head_vertex] = ids[idx];
        } else if (queue.element_key(head_vertex) > head_value) {
          queue.decrease_key(head_vertex, head_value);
          predecessors[head_vertex] = tail_vertex;
          connectors[head_vertex] = ids[idx];
        }
      }
    }
  }

  queue.free_heap();
  return found - 1;
}

template <class Queue>
void a_star(size_t origin, size_t destination, const size_t max_size,
            const double *costs, const size_t *csr, const size_t *fs,
            const size_t * /*nodes_to_indices*/, const double *lats,
            const double *lons, size_t *predecessors, const size_t *ids,
            size_t *connectors, HeuristicFn heur, void *heuristic_data,
            AeqLogClosure *b) noexcept {
  static_assert(std::is_base_of<PriorityQueueBase<Queue>, Queue>::value,
                "Queue provided does not derive from PriorityQueueBase");

  Queue queue{};
  queue.attach_logger(b);
  const size_t destination_vert = (destination != SENTINEL) ? destination : 0;

  AEQ_LOG(b, AEQ_LOG_DEBUG,
          aeq_format_string("Running A* with origin = ", origin,
                            " destination = ", destination_vert));

  std::vector<double> gScore(max_size, std::numeric_limits<double>::infinity());

  for (size_t i = 0; i < max_size; i++) {
    predecessors[i] = SENTINEL;
    connectors[i] = SENTINEL;
  }

  const double lat1_rad = lats[destination_vert] * DEG_TO_RAD;
  const double lon1_rad = lons[destination_vert] * DEG_TO_RAD;

  queue.init_heap(max_size);
  queue.insert(origin, 0.0);
  gScore[origin] = 0.0;

  while (!queue.is_empty()) {
    const size_t current = queue.extract_min();

    if (current == destination_vert) {
      break;
    }

    for (size_t idx = fs[current]; idx < fs[current + 1]; idx++) {
      const size_t neighbour = csr[idx];
      const double tentative_gScore = gScore[current] + costs[idx];

      if (tentative_gScore < gScore[neighbour]) {
        predecessors[neighbour] = current;
        connectors[neighbour] = ids[idx];
        gScore[neighbour] = tentative_gScore;

        const double h = heur(lat1_rad, lon1_rad, lats[neighbour] * DEG_TO_RAD,
                              lons[neighbour] * DEG_TO_RAD, heuristic_data);

        if (queue.effective_state(neighbour) != IN_HEAP) {
          queue.insert(neighbour, tentative_gScore + h);
        } else {
          queue.decrease_key(neighbour, tentative_gScore + h);
        }
      }
    }
  }

  queue.free_heap();
}

} // namespace aequilibrae::paths::cpp

namespace aequilibrae::paths::cpp::mvp {

template <class Queue>
void dijkstra(const NodeBasedContext &context, std::size_t origin,
              SearchResults &results) noexcept {
  static_assert(std::is_base_of<PriorityQueueBase<Queue>, Queue>::value,
                "Queue provided does not derive from PriorityQueueBase");

  results.origin = origin;
  results.root = origin;
  results.settled_count = 0;
  results.destination_count = 0;
  results.reached_destination_count = 0;

  std::fill_n(results.predecessors, context.node_count, SENTINEL);
  std::fill_n(results.connectors, context.node_count, SENTINEL);
  std::fill_n(results.reached_first, context.node_count, SENTINEL);
  std::fill_n(results.distances, context.node_count, kInfinity);
  std::fill_n(results.turn_costs, context.node_count, kInfinity);
  std::fill_n(results.terminal_states, context.node_count, SENTINEL);

  for (std::size_t node = 0; node < context.node_count; ++node) {
    results.destination_count += results.destination_mask[node] != 0;
  }

  Queue queue;
  queue.init_heap(context.node_count);
  queue.insert(origin, 0.0);

  while (!queue.is_empty()) {
    const auto state = queue.extract_min();
    const double cost = queue.element_key(state);

    results.reached_first[results.settled_count++] = state;
    results.distances[state] = cost;
    results.turn_costs[state] = 0.0;
    results.terminal_states[state] = state;

    if (results.destination_mask[state] != 0) {
      ++results.reached_destination_count;

      if (results.reached_destination_count == results.destination_count) {
        break;
      }
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
      results.connectors[next] = context.link_ids[link];
    }
  }

  // Early exit must not expose tentative paths as finalized.
  for (std::size_t state = 0; state < context.node_count; ++state) {
    if (queue.effective_state(state) == IN_HEAP) {
      results.predecessors[state] = SENTINEL;
      results.connectors[state] = SENTINEL;
    }
  }
}

template <class Queue>
void dijkstra(const TurnBasedContext &context, std::size_t origin,
              SearchResults &results) noexcept {
  const auto &graph = context.graph;
  // State s < link_count means arrival via directed link s. The last state is
  // a virtual source, so first links pay no turn cost and even an edgeless
  // graph has a valid root. No expanded transition graph is allocated.
  const auto root = graph.link_count;
  const auto state_count = graph.link_count + 1;

  results.origin = origin;
  results.root = root;
  results.settled_count = 0;
  results.destination_count = 0;
  results.reached_destination_count = 0;

  std::fill_n(results.predecessors, state_count, SENTINEL);
  std::fill_n(results.connectors, state_count, SENTINEL);
  std::fill_n(results.reached_first, state_count, SENTINEL);
  std::fill_n(results.distances, state_count, kInfinity);
  std::fill_n(results.turn_costs, state_count, kInfinity);
  std::fill_n(results.terminal_states, graph.node_count, SENTINEL);

  for (std::size_t node = 0; node < graph.node_count; ++node) {
    results.destination_count += results.destination_mask[node] != 0;
  }
  results.turn_costs[root] = 0.0;

  Queue queue;
  queue.init_heap(state_count);
  queue.insert(root, 0.0);

  while (!queue.is_empty()) {
    const auto state = queue.extract_min();
    const double cost = queue.element_key(state);
    const auto node = state == root ? origin : graph.heads[state];
    results.reached_first[results.settled_count++] = state;
    results.distances[state] = cost;

    if (results.terminal_states[node] == SENTINEL) {
      results.terminal_states[node] = state;

      if (results.destination_mask[node] != 0) {
        ++results.reached_destination_count;

        if (results.reached_destination_count == results.destination_count) {
          break;
        }
      }
    }

    auto turn = state == root ? 0 : context.turn_fs[state];
    const auto turn_end = state == root ? 0 : context.turn_fs[state + 1];
    for (auto next = graph.fs[node]; next < graph.fs[node + 1]; ++next) {
      if (queue.effective_state(next) == SCANNED) {
        continue;
      }

      // Merge the sorted sparse turn row with the outgoing forward-star row.
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
      results.connectors[next] = graph.link_ids[next];
      results.turn_costs[next] = results.turn_costs[state] + penalty;
    }
  }

  // Early exit must not expose tentative paths or turn labels as finalized.
  for (std::size_t state = 0; state < state_count; ++state) {
    if (queue.effective_state(state) == IN_HEAP) {
      results.predecessors[state] = SENTINEL;
      results.connectors[state] = SENTINEL;
      results.turn_costs[state] = kInfinity;
    }
  }
}

} // namespace aequilibrae::paths::cpp::mvp
