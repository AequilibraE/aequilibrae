#pragma once
#include "aeq_log.hpp"
#include "pq_heap_base.hpp"
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
