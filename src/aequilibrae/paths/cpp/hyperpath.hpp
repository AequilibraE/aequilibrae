#pragma once

// Spiess and Florian's optimal strategies: label setting towards one
// destination, then loading of the attractive edges in reverse extraction order.
//
// Spiess, H. and Florian, M. (1989). Optimal strategies: A new assignment model
// for transit networks. Transportation Research Part B 23(2), 83-102.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "pq_4ary_heap.hpp"

namespace aequilibrae::paths::cpp::hyperpath {

inline constexpr std::uint32_t kNoEdge = std::numeric_limits<std::uint32_t>::max();
inline constexpr double kInf = std::numeric_limits<double>::infinity();

// How a skim field combines an edge with the strategy beyond the edge's head.
// A "waiting" edge is one with finite frequency.
enum SkimOperator : std::uint8_t {
  SKIM_ADD = 0,           // edge value plus the head's skim
  SKIM_ANY_WAIT = 1,      // 1 on a waiting edge, otherwise the head's skim
  SKIM_RESET_AT_WAIT = 2, // 0 on a waiting edge, otherwise the head's skim
  SKIM_COUNT_WAITS = 3,   // the head's skim, plus 1 on a waiting edge
};

// Edges into vertex v are csc_edges[csc_indptr[v]:csc_indptr[v + 1]].
struct Graph {
  std::size_t vertex_count = 0;
  std::size_t edge_count = 0;
  const std::uint32_t *csc_indptr = nullptr; // [vertices + 1]
  const std::uint32_t *csc_edges = nullptr;  // [edges]
  const std::uint32_t *tail = nullptr;       // [edges]
  const std::uint32_t *head = nullptr;       // [edges]
};

// Costs are checked nonnegative by the caller. Frequency +inf marks an edge
// with no wait, and an edge with zero frequency is never used. Each vertex
// waits kappa_i / F_i for the attractive set it boards, where F_i is the set's
// combined frequency, and reports alpha_i / F_i as its waiting time.
struct Costs {
  const double *cost = nullptr;             // c_a [edges]
  const double *frequency = nullptr;        // f_a [edges]
  const double *wait_constant = nullptr;    // kappa_i [vertices]
  const double *headway_fraction = nullptr; // alpha_i [vertices]
  // Raises every cost to at least this value, so labels strictly increase
  // along attractive edges and the extraction order is a loading order.
  double cost_floor = 0;
};

struct Skims {
  std::size_t field_count = 0;
  const double *edge_values = nullptr;     // [edges, fields]
  const std::uint8_t *operators = nullptr; // [fields]
  const std::uint8_t *waits = nullptr;     // [fields], 1 adds alpha_i / F_i
};

// Per-thread scratch, kept between destinations and assignments.
struct Workspace {
  Workspace(std::size_t vertex_count, std::size_t edge_count,
            std::size_t field_count)
      : label(vertex_count, kInf), frequency(vertex_count, 0.0),
        infinite_edge(vertex_count, kNoEdge), volume(vertex_count, 0.0),
        skims(vertex_count * field_count, 0.0) {
    order.reserve(edge_count);
    heap.alloc_heap(edge_count);
  }

  Workspace(const Workspace &) = delete;
  Workspace &operator=(const Workspace &) = delete;

  std::vector<double> label;                // u_i
  std::vector<double> frequency;            // F_i, finite attractive edges only
  std::vector<std::uint32_t> infinite_edge; // attractive edge with f_a = inf
  std::vector<double> volume;               // v_i
  std::vector<std::uint32_t> order;         // attractive edges as extracted
  std::vector<double> skims;                // [vertices, fields]
  FourAryHeap<double> heap;
};

inline double edge_skim(std::uint8_t op, double value, double head_skim,
                        bool waits) noexcept {
  switch (op) {
  case SKIM_ANY_WAIT:
    return waits ? 1.0 : head_skim;
  case SKIM_RESET_AT_WAIT:
    return waits ? 0.0 : head_skim;
  case SKIM_COUNT_WAITS:
    return head_skim + (waits ? 1.0 : 0.0);
  default:
    return head_skim + value;
  }
}

inline void relax_incoming(const Graph &graph, const Costs &costs,
                           std::uint32_t vertex, double label,
                           Workspace &ws) noexcept {
  for (std::size_t p = graph.csc_indptr[vertex];
       p < graph.csc_indptr[vertex + 1]; ++p) {
    const std::uint32_t edge = graph.csc_edges[p];
    if (!(costs.frequency[edge] > 0.0)) {
      continue;
    }
    const ElementState state = ws.heap.effective_state(edge);
    if (state == SCANNED) {
      continue;
    }
    const double key = label + std::max(costs.cost[edge], costs.cost_floor);
    if (!(key < kInf)) {
      continue;
    }
    if (state == NOT_IN_HEAP) {
      ws.heap.insert(edge, key);
    } else if (ws.heap.element_key(edge) > key) {
      ws.heap.decrease_key(edge, key);
    }
  }
}

// Labels, attractive set and skims of every vertex towards the destination.
inline void compute_labels(const Graph &graph, const Costs &costs,
                           const Skims &skims, std::uint32_t destination,
                           Workspace &ws) noexcept {
  const std::size_t fields = skims.field_count;
  std::fill(ws.label.begin(), ws.label.end(), kInf);
  std::fill(ws.frequency.begin(), ws.frequency.end(), 0.0);
  std::fill(ws.infinite_edge.begin(), ws.infinite_edge.end(), kNoEdge);
  ws.order.clear();
  ws.heap.reset_heap();

  ws.label[destination] = 0.0;
  std::fill_n(ws.skims.begin() + destination * fields, fields, 0.0);
  relax_incoming(graph, costs, destination, 0.0, ws);

  while (!ws.heap.is_empty()) {
    const std::size_t edge = ws.heap.extract_min();
    const double key = ws.heap.element_key(edge);
    const std::uint32_t tail = graph.tail[edge];

    // Strict, so a tie never joins the attractive set without lowering u_i.
    if (!(key < ws.label[tail])) {
      continue;
    }

    const double f = costs.frequency[edge];
    const bool waits = !std::isinf(f);
    const double *head_skims = ws.skims.data() + graph.head[edge] * fields;
    double *tail_skims = ws.skims.data() + tail * fields;
    const double *values = skims.edge_values + edge * fields;

    if (!waits) {
      // An edge without a wait takes all of the vertex's flow, so any finite
      // frequency edges extracted before it keep a zero share.
      ws.label[tail] = key;
      ws.infinite_edge[tail] = static_cast<std::uint32_t>(edge);
      for (std::size_t k = 0; k < fields; ++k) {
        tail_skims[k] =
            edge_skim(skims.operators[k], values[k], head_skims[k], false);
      }
    } else {
      const double previous = ws.frequency[tail];
      const double total = previous + f;
      if (previous > 0.0) {
        ws.label[tail] = (previous * ws.label[tail] + f * key) / total;
        for (std::size_t k = 0; k < fields; ++k) {
          const double s =
              edge_skim(skims.operators[k], values[k], head_skims[k], true);
          tail_skims[k] = (previous * tail_skims[k] + f * s) / total;
        }
      } else {
        ws.label[tail] = (costs.wait_constant[tail] + f * key) / f;
        const double alpha = costs.headway_fraction[tail];
        for (std::size_t k = 0; k < fields; ++k) {
          const double s =
              edge_skim(skims.operators[k], values[k], head_skims[k], true);
          tail_skims[k] = ((skims.waits[k] ? alpha : 0.0) + f * s) / f;
        }
      }
      ws.frequency[tail] = total;
    }

    ws.order.push_back(static_cast<std::uint32_t>(edge));
    relax_incoming(graph, costs, tail, ws.label[tail], ws);
  }
}

// Loads demand from the origins onto the strategy of the last call to
// compute_labels. Every attractive edge is extracted before the attractive
// edges into its tail, so walking the order backwards meets each vertex with
// all of its inflow. Demand whose origin cannot reach the destination is added
// to unassigned instead. Returns the kappa-weighted waiting time sum_i v_i
// kappa_i / F_i, so that sum_a c_a v_a plus this equals sum_o d_o u_o.
inline double load(const Graph &graph, const Costs &costs,
                   const std::uint32_t *origins, const double *demand,
                   std::size_t count, double *edge_volume, double *unassigned,
                   Workspace &ws) noexcept {
  for (std::size_t k = 0; k < count; ++k) {
    if (ws.label[origins[k]] < kInf) {
      ws.volume[origins[k]] += demand[k];
    } else {
      unassigned[k] += demand[k];
    }
  }

  double weighted_wait = 0.0;
  for (std::size_t idx = ws.order.size(); idx-- > 0;) {
    const std::uint32_t edge = ws.order[idx];
    const std::uint32_t tail = graph.tail[edge];
    const double vertex_volume = ws.volume[tail];
    if (vertex_volume == 0.0) {
      continue;
    }

    double share = 1.0;
    if (ws.infinite_edge[tail] != kNoEdge) {
      if (edge != ws.infinite_edge[tail]) {
        continue;
      }
    } else {
      share = costs.frequency[edge] / ws.frequency[tail];
      weighted_wait += vertex_volume * share * costs.wait_constant[tail] /
                       ws.frequency[tail];
    }

    const double edge_flow = vertex_volume * share;
    edge_volume[edge] += edge_flow;
    ws.volume[graph.head[edge]] += edge_flow;
  }

  // Leave the node volumes zeroed for the next destination.
  for (std::size_t k = 0; k < count; ++k) {
    ws.volume[origins[k]] = 0.0;
  }
  for (const std::uint32_t edge : ws.order) {
    ws.volume[graph.tail[edge]] = 0.0;
    ws.volume[graph.head[edge]] = 0.0;
  }
  return weighted_wait;
}

} // namespace aequilibrae::paths::cpp::hyperpath
