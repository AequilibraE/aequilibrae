#pragma once

#include <cstddef>

namespace aequilibrae::paths::cpp::mvp {

struct NodeBasedContext {
  std::size_t node_count = 0;
  std::size_t link_count = 0;
  const std::size_t *fs = nullptr;
  const std::size_t *heads = nullptr;
  const double *costs = nullptr;
  const std::size_t *link_ids = nullptr;
};

struct TurnBasedContext {
  NodeBasedContext graph;
  const std::size_t *tails = nullptr;
  // Sparse turns grouped by incoming link, sorted by outgoing link.
  const std::size_t *turn_fs = nullptr;
  const std::size_t *turn_to_links = nullptr;
  const double *turn_penalties = nullptr;
  bool allow_uturns = true;
};

} // namespace aequilibrae::paths::cpp::mvp
