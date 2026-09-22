#pragma once

#include <cstddef>

namespace aequilibrae::paths::cpp::mvp {

struct NodeBasedContext {
  std::size_t node_count = 0;
  std::size_t link_count = 0;
  const std::size_t *fs = nullptr;
  const std::size_t *heads = nullptr;
  const double *costs = nullptr;
  // Nodes in this prefix may start or end a path, but cannot be used through.
  std::size_t blocked_centroid_count = 0;
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

// Link fields come first, then the optional cost and turn-cost labels.
// The layout is fixed at setup and reused for every origin.
template <typename T> struct SkimmingContext {
  std::size_t link_count = 0;
  std::size_t field_count = 0;
  std::size_t additive_field_count = 0;
  const T *const *link_fields = nullptr; // [additive fields], each [links]

  // Labels follow the additive fields but need no columns in state scratch.
  std::size_t cost_field_index = 0;
  std::size_t turn_cost_field_index = 0;
  std::size_t cost_field_count = 0;      // Zero or one.
  std::size_t turn_cost_field_count = 0; // Zero or one.

  bool needs_state_sums() const noexcept { return additive_field_count > 0; }

  bool has_cost_field() const noexcept { return cost_field_count > 0; }

  bool has_turn_cost_field() const noexcept {
    return turn_cost_field_count > 0;
  }
};

// Masks are fixed at setup and shared across origins and workers.
struct SelectLinkContext {
  std::size_t link_count = 0;
  std::size_t set_count = 0;
  const bool *masks = nullptr; // [sets, links]

  const bool *selection(std::size_t index) const noexcept {
    return link_count ? masks + index * link_count : nullptr;
  }
};

} // namespace aequilibrae::paths::cpp::mvp
