#pragma once

#include <cstddef>

namespace aequilibrae::paths::cpp::mvp {

// Plain link fields come first, then link fields with turn costs, then the
// optional label matrices. Counts and positions are fixed at setup, so the
// same layout can be reused for every origin.
template <typename T> struct SkimmingContext {
  std::size_t link_count = 0;
  std::size_t field_count = 0;
  std::size_t additive_field_count = 0;
  const T *const *link_fields = nullptr; // [additive fields], each [links]

  // Both additive groups occupy the same positions in scratch and output.
  std::size_t plain_field_count = 0; // Plain fields always start at zero.
  std::size_t turn_field_offset = 0;
  std::size_t turn_field_count = 0;

  // Labels follow the additive fields but need no columns in state scratch.
  std::size_t cost_field_index = 0;
  std::size_t turn_cost_field_index = 0;
  std::size_t cost_field_count = 0;      // Zero or one.
  std::size_t turn_cost_field_count = 0; // Zero or one.

  bool needs_state_sums() const noexcept { return additive_field_count > 0; }

  bool has_link_fields() const noexcept { return plain_field_count > 0; }

  bool has_link_fields_with_turn_costs() const noexcept {
    return turn_field_count > 0;
  }

  bool has_cost_field() const noexcept { return cost_field_count > 0; }

  bool has_turn_cost_field() const noexcept {
    return turn_cost_field_count > 0;
  }
};

} // namespace aequilibrae::paths::cpp::mvp
