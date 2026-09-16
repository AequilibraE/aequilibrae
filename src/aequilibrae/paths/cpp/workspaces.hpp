#pragma once

#include <cstddef>

namespace aequilibrae::paths::cpp::mvp {

// Ordinary and selected loading can reuse this cascade scratch sequentially
// because each operation replaces every state total.
template <typename T> struct LoadingWorkspace {
  std::size_t state_count = 0;
  std::size_t class_count = 0;
  T *state_loads = nullptr; // [states, classes]
};

// Only additive fields need state sums. Direct cost projection needs no
// scratch.
template <typename T> struct SkimmingWorkspace {
  std::size_t state_count = 0;
  std::size_t field_count = 0;
  T *state_skims = nullptr; // [states, additive fields]
};

// Process sets sequentially so membership scratch does not grow with set count.
// Demand cascades use a separately supplied LoadingWorkspace.
struct SelectLinkWorkspace {
  std::size_t state_count = 0;
  bool *selected_paths = nullptr;
};

// Group worker scratch for the assignment driver. Each kernel accepts only
// the small workspace it needs; none depends on this aggregate.
template <typename T> struct AoNWorkspace {
  LoadingWorkspace<T> loading;
  SkimmingWorkspace<T> skimming;
  SelectLinkWorkspace select_link;
};

} // namespace aequilibrae::paths::cpp::mvp
