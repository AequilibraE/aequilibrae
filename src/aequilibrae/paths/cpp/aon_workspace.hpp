#pragma once

#include <cstddef>

namespace aequilibrae::paths::cpp::mvp {

// Borrowed, per-worker scratch buffers. The owner allocates before releasing
// the GIL and must serialize access, just as for SearchResults. This workspace
// is intentionally separate from finalized search labels; future routing
// scratch (including heaps) can live here without changing that contract.
template <typename T> struct AoNWorkspace {
  std::size_t state_count = 0;
  std::size_t skim_field_count = 0;
  // Packed row-major [state_count, skim_field_count].
  T *state_skims = nullptr;

  std::size_t loading_class_count = 0;
  // Packed row-major [state_count, loading_class_count], reset per origin.
  // Link loads are caller-owned [link_count, loading_class_count] accumulators
  // passed to network_loading, normally one slice of a per-thread allocation.
  T *state_loads = nullptr;

  // One flag per state: does its path use a selected link? Reuse these flags
  // across sets so scratch size does not grow with the number of sets.
  bool *selected_paths = nullptr;
};

} // namespace aequilibrae::paths::cpp::mvp
