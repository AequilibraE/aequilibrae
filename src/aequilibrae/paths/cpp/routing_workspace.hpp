#pragma once

#include <cstddef>

namespace aequilibrae::paths::cpp::mvp {

// Borrowed, per-worker scratch buffers. The owner allocates before releasing
// the GIL and must serialize access, just as for SearchResults. This workspace
// is intentionally separate from finalized search labels; future routing
// scratch (including heaps) can live here without changing that contract.
template <typename T> struct RoutingWorkspace {
  std::size_t state_count = 0;
  std::size_t skim_field_count = 0;
  // Packed row-major [state_count, skim_field_count].
  T *state_skims = nullptr;
};

} // namespace aequilibrae::paths::cpp::mvp
