#pragma once

#include <algorithm>
#include <cstddef>

namespace aequilibrae::paths::cpp::mvp {

// Borrowed link accumulators, separate from the state cascade scratch. The
// caller decides when to reset: loading adds origins until an iteration ends.
template <typename T> struct LoadingOutputs {
  std::size_t link_count = 0;
  std::size_t class_count = 0;
  T *link_loads = nullptr; // [links, classes]

  void reset() const noexcept {
    if (link_count && class_count) {
      std::fill_n(link_loads, link_count * class_count, T{0});
    }
  }
};

} // namespace aequilibrae::paths::cpp::mvp
