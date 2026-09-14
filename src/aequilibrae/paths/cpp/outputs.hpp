#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

namespace aequilibrae::paths::cpp::mvp {

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

// Borrow one origin's contiguous [fields, destinations] block.
// Callers supply valid indices and ranges. Empty views do not offset pointers.
template <typename T> struct SkimmingOriginView {
  std::size_t field_count = 0;
  std::size_t destination_count = 0;
  T *data = nullptr;

  T *field_data(std::size_t index) const noexcept {
    if (field_count == 0 || destination_count == 0) {
      return nullptr;
    }
    return data + index * destination_count;
  }

  SkimmingOriginView<T> subfields(std::size_t first,
                                  std::size_t count) const noexcept {
    return {count, destination_count, count ? field_data(first) : nullptr};
  }
};

// Borrow the complete contiguous output. Neither view owns its buffer.
template <typename T> struct SkimmingOutputsView {
  std::size_t origin_count = 0;
  std::size_t field_count = 0;
  std::size_t destination_count = 0;
  T *data = nullptr; // [origins, fields, destinations]

  SkimmingOriginView<T> origin(std::size_t index) const noexcept {
    T *values = nullptr;
    if (field_count && destination_count) {
      values = data + index * field_count * destination_count;
    }
    return {field_count, destination_count, values};
  }

  void reset() const noexcept {
    if (origin_count && field_count && destination_count) {
      std::fill_n(data, origin_count * field_count * destination_count,
                  std::numeric_limits<T>::infinity());
    }
  }
};

} // namespace aequilibrae::paths::cpp::mvp
