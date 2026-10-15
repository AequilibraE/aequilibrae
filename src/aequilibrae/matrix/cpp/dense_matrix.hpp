#pragma once

#include <cstddef>
#include <type_traits>

namespace aequilibrae::matrix::cpp {

// Borrow one contiguous [destinations, fields] block. The owner must outlive the view.
// Callers supply valid indices. Empty views do not offset pointers.
template <typename T> struct DenseMatrixOriginView {
  static_assert(std::is_floating_point_v<T>);

  std::size_t destination_count = 0;
  std::size_t field_count = 0;
  T *data = nullptr;

  T *destination_data(std::size_t index) const noexcept {
    return destination_count && field_count ? data + index * field_count
                                           : nullptr;
  }
};

// Borrow a contiguous [origins, destinations, fields] buffer without owning it.
template <typename T> struct DenseMatrixView {
  static_assert(std::is_floating_point_v<T>);

  std::size_t origin_count = 0;
  std::size_t destination_count = 0;
  std::size_t field_count = 0;
  T *data = nullptr;

  DenseMatrixOriginView<T> origin(std::size_t index) const noexcept {
    T *values = nullptr;
    if (origin_count && destination_count && field_count) {
      values = data + index * destination_count * field_count;
    }
    return {destination_count, field_count, values};
  }
};

} // namespace aequilibrae::matrix::cpp
