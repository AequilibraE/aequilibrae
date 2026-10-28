#pragma once

#include <cstddef>
#include <type_traits>

#include "matrix_base.hpp"

namespace aequilibrae::matrix::cpp {

// Borrow one contiguous [destinations, fields] block. The owner must outlive
// the view. Callers supply valid indices. Empty views do not offset pointers.
template <typename T> struct DenseMatrixOriginView {
  static_assert(std::is_floating_point_v<T>);

  std::size_t destination_count = 0;
  std::size_t field_count = 0;
  T *data = nullptr;

  std::size_t entry_count() const noexcept { return destination_count; }

  std::size_t destination(std::size_t entry) const noexcept { return entry; }

  const T *values(std::size_t entry) const noexcept {
    return destination_data(entry);
  }

  T *destination_data(std::size_t index) const noexcept {
    return destination_count && field_count ? data + index * field_count
                                            : nullptr;
  }
};

// Borrow a contiguous [origins, destinations, fields] buffer without owning it.
template <typename T> class DenseMatrix : public MatrixBase<T> {
public:
  DenseMatrixOriginView<T> origin(std::size_t index) const noexcept {
    T *values = nullptr;
    if (this->origin_count && this->destination_count && this->field_count) {
      values = this->data + index * this->destination_count * this->field_count;
    }
    return {this->destination_count, this->field_count, values};
  }
};

} // namespace aequilibrae::matrix::cpp
