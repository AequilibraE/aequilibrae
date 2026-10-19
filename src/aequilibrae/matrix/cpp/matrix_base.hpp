#pragma once

#include <cstddef>
#include <type_traits>

namespace aequilibrae::matrix::cpp {

template <typename T> class MatrixBase {
  static_assert(std::is_floating_point_v<T>);

public:
  std::size_t origin_count = 0;
  std::size_t destination_count = 0;
  std::size_t field_count = 0;
  T *data = nullptr;

  // Deleted functions, null-ary copy, and copy assignment are all disabled.
  MatrixBase() = delete;
  MatrixBase(const MatrixBase &other) = delete;
  MatrixBase &operator=(const MatrixBase &) = delete;
};

} // namespace aequilibrae::matrix::cpp
