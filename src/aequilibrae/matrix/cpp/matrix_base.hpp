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
};

} // namespace aequilibrae::matrix::cpp
