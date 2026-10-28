#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace aequilibrae::paths::cpp::routing {

// Coordinates use local node order. Views borrow their Cython owner's arrays.
template <typename T> struct EuclideanContext {
  using value_type = T;

  std::size_t node_count = 0;
  const T *x = nullptr;
  const T *y = nullptr;
  T scale = 0;

  T distance(std::size_t from, std::size_t to) const noexcept {
    return std::hypot(x[to] - x[from], y[to] - y[from]);
  }

  T operator()(std::size_t node, std::size_t destination) const noexcept {
    return scale == T{0} ? T{0} : scale * distance(node, destination);
  }
};

template <typename T> struct HaversineContext {
  using value_type = T;

  std::size_t node_count = 0;
  const T *latitudes = nullptr;  // Radians.
  const T *longitudes = nullptr; // Radians.
  const T *cos_latitudes = nullptr;
  T scale = 0;

  T distance(std::size_t from, std::size_t to) const noexcept {
    constexpr T earth_radius_metres = T{6371000};
    const T sin_lat = std::sin((latitudes[to] - latitudes[from]) / T{2});
    const T sin_lon = std::sin((longitudes[to] - longitudes[from]) / T{2});
    const T a = sin_lat * sin_lat +
                cos_latitudes[from] * cos_latitudes[to] * sin_lon * sin_lon;

    // Round-off near antipodal points must not put asin outside its domain.
    return T{2} * earth_radius_metres *
           std::asin(std::sqrt(std::clamp(a, T{0}, T{1})));
  }

  T operator()(std::size_t node, std::size_t destination) const noexcept {
    return scale == T{0} ? T{0} : scale * distance(node, destination);
  }
};

} // namespace aequilibrae::paths::cpp::routing
