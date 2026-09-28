#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace aequilibrae::paths::cpp::mvp {

// Coordinates use local node order. Views borrow their Cython owner's arrays.
struct EuclideanContext {
  std::size_t node_count = 0;
  const double *x = nullptr;
  const double *y = nullptr;
  double scale = 0.0;

  double distance(std::size_t from, std::size_t to) const noexcept {
    return std::hypot(x[to] - x[from], y[to] - y[from]);
  }

  double operator()(std::size_t node, std::size_t destination) const noexcept {
    return scale == 0.0 ? 0.0 : scale * distance(node, destination);
  }
};

struct HaversineContext {
  std::size_t node_count = 0;
  const double *latitudes = nullptr;  // Radians.
  const double *longitudes = nullptr; // Radians.
  const double *cos_latitudes = nullptr;
  double scale = 0.0;

  double distance(std::size_t from, std::size_t to) const noexcept {
    constexpr double earth_radius_metres = 6371000.0;
    const double sin_lat = std::sin((latitudes[to] - latitudes[from]) / 2.0);
    const double sin_lon = std::sin((longitudes[to] - longitudes[from]) / 2.0);
    const double a = sin_lat * sin_lat + cos_latitudes[from] *
                                             cos_latitudes[to] * sin_lon *
                                             sin_lon;

    // Round-off near antipodal points must not put asin outside its domain.
    return 2.0 * earth_radius_metres *
           std::asin(std::sqrt(std::clamp(a, 0.0, 1.0)));
  }

  double operator()(std::size_t node, std::size_t destination) const noexcept {
    return scale == 0.0 ? 0.0 : scale * distance(node, destination);
  }
};

} // namespace aequilibrae::paths::cpp::mvp
