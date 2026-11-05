#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

#include "dense_matrix.hpp"
#include "sparse_matrix_builder.hpp"

namespace aequilibrae::paths::cpp::routing {

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

using aequilibrae::matrix::cpp::DenseMatrixOriginView;

// Borrow the complete contiguous output. Neither view owns its buffer.
template <typename T> struct SkimmingOutputsView {
  std::size_t origin_count = 0;
  std::size_t field_count = 0;
  std::size_t destination_count = 0;
  T *data = nullptr; // [origins, destinations, fields]

  // Borrow one origin's contiguous [destinations, fields] block.
  // Callers supply valid indices and ranges. Empty views do not offset
  // pointers.
  DenseMatrixOriginView<T> origin(std::size_t index) const noexcept {
    T *values = nullptr;
    if (field_count && destination_count) {
      values = data + index * field_count * destination_count;
    }
    return {destination_count, field_count, values};
  }

  void reset() const noexcept {
    if (origin_count && field_count && destination_count) {
      std::fill_n(data, origin_count * field_count * destination_count,
                  std::numeric_limits<T>::infinity());
    }
  }
};

// Selected link loads have the same per-set layout as ordinary link loads.
template <typename T> struct SelectLinkLoadingOutputsView {
  std::size_t set_count = 0;
  std::size_t link_count = 0;
  std::size_t class_count = 0;
  T *data = nullptr; // [sets, links, classes]

  LoadingOutputs<T> selection(std::size_t index) const noexcept {
    T *values = nullptr;
    if (link_count && class_count) {
      values = data + index * link_count * class_count;
    }
    return {link_count, class_count, values};
  }

  void reset() const noexcept {
    if (set_count && link_count && class_count) {
      std::fill_n(data, set_count * link_count * class_count, T{0});
    }
  }
};

using aequilibrae::matrix::cpp::SparseMatrixBuilderView;

// Borrow worker-local builders and bind the origin for one routing operation.
template <typename T> struct SelectLinkODOriginView {
  std::size_t set_count = 0;
  std::size_t origin_index = 0;
  SparseMatrixBuilderView<T> *builders = nullptr;

  void append(std::size_t set, std::size_t destination,
              const T *values) const noexcept {
    builders[set].append(origin_index, destination, values);
  }
};

template <typename T> struct SelectLinkODOutputsView {
  std::size_t set_count = 0;
  SparseMatrixBuilderView<T> *builders = nullptr;

  SelectLinkODOriginView<T> origin(std::size_t index) const noexcept {
    return {set_count, index, builders};
  }

  void reset() const noexcept {
    for (std::size_t set = 0; set < set_count; ++set)
      builders[set].reset();
  }
};

// Borrow the assignment's components for one run. This is a driver view;
// individual operation kernels receive only the component they write.
template <typename T> struct AoNOutputsView {
  LoadingOutputs<T> loading;
  SkimmingOutputsView<T> skimming;
  SelectLinkLoadingOutputsView<T> selected_loading;
  SelectLinkODOutputsView<T> selected_od;

  void reset() const noexcept {
    loading.reset();
    skimming.reset();
    selected_loading.reset();
    selected_od.reset();
  }
};

} // namespace aequilibrae::paths::cpp::routing
