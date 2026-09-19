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

// An origin corresponds to a contiguous block.
template <typename T> struct SelectLinkODOriginView {
  std::size_t set_count = 0;
  std::size_t destination_count = 0;
  std::size_t class_count = 0;
  T *data = nullptr;

  T *selection_data(std::size_t index) const noexcept {
    return destination_count && class_count
               ? data + index * destination_count * class_count
               : nullptr;
  }
};

template <typename T> struct SelectLinkODOutputsView {
  std::size_t origin_count = 0;
  std::size_t set_count = 0;
  std::size_t destination_count = 0;
  std::size_t class_count = 0;
  T *data = nullptr; // [origins, sets, destinations, classes]

  SelectLinkODOriginView<T> origin(std::size_t index) const noexcept {
    T *values = nullptr;
    if (set_count && destination_count && class_count) {
      values = data + index * set_count * destination_count * class_count;
    }
    return {set_count, destination_count, class_count, values};
  }

  void reset() const noexcept {
    if (origin_count && set_count && destination_count && class_count) {
      std::fill_n(data,
                  origin_count * set_count * destination_count * class_count,
                  T{0});
    }
  }
};

// Borrow the assignment's components for one run. This is a driver view;
// individual operation kernels receive only the component they write.
struct AoNOutputsView {
  LoadingOutputs<double> loading;
  SkimmingOutputsView<double> skimming;
  SelectLinkLoadingOutputsView<double> selected_loading;
  SelectLinkODOutputsView<double> selected_od;

  void reset() const noexcept {
    loading.reset();
    skimming.reset();
    selected_loading.reset();
    selected_od.reset();
  }
};

} // namespace aequilibrae::paths::cpp::mvp
