#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace aequilibrae::matrix::cpp {

template <typename T> struct COOStorage {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
  std::vector<std::size_t> rows, columns;
  std::vector<T> values; // [entries, fields]
};

template <typename T> struct SparseMatrixBuilderView {
  std::size_t origin_count = 0;
  std::size_t destination_count = 0;
  std::size_t field_count = 0;
  COOStorage<T> *storage = nullptr;

  std::size_t entry_count() const noexcept { return storage->rows.size(); }

  std::size_t row(std::size_t entry) const noexcept {
    return storage->rows[entry];
  }
  std::size_t column(std::size_t entry) const noexcept {
    return storage->columns[entry];
  }

  const T *values(std::size_t entry) const noexcept {
    return field_count ? storage->values.data() + entry * field_count : nullptr;
  }

  void reset() const noexcept {
    if (!storage)
      return;

    storage->rows.clear();
    storage->columns.clear();
    storage->values.clear();
  }

  // Kernels supply valid coordinates and one complete field row. Allocation
  // failure terminates the process; appends do not recover partial entries.
  void append_scaled(std::size_t origin, std::size_t destination, const T *data,
                     T weight) const noexcept {
    storage->rows.push_back(origin);
    storage->columns.push_back(destination);

    for (std::size_t field = 0; field < field_count; ++field)
      storage->values.push_back(data[field] * weight);
  }

  void append(std::size_t origin, std::size_t destination,
              const T *data) const noexcept {
    storage->rows.push_back(origin);
    storage->columns.push_back(destination);

    for (std::size_t field = 0; field < field_count; ++field)
      storage->values.push_back(data[field]);
  }

  // Reserve for a total entry count, allowing a reducer to size the destination
  // once after counting all worker-local builders.
  void reserve_entries(std::size_t total_entries) const {
    if (total_entries > storage->rows.max_size() ||
        total_entries > storage->columns.max_size() ||
        (field_count &&
         total_entries > storage->values.max_size() / field_count))
      throw std::length_error("COO builder capacity exceeded");

    storage->rows.reserve(total_entries);
    storage->columns.reserve(total_entries);
    storage->values.reserve(total_entries * field_count);
  }

  void extend(const SparseMatrixBuilderView &source) const {
    if (origin_count != source.origin_count ||
        destination_count != source.destination_count ||
        field_count != source.field_count || storage == source.storage)
      throw std::invalid_argument("incompatible or aliased COO builders");

    storage->rows.insert(storage->rows.end(), source.storage->rows.begin(),
                         source.storage->rows.end());
    storage->columns.insert(storage->columns.end(),
                            source.storage->columns.begin(),
                            source.storage->columns.end());
    storage->values.insert(storage->values.end(),
                           source.storage->values.begin(),
                           source.storage->values.end());
  }
};

} // namespace aequilibrae::matrix::cpp
