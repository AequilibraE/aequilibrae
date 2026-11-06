#pragma once

#include <cstring>
#include <type_traits>

#include "matrix_base.hpp"

namespace aequilibrae::matrix::cpp {

template <typename T> struct SparseMatrixOriginView {
  static_assert(std::is_floating_point_v<T>);

  std::size_t destination_count = 0;
  std::size_t field_count = 0;
  std::size_t stored_count = 0;
  const std::size_t *col_index = nullptr;
  T *data = nullptr;

  std::size_t entry_count() const noexcept { return stored_count; }

  std::size_t destination(std::size_t entry) const noexcept {
    return col_index[entry];
  }

  const T *values(std::size_t entry) const noexcept {
    return stored_count && field_count ? data + entry * field_count : nullptr;
  }
};

// Borrow canonical CSR indices and a packed [stored positions, fields] buffer
template <typename T> class SparseMatrix : public MatrixBase<T> {
public:
  // In a SparseMatrix, n_alloc stores the number of stored positions shared by
  // all cores. The non-zero nature of cells is not enforced.
  //
  // Columns in each row must be sorted and unique. The structure stays fixed
  // while views are in use. This view does not support insertion.
  //
  //
  //           row_index     col_index   cores (3 cores)      No entries
  //            ┌─────┐       ┌─────┐  ┌─────────────────┐     in row 0
  //            │     │       │     │  │  ┌───┬───┬───┐  │
  //   Row 0  │ │  0 ─┼───┬───┼► 0  │  │  │   │   │   ├┐ │  │ Row 1 has
  //          │ │     │   │   │     │  │  └───┴───┴───┘│ │  │ 3 entries
  //          │ │     │   │   │     │  │ ┌─────────────┘ │  │
  //          │ │     │   │   │     │  │ │┌───┬───┬───┐  │  │
  //   Row 1 ││ │  0 ─┼───┘   │  2  │  │ └►   │   │   ├┐ │  │
  //         │  │     │       │     │  │  └───┴───┴───┘│ │  │
  //         │  │     │       │     │  │ ┌─────────────┘ │  │
  //         │  │     │       │     │  │ │┌───┬───┬───┐  │  │
  //   Row 2 ││ │  3 ─┼───┐   │  5  │  │ └►   │   │   ├┐ │  │
  //          │ │     │   │   │     │  │  └───┴───┴───┘│ │
  //          │ │     │   │   │     │  │ ┌─────────────┘ │
  //          │ │     │   │   │     │  │ │┌───┬───┬───┐  │
  //   Row 3 ││ │  5 ─┼─┐ └───┼► 1  │  │ └►   │   │   ├┐ │  │ Row 2 has
  //         │  │     │ │     │     │  │  └───┴───┴───┘│ │  │ 2 entries
  //         │  │     │ │     │     │  │ ┌─────────────┘ │  │
  //         │  │     │ │     │     │  │ │┌───┬───┬───┐  │  │
  //         │  │  6 ─┼┐│     │  2  │  │ └►   │   │   ├┐ │  │
  //            │     │││     │     │  │  └───┴───┴───┘│ │
  //            └─────┘││     │     │  │ ┌─────────────┘ │
  //                   ││     │     │  │ │┌───┬───┬───┐  │
  //                   │└─────┼► 4  │  │ └►   │   │   │  │  │ Row 3 has
  //                   │      │     │  │  └───┴───┴───┘  │    1 entry
  //                   │      └─────┘  └─────────────────┘
  //                   └───────►
  //
  //   Last row_index value points          This example stores 3
  //   off the end of the col_index         cores of a 4x6 matrix
  //   and is the NNZ.
  //                                               012345
  //                                              ┌──────┐
  //   All fields of a cell are stored           0│      │
  //   contiguously in memory.                   1│. .  .│
  //                                             2│ ..   │
  //                                             3│    . │
  //   Example Python iteration.                  └──────┘
  //
  //   for row in range(origin_count):
  //       row_start = row_index[row]
  //       row_end = row_index[row + 1]
  //
  //       for col_idx in range(row_start, row_end):
  //           col = col_index[col_idx]
  //
  //           for field in range(field_count):
  //               value = data[col_idx * field_count + field]

  // row_index is of length origin_count + 1. row_index[i]:row_index[i + 1]
  // defines the range of indices for the values and column indices in each
  // core, and col_index.
  //
  // row_index[origin_count] is equal to n_alloc.
  size_t n_alloc = 0;
  size_t *row_index = nullptr;

  // Each core and col_index are of length n_alloc.
  size_t *col_index = nullptr;

  SparseMatrixOriginView<T> origin(std::size_t index) const noexcept {
    const auto start = row_index[index];
    const auto count = row_index[index + 1] - start;
    const auto *columns = count ? col_index + start : nullptr;
    T *values = count && this->field_count
                    ? this->data + start * this->field_count
                    : nullptr;
    return {this->destination_count, this->field_count, count, columns, values};
  }
};
} // namespace aequilibrae::matrix::cpp
