#pragma once

#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <iomanip>
#include <sstream>
#include <string>

#include "matrix_base.hpp"

namespace aequilibrae::matrix::cpp {

template <typename T> class SparseMatrix : public MatrixBase<T> {
public:
  // In a SparseMatrix, n_alloc stores the number of non-zero cells across all
  // cores. The non-zero nature of cells is not enforced.
  //
  // It supports coordinated insertion from a multi-threaded environment (thread
  // safety is not enforced).
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
  //   All values of a core are stored           0│      │
  //   contiguously in memory.                   1│. .  .│
  //                                             2│ ..   │
  //                                             3│    . │
  //   Example Python iteration.                  └──────┘
  //
  //   for row in range(n_rows):
  //       row_start = row_index[row]
  //       row_end = row_index[row + 1]
  //
  //       for col_idx in range(row_start, row_end):
  //           col = col_index[col_idx]
  //
  //           for core in range(n_cores):
  //               value = cores[col_idx * n_cores + core]

  // row_index is of length n_rows + 1. row_index[i]:row_index[i + 1] defines
  // the range of indices for the values and column indices in each core, and
  // col_index.
  //
  // row_index[n_rows] is equal to n_alloc.
  size_t *row_index;

  // Each core and col_index are of length n_alloc.
  size_t *col_index;

  // // The insertion cache is intended to provide a common way to write values
  // // into the matrix between a sparse and dense matrix.
  // //
  // // The 'insertion_cache' contains a vector per row, with tuples of the
  // column
  // // number and pointers to where data can be written without concern. This
  // is
  // // separated by row to allow lock free row based parallelised usage.
  // //
  // // In the dense case, each tuple will contain a pointer to the start of the
  // // appropriate segment within the real matrix.
  // //
  // // In the sparse case, each tuple will contain a pointer to a new (zero'd
  // and
  // // initialised) memory segment. These segments will then be merged into the
  // // CSR matrix at a later time.
  // //
  // // Because of this difference in implementation. Users should always
  // // accumulate into the returned buffers. There is no guarantee the returned
  // // buffers
  // //
  // //   1: have data,
  // //   2: are canonical, and
  // //   3: are the same pointers been returned previously.
  // //
  // // The returned pointers should be considered managed by the matrix and are
  // // not to be stored or freed.
  // std::vector<std::vector<std::tuple<size_t, T *>>> insertion_cache;
  // bool dirty = false;

  /* ------------------------- Iterators ------------------------- */

  // This iterator makes the iteration over all cells a little nicer. Compared
  // to it's dense counter part it needs to do a little more bookkeeping, but
  // not much. Again it is essentially a counter, but sometimes it skips rows.
  class iterator {
    // private:
  public:
    const SparseMatrix<T> &matrix_;
    size_t row_pos_;
    size_t col_pos_;

    iterator(const SparseMatrix<T> &matrix, size_t row)
        : matrix_(matrix), row_pos_(row), col_pos_(0) {

      // Find the first non-zero row
      while (row_pos_ < matrix_.n_rows &&
             matrix_.row_index[row_pos_] == matrix_.row_index[row_pos_ + 1]) {
        row_pos_++;
      }

      // If this is the end, then row_pos_ is n_rows, and col_pos_ is n_alloc
      col_pos_ = matrix_.row_index[row_pos_];
    }

    // Move constructor
    iterator(iterator &&other) noexcept
        : matrix_(other.matrix_), row_pos_(other.row_pos_),
          col_pos_(other.col_pos_) {}

    // Copy constructor
    iterator(const iterator &other)
        : matrix_(other.matrix_), row_pos_(other.row_pos_),
          col_pos_(other.col_pos_) {}

    // Move assignment operator
    iterator &operator=(iterator &&other) noexcept {
      if (this != &other) {
        row_pos_ = other.row_pos_;
        col_pos_ = other.col_pos_;
      }
      return *this;
    }

    iterator(iterator &other)
        : matrix_(other.matrix_), row_pos_(other.row_pos_),
          col_pos_(other.col_pos_) {}

    iterator &operator++() {
      ++col_pos_;

      // If the column index is equal to the end of the row then we're
      // finished the row and need to increment it till we find the next
      // non-zero row.
      while (row_pos_ < matrix_.n_rows &&
             col_pos_ == matrix_.row_index[row_pos_ + 1]) {
        row_pos_++;
      }

      return *this;
    }

    bool operator==(const iterator &other) const {
      // FIXME: should we compare matrix_ as well?
      return row_pos_ == other.row_pos_ && col_pos_ == other.col_pos_;
    }

    bool operator!=(const iterator &other) const { return !(*this == other); }

    /* row() and col() can be used to obtain the currently looked at
     * values respective row and column. Their results between iterations may
     * change in a non-contiguous manner. */
    size_t row() const {
      if (row_pos_ >= matrix_.n_alloc) {
        throw std::out_of_range("Iterator position out of range");
      }

      return row_pos_;
    }

    size_t col() const { return matrix_.col_index[col_pos_]; }

    std::pair<size_t, size_t> operator*() const {
      return std::make_pair(row(), col());
    }

    T *values() const { return &matrix_.cores[col_pos_ * matrix_.n_cores]; }
  };

  iterator begin() const { return iterator(*this, 0); }
  iterator end() const { return iterator(*this, this->n_rows); }
};
} // namespace aequilibrae::matrix::cpp
