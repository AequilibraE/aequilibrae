#pragma once

#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
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

  // The insertion cache is intended to provide a common way to write values
  // into the matrix between a sparse and dense matrix.
  //
  // The 'insertion_cache' contains a vector per row, of a vector of tuples of
  // the column number, and pointers to where data can be written without
  // concern. This is separated first by row allow lock free row based
  // parallelised usage.
  //
  // In the dense case, these pairs will contain pointers to the start of the
  // appropriate segment within the real matrix.
  //
  // In the sparse case, these pairs will contain pointers to new (zero'd and
  // initialised) memory segments. These individual segments will then be
  // merged together into the CSR matrix at a later time.
  //
  // Because of this difference in implementation. Users should always
  // accumulate into the returned buffers. There is no guarantee the returned
  // buffers
  //
  //   1: have data,
  //   2: are canonical, and
  //   3: are the same pointers been returned previously.
  //
  // The returned pointers should be considered managed by the matrix and are
  // not to be stored or freed.
  std::vector<std::vector<std::tuple<size_t, T *>>> insertion_cache;
  bool dirty = false;

  // Allows for over-allocation
  size_t true_n_alloc;

  SparseMatrix() = delete;
  explicit SparseMatrix(const SparseMatrix &other)
      : MatrixBase<T1, T2>(other.n_rows, other.n_cols, other.n_cores,
                           other.n_alloc),
        row_index(nullptr), col_index(nullptr), insertion_cache(other.n_rows),
        true_n_alloc(other.n_alloc) {
    this->row_index = memory::acquire<size_t>(this->n_rows + 1);
    std::copy(other.row_index, other.row_index + other.n_rows + 1,
              this->row_index);

    this->col_index = memory::acquire<size_t>(this->n_alloc);
    std::copy(other.col_index, other.col_index + other.n_alloc,
              this->col_index);

    this->cores.first =
        memory::acquire<T1>(this->n_alloc * this->n_cores.first);
    this->cores.second =
        memory::acquire<T2>(this->n_alloc * this->n_cores.second);

    std::copy(other.cores.first,
              other.cores.first + other.n_alloc * other.n_cores.first,
              this->cores.first);
    std::copy(other.cores.second,
              other.cores.second + other.n_alloc * other.n_cores.second,
              this->cores.second);
  };

  SparseMatrix &operator=(const SparseMatrix &) = delete;

  // Allocate an sparse matrix, fields are initialised but may be
  // null/pointers to nothing. By default the matrix is empty
  SparseMatrix(size_t rows, size_t cols, std::pair<size_t, size_t> n_cores,
               size_t n_alloc = 0)
      : MatrixBase<T1, T2>(rows, cols, n_cores, n_alloc), row_index(nullptr),
        col_index(nullptr), insertion_cache(rows), true_n_alloc(n_alloc) {
    row_index = memory::acquire<size_t>(this->n_rows + 1);
    std::fill(row_index, row_index + this->n_rows, 0);
    row_index[this->n_rows] = this->n_alloc;

    col_index = memory::acquire<size_t>(this->n_alloc);
    this->cores.first =
        memory::acquire<T1>(this->n_alloc * this->n_cores.first);
    this->cores.second =
        memory::acquire<T2>(this->n_alloc * this->n_cores.second);
  }

  SparseMatrix(SparseMatrix &&other) noexcept
      : MatrixBase<T1, T2>(std::move(other)), // Move the base class
        row_index(other.row_index), col_index(other.col_index),
        insertion_cache(std::move(other.insertion_cache)),
        true_n_alloc(other.n_alloc) {
    other.row_index = nullptr;
    other.col_index = nullptr;
  }

  // Constructor from COO format
  SparseMatrix(const size_t nnz, const size_t rows, const size_t cols,
               const std::pair<size_t, size_t> n_cores,
               const size_t *coo_row_index, const size_t *coo_col_index,
               std::pair<T1 **, T2 **> coo_values, bool verify)
      : MatrixBase<T1, T2>(rows, cols, n_cores, 0), row_index(nullptr),
        col_index(nullptr), insertion_cache(rows), true_n_alloc(0) {

    if (coo_row_index == nullptr || coo_col_index == nullptr) {
      throw std::logic_error("COO index pointers are null");
    } else if (nnz == 0) {
      throw std::length_error(
          "Cannot have a core with a row or column of length 0");
    }

    if (verify) {
      if (std::any_of(
              coo_row_index, coo_row_index + nnz,
              [rows](const auto &element) { return element >= rows; }) ||
          std::any_of(
              coo_col_index, coo_col_index + nnz,
              [cols](const auto &element) { return element >= cols; })) {
        throw std::logic_error("COO entry exceeds specified dimensions");
      }

      for (size_t i = 0; i < n_cores.first; i++) {
        if (coo_values.first[i] == nullptr) {
          throw std::logic_error("Null pointer found in coo_values");
        }
      }

      for (size_t i = 0; i < n_cores.second; i++) {
        if (coo_values.second[i] == nullptr) {
          throw std::logic_error("Null pointer found in coo_values");
        }
      }

      for (size_t i = 0; i < nnz - 1; ++i) {
        if (coo_row_index[i] > coo_row_index[i + 1] ||
            (coo_row_index[i] == coo_row_index[i + 1] &&
             coo_col_index[i] >= coo_col_index[i + 1])) {
          throw std::logic_error("COO values are not in canonical form");
        }
      }
    }

    // The number of cells we'll allocate is the length of the column index
    this->n_alloc = nnz;
    this->true_n_alloc = nnz;
    this->n_cores = n_cores;

    this->row_index = memory::acquire<size_t>(this->n_rows + 1);
    this->col_index = memory::acquire<size_t>(this->n_alloc);

    this->cores.first =
        memory::acquire<T1>(this->n_alloc * this->n_cores.first);
    this->cores.second =
        memory::acquire<T2>(this->n_alloc * this->n_cores.second);

    // CSR construction

    std::fill(this->row_index, this->row_index + this->n_rows, 0);

    // Initially we use the row_index as a buffer to count the number of entries
    // per row.
    for (size_t n = 0; n < this->n_alloc; n++) {
      this->row_index[coo_row_index[n]]++;
    }

    // We then turn those row counts into a cumulative sum, but shuffled down
    // one element. This forms the row_index correctly, however we're not done
    // with it.
    for (size_t i = 0, cumulative_sum = 0; i < this->n_rows; i++) {
      size_t tmp = this->row_index[i];
      this->row_index[i] = cumulative_sum;
      cumulative_sum += tmp;
    }
    this->row_index[this->n_rows] = nnz;

    // We then build the column index, and place the values into the correct
    // spots within the cores. We keep track of where to place the column and
    // column values by incrementing the row_index buffer. This essentially
    // undoes the shuffle down above, so we redo that shuffle last.
    for (size_t n = 0; n < nnz; n++) {
      // For each row, we grab it's label and index.
      size_t row = coo_row_index[n];
      size_t dest = this->row_index[row];

      // Then write the column value into col_index at the index from the
      // row_index (which has been incremented by the number of columns in this
      // row seen so far).
      this->col_index[dest] = coo_col_index[n];

      // We then do the same for each value within the cores.
      for (size_t core = 0; core < this->n_cores.first; core++) {
        this->cores.first[this->n_cores.first * dest + core] =
            coo_values.first[core][n];
      }

      for (size_t core = 0; core < this->n_cores.second; core++) {
        this->cores.second[this->n_cores.second * dest + core] =
            coo_values.second[core][n];
      }

      this->row_index[row]++;
    }

    // Now we re-shuffle the row_index values down.
    for (size_t i = 0, last = 0; i <= this->n_rows; i++) {
      size_t tmp = this->row_index[i];
      this->row_index[i] = last;
      last = tmp;
    }
  }

  ~SparseMatrix() noexcept {
    this->clear_cache();

    memory::release(this->row_index);
    memory::release(this->col_index);
    memory::release(this->cores.first);
    memory::release(this->cores.second);
  }

  friend void swap(SparseMatrix &first, SparseMatrix &second) {
    using std::swap;

    swap(static_cast<MatrixBase<T1, T2> &>(first),
         static_cast<MatrixBase<T1, T2> &>(second));
    swap(first.row_index, second.row_index);
    swap(first.col_index, second.col_index);
    swap(first.insertion_cache, second.insertion_cache);
  }

  // Copying a sparse matrix with a different spatial index is slow, instead of
  // checking if they're compatible, then cloning, or memcpy the values, we just
  // clone it and swap the new clone onto the existing matrix.
  void copy(SparseMatrix &other) {
    if (this->n_rows == other.n_rows && this->n_cols == other.n_cols &&
        this->n_cores == other.n_cores) {
      SparseMatrix<T1, T2> tmp(other);
      swap(*this, tmp);
      this->insert_cache(other);
    } else {
      throw std::logic_error(
          "matrices are not compatible, check shape and core counts");
    }
  }

  // Create a new matrix with the same sparsity structure by all zeros
  SparseMatrix<T1, T2> *like() const {
    auto new_mat = new SparseMatrix<T1, T2>(this->n_rows, this->n_cols,
                                            this->n_cores, this->n_alloc);

    std::copy(this->row_index, this->row_index + this->n_rows + 1,
              access::rwp(new_mat->row_index));

    std::copy(this->col_index, this->col_index + this->n_alloc,
              access::rwp(new_mat->col_index));

    std::fill(new_mat->cores.first,
              new_mat->cores.first + new_mat->n_alloc * new_mat->n_cores.first,
              0.0);
    std::fill(new_mat->cores.second,
              new_mat->cores.second +
                  new_mat->n_alloc * new_mat->n_cores.second,
              0.0);

    return new_mat;
  }

  // Construct from an existing CSR index in pointer form (e.g. memory views)
  // The length of row_index must be n_rows, the sentinel value at
  // row_index[n_rows] is not copied, we set it ourselves.
  void set_index(const size_t *const row_index, const size_t *const col_index,
                 size_t nnz, bool verify) {
    if (row_index == nullptr || col_index == nullptr) {
      return; // FIXME: should we throw something instead?
    } else if (nnz == 0) {
      throw std::length_error(
          "Cannot have a core with a row or column of length 0");
    }

    if (verify) {
      if (row_index[this->n_rows] != nnz) {
        throw std::logic_error("row_index not properly terminated.");
      }

      for (size_t row = 0; row < this->n_rows; row++) {
        size_t row_start = row_index[row];
        size_t row_end = row_index[row + 1];

        if (!(row_end >= row_start)) {
          throw std::logic_error("Row indices must be ascending");
        } else if (row_start > nnz) {
          throw std::logic_error("A value greater than the row_index sentinel "
                                 "value must not appear in the index.");
        }

        for (size_t i = row_start; i < row_end; i++) {
          size_t col = col_index[i];

          if (col > this->n_cols) {
            throw std::logic_error(
                "Column value exceeds previously supplied bounds");
          }
        }
      }
    }

    // The number of cells we'll allocate is the length of the column index
    this->n_alloc = nnz;
    this->true_n_alloc = nnz;

    // Copy the row index, setting the sentential value
    memory::release(this->row_index);
    access::rwp(this->row_index) = memory::acquire<size_t>(this->n_rows + 1);
    std::memcpy(access::rwp(this->row_index), row_index,
                this->n_rows * sizeof(size_t));
    access::rwp(this->row_index)[this->n_rows] = this->n_alloc;

    // Copy the col index
    memory::release(this->col_index);
    access::rwp(this->col_index) = memory::acquire<size_t>(this->n_alloc);
    std::memcpy(access::rwp(this->col_index), col_index,
                this->n_alloc * sizeof(size_t));
  }

  void set_index(const size_t *const row_index, const size_t *const col_index,
                 size_t col_length) {
    this->set_index(row_index, col_index, col_length, true);
  }

  RowSlice<T1, T2> row_slice(size_t row) {
    return RowSlice(row, this->row_index[row + 1] - this->row_index[row], this);
  }

  // Find the row that an index in the range [0, n_alloc) corresponds to.
  size_t row_from_index(const size_t column_index) const {
    if (column_index >= this->n_alloc) {
      throw std::logic_error(
          "Requested column_index is larger than the length of col_index.");
    }

    auto begin = row_index;
    auto end = begin + this->n_rows + 1;

    auto iter = std::upper_bound(begin, end, column_index);

    if (begin < iter && iter < end) {
      // std::upper_bound finds the first value i, such that column_index < i.
      // In the case of a row_index, that is the row + 1 that column_index
      // belongs to.
      return iter - 1 - begin;
    } else if (iter == begin) {
      // If std::upper_bound returns begin then the value is less than 0
      // (row_index[0] is always 0), but size_t is unsigned. This should be
      // impossible.
      throw std::logic_error("Requested column_index is less than 0.");
    } else if (iter == end) {
      // If std::upper_bound returns end then the value wasn't found.
      // This should be impossible.
      throw std::logic_error("Requested column_index is larger than the "
                             "row_index sentinel.");
    } else {
      throw std::logic_error("Reverse index lookup failed");
    }
  }

  size_t col_from_index(const size_t index) const {
    return this->col_index[index];
  }

  size_t col(const size_t row, const size_t offset) const {
    return this->col_index[this->row_index[row] + offset];
  }

  template <typename T>
  T &get(const size_t row, const size_t offset, const size_t core) const {
    auto idx = this->row_index[row] + offset;
    return this->template get<T>(idx, core);
  };

  T &get_first(const size_t row, const size_t offset, const size_t core) const {
    return this->template get<T>(row, offset, core);
  }

  T2 &get_second(const size_t row, const size_t offset,
                 const size_t core) const {
    return this->template get<T2>(row, offset, core);
  }

  template <typename T>
  T *get_address(const size_t idx, const size_t core) const {
    if constexpr (std::is_same_v<T, T>) {
      return std::get<T *>(this->cores) + (idx * this->n_cores.first + core);
    } else if constexpr (std::is_same_v<T, T2>) {
      return std::get<T2 *>(this->cores) + (idx * this->n_cores.second + core);
    } else {
      static_assert(std::is_same_v<T, T1> || std::is_same_v<T, T2>,
                    "Unsupported type.");
    }
  };

  template <typename T> T &get(const size_t idx, const size_t core) const {
    if constexpr (std::is_same_v<T, T>) {
      return *this->get_address<T>(idx, core);
    } else if constexpr (std::is_same_v<T, T2>) {
      return *this->get_address<T2>(idx, core);
    } else {
      static_assert(std::is_same_v<T, T1> || std::is_same_v<T, T2>,
                    "Unsupported type.");
    }
  };

  T &get_first(const size_t idx, const size_t core) const {
    return this->template get<T>(idx, core);
  }

  T2 &get_second(const size_t idx, const size_t core) const {
    return this->template get<T2>(idx, core);
  }

  // TODO reassess the unified methods, perhaps they'll find a use
  inline T get_unifed(const size_t offset, const size_t core) {
    if (core < this->n_cores.first) {
      return this->get_first(col_index, core);
    } else {
      return this->get_second(col_index, core - this->n_cores.first);
    }
  }

  inline void set_unifed(const size_t col_index, const size_t core,
                         const T value) {
    if (core < this->n_cores.first) {
      this->get_first(col_index, core) = value;
    } else {
      this->get_second(col_index, core - this->n_cores.first) = value;
    }
  }

  bool cell_is_nonzero(const size_t idx) const {
    for (size_t j = 0; j < this->template core_count<T1>(); j++) {
      if (this->template get<T>(idx, j) != 0.0) {
        return true;
      }
    }

    for (size_t j = 0; j < this->template core_count<T2>(); j++) {
      if (this->template get<T2>(idx, j) != 0.0) {
        return true;
      }
    }

    return false;
  }

  // It's useful to know is a row is truly nonzero.
  bool row_is_nonzero(const size_t row) const {
    if (row >= this->n_rows) {
      throw std::logic_error("row index out of bounds");
    } else if (row_index[row + 1] - row_index[row] == 0) {
      return false;
    }

    for (size_t i = row_index[row]; i < row_index[row + 1]; i++) {
      for (size_t j = 0; j < this->template core_count<T1>(); j++) {
        if (this->template get<T>(i, j) != 0.0) {
          return true;
        }
      }

      for (size_t j = 0; j < this->template core_count<T2>(); j++) {
        if (this->template get<T2>(i, j) != 0.0) {
          return true;
        }
      }
    }

    return false;
  }

  // Mark the non zero columns within a particularly row inside the 'out'
  // buffer. This is used for destination set construction
  size_t mark_nonzero_row(const size_t row, unsigned char *out,
                          size_t length) const {
    if (row >= this->n_rows) {
      throw std::logic_error("row index out of bounds");
    } else if (length < this->n_cols) {
      throw std::logic_error(
          "length of array to mark is shorted than the number of columns");
    }

    size_t count = 0;
    std::fill(out, out + length, false);

    for (size_t i = row_index[row]; i < row_index[row + 1]; i++) {
      bool found = false;

      for (size_t j = 0; j < this->template core_count<T1>(); j++) {
        if (this->template get<T>(i, j) > 0.0) {
          found = true;
          break;
        }
      }

      for (size_t j = 0; !found && j < this->template core_count<T2>(); j++) {
        if (this->template get<T2>(i, j) > 0.0) {
          found = true;
          break;
        }
      }

      if (found) {
        out[this->col_index[i]] = 1;
        count++;
      }
    }

    return count;
  }

  /* ------------------------- Iterators ------------------------- */

  // This iterator makes the iteration over all cells a little nicer. Compared
  // to it's dense counter part it needs to do a little more bookkeeping, but
  // not much. Again it is essentially a counter, but sometimes it skips rows.
  class iterator {
    // private:
  public:
    const SparseMatrix<T1, T2> &matrix_;
    size_t row_pos_;
    size_t col_pos_;

    iterator(const SparseMatrix<T1, T2> &matrix, size_t row)
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
      return std::make_pair(row_pos_, matrix_.col_index[col_pos_]);
    }

    std::pair<T1 *, T2 *> values() const {
      return std::make_pair(
          &matrix_.cores.first[col_pos_ * matrix_.n_cores.first],
          &matrix_.cores.second[col_pos_ * matrix_.n_cores.second]);
    }
  };

  iterator begin() const { return iterator(*this, 0); }
  iterator end() const { return iterator(*this, this->n_rows); }

  // Returned pointers may be invalidated by successive insert calls
  std::pair<T1 *, T2 *> insert(size_t row, size_t col) {
    this->dirty = true;
    T *tmp1 = memory::acquire<T1>(this->n_cores.first);
    std::fill(tmp1, tmp1 + this->n_cores.first, 0.0);

    T2 *tmp2 = memory::acquire<T2>(this->n_cores.second);
    std::fill(tmp2, tmp2 + this->n_cores.second, 0.0f);

    insertion_cache[row].emplace_back(col, tmp1, tmp2);

    return std::make_pair(tmp1, tmp2);
  }

  // Insert the cache of another Sparse matrix into this one. This is useful
  // when copying the matrix
  void insert_cache(const SparseMatrix &other) {
    if (this->n_rows == other.n_rows && this->n_cols == other.n_cols &&
        this->n_cores == other.n_cores) {
      for (size_t i = 0; i < insertion_cache.size(); i++) {
        auto &row = other.insertion_cache[i];
        for (auto &col : row) {
          auto tmp = this->insert(i, std::get<0>(col));

          std::copy(std::get<1>(col), std::get<1>(col) + this->n_cores.first,
                    tmp.first);
          std::copy(std::get<2>(col), std::get<2>(col) + this->n_cores.second,
                    tmp.second);
        }
      }
    } else {
      throw std::logic_error(
          "matrices are not compatible, check shape and core counts");
    }
  }

  // Flushing the insertion cache is not a particularly optimised operation
  // ATM, however it's not often used for the current use-cases. The process
  // involves going to COO, then making a new CSR matrix.
  void flush_to_csr(void) {
    if (!this->dirty) {
      return;
    }

    size_t nnz = 0;
    for (auto &row : insertion_cache) {
      nnz += row.size();
    }

    // Convert CSR to COO
    std::vector<size_t> coo_row_indices;
    std::vector<size_t> coo_col_indices;
    std::vector<std::pair<T1 *, T2 *>> coo_values;

    coo_row_indices.reserve(this->n_alloc + nnz);
    coo_col_indices.reserve(this->n_alloc + nnz);
    coo_values.reserve(this->n_alloc + nnz);

    // Extract nonzero entries from the existing CSR matrix
    for (auto iter = this->begin(), end = this->end(); iter != end; ++iter) {
      coo_row_indices.push_back(iter.row());
      coo_col_indices.push_back(iter.col());
      coo_values.emplace_back(iter.values());
    }

    // Concatenate COO entries
    for (size_t i = 0; i < insertion_cache.size(); i++) {
      auto &row = insertion_cache[i];
      for (auto &col : row) {
        coo_row_indices.push_back(i);
        coo_col_indices.push_back(std::get<0>(col));
        coo_values.emplace_back(std::get<1>(col), std::get<2>(col));
      }
    }

    // Arg sort entries
    std::vector<size_t> indices(coo_row_indices.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
      return std::tie(coo_row_indices[i], coo_col_indices[i]) <
             std::tie(coo_row_indices[j], coo_col_indices[j]);
    });

    std::vector<size_t> sorted_row_indices;
    std::vector<size_t> sorted_col_indices;
    std::vector<std::pair<T1 *, T2 *>> sorted_values;

    sorted_row_indices.reserve(coo_row_indices.size());
    sorted_col_indices.reserve(coo_col_indices.size());
    sorted_values.reserve(coo_values.size());

    for (size_t idx : indices) {
      sorted_row_indices.push_back(coo_row_indices[idx]);
      sorted_col_indices.push_back(coo_col_indices[idx]);
      sorted_values.push_back(coo_values[idx]);
    }

    // Merge Duplicates
    std::vector<size_t> final_row_indices;
    std::vector<size_t> final_col_indices;
    std::pair<std::vector<std::vector<T1>>, std::vector<std::vector<T2>>>
        final_values;

    final_values.first.resize(this->n_cores.first);
    final_values.second.resize(this->n_cores.second);

    for (size_t i = 0; i < sorted_row_indices.size(); ++i) {
      if (!final_row_indices.empty() &&
          final_row_indices.back() == sorted_row_indices[i] &&
          final_col_indices.back() == sorted_col_indices[i]) {
        // Merge values for duplicate entries

        std::vector<std::vector<T1>> &values1 = final_values.first;
        std::vector<std::vector<T2>> &values2 = final_values.second;

        for (size_t j = 0; j < this->n_cores.first; j++) {
          values1[j].back() += sorted_values[i].first[j];
        }

        for (size_t j = 0; j < this->n_cores.second; j++) {
          values2[j].back() += sorted_values[i].second[j];
        }

      } else {
        // Add new entry
        final_row_indices.push_back(sorted_row_indices[i]);
        final_col_indices.push_back(sorted_col_indices[i]);

        for (size_t j = 0; j < this->n_cores.first; j++) {
          final_values.first[j].push_back(sorted_values[i].first[j]);
        }

        for (size_t j = 0; j < this->n_cores.second; j++) {
          final_values.second[j].push_back(sorted_values[i].second[j]);
        }
      }
    }

    std::pair<T1 **, T2 **> cores(memory::acquire<T1 *>(this->n_cores.first),
                                  memory::acquire<T2 *>(this->n_cores.second));

    for (size_t j = 0; j < this->n_cores.first; j++) {
      cores.first[j] = final_values.first[j].data();
    }

    for (size_t j = 0; j < this->n_cores.second; j++) {
      cores.second[j] = final_values.second[j].data();
    }

    auto tmp_mat = SparseMatrix<T1, T2>(
        final_row_indices.size(), this->n_rows, this->n_cols, this->n_cores,
        final_row_indices.data(), final_col_indices.data(), cores, true);

    swap(*this, tmp_mat);

    this->clear_cache();

    memory::release(cores.first);
    memory::release(cores.second);
    this->dirty = false;
  };

  void clear() {
    this->clear_cache();

    auto tmp = SparseMatrix<T1, T2>(this->n_rows, this->n_cols, this->n_cores);
    swap(*this, tmp);
  }

  void clear_cache() {
    for (auto &row : insertion_cache) {
      for (auto &col : row) {
        memory::release(std::get<1>(col));
        memory::release(std::get<2>(col));
      }
      row.clear();
    }
  }

  // Reduce memory usage to match the exact number of non-zero elements
  void shrink_to_fit() {
    // 1. Ensure all pending insertions are merged into the CSR structure.
    // This might temporarily increase memory usage but is necessary to know the
    // final size.
    if (this->dirty) {
      this->flush_to_csr();
    }

    // The last element of row_index tells us exactly how many elements are
    // currently in the CSR.
    size_t actual_nnz = this->row_index[this->n_rows];

    // If the allocated size matches the actual size, there is nothing to do.
    if (actual_nnz == this->true_n_alloc) {
      return;
    }

    // 2. Reallocate col_index to the exact size
    size_t *new_col_index = memory::acquire<size_t>(actual_nnz);
    std::copy(this->col_index, this->col_index + actual_nnz, new_col_index);
    memory::release(this->col_index);
    this->col_index = new_col_index;

    // 3. Reallocate cores.first (T1)
    if (this->n_cores.first > 0) {
      T *new_core_t1 = memory::acquire<T1>(actual_nnz * this->n_cores.first);
      std::copy(this->cores.first,
                this->cores.first + (actual_nnz * this->n_cores.first),
                new_core_t1);
      memory::release(this->cores.first);
      this->cores.first = new_core_t1;
    }

    // 4. Reallocate cores.second (T2)
    if (this->n_cores.second > 0) {
      T2 *new_core_t2 = memory::acquire<T2>(actual_nnz * this->n_cores.second);
      std::copy(this->cores.second,
                this->cores.second + (actual_nnz * this->n_cores.second),
                new_core_t2);
      memory::release(this->cores.second);
      this->cores.second = new_core_t2;
    }

    // 5. Update allocation counters
    this->n_alloc = actual_nnz;
    this->true_n_alloc = actual_nnz;

    // 6. Force the insertion_cache vectors to release their capacity
    // std::vector::clear() does not reduce capacity, shrink_to_fit() does.
    for (auto &row : this->insertion_cache) {
      row.shrink_to_fit();
    }
    this->insertion_cache.shrink_to_fit();
  }

  std::string status() const {
    std::stringstream ss;

    // --- Helper for formatting bytes ---
    auto f = [](size_t bytes) -> std::string {
      const char *suffixes[] = {"B", "KiB", "MiB", "GiB", "TiB"};
      int i = 0;
      double dblBytes = static_cast<double>(bytes);
      if (bytes > 1024) {
        for (i = 0; (bytes / 1024) > 0 && i < 4; i++, bytes /= 1024)
          dblBytes = bytes / 1024.0;
      }
      std::stringstream bss;
      bss << std::fixed << std::setprecision(2) << dblBytes << " "
          << suffixes[i];
      return bss.str();
    };

    // --- 1. CSR Data Usage ---
    size_t csr_t1_mem = this->n_alloc * this->n_cores.first * sizeof(T1);
    size_t csr_t2_mem = this->n_alloc * this->n_cores.second * sizeof(T2);
    size_t true_csr_t1_mem =
        this->true_n_alloc * this->n_cores.first * sizeof(T1);
    size_t true_csr_t2_mem =
        this->true_n_alloc * this->n_cores.second * sizeof(T2);
    size_t size_of_self = sizeof(SparseMatrix<T1, T2>);

    // --- 2. Cache Data Usage & Overhead ---
    size_t cache_entries = 0;
    size_t cache_struct_overhead = 0; // Vector + Tuple overhead

    // Outer vector overhead
    cache_struct_overhead +=
        this->insertion_cache.capacity() *
        sizeof(std::vector<std::tuple<size_t, T1 *, T2 *>>);

    for (const auto &row_vec : this->insertion_cache) {
      cache_entries += row_vec.size();
      // Inner vector/tuple overhead
      cache_struct_overhead +=
          row_vec.capacity() * sizeof(std::tuple<size_t, T1 *, T2 *>);
    }

    size_t cache_t1_mem = cache_entries * this->n_cores.first * sizeof(T1);
    size_t cache_t2_mem = cache_entries * this->n_cores.second * sizeof(T2);

    // --- 3. Index Overhead (CSR) ---
    size_t index_overhead = (this->n_rows + 1) * sizeof(size_t); // row_index
    size_t true_index_overhead = index_overhead;
    if (this->col_index) {
      index_overhead += this->n_alloc * sizeof(size_t);           // col_index
      true_index_overhead += this->true_n_alloc * sizeof(size_t); // col_index
    }

    // --- 4. Totals ---
    size_t total_t1 = csr_t1_mem + cache_t1_mem;
    size_t total_t2 = csr_t2_mem + cache_t2_mem;
    size_t true_total_t1 = true_csr_t1_mem + cache_t1_mem;
    size_t true_total_t2 = true_csr_t2_mem + cache_t2_mem;
    size_t total_overhead = index_overhead + cache_struct_overhead;
    size_t true_total_overhead = true_index_overhead + cache_struct_overhead;
    size_t total_mem = total_t1 + total_t2 + total_overhead + size_of_self;
    size_t true_total_mem =
        true_total_t1 + true_total_t2 + true_total_overhead + size_of_self;

    // --- 5. Report ---
    ss << "SparseMatrix Status Report" << "\n";
    ss << "==========================" << "\n";
    ss << "Dimensions : " << this->n_rows << " x " << this->n_cols << "\n";
    ss << "State      : " << (this->dirty ? "Dirty (Flush Required)" : "Clean")
       << "\n";
    ss << "Entries    : " << this->n_alloc << " (CSR) + " << cache_entries
       << " (Cache)\n";
    ss << "\n";

    ss << "[Data Memory]" << "\n";
    ss << "  T1 Usage : " << f(total_t1) << " (" << f(true_total_t1) << " true)"
       << " (Cores: " << this->n_cores.first << ")\n";
    ss << "  T2 Usage : " << f(total_t2) << " (" << f(true_total_t2) << " true)"
       << " (Cores: " << this->n_cores.second << ")\n";
    ss << "\n";

    ss << "[Structure Overhead]" << "\n";
    ss << "  Self : " << f(size_of_self) << "\n";
    ss << "  Indices  : " << f(index_overhead) << " (" << f(true_index_overhead)
       << " true)" << " (CSR Row/Col arrays)\n";
    ss << "  Cache    : " << f(cache_struct_overhead)
       << " (Vectors & Tuples)\n";
    ss << "--------------------------" << "\n";
    ss << "Total Mem  : " << f(total_mem) << " (" << f(true_total_mem)
       << " true)" << "\n";

    return ss.str();
  }

  std::pair<std::vector<T1>, std::vector<T2>> sum(void) {
    return std::make_pair(
        SparseMatrix::_sum<T1>(this->n_cores.first, this->n_alloc,
                               this->cores.first),
        SparseMatrix::_sum<T2>(this->n_cores.second, this->n_alloc,
                               this->cores.second));
  }

  template <typename T>
  static std::vector<T> _sum(const size_t stride, const size_t length,
                             const T *const values) {
    std::vector<T> result(stride);

    for (size_t i = 0; i < length; i++) {
      for (size_t j = 0; j < stride; j++) {
        result[j] += values[i * stride + j];
      }
    }

    return result;
  }

  void inplace_linear_combination_skims(SparseMatrix &other, T1 step_size) {
    this->inplace_mult_scalar(step_size);

    const T1 conjugate_step_size = 1.0 - step_size;

    this->inplace_fma_element_wise_scalar(other, conjugate_step_size);
  }

  // Computes A = A*x1 + B*x2 + C*x3
  void inplace_triple_linear_combination_skims(SparseMatrix &B, SparseMatrix &C,
                                               T1 x1, T1 x2, T1 x3) {
    this->inplace_mult_scalar(x1);

    this->inplace_fma_element_wise_scalar(B, x2);
    this->inplace_fma_element_wise_scalar(C, x3);
  }

  void inplace_mult_scalar(T1 x) {
    this->_inplace_mult_scalar<T1>(x);
    this->_inplace_mult_scalar<T2>(x);
  }

  // Performs A = A*x
  template <typename T> void _inplace_mult_scalar(T x) {
    T *core = std::get<T *>(this->cores);
    const size_t core_count = this->template core_count<T>();

    // Modify the current CSR matrix (* step_size)
    for (size_t i = 0; i < this->n_alloc; i++) {
      for (size_t j = 0; j < core_count; j++) {
        core[i * core_count + j] *= x;
      }
    }

    // Modify the current insertion cache (* step_size)
    for (auto &row : insertion_cache) {
      for (auto &col : row) {
        T *values = std::get<T *>(col);
        for (size_t j = 0; j < core_count; j++) {
          values[j] *= x;
        }
      }
    }
  }

// These methods were a little repetitive so we use a macro to apply an
// operation to all values in the insertion cache and in the cores.
#define SPARSE_MATRIX_ELEMENT_WISE_OP(OTHER, OP)                               \
  do {                                                                         \
    const auto core_counts = this->n_cores;                                    \
    auto &_other = (OTHER);                                                    \
                                                                               \
    for (auto iter = _other.begin(), end = _other.end(); iter != end;          \
         ++iter) {                                                             \
      auto A_values = this->insert(iter.row(), iter.col());                    \
      auto B_values = iter.values();                                           \
                                                                               \
      for (size_t i = 0; i < core_counts.first; i++) {                         \
        A_values.first[i] += B_values.first[i] OP;                             \
      }                                                                        \
                                                                               \
      for (size_t i = 0; i < core_counts.second; i++) {                        \
        A_values.second[i] += B_values.second[i] OP;                           \
      }                                                                        \
    }                                                                          \
                                                                               \
    for (size_t i = 0; i < _other.insertion_cache.size(); i++) {               \
      auto &row = _other.insertion_cache[i];                                   \
      for (auto &col : row) {                                                  \
        auto A_values = this->insert(i, std::get<0>(col));                     \
                                                                               \
        for (size_t i = 0; i < core_counts.first; i++) {                       \
          A_values.first[i] += std::get<1>(col)[i] OP;                         \
        }                                                                      \
                                                                               \
        for (size_t i = 0; i < core_counts.second; i++) {                      \
          A_values.second[i] += std::get<2>(col)[i] OP;                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (0)

  void inplace_fma_element_wise_scalar(const SparseMatrix &B, T1 x) {
    SPARSE_MATRIX_ELEMENT_WISE_OP(B, *x);
  }

  void inplace_fda_element_wise_scalar(const SparseMatrix &B, T1 x) {
    SPARSE_MATRIX_ELEMENT_WISE_OP(B, / x);
  }

  void inplace_add_element_wise(const SparseMatrix &B) {
    SPARSE_MATRIX_ELEMENT_WISE_OP(B, );
  }

  void inplace_sub_element_wise(const SparseMatrix &B) {
    SPARSE_MATRIX_ELEMENT_WISE_OP(B, *-1.0);
  }
};

// Operates on two cells, writes results into C, returns true if writes
// contained non-zero values
template <typename T, typename binary_op>
bool _element_wise_cell_op(const T *A, const T *B, T *C, const size_t n,
                           const binary_op &op) {
  bool has_non_zero_result = false;

  T res;
  for (size_t i = 0; i < n; i++) {
    res = op(A[i], B[i]);
    has_non_zero_result |= (res != 0.0);
    C[i] = res;
  }

  return has_non_zero_result;
};

// Operates on one cell and a scalar, writes results into C, returns true if
// writes contained non-zero values
template <typename T, typename binary_op>
bool _element_wise_cell_op(const T *A, const T b, T *C, const size_t n,
                           const binary_op &op) {
  bool has_non_zero_result = false;

  T res;
  for (size_t i = 0; i < n; i++) {
    res = op(A[i], b);
    has_non_zero_result |= (res != 0.0);
    C[i] = res;
  }

  return has_non_zero_result;
};

template <typename T, typename binary_op>
bool _element_wise_cell_op(const T a, const T *B, T *C, const size_t n,
                           const binary_op &op) {
  bool has_non_zero_result = false;

  T res;
  for (size_t i = 0; i < n; i++) {
    res = op(a, B[i]);
    has_non_zero_result |= (res != 0.0);
    C[i] = res;
  }

  return has_non_zero_result;
};

// Adds two rows, writes results into out, returns the number of non-zero cells.
template <typename T1, typename T2, typename binary_op>
size_t _element_wise_row_op(const SparseMatrix<T1, T2> &A,
                            const SparseMatrix<T1, T2> &B,
                            SparseMatrix<T1, T2> &C, const size_t row,
                            const binary_op &op) {

  size_t A_idx = A.row_index[row];
  size_t A_row_end = A.row_index[row + 1];
  size_t B_idx = B.row_index[row];
  size_t B_row_end = B.row_index[row + 1];
  size_t C_idx = C.row_index[row];

  while ((A_idx < A_row_end) || (B_idx < B_row_end)) {
    // Walk all columns of A that are less than B, we also need to account for
    // there being no values in B
    while ((A_idx < A_row_end) && ((B_idx == B_row_end) ||
                                   (A.col_index[A_idx] < B.col_index[B_idx]))) {
      C.col_index[C_idx] = A.col_index[A_idx];
      bool any_nonzero = false;
      any_nonzero |= _element_wise_cell_op(A.template get_address<T1>(A_idx, 0),
                                           static_cast<T1>(0.0),
                                           C.template get_address<T1>(C_idx, 0),
                                           A.template core_count<T1>(), op);

      any_nonzero |= _element_wise_cell_op(A.template get_address<T2>(A_idx, 0),
                                           static_cast<T2>(0.0),
                                           C.template get_address<T2>(C_idx, 0),
                                           A.template core_count<T2>(), op);

      A_idx++;
      C_idx += any_nonzero;
    }

    // Walk all columns of B that are less than A, we also need to account for
    // there being no values in A
    while ((B_idx < B_row_end) && ((A_idx == A_row_end) ||
                                   (B.col_index[B_idx] < A.col_index[A_idx]))) {
      C.col_index[C_idx] = B.col_index[B_idx];
      bool any_nonzero = false;
      any_nonzero |= _element_wise_cell_op(static_cast<T1>(0.0),
                                           B.template get_address<T1>(B_idx, 0),
                                           C.template get_address<T1>(C_idx, 0),
                                           A.template core_count<T1>(), op);

      any_nonzero |= _element_wise_cell_op(static_cast<T2>(0.0),
                                           B.template get_address<T2>(B_idx, 0),
                                           C.template get_address<T2>(C_idx, 0),
                                           A.template core_count<T2>(), op);

      B_idx++;
      C_idx += any_nonzero;
    }

    // Walk all columns of A and B that are equal
    while ((A_idx < A_row_end) && (B_idx < B_row_end) &&
           (A.col_index[A_idx] == B.col_index[B_idx])) {
      C.col_index[C_idx] = A.col_index[A_idx];

      bool any_nonzero = false;
      any_nonzero |= _element_wise_cell_op(A.template get_address<T1>(A_idx, 0),
                                           B.template get_address<T1>(B_idx, 0),
                                           C.template get_address<T1>(C_idx, 0),
                                           A.template core_count<T1>(), op);

      any_nonzero |= _element_wise_cell_op(A.template get_address<T2>(A_idx, 0),
                                           B.template get_address<T2>(B_idx, 0),
                                           C.template get_address<T2>(C_idx, 0),
                                           A.template core_count<T2>(), op);

      A_idx++;
      B_idx++;
      C_idx += any_nonzero;
    }
  }
  return C_idx - C.row_index[row];
};

template <typename T1, typename T2, typename binary_op>
SparseMatrix<T1, T2> *element_wise_op(const SparseMatrix<T1, T2> &A,
                                      const SparseMatrix<T1, T2> &B,
                                      const binary_op &op) {
  if (!(A.n_rows == B.n_rows && A.n_cols == B.n_cols &&
        A.n_cores == B.n_cores)) {
    throw std::logic_error(
        "matrices are not compatible, check shape and core counts");
  } else if (A.dirty || B.dirty) {
    throw std::logic_error(
        "matrices are dirty, flush before operating on them");
  }

  // We cannot know how many elements will be in the resulting matrix so we
  // allocate for the worst case scenario, i.e. there is no overlap. We then
  // track how many cells have been actually added and "shrink" our structure.
  // This means we over allocate each object during operations but successive
  // additions don't get worse and worse if they don't add more elements.
  auto *new_mat = new SparseMatrix<T1, T2>(A.n_rows, A.n_cols, A.n_cores,
                                           A.n_alloc + B.n_alloc);

  size_t cells_added = 0;
  for (size_t row = 0; row < A.n_rows; row++) {
    cells_added += _element_wise_row_op(A, B, *new_mat, row, op);
    new_mat->row_index[row + 1] = cells_added;
  }

  new_mat->n_alloc = cells_added;

  return new_mat;
};

template <typename T1, typename T2>
SparseMatrix<T1, T2> *addition(const SparseMatrix<T1, T2> &A,
                               const SparseMatrix<T1, T2> &B) {
  return element_wise_op(A, B, std::plus());
};

template <typename T1, typename T2>
SparseMatrix<T1, T2> *subtraction(const SparseMatrix<T1, T2> &A,
                                  const SparseMatrix<T1, T2> &B) {
  return element_wise_op(A, B, std::minus());
};

template <typename T1, typename T2>
SparseMatrix<T1, T2> *multiplication(const SparseMatrix<T1, T2> &A,
                                     const SparseMatrix<T1, T2> &B) {
  return element_wise_op(A, B, std::multiplies());
};

template <typename T1, typename T2>
SparseMatrix<T1, T2> *division(const SparseMatrix<T1, T2> &A,
                               const SparseMatrix<T1, T2> &B) {
  return element_wise_op(A, B, std::divides());
};

} // namespace aequilibrae::matrix::cpp
