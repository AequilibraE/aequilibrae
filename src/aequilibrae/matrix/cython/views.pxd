from libc.stddef cimport size_t


cdef extern from "dense_matrix.hpp" namespace "aequilibrae::matrix::cpp" nogil:
    cdef cppclass CppDenseMatrixOriginView "aequilibrae::matrix::cpp::DenseMatrixOriginView"[T]:
        CppDenseMatrixOriginView() noexcept

        size_t destination_count
        size_t field_count
        T *data

        size_t entry_count() noexcept
        size_t destination(size_t entry) noexcept
        const T *values(size_t entry) noexcept
        T *destination_data(size_t index) noexcept

    cdef cppclass CppDenseMatrix "aequilibrae::matrix::cpp::DenseMatrix"[T]:
        CppDenseMatrix() noexcept

        size_t origin_count
        size_t destination_count
        size_t field_count
        T *data

        CppDenseMatrixOriginView[T] origin(size_t index) noexcept


cdef extern from "sparse_matrix.hpp" namespace "aequilibrae::matrix::cpp" nogil:
    cdef cppclass CppSparseMatrixOriginView "aequilibrae::matrix::cpp::SparseMatrixOriginView"[T]:
        CppSparseMatrixOriginView() noexcept

        size_t destination_count
        size_t field_count
        size_t stored_count
        const size_t *col_index
        T *data

        size_t entry_count() noexcept
        size_t destination(size_t entry) noexcept
        const T *values(size_t entry) noexcept
        # NOTE: no destination_data ?

    cdef cppclass CppSparseMatrix "aequilibrae::matrix::cpp::SparseMatrix"[T]:
        CppSparseMatrix() noexcept

        size_t origin_count
        size_t destination_count
        size_t field_count
        T *data

        size_t n_alloc
        size_t *row_index
        size_t *col_index
        CppSparseMatrixOriginView[T] origin(size_t index) noexcept
