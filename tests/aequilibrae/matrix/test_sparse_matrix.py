import gc

import numpy as np
import pytest
from aequilibrae.matrix.cython.sparse_matrix import SparseMatrix
from scipy import sparse

from aequilibrae.matrix import AequilibraeMatrix, AequilibraeSparseMatrix


@pytest.fixture(params=[np.float32, np.float64])
def dtype(request):
    return request.param


def test_from_dict_aligns_fields_on_the_union_of_stored_coordinates(dtype):
    # Include empty leading/trailing rows and overlapping coordinates across fields.
    bus = sparse.coo_array((np.array([9, 4, 2], dtype=dtype), ([2, 1, 1], [4, 3, 1])), shape=(4, 5))
    car = np.zeros((4, 5), dtype=dtype)
    car[1, 1], car[2, 0] = 7, 3
    packed = SparseMatrix.from_dict({"bus": bus, "car": car, "empty": sparse.csr_array((4, 5), dtype=dtype)})

    assert (packed.origin_count, packed.destination_count, packed.field_count) == (4, 5, 3)
    assert packed.field_names == ("bus", "car", "empty")
    assert packed.n_alloc == 4
    np.testing.assert_array_equal(packed.row_index, [0, 0, 2, 4, 4])
    np.testing.assert_array_equal(packed.col_index, [1, 3, 0, 4])
    fields = packed.to_scipy()
    assert tuple(fields) == packed.field_names
    for name, expected in {"bus": [2, 4, 0, 9], "car": [7, 0, 3, 0], "empty": [0, 0, 0, 0]}.items():
        np.testing.assert_array_equal(fields[name].data, expected)


def test_from_dict_copies_strided_dense_input(dtype):
    backing = np.arange(24, dtype=dtype).reshape(3, 8)
    source = backing[:, ::2]
    expected = source.copy()
    packed = SparseMatrix.from_dict({"bus": source})
    backing[:] = -1

    np.testing.assert_array_equal(packed.to_scipy()["bus"].toarray(), expected)


def test_from_scipy_casts_values_before_summing_duplicates(dtype):
    # Summing the first two int8 values before casting would overflow.
    source = sparse.coo_array((np.array([100, 100], dtype=np.int8), ([0, 0], [1, 1])), shape=(2, 3))
    packed = SparseMatrix.from_scipy(source, dtype=dtype)

    assert packed.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(packed.to_scipy()["matrix"].toarray(), [[0, 200, 0], [0, 0, 0]])


def test_from_scipy_removes_explicit_zeros_and_cancelled_duplicates(dtype):
    source = sparse.coo_array((np.array([5, -5, 0, 4], dtype=dtype), ([1, 1, 0, 2], [2, 2, 1, 3])), shape=(3, 4))
    packed = SparseMatrix.from_scipy(source)

    assert packed.n_alloc == 1
    np.testing.assert_array_equal(packed.row_index, [0, 0, 0, 1])
    np.testing.assert_array_equal(packed.col_index, [3])
    np.testing.assert_array_equal(packed.to_scipy()["matrix"].data, [4])


def test_from_scipy_leaves_source_data_coordinates_and_canonical_flag_unchanged(dtype):
    source = sparse.coo_array((np.array([2, 3, 0], dtype=dtype), ([1, 0, 1], [0, 1, 0])), shape=(2, 2))
    data, rows, cols = source.data.copy(), source.row.copy(), source.col.copy()
    canonical = source.has_canonical_format
    SparseMatrix.from_scipy(source)

    np.testing.assert_array_equal(source.data, data)
    np.testing.assert_array_equal(source.row, rows)
    np.testing.assert_array_equal(source.col, cols)
    assert source.has_canonical_format == canonical


def test_from_dict_transposes_rectangular_fields(dtype):
    bus = np.array([[0, 2, 0], [3, 0, 4]], dtype=dtype)
    car = np.array([[5, 0, 6], [0, 7, 0]], dtype=dtype)
    packed = SparseMatrix.from_dict({"bus": sparse.coo_array(bus), "car": car}, transpose=True)

    assert (packed.origin_count, packed.destination_count) == (3, 2)
    fields = packed.to_scipy()
    np.testing.assert_array_equal(fields["bus"].toarray(), bus.T)
    np.testing.assert_array_equal(fields["car"].toarray(), car.T)


def test_from_scipy_forwards_transpose(dtype):
    source = sparse.coo_array(np.array([[0, 2, 0], [3, 0, 4]], dtype=dtype))
    packed = SparseMatrix.from_scipy(source, transpose=True)

    np.testing.assert_array_equal(packed.to_scipy()["matrix"].toarray(), [[0, 3], [2, 0], [0, 4]])


def test_from_csr_removes_only_coordinates_zero_in_every_field(dtype):
    packed = SparseMatrix.from_csr(
        (4, 5),
        [0, 2, 3, 5, 6],
        [0, 4, 2, 1, 3, 4],
        {"bus": np.array([0, 5, 0, 0, 8, 0], dtype=dtype), "car": np.array([3, 0, 0, 0, 0, 0], dtype=dtype)},
    )

    assert packed.n_alloc == 3
    np.testing.assert_array_equal(packed.row_index, [0, 2, 2, 3, 3])
    np.testing.assert_array_equal(packed.col_index, [0, 4, 3])
    fields = packed.to_scipy()
    np.testing.assert_array_equal(fields["bus"].data, [0, 5, 8])
    np.testing.assert_array_equal(fields["car"].data, [3, 0, 0])


def test_from_csr_copies_index_and_value_arrays(dtype):
    rows, cols = np.array([0, 1, 2]), np.array([1, 2])
    values = np.array([2, 3], dtype=dtype)
    packed = SparseMatrix.from_csr((2, 3), rows, cols, {"bus": values})
    rows[:] = 0
    cols[:] = 0
    values[:] = -1

    np.testing.assert_array_equal(packed.row_index, [0, 1, 2])
    np.testing.assert_array_equal(packed.col_index, [1, 2])
    np.testing.assert_array_equal(packed.to_scipy()["bus"].data, [2, 3])


def test_from_csr_removes_all_explicit_zero_entries(dtype):
    packed = SparseMatrix.from_csr((3, 4), [0, 1, 1, 3], [1, 0, 2], {"bus": np.zeros(3, dtype=dtype)})

    assert packed.field_names == ("bus",)
    assert packed.n_alloc == 0
    np.testing.assert_array_equal(packed.row_index, [0, 0, 0, 0])
    assert len(packed.col_index) == 0
    bus = packed.to_scipy()["bus"]
    assert bus.shape == (3, 4)
    assert bus.nnz == 0


def test_from_csr_with_no_fields_retains_supplied_coordinates(dtype):
    packed = SparseMatrix.from_csr((3, 4), [0, 1, 1, 3], [1, 0, 2], {}, dtype=dtype)

    assert (packed.origin_count, packed.destination_count) == (3, 4)
    assert packed.field_names == ()
    assert packed.n_alloc == 3
    np.testing.assert_array_equal(packed.row_index, [0, 1, 1, 3])
    np.testing.assert_array_equal(packed.col_index, [1, 0, 2])
    assert packed.to_scipy() == {}


@pytest.mark.parametrize(
    "rows, cols, error, match",
    [
        pytest.param([0, 2, 3], [0, 2, 3], ValueError, "length", id="wrong_row_pointer_count"),
        pytest.param([[0, 2, 2, 3]], [0, 2, 3], TypeError, "one-dimensional integer", id="2d_row_pointers"),
        pytest.param([0.0, 2.0, 2.0, 3.0], [0, 2, 3], TypeError, "one-dimensional integer", id="float_row_pointers"),
        pytest.param([0, -1, 2, 3], [0, 2, 3], ValueError, "nonnegative", id="negative_row_pointer"),
        pytest.param([1, 2, 2, 3], [0, 2, 3], ValueError, "start at zero", id="nonzero_first_row_pointer"),
        pytest.param([0, 2, 2, 2], [0, 2, 3], ValueError, "terminated", id="wrong_final_row_pointer"),
        pytest.param([0, 2, 1, 3], [0, 2, 3], ValueError, "ascending", id="decreasing_row_pointers"),
        pytest.param([0, 4, 4, 3], [0, 2, 3], ValueError, "greater than", id="row_pointer_exceeds_storage"),
        pytest.param([0, 2, 2, 3], [[0, 2, 3]], TypeError, "one-dimensional integer", id="2d_column_indices"),
        pytest.param([0, 2, 2, 3], [0.0, 2.0, 3.0], TypeError, "one-dimensional integer", id="float_column_indices"),
        pytest.param([0, 2, 2, 3], [-1, 2, 3], ValueError, "nonnegative", id="negative_column_index"),
        pytest.param([0, 2, 2, 3], [0, 2, 4], ValueError, "exceeds shape", id="column_outside_shape"),
        pytest.param([0, 2, 2, 3], [2, 0, 3], ValueError, "sorted and unique", id="unsorted_columns"),
        pytest.param([0, 2, 2, 3], [0, 0, 3], ValueError, "sorted and unique", id="duplicate_columns"),
    ],
)
def test_from_csr_rejects_invalid_index_arrays(rows, cols, error, match):
    with pytest.raises(error, match=match):
        SparseMatrix.from_csr((3, 4), rows, cols, {"bus": np.ones(3)})


@pytest.mark.parametrize(
    "values, error, match",
    [
        pytest.param(np.ones((1, 3)), ValueError, "1D", id="2d_values"),
        pytest.param(np.ones(2), ValueError, "mismatched length", id="wrong_value_count"),
        pytest.param(np.ones(3, dtype=complex), TypeError, "cast", id="complex_values"),
        pytest.param(np.array(["a", "b", "c"]), TypeError, "cast", id="string_values"),
    ],
)
def test_from_csr_rejects_invalid_field_arrays(values, error, match):
    with pytest.raises(error, match=match):
        SparseMatrix.from_csr((3, 4), [0, 2, 2, 3], [0, 2, 3], {"bus": values}, dtype=np.float64)


def test_from_csr_rejects_mixed_dtypes_without_a_target_dtype():
    with pytest.raises(TypeError, match="mixed dtypes"):
        SparseMatrix.from_csr(
            (2, 2), [0, 1, 2], [1, 0], {"bus": np.array([1, 2], dtype=np.int16), "car": np.array([3.0, 4.0])}
        )


def test_from_csr_converts_mixed_dtypes_to_the_requested_dtype(dtype):
    packed = SparseMatrix.from_csr(
        (2, 2), [0, 1, 2], [1, 0], {"bus": np.array([1, 2], dtype=np.int16), "car": np.array([3.0, 4.0])}, dtype=dtype
    )

    assert packed.dtype == np.dtype(dtype)
    for name, expected in {"bus": [1, 2], "car": [3, 4]}.items():
        field = packed.to_scipy()[name]
        assert field.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(field.data, expected)


def test_from_dict_rejects_mixed_dtypes_without_a_target_dtype():
    with pytest.raises(TypeError, match="mixed dtypes"):
        SparseMatrix.from_dict({"bus": np.eye(2, dtype=np.int16), "car": np.eye(2, dtype=np.float64)})


def test_from_dict_converts_mixed_dtypes_to_the_requested_dtype(dtype):
    packed = SparseMatrix.from_dict(
        {"bus": np.array([[0, 2], [3, 0]], dtype=np.int16), "car": np.eye(2, dtype=np.float64)}, dtype=dtype
    )

    assert packed.dtype == np.dtype(dtype)
    fields = packed.to_scipy()
    for field in fields.values():
        assert field.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(fields["bus"].toarray(), [[0, 2], [3, 0]])
    np.testing.assert_array_equal(fields["car"].toarray(), np.eye(2))


@pytest.mark.parametrize(
    "sources, error, match",
    [
        pytest.param([], TypeError, "mapping", id="not_a_mapping"),
        pytest.param({}, ValueError, "at least one", id="empty_mapping"),
        pytest.param({"bus": np.array(1.0)}, ValueError, "two-dimensional", id="scalar_field"),
        pytest.param({"bus": np.ones(2)}, ValueError, "two-dimensional", id="1d_field"),
        pytest.param({"bus": np.ones((2, 2, 1))}, ValueError, "two-dimensional", id="3d_field"),
        pytest.param({"bus": np.ones((2, 3)), "car": np.ones((3, 2))}, ValueError, "same shape", id="different_shapes"),
        pytest.param({"bus": np.ones((2, 2), dtype=bool)}, TypeError, "real numbers", id="boolean_field"),
        pytest.param({"bus": np.ones((2, 2), dtype=complex)}, TypeError, "real numbers", id="complex_field"),
        pytest.param({"bus": np.full((2, 2), "a")}, TypeError, "real numbers", id="string_field"),
        pytest.param(
            {"bus": np.ones((2, 2), dtype=int)}, TypeError, "float32 or float64", id="integer_without_conversion"
        ),
    ],
)
def test_from_dict_rejects_invalid_inputs(sources, error, match):
    with pytest.raises(error, match=match):
        SparseMatrix.from_dict(sources)


@pytest.mark.parametrize("matrix_type", [AequilibraeMatrix, AequilibraeSparseMatrix], ids=["dense", "sparse"])
def test_from_matrix_copies_fields_in_collection_order(dtype, matrix_type):
    matrix = matrix_type(index=[40, 10])
    expected = {"bus": np.array([[0, 2], [3, 0]], dtype=dtype), "car": np.array([[4, 0], [0, 5]], dtype=dtype)}
    for name, values in expected.items():
        matrix[name] = sparse.csr_array(values) if matrix_type is AequilibraeSparseMatrix else values
    packed = SparseMatrix.from_matrix(matrix)
    matrix["bus"][0, 1] = 99

    assert packed.field_names == ("bus", "car")
    for name, values in packed.to_scipy().items():
        np.testing.assert_array_equal(values.toarray(), expected[name])


@pytest.mark.parametrize("matrix_type", [AequilibraeMatrix, AequilibraeSparseMatrix], ids=["dense", "sparse"])
def test_from_matrix_forwards_transpose(dtype, matrix_type):
    matrix = matrix_type(index=[40, 10])
    values = np.array([[0, 2], [3, 0]], dtype=dtype)
    matrix["bus"] = sparse.csr_array(values) if matrix_type is AequilibraeSparseMatrix else values
    packed = SparseMatrix.from_matrix(matrix, transpose=True)

    np.testing.assert_array_equal(packed.to_scipy()["bus"].toarray(), [[0, 3], [2, 0]])


@pytest.mark.parametrize("matrix_type", [AequilibraeMatrix, AequilibraeSparseMatrix], ids=["dense", "sparse"])
def test_from_matrix_accepts_collections_without_fields(dtype, matrix_type):
    packed = SparseMatrix.from_matrix(matrix_type(index=[40, 10]), dtype=dtype)

    assert (packed.origin_count, packed.destination_count, packed.field_count) == (2, 2, 0)
    assert packed.dtype == np.dtype(dtype)
    assert packed.n_alloc == 0
    np.testing.assert_array_equal(packed.row_index, [0, 0, 0])
    assert packed.to_scipy() == {}


def test_from_matrix_rejects_objects_that_are_not_matrix_collections():
    with pytest.raises(TypeError, match="AequilibraeMatrix or AequilibraeSparseMatrix"):
        SparseMatrix.from_matrix({"bus": np.ones((2, 2))})


def test_from_dict_all_zero_fields_have_no_stored_coordinates(dtype):
    packed = SparseMatrix.from_dict(
        {"bus": sparse.csr_array((2, 3), dtype=dtype), "car": np.zeros((2, 3), dtype=dtype)}
    )

    assert packed.field_names == ("bus", "car")
    assert packed.n_alloc == 0
    np.testing.assert_array_equal(packed.row_index, [0, 0, 0])
    assert len(packed.col_index) == 0
    for field in packed.to_scipy().values():
        assert field.shape == (2, 3)
        assert field.dtype == np.dtype(dtype)
        assert field.nnz == 0


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (2, 0)], ids=["no_zones", "no_origins", "no_destinations"])
def test_from_dict_accepts_zero_sized_axes(dtype, shape):
    packed = SparseMatrix.from_dict({"bus": np.zeros(shape, dtype=dtype)})

    assert (packed.origin_count, packed.destination_count) == shape
    assert packed.n_alloc == 0
    np.testing.assert_array_equal(packed.row_index, np.zeros(shape[0] + 1))
    assert len(packed.col_index) == 0
    assert packed.to_scipy()["bus"].shape == shape


@pytest.mark.parametrize("field_count", [1, 3], ids=["single_field", "multiple_fields"])
def test_to_matrix_preserves_values_field_order_dtype_and_zone_mapping(dtype, field_count):
    sources = {f"field_{field}": np.arange(4, dtype=dtype).reshape(2, 2) + field + 1 for field in range(field_count)}
    matrix = SparseMatrix.from_dict(sources).to_matrix([309, 0], index_name="zones")

    assert isinstance(matrix, AequilibraeSparseMatrix)
    assert matrix.index_name == "zones"
    np.testing.assert_array_equal(matrix.index, [309, 0])
    assert tuple(matrix) == tuple(sources)
    for name, values in matrix.items():
        assert isinstance(values, sparse.csr_array)
        assert values.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(values.toarray(), sources[name])


@pytest.mark.parametrize("field_count", [1, 3], ids=["single_field", "multiple_fields"])
def test_to_matrix_arrays_remain_valid_after_owner_deletion(dtype, field_count):
    sources = {f"field_{field}": np.full((2, 2), field + 1, dtype=dtype) for field in range(field_count)}
    packed = SparseMatrix.from_dict(sources)
    matrix = packed.to_matrix([309, 0])
    retained = dict(matrix.items())
    del packed, matrix
    gc.collect()

    for name, values in retained.items():
        np.testing.assert_array_equal(values.toarray(), sources[name])


def test_to_matrix_accepts_zero_zones(dtype):
    matrix = SparseMatrix(0, ("bus", "car"), dtype=dtype).to_matrix([])

    assert tuple(matrix) == ("bus", "car")
    assert matrix.shape == (0, 0)
    for field in matrix.values():
        assert field.shape == (0, 0)
        assert field.dtype == np.dtype(dtype)


@pytest.mark.parametrize("index", [[1], [1, 2, 3]], ids=["too_short", "too_long"])
def test_to_matrix_rejects_wrong_index_length(index):
    with pytest.raises(ValueError, match="index length"):
        SparseMatrix(2, ["bus"]).to_matrix(index)


def test_to_matrix_rejects_rectangular_buffers():
    with pytest.raises(ValueError, match="only square"):
        SparseMatrix(3, ["bus"], origin_count=2).to_matrix([1, 2])
