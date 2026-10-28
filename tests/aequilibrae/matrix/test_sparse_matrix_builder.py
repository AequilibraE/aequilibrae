"""Our COO builder owns positional records, not zone labels or SciPy arithmetic."""

import gc

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from aequilibrae.matrix.cython.sparse_matrix_builder import SparseMatrixBuilder


@pytest.fixture(params=[np.float32, np.float64])
def dtype(request):
    return request.param


@pytest.fixture
def od_records():
    return pd.DataFrame({"o": [10, 40, 10], "d": [40, 99, 40], "bus": [2, 0, 3], "car": [0, 5, 7]})


def assert_records(builder, rows, cols, values):
    actual_rows, actual_cols, actual_values = builder.records()
    np.testing.assert_array_equal(actual_rows, rows)
    np.testing.assert_array_equal(actual_cols, cols)
    np.testing.assert_array_equal(actual_values, values)
    assert actual_values.dtype == builder.dtype
    assert actual_values.flags.c_contiguous
    assert builder.entry_count == len(rows)


def test_append_many_keeps_duplicate_and_zero_records_in_input_order(dtype):
    builder = SparseMatrixBuilder(4, ["bus", "car"], dtype=dtype, origin_count=3)
    builder.append_many([2, 0, 2], [1, 3, 1], [[2, 0], [0, 0], [6, 7]])

    # Duplicates and all-zero records remain distinct in our COO storage.
    assert_records(builder, [2, 0, 2], [1, 3, 1], [[2, 0], [0, 0], [6, 7]])


def test_append_adds_one_complete_field_record(dtype):
    builder = SparseMatrixBuilder(4, ["bus", "car"], dtype=dtype, origin_count=3)
    builder.append(1, 2, [8, 9])

    assert_records(builder, [1], [2], [[8, 9]])


def test_append_many_copies_strided_coordinates_and_values(dtype):
    builder = SparseMatrixBuilder(4, ["bus", "car"], dtype=dtype, origin_count=3)
    rows = np.array([2, -1, 0, -1])[::2]
    cols = np.array([1, -1, 3, -1])[::2]
    backing = np.zeros((4, 4), dtype=dtype)
    values = backing[::2, ::2]
    values[:] = [[2, 3], [4, 5]]
    builder.append_many(rows, cols, values)
    rows[:] = 0
    cols[:] = 0
    values[:] = -1

    assert_records(builder, [2, 0], [1, 3], [[2, 3], [4, 5]])


def test_to_scipy_aligns_each_fields_nonzero_values_with_its_coordinates(dtype):
    builder = SparseMatrixBuilder(4, ["bus", "car"], dtype=dtype, origin_count=3)
    builder.append_many([2, 0, 2, 1], [1, 3, 1, 2], [[2, 0], [0, 5], [6, 7], [0, 0]])
    matrices = builder.to_scipy()

    assert tuple(matrices) == ("bus", "car")
    for name, expected in {
        "bus": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 8, 0, 0]],
        "car": [[0, 0, 0, 5], [0, 0, 0, 0], [0, 7, 0, 0]],
    }.items():
        assert isinstance(matrices[name], sparse.csr_array)
        assert matrices[name].dtype == np.dtype(dtype)
        np.testing.assert_array_equal(matrices[name].toarray(), expected)


def test_to_scipy_returns_arrays_independent_of_the_builder_and_other_exports(dtype):
    builder = SparseMatrixBuilder(2, ["bus", "car"], dtype=dtype)
    builder.append(0, 1, [2, 3])
    first = builder.to_scipy()
    second = builder.to_scipy()
    first["bus"].data[:] = -1

    assert_records(builder, [0], [1], [[2, 3]])
    np.testing.assert_array_equal(second["bus"].toarray(), [[0, 2], [0, 0]])
    np.testing.assert_array_equal(first["car"].toarray(), [[0, 3], [0, 0]])


def test_records_returns_independent_coordinate_and_value_arrays(dtype):
    builder = SparseMatrixBuilder(3, ["bus", "car"], dtype=dtype)
    builder.append(2, 1, [2, 3])
    rows, cols, values = builder.records()
    rows[:] = 0
    cols[:] = 0
    values[:] = -1

    assert_records(builder, [2], [1], [[2, 3]])


def test_records_snapshot_survives_builder_growth(dtype):
    builder = SparseMatrixBuilder(3, ["bus", "car"], dtype=dtype)
    builder.append_many([2, 0], [1, 2], [[2, 3], [4, 5]])
    rows, cols, values = builder.records()
    builder.append_many(np.zeros(1000, dtype=int), np.ones(1000, dtype=int), np.ones((1000, 2)))

    np.testing.assert_array_equal(rows, [2, 0])
    np.testing.assert_array_equal(cols, [1, 2])
    np.testing.assert_array_equal(values, [[2, 3], [4, 5]])


def test_reset_clears_records_and_allows_reuse(dtype):
    builder = SparseMatrixBuilder(3, ["bus", "car"], dtype=dtype)
    builder.append(2, 1, [2, 3])
    builder.reset()
    assert_records(builder, [], [], np.empty((0, 2)))

    builder.append(1, 0, [6, 7])
    assert_records(builder, [1], [0], [[6, 7]])


def test_to_df_returns_independent_positional_records_with_requested_column_names(dtype):
    builder = SparseMatrixBuilder(3, ["bus", "car"], dtype=dtype)
    builder.append_many([2, 0], [1, 2], [[2, 3], [4, 5]])
    frame = builder.to_df(row="o", col="d")

    assert list(frame) == ["o", "d", "bus", "car"]
    assert frame["bus"].dtype == np.dtype(dtype)
    assert frame["car"].dtype == np.dtype(dtype)
    np.testing.assert_array_equal(frame.to_numpy(), [[2, 1, 2, 3], [0, 2, 4, 5]])
    frame.loc[:, "bus"] = -1
    assert_records(builder, [2, 0], [1, 2], [[2, 3], [4, 5]])


def test_exported_snapshots_survive_reset_and_builder_deletion(dtype):
    builder = SparseMatrixBuilder(2, ["bus"], dtype=dtype)
    builder.append(0, 1, [2])
    records = builder.records()
    frame = builder.to_df()
    matrices = builder.to_scipy()
    builder.reset()
    del builder
    gc.collect()

    np.testing.assert_array_equal(records[0], [0])
    np.testing.assert_array_equal(records[1], [1])
    np.testing.assert_array_equal(records[2], [[2]])
    np.testing.assert_array_equal(frame.to_numpy(), [[0, 1, 2]])
    np.testing.assert_array_equal(matrices["bus"].toarray(), [[0, 2], [0, 0]])


def test_extend_appends_copies_of_source_records_in_order(dtype):
    destination = SparseMatrixBuilder(4, ["bus", "car"], dtype=dtype, origin_count=3)
    source = SparseMatrixBuilder(4, ["bus", "car"], dtype=dtype, origin_count=3)
    destination.append(1, 2, [1, 2])
    source.append_many([2, 0], [1, 3], [[3, 4], [5, 6]])
    destination.extend(source)
    assert_records(source, [2, 0], [1, 3], [[3, 4], [5, 6]])
    destination.extend(source)
    source.reset()

    assert_records(destination, [1, 2, 0, 2, 0], [2, 1, 3, 1, 3], [[1, 2], [3, 4], [5, 6], [3, 4], [5, 6]])


@pytest.mark.parametrize(
    "options, match",
    [
        pytest.param({"origin_count": 2}, "incompatible", id="different_origin_counts"),
        pytest.param({"zone_count": 5}, "incompatible", id="different_destination_counts"),
        pytest.param({"field_names": ["bus", "other"]}, "ordered field names", id="different_names"),
        pytest.param({"field_names": ["car", "bus"]}, "ordered field names", id="different_field_order"),
        pytest.param({"field_names": ["bus"]}, "ordered field names", id="different_field_counts"),
    ],
)
def test_extend_rejects_incompatible_dimensions_or_fields_without_changing_either_builder(dtype, options, match):
    destination = SparseMatrixBuilder(4, ["bus", "car"], dtype=dtype, origin_count=3)
    destination.append(1, 2, [1, 2])
    arguments = {"zone_count": 4, "field_names": ["bus", "car"], "dtype": dtype, "origin_count": 3}
    arguments.update(options)
    source = SparseMatrixBuilder(**arguments)
    source.append(0, 0, [3] * source.field_count)
    before = source.records()

    with pytest.raises(ValueError, match=match):
        destination.extend(source)
    assert_records(destination, [1], [2], [[1, 2]])
    assert_records(source, *before)


@pytest.mark.parametrize("destination_dtype, source_dtype", [(np.float32, np.float64), (np.float64, np.float32)])
def test_extend_rejects_different_dtypes(destination_dtype, source_dtype):
    destination = SparseMatrixBuilder(2, ["bus"], dtype=destination_dtype)
    source = SparseMatrixBuilder(2, ["bus"], dtype=source_dtype)
    destination.append(0, 1, [2])
    source.append(1, 0, [3])

    with pytest.raises(ValueError, match="dtype"):
        destination.extend(source)
    assert_records(destination, [0], [1], [[2]])
    assert_records(source, [1], [0], [[3]])


def test_extend_rejects_self_extension_without_changing_records(dtype):
    builder = SparseMatrixBuilder(2, ["bus"], dtype=dtype)
    builder.append(0, 1, [2])

    with pytest.raises(ValueError, match="aliased"):
        builder.extend(builder)
    assert_records(builder, [0], [1], [[2]])


@pytest.mark.parametrize(
    "rows, cols, values, match",
    [
        pytest.param([[0]], [1], [[1, 2]], "equally sized one-dimensional", id="2d_rows"),
        pytest.param([0, 1], [1], [[1, 2], [3, 4]], "equally sized one-dimensional", id="different_coordinate_counts"),
        pytest.param([0], [[1]], [[1, 2]], "equally sized one-dimensional", id="2d_columns"),
        pytest.param([-1], [1], [[1, 2]], "integer indices within", id="negative_origin"),
        pytest.param([3], [1], [[1, 2]], "integer indices within", id="origin_outside_shape"),
        pytest.param([0], [-1], [[1, 2]], "integer indices within", id="negative_destination"),
        pytest.param([0], [4], [[1, 2]], "integer indices within", id="destination_outside_shape"),
        pytest.param([0.0], [1], [[1, 2]], "integer indices within", id="float_origin"),
        pytest.param([0], [1.0], [[1, 2]], "integer indices within", id="float_destination"),
        pytest.param([True], [1], [[1, 2]], "integer indices within", id="boolean_origin"),
        pytest.param([0], [False], [[1, 2]], "integer indices within", id="boolean_destination"),
        pytest.param([0], [1], [1, 2], "values must have shape", id="1d_values"),
        pytest.param([0], [1], [[1]], "values must have shape", id="wrong_field_count"),
        pytest.param([0], [1], [[1, 2], [3, 4]], "values must have shape", id="wrong_record_count"),
    ],
)
def test_append_many_rejects_invalid_records_before_changing_storage(rows, cols, values, match):
    builder = SparseMatrixBuilder(4, ["bus", "car"], origin_count=3)
    builder.append(1, 2, [5, 6])

    with pytest.raises(ValueError, match=match):
        builder.append_many(rows, cols, values)
    assert_records(builder, [1], [2], [[5, 6]])


def test_from_df_uses_the_supplied_zone_order_to_map_coordinates(dtype, od_records):
    builder = SparseMatrixBuilder.from_df(od_records, "o", "d", index=[40, 10, 99], dtype=dtype)

    assert_records(builder, [1, 0, 1], [0, 2, 0], [[2, 0], [0, 5], [3, 7]])


def test_from_df_infers_sorted_zone_order_when_no_index_is_supplied(dtype, od_records):
    builder = SparseMatrixBuilder.from_df(od_records, "o", "d", dtype=dtype)

    assert_records(builder, [0, 1, 0], [1, 2, 1], [[2, 0], [0, 5], [3, 7]])


@pytest.mark.parametrize("subset", [None, ["car", "bus"], ["car"]], ids=["all_fields", "reordered_fields", "one_field"])
def test_from_df_selects_value_columns_in_requested_order(dtype, od_records, subset):
    builder = SparseMatrixBuilder.from_df(od_records, "o", "d", subset, dtype=dtype)
    names = ["bus", "car"] if subset is None else subset

    assert builder.field_names == tuple(names)
    assert_records(builder, [0, 1, 0], [1, 2, 1], od_records[names].to_numpy(dtype=dtype))


@pytest.mark.parametrize(
    "index, match",
    [pytest.param([10, 10], "unique", id="duplicate_zones"), pytest.param([10], "not in the index", id="missing_zone")],
)
def test_from_df_rejects_invalid_zone_mappings(index, match):
    frame = pd.DataFrame({"o": [10], "d": [40], "bus": [2.0]})
    with pytest.raises(ValueError, match=match):
        SparseMatrixBuilder.from_df(frame, "o", "d", index=index)


@pytest.mark.parametrize(
    "rows, cols",
    [("o", "o"), ("bus", "d"), ("o", "car")],
    ids=["same_name", "row_matches_field", "column_matches_field"],
)
def test_to_df_rejects_coordinate_column_collisions(rows, cols):
    builder = SparseMatrixBuilder(2, ["bus", "car"])
    with pytest.raises(ValueError, match="coordinate column names"):
        builder.to_df(rows, cols)


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (2, 0)], ids=["no_zones", "no_origins", "no_destinations"])
def test_append_many_accepts_an_empty_batch_on_zero_sized_axes(dtype, shape):
    builder = SparseMatrixBuilder(shape[1], ["bus", "car"], dtype=dtype, origin_count=shape[0])
    builder.append_many([], [], np.empty((0, 2)))

    assert_records(builder, [], [], np.empty((0, 2)))
    for values in builder.to_scipy().values():
        assert values.shape == shape
        assert values.dtype == np.dtype(dtype)
        assert values.nnz == 0


def test_builder_without_fields_keeps_appended_coordinates_until_reset(dtype):
    builder = SparseMatrixBuilder(3, [], dtype=dtype, origin_count=2)
    builder.append_many([0, 1], [1, 2], np.empty((2, 0)))

    assert_records(builder, [0, 1], [1, 2], np.empty((2, 0)))
    assert builder.to_scipy() == {}
    builder.reset()
    assert_records(builder, [], [], np.empty((0, 0)))


def test_from_df_accepts_empty_records(dtype):
    frame = pd.DataFrame({"o": pd.Series(dtype=int), "d": pd.Series(dtype=int), "bus": pd.Series(dtype=dtype)})
    builder = SparseMatrixBuilder.from_df(frame, "o", "d", dtype=dtype)

    assert (builder.origin_count, builder.destination_count) == (0, 0)
    assert builder.field_names == ("bus",)
    assert_records(builder, [], [], np.empty((0, 1)))
    assert list(builder.to_df()) == ["origin", "destination", "bus"]
