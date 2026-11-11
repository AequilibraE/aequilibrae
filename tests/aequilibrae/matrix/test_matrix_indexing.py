"""Our zone-ID validation and translation, not general array indexing semantics."""

import numpy as np
import pytest
from scipy import sparse

from aequilibrae.matrix import AequilibraeMatrix, AequilibraeSparseMatrix, MatrixStore, from_dict


@pytest.fixture
def label_matrix():
    return from_dict({"demand": np.arange(16, dtype=np.int16).reshape(4, 4)}, index=[309, 0, 101, 7])


@pytest.fixture(params=["dense", "sparse", "store"])
def indexed_matrix(request, label_matrix, tmp_path):
    if request.param == "dense":
        yield label_matrix
    elif request.param == "sparse":
        yield from_dict({"demand": sparse.csr_array(label_matrix["demand"])}, index=label_matrix.index)
    else:
        path = tmp_path / "dense.omx"
        label_matrix.save_as_omx(path)
        with MatrixStore(path, mode="r+") as store:
            yield store


@pytest.fixture
def sparse_store(label_matrix, tmp_path):
    path = tmp_path / "sparse.omx"
    matrix = from_dict({"demand": sparse.csr_array(label_matrix["demand"])}, index=label_matrix.index)
    matrix.save_as_omx(path)
    with MatrixStore(path, mode="r+") as store:
        yield store


def dense(value):
    return value.toarray() if sparse.issparse(value) else value


@pytest.mark.parametrize(
    "index, error, match",
    [
        pytest.param([1, 1], ValueError, "duplicate", id="duplicate_ids"),
        pytest.param([-1, 0], ValueError, "between 0", id="negative_id"),
        pytest.param([0, 2**64], ValueError, "between 0", id="id_exceeds_uint64"),
        pytest.param([False, True], TypeError, "integers", id="boolean_ids"),
        pytest.param([1, True], TypeError, "integers", id="boolean_mixed_with_integer"),
        pytest.param(np.array([1, True], dtype=object), TypeError, "integers", id="object_boolean"),
        pytest.param([1.0, 2.0], TypeError, "integers", id="float_ids"),
        pytest.param(["1", "2"], TypeError, "integers", id="string_ids"),
        pytest.param(np.array([1, None], dtype=object), TypeError, "integers", id="non_integer_object"),
        pytest.param(1, ValueError, "one dimension", id="scalar_index"),
        pytest.param([[1, 2]], ValueError, "one dimension", id="2d_index"),
    ],
)
def test_constructor_rejects_invalid_zone_ids(index, error, match):
    with pytest.raises(error, match=match):
        AequilibraeMatrix(index=index)


@pytest.mark.parametrize(
    "name, error",
    [
        pytest.param("", ValueError, id="empty"),
        pytest.param(".", ValueError, id="current_directory"),
        pytest.param("..", ValueError, id="parent_directory"),
        pytest.param("lookup/zones", ValueError, id="slash"),
        pytest.param("lookup\\zones", ValueError, id="backslash"),
        pytest.param("zone\0ids", ValueError, id="null_character"),
        pytest.param(1, TypeError, id="not_a_string"),
    ],
)
def test_constructor_rejects_invalid_index_names(name, error):
    with pytest.raises(error, match="index name"):
        AequilibraeMatrix(index=[10, 20], index_name=name)


def test_index_is_a_read_only_uint64_copy_of_the_supplied_ids():
    original = np.array([309, 0, 101], dtype=np.int64)
    matrix = AequilibraeMatrix(index=original)
    original[:] = 99
    returned = matrix.index

    assert returned.dtype == np.dtype(np.uint64)
    assert not returned.flags.writeable
    np.testing.assert_array_equal(returned, [309, 0, 101])
    with pytest.raises(ValueError):
        returned[0] = 42
    np.testing.assert_array_equal(matrix.index, [309, 0, 101])


@pytest.mark.parametrize("matrix_type", [AequilibraeMatrix, AequilibraeSparseMatrix], ids=["dense", "sparse"])
def test_empty_zone_index_defines_a_zero_sized_collection(matrix_type):
    matrix = matrix_type(index=[])

    assert matrix.shape == (0, 0)
    assert matrix.index.dtype == np.dtype(np.uint64)
    assert len(matrix.index) == 0


@pytest.mark.parametrize("sequence_type", [list, tuple], ids=["list", "tuple"])
def test_large_zone_ids_remain_exact_in_scalar_and_sequence_selections(sequence_type):
    ids = [0, 2**53 + 1, 2**64 - 1]
    matrix = from_dict({"demand": np.arange(9).reshape(3, 3)}, index=ids)

    np.testing.assert_array_equal(matrix.index, np.array(ids, dtype=np.uint64))
    assert matrix.mat["demand", ids[1], ids[2]] == 5
    rows = sequence_type([ids[2], 0])
    np.testing.assert_array_equal(matrix.mat["demand", rows, ids[1]], [7, 1])


def test_zone_lookup_treats_zero_as_a_label_not_a_position(indexed_matrix):
    assert indexed_matrix.mat["demand", 0, 101] == 6
    assert indexed_matrix.imat["demand", 0, 2] == 2


def test_zone_slice_uses_stored_order_and_includes_the_stop_label(indexed_matrix):
    selected = indexed_matrix.mat["demand", 309:101]

    np.testing.assert_array_equal(dense(selected), np.arange(12).reshape(3, 4))


def test_position_slice_does_not_translate_labels_and_excludes_the_stop_position(indexed_matrix):
    selected = indexed_matrix.imat["demand", :2]

    np.testing.assert_array_equal(dense(selected), np.arange(8).reshape(2, 4))


@pytest.mark.parametrize(
    "selection, expected_rows",
    [
        pytest.param(slice(7, 309, -1), [3, 2, 1, 0], id="include_first_zone_as_stop"),
        pytest.param(slice(7, 0, -1), [3, 2, 1], id="include_nonfirst_zone_as_stop"),
        pytest.param(slice(None, 309, -2), [3, 1], id="open_start_with_negative_step"),
        pytest.param(slice(309, 7, 2), [0, 2], id="positive_step"),
        pytest.param(slice(None, 0), [0, 1], id="open_start"),
        pytest.param(slice(101, None), [2, 3], id="open_stop"),
    ],
)
def test_zone_slice_translates_bounds_and_preserves_step(label_matrix, selection, expected_rows):
    np.testing.assert_array_equal(label_matrix.mat["demand", selection], label_matrix["demand"][expected_rows])


def test_sequence_of_zone_labels_is_translated_to_row_positions(indexed_matrix):
    selected = indexed_matrix.mat["demand", [0, 7], :]

    np.testing.assert_array_equal(dense(selected), [[4, 5, 6, 7], [12, 13, 14, 15]])


def test_boolean_mask_is_passed_through_instead_of_looked_up_as_zone_ids(indexed_matrix):
    selected = indexed_matrix.mat["demand", [False, True, False, True], :]

    np.testing.assert_array_equal(dense(selected), [[4, 5, 6, 7], [12, 13, 14, 15]])


def test_label_arrays_keep_their_shape_during_translation(label_matrix):
    rows = np.array([[0], [7]])
    columns = np.array([[309, 101]])

    np.testing.assert_array_equal(label_matrix.mat["demand", rows, columns], [[4, 6], [12, 14]])


def test_ellipsis_and_new_axis_are_passed_through_without_zone_lookup(label_matrix):
    np.testing.assert_array_equal(label_matrix.mat["demand", ...], label_matrix["demand"])
    np.testing.assert_array_equal(label_matrix.mat["demand", None], np.arange(16).reshape(1, 4, 4))


@pytest.mark.parametrize(
    "selection",
    [
        pytest.param((999, slice(None)), id="unknown_row"),
        pytest.param((slice(None), 999), id="unknown_column"),
        pytest.param((slice(999, 101),), id="unknown_slice_start"),
        pytest.param((slice(309, 999),), id="unknown_slice_stop"),
    ],
)
def test_unknown_zone_ids_raise_key_error_before_backend_indexing(indexed_matrix, selection):
    with pytest.raises(KeyError) as error:
        indexed_matrix.mat[("demand", *selection)]
    assert error.value.args == (999,)


def test_failed_zone_translation_does_not_partially_write_values(indexed_matrix):
    before = dense(indexed_matrix["demand"]).copy()
    with pytest.raises(KeyError):
        indexed_matrix.mat["demand", [309, 999], :] = -1

    np.testing.assert_array_equal(dense(indexed_matrix["demand"]), before)


def test_zone_assignment_updates_the_selected_position_without_replacing_its_dtype(indexed_matrix):
    indexed_matrix.mat["demand", 0, 101] = 42
    expected = np.arange(16, dtype=np.int16).reshape(4, 4)
    expected[1, 2] = 42

    assert indexed_matrix["demand"].dtype == np.dtype(np.int16)
    np.testing.assert_array_equal(dense(indexed_matrix["demand"]), expected)


def test_relabelling_rebuilds_zone_lookup_without_moving_values(indexed_matrix):
    indexed_matrix.index = [100, 200, 300, 400]

    assert indexed_matrix.mat["demand", 200, 300] == 6
    np.testing.assert_array_equal(dense(indexed_matrix["demand"]), np.arange(16).reshape(4, 4))
    with pytest.raises(KeyError):
        indexed_matrix.mat["demand", 0, 101]


@pytest.mark.parametrize(
    "replacement, error, match",
    [
        pytest.param([10, 20], ValueError, "shape may not change", id="different_length"),
        pytest.param([10, 20, 20, 40], ValueError, "duplicate", id="duplicate_ids"),
        pytest.param([10.0, 20.0, 30.0, 40.0], TypeError, "integers", id="float_ids"),
        pytest.param([-1, 20, 30, 40], ValueError, "between 0", id="negative_id"),
    ],
)
def test_invalid_relabelling_leaves_index_and_zone_lookup_unchanged(indexed_matrix, replacement, error, match):
    with pytest.raises(error, match=match):
        indexed_matrix.index = replacement

    np.testing.assert_array_equal(indexed_matrix.index, [309, 0, 101, 7])
    assert indexed_matrix.mat["demand", 0, 101] == 6


@pytest.mark.parametrize("accessor", ["mat", "imat"])
@pytest.mark.parametrize("key", ["demand", ("demand",)], ids=["name", "single_element_tuple"])
def test_name_only_indexer_lookup_returns_the_whole_matrix(label_matrix, accessor, key):
    np.testing.assert_array_equal(getattr(label_matrix, accessor)[key], label_matrix["demand"])


@pytest.mark.parametrize("accessor", ["mat", "imat"])
def test_name_only_indexer_assignment_replaces_the_whole_matrix(indexed_matrix, accessor):
    replacement = np.full((4, 4), 42, dtype=np.float32)
    value = sparse.csr_array(replacement) if isinstance(indexed_matrix, AequilibraeSparseMatrix) else replacement
    getattr(indexed_matrix, accessor)["demand"] = value

    assert indexed_matrix["demand"].dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(dense(indexed_matrix["demand"]), replacement)


@pytest.mark.parametrize("accessor", ["mat", "imat"])
@pytest.mark.parametrize(
    "key",
    [pytest.param((), id="no_arguments"), pytest.param(("demand", 0, 0, 0), id="too_many_arguments")],
)
def test_indexers_reject_keys_with_invalid_argument_counts(label_matrix, accessor, key):
    with pytest.raises(TypeError, match="expected a matrix name"):
        getattr(label_matrix, accessor)[key]


@pytest.mark.parametrize("accessor", ["mat", "imat"])
@pytest.mark.parametrize(
    "selection",
    [pytest.param((0, 0), id="scalar"), pytest.param((slice(None), slice(None)), id="all_values")],
)
def test_stored_sparse_partial_reads_are_rejected_even_when_selecting_all_values(sparse_store, accessor, selection):
    with pytest.raises(NotImplementedError, match="whole-matrix reads"):
        getattr(sparse_store, accessor)[("demand", *selection)]


@pytest.mark.parametrize("accessor", ["mat", "imat"])
@pytest.mark.parametrize(
    "selection",
    [pytest.param((0, 0), id="scalar"), pytest.param((slice(None), slice(None)), id="all_values")],
)
def test_stored_sparse_partial_writes_are_rejected_without_changing_values(sparse_store, accessor, selection):
    with pytest.raises(NotImplementedError, match="whole-matrix writes"):
        getattr(sparse_store, accessor)[("demand", *selection)] = 42

    np.testing.assert_array_equal(sparse_store["demand"].toarray(), np.arange(16).reshape(4, 4))


@pytest.mark.parametrize("accessor", ["mat", "imat"])
def test_stored_sparse_indexers_support_whole_matrix_reads_and_writes(sparse_store, accessor):
    indexer = getattr(sparse_store, accessor)
    np.testing.assert_array_equal(indexer["demand"].toarray(), np.arange(16).reshape(4, 4))
    indexer["demand"] = sparse.csr_array(np.full((4, 4), 42))

    np.testing.assert_array_equal(sparse_store["demand"].toarray(), np.full((4, 4), 42))
