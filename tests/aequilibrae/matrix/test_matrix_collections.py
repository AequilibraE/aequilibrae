"""Mapping, ownership, subset and equality contracts of matrix collections."""

import operator

import numpy as np
import pytest
from scipy import sparse

from aequilibrae.matrix import AequilibraeMatrix, AequilibraeSparseMatrix, MatrixStore, from_dict
from aequilibrae.matrix.aequilibrae_matrix import OMX_SPECIAL_ATTRS


@pytest.fixture(params=[AequilibraeMatrix, AequilibraeSparseMatrix], ids=["dense", "sparse"])
def memory_collection(request):
    matrix = request.param(index=[20, 10], index_name="zones")
    for name, values in {"car": [[0, 2], [3, 0]], "bus": [[4, 0], [0, 5]]}.items():
        array = np.array(values, dtype=np.int16)
        matrix[name] = sparse.csr_array(array) if request.param is AequilibraeSparseMatrix else array
    matrix.metadata["year"] = 2026
    matrix.matrix_metadata("car")["purpose"] = "work"
    matrix.matrix_metadata("bus")["purpose"] = "other"
    return matrix


@pytest.fixture(params=["memory", "store"])
def collection(request, memory_collection, tmp_path):
    if request.param == "memory":
        yield memory_collection
    else:
        path = tmp_path / "collection.omx"
        memory_collection.save_as_omx(path)
        with MatrixStore(path, index_name="zones", mode="r+") as store:
            yield store


def dense(value):
    return value.toarray() if sparse.issparse(value) else value


def test_iteration_length_and_membership_describe_named_matrices(collection):
    assert len(collection) == 2
    assert set(collection) == {"car", "bus"}
    assert "car" in collection
    assert "missing" not in collection


def test_totals_returns_python_scalars_keyed_by_matrix_name(collection):
    totals = collection.totals()

    assert totals == {"car": 5, "bus": 9}
    assert all(isinstance(value, int) for value in totals.values())


def test_in_memory_lookup_allows_edits_to_the_stored_values(memory_collection):
    returned = memory_collection["car"]
    returned[0, 1] = 42

    np.testing.assert_array_equal(dense(memory_collection["car"]), [[0, 42], [3, 0]])
    np.testing.assert_array_equal(dense(memory_collection["bus"]), [[4, 0], [0, 5]])


def test_replacing_values_keeps_existing_matrix_metadata(collection):
    collection["car"] = collection["bus"]

    np.testing.assert_array_equal(dense(collection["car"]), [[4, 0], [0, 5]])
    assert collection.matrix_metadata("car")["purpose"] == "work"


def test_deleting_a_matrix_removes_its_metadata_but_keeps_other_state(collection):
    del collection["car"]

    assert set(collection) == {"bus"}
    with pytest.raises(KeyError):
        collection.matrix_metadata("car")
    np.testing.assert_array_equal(dense(collection["bus"]), [[4, 0], [0, 5]])
    np.testing.assert_array_equal(collection.index, [20, 10])
    assert collection.metadata["year"] == 2026


def test_recreating_a_deleted_name_starts_without_its_old_metadata(collection):
    replacement = collection["bus"]
    del collection["car"]
    collection["car"] = replacement

    assert "purpose" not in collection.matrix_metadata("car")


def test_clear_removes_matrices_and_their_metadata_but_keeps_index_and_file_metadata(collection):
    collection.clear()

    assert len(collection) == 0
    assert collection.totals() == {}
    with pytest.raises(KeyError):
        collection.matrix_metadata("car")
    with pytest.raises(KeyError):
        collection.matrix_metadata("bus")
    np.testing.assert_array_equal(collection.index, [20, 10])
    assert collection.index_name == "zones"
    assert collection.shape == (2, 2)
    assert collection.metadata["year"] == 2026


def test_setdefault_does_not_validate_or_replace_an_existing_matrix(collection):
    returned = collection.setdefault("car", None)

    np.testing.assert_array_equal(dense(returned), [[0, 2], [3, 0]])
    assert set(collection) == {"car", "bus"}
    assert collection.matrix_metadata("car")["purpose"] == "work"


def test_setdefault_inserts_missing_values_without_copying_another_matrices_metadata(collection):
    returned = collection.setdefault("new", collection["bus"])

    np.testing.assert_array_equal(dense(returned), [[4, 0], [0, 5]])
    np.testing.assert_array_equal(dense(collection["new"]), [[4, 0], [0, 5]])
    assert "purpose" not in collection.matrix_metadata("new")


def test_setdefault_rejects_a_missing_matrix_without_default_values(collection):
    with pytest.raises(TypeError):
        collection.setdefault("missing")
    assert set(collection) == {"car", "bus"}


@pytest.mark.parametrize(
    "name, error",
    [
        pytest.param("", ValueError, id="empty_name"),
        pytest.param(".", ValueError, id="current_directory"),
        pytest.param("..", ValueError, id="parent_directory"),
        pytest.param("a/b", ValueError, id="slash"),
        pytest.param("a\\b", ValueError, id="backslash"),
        pytest.param("a\0b", ValueError, id="null_character"),
        pytest.param(1, TypeError, id="non_string_name"),
    ],
)
def test_assignment_rejects_invalid_matrix_names(name, error):
    matrix = AequilibraeMatrix(index=[20, 10])
    with pytest.raises(error, match="matrix name"):
        matrix[name] = np.eye(2)
    assert len(matrix) == 0


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(operator.getitem, id="lookup"),
        pytest.param(operator.contains, id="membership"),
        pytest.param(operator.delitem, id="deletion"),
        pytest.param(lambda matrix, key: matrix.matrix_metadata(key), id="metadata"),
    ],
)
def test_mapping_entry_points_reject_path_keys_before_backend_access(collection, operation):
    with pytest.raises(ValueError, match="matrix name"):
        operation(collection, "data/car")


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(operator.getitem, id="lookup"),
        pytest.param(operator.delitem, id="deletion"),
        pytest.param(lambda matrix, key: matrix.matrix_metadata(key), id="metadata"),
    ],
)
def test_missing_matrix_names_raise_key_error_across_backends(collection, operation):
    with pytest.raises(KeyError):
        operation(collection, "missing")


@pytest.mark.parametrize("shape", [(3, 3), (2, 3)], ids=["wrong_size", "not_square"])
def test_assignment_rejects_wrong_shapes_without_replacing_existing_values(collection, shape):
    replacement = np.zeros(shape)
    if sparse.issparse(collection["car"]):
        replacement = sparse.csr_array(replacement)
    with pytest.raises(ValueError, match="bad shape"):
        collection["car"] = replacement

    np.testing.assert_array_equal(dense(collection["car"]), [[0, 2], [3, 0]])
    assert collection.matrix_metadata("car")["purpose"] == "work"


@pytest.mark.parametrize("dtype", [np.bool_, np.int16, np.float32, np.complex128])
def test_in_memory_assignment_accepts_supported_numeric_categories_and_preserves_dtype(memory_collection, dtype):
    replacement = np.eye(2, dtype=dtype)
    if isinstance(memory_collection, AequilibraeSparseMatrix):
        replacement = sparse.csr_array(replacement)
    memory_collection["new"] = replacement

    assert memory_collection["new"].dtype == np.dtype(dtype)
    np.testing.assert_array_equal(dense(memory_collection["new"]), np.eye(2))


@pytest.mark.parametrize("dtype", [object, "U1"], ids=["object_values", "string_values"])
def test_dense_assignment_rejects_nonnumeric_values(dtype):
    matrix = AequilibraeMatrix(index=[20, 10])
    with pytest.raises(TypeError, match="matrix values"):
        matrix["car"] = np.array([["a", "b"], ["c", "d"]], dtype=dtype)
    assert len(matrix) == 0


def test_sparse_in_memory_assignment_rejects_dense_values():
    matrix = AequilibraeSparseMatrix(index=[20, 10])
    with pytest.raises(TypeError, match="SciPy sparse arrays"):
        matrix["car"] = np.eye(2)
    assert len(matrix) == 0


def test_subset_copies_selected_values_and_zone_mapping(collection):
    selected = collection.subset(["car"])
    assert tuple(selected) == ("car",)
    assert selected.index_name == "zones"
    selected["car"][0, 1] = 99
    selected.index = [30, 40]

    np.testing.assert_array_equal(dense(collection["car"]), [[0, 2], [3, 0]])
    np.testing.assert_array_equal(collection.index, [20, 10])


def test_subset_copies_file_and_matrix_metadata_arrays(collection):
    collection.metadata["weights"] = np.array([1.0, 2.0])
    collection.matrix_metadata("car")["weights"] = np.array([3.0, 4.0])
    selected = collection.subset(["car"])
    selected.metadata["weights"][0] = 99
    selected.matrix_metadata("car")["weights"][0] = 99

    np.testing.assert_array_equal(collection.metadata["weights"], [1, 2])
    np.testing.assert_array_equal(collection.matrix_metadata("car")["weights"], [3, 4])


def test_in_memory_subset_deep_copies_nested_metadata(memory_collection):
    memory_collection.metadata["nested"] = {"years": [2026]}
    memory_collection.matrix_metadata("car")["nested"] = {"purposes": ["work"]}
    selected = memory_collection.subset(["car"])
    selected.metadata["nested"]["years"].append(2030)
    selected.matrix_metadata("car")["nested"]["purposes"].append("leisure")

    assert memory_collection.metadata["nested"] == {"years": [2026]}
    assert memory_collection.matrix_metadata("car")["nested"] == {"purposes": ["work"]}


def test_subset_copies_only_user_file_metadata(memory_collection):
    memory_collection.metadata.update(dict.fromkeys(OMX_SPECIAL_ATTRS, "managed value"))
    selected = memory_collection.subset(["car"])

    assert selected.metadata == {"year": 2026}
    assert memory_collection.metadata.keys() & OMX_SPECIAL_ATTRS == OMX_SPECIAL_ATTRS


def test_subset_query_requires_all_attributes_to_match(collection):
    collection.matrix_metadata("bus")["purpose"] = "work"
    collection.matrix_metadata("car")["year"] = 2030
    collection.matrix_metadata("bus")["year"] = 2026
    selected = collection.subset([{"purpose": "work", "year": 2030}])

    assert tuple(selected) == ("car",)


def test_subset_query_compares_array_metadata_with_nans(collection):
    collection.matrix_metadata("car")["weights"] = np.array([1.0, np.nan])
    collection.matrix_metadata("bus")["weights"] = np.array([2.0, np.nan])
    selected = collection.subset([{"weights": np.array([1.0, np.nan])}])

    assert tuple(selected) == ("car",)


def test_subset_includes_each_matching_name_only_once(collection):
    selected = collection.subset(["car", {"purpose": "work"}, "car"])

    assert tuple(selected) == ("car",)


def test_empty_attribute_query_matches_all_matrices(collection):
    assert set(collection.subset([{}])) == {"car", "bus"}


def test_empty_subset_keeps_zone_mapping_and_file_metadata(collection):
    selected = collection.subset([])

    assert len(selected) == 0
    assert selected.shape == (2, 2)
    assert selected.index_name == "zones"
    np.testing.assert_array_equal(selected.index, [20, 10])
    assert selected.metadata["year"] == 2026


def test_subset_rejects_a_missing_name(collection):
    with pytest.raises(KeyError):
        collection.subset(["missing"])


def test_subset_rejects_entries_that_are_neither_names_nor_attribute_queries(collection):
    with pytest.raises(TypeError, match="subset entries"):
        collection.subset([42])


@pytest.fixture
def mixed_store(tmp_path):
    with MatrixStore(tmp_path / "mixed.omx", index=[20, 10], mode="x") as store:
        store["dense"] = np.eye(2)
        store["sparse"] = sparse.csr_array(np.eye(2))
        yield store


@pytest.mark.parametrize("names", [["dense", "sparse"], ["sparse", "dense"]], ids=["dense_first", "sparse_first"])
def test_store_subset_rejects_mixed_storage_regardless_of_first_selected_type(mixed_store, names):
    with pytest.raises(ValueError, match="inferred from the first selected matrix"):
        mixed_store.subset(names)


@pytest.mark.parametrize("name, sparse_required", [("dense", True), ("sparse", False)])
def test_store_subset_rejects_values_not_matching_the_requested_storage_type(mixed_store, name, sparse_required):
    with pytest.raises(ValueError, match="required by sparse="):
        mixed_store.subset([name], sparse=sparse_required)


@pytest.mark.parametrize("name", ["dense", "sparse"])
def test_store_subset_remains_usable_after_store_closes(mixed_store, name):
    selected = mixed_store.subset([name])
    mixed_store.close()
    selected[name][0, 0] = 42

    np.testing.assert_array_equal(dense(selected[name]), [[42, 0], [0, 1]])


@pytest.mark.parametrize("group_name", ["data", "sparse"])
def test_store_observes_storage_groups_replaced_through_the_omx_handle(tmp_path, group_name):
    with MatrixStore(tmp_path / "replace.omx", index=[20, 10], mode="x") as store:
        store["car"] = sparse.csr_array(np.eye(2)) if group_name == "sparse" else np.eye(2)
        root = store.omx["/"]
        root.move(group_name, "old_group")
        root.copy("old_group", group_name)
        root[group_name].move("car", "bus")

        assert set(store) == {"bus"}
        np.testing.assert_array_equal(dense(store["bus"]), np.eye(2))
        with pytest.raises(KeyError):
            store["car"]


def test_collection_equals_itself_and_an_independent_subset_copy(collection):
    other = collection.subset(list(collection))

    assert collection == collection
    assert collection == other
    assert other == collection


def test_equality_ignores_insertion_order_numeric_dtypes_and_dense_vs_sparse_storage(memory_collection):
    other = from_dict(
        {name: dense(memory_collection[name]).astype(np.float64) for name in ("bus", "car")},
        index=memory_collection.index,
        index_name=memory_collection.index_name,
    )
    other.metadata.update(memory_collection.metadata)
    for name in other:
        other.matrix_metadata(name).update(memory_collection.matrix_metadata(name))

    assert tuple(memory_collection) == ("car", "bus")
    assert tuple(other) == ("bus", "car")
    assert memory_collection == other
    assert other == memory_collection


def test_equality_requires_the_same_zone_labels(collection):
    other = collection.subset(list(collection))
    other.index = [10, 20]

    assert collection != other


def test_equality_requires_the_same_zone_mapping_name():
    left = from_dict({"car": np.eye(2)}, index=[20, 10], index_name="zones")
    right = from_dict({"car": np.eye(2)}, index=[20, 10], index_name="other")

    assert left != right


def test_equality_requires_the_same_matrix_names(collection):
    other = collection.subset(list(collection))
    del other["bus"]

    assert collection != other


def test_equality_detects_changed_matrix_values(collection):
    other = collection.subset(list(collection))
    other["car"][0, 1] = 99

    assert collection != other


def test_equality_detects_changed_user_file_metadata(collection):
    other = collection.subset(list(collection))
    other.metadata["year"] = 2030

    assert collection != other


def test_equality_detects_changed_matrix_metadata(collection):
    other = collection.subset(list(collection))
    other.matrix_metadata("car")["purpose"] = "leisure"

    assert collection != other


def test_equality_compares_nested_metadata_values(memory_collection):
    memory_collection.metadata["nested"] = {"years": [2026], "weights": np.array([1.0, np.nan])}
    other = memory_collection.subset(list(memory_collection))
    assert memory_collection == other
    other.metadata["nested"]["years"][0] = 2030

    assert memory_collection != other


def test_equality_treats_matching_matrix_nans_as_equal(collection):
    values = collection["car"].astype(np.float64)
    values[0, 1] = np.nan
    collection["car"] = values
    other = collection.subset(list(collection))
    assert collection == other
    other["car"][1, 0] = np.nan

    assert collection != other


def test_equality_treats_matching_metadata_nans_as_equal(collection):
    collection.metadata["weights"] = np.array([1.0, np.nan])
    collection.matrix_metadata("car")["weights"] = np.array([2.0, np.nan])
    other = collection.subset(list(collection))

    assert collection == other
    assert other == collection


def test_sparse_equality_ignores_explicit_zeros_and_duplicate_coordinates():
    values = sparse.coo_array(([2.0, 3.0, 0.0], ([0, 0, 1], [1, 1, 0])), shape=(2, 2))
    left = from_dict({"car": values}, index=[20, 10])
    right = from_dict({"car": [[0, 5], [0, 0]]}, index=[20, 10])

    assert left == right
    assert right == left


def test_equality_ignores_file_attributes_managed_by_omx(memory_collection):
    other = memory_collection.subset(list(memory_collection))
    memory_collection.metadata.update(dict.fromkeys(OMX_SPECIAL_ATTRS, "managed value"))

    assert memory_collection == other
    assert other == memory_collection


def test_sparse_equality_ignores_the_storage_format_marker():
    left = from_dict({"car": sparse.csr_array(np.eye(2))}, index=[20, 10])
    right = left.subset(["car"])
    left.matrix_metadata("car")["FORMAT"] = "coo"
    right.matrix_metadata("car")["FORMAT"] = "csr"

    assert left == right


def test_comparison_with_an_unrelated_type_returns_not_implemented():
    matrix = AequilibraeMatrix(index=[])

    assert matrix.__eq__(object()) is NotImplemented
