"""Construction, file loading and read-only access through the public helpers."""

import h5py
import numpy as np
import openmatrix as omx
import pandas as pd
import pytest
from scipy import sparse

from aequilibrae.matrix import AequilibraeMatrix, AequilibraeSparseMatrix, MatrixStore, from_df, from_dict, from_file


@pytest.fixture(params=[False, True], ids=["dense", "sparse"])
def sparse_output(request):
    return request.param


@pytest.fixture
def od_records():
    return pd.DataFrame({"origin": [20, 10, 20], "destination": [10, 20, 10], "car": [2, 3, 4], "bus": [1, 0, 5]})


@pytest.fixture
def mixed_file(tmp_path):
    path = tmp_path / "mixed.omx"
    with MatrixStore(path, index=[20, 10], index_name="zones", mode="x") as store:
        store["dense"] = np.array([[0, 2], [3, 0]], dtype=np.float32)
        store["sparse"] = sparse.coo_array(np.array([[0, 5], [7, 0]], dtype=np.int16))
        store.metadata["year"] = 2026
        store.matrix_metadata("dense")["purpose"] = "work"
        store.matrix_metadata("sparse")["purpose"] = "other"
    return path


def dense(value):
    return value.toarray() if sparse.issparse(value) else value


def test_from_dict_preserves_field_order_individual_dtypes_and_zone_mapping():
    matrix = from_dict(
        {"bus": np.array([[0, 2], [3, 0]], dtype=np.int16), "car": np.eye(2, dtype=np.float32)},
        index=[20, 0],
        index_name="zones",
    )

    assert isinstance(matrix, AequilibraeMatrix)
    assert tuple(matrix) == ("bus", "car")
    assert matrix.index_name == "zones"
    np.testing.assert_array_equal(matrix.index, [20, 0])
    assert matrix["bus"].dtype == np.dtype(np.int16)
    assert matrix["car"].dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(matrix["bus"], [[0, 2], [3, 0]])
    np.testing.assert_array_equal(matrix["car"], np.eye(2))


def test_from_dict_copies_dense_input_arrays():
    source = np.array([[0, 2], [3, 0]])
    matrix = from_dict({"car": source}, index=[20, 10])
    source[:] = -1

    np.testing.assert_array_equal(matrix["car"], [[0, 2], [3, 0]])


def test_from_dict_copies_sparse_arrays_without_changing_their_formats_or_dtypes():
    bus = sparse.coo_array(np.array([[0, 2], [3, 0]], dtype=np.float32))
    car = sparse.csc_array(np.eye(2, dtype=np.int16))
    matrix = from_dict({"bus": bus, "car": car}, index=[20, 10])
    bus.data[:] = -1
    car.data[:] = -1

    assert isinstance(matrix, AequilibraeSparseMatrix)
    assert tuple(matrix) == ("bus", "car")
    assert matrix["bus"].format == "coo"
    assert matrix["car"].format == "csc"
    assert matrix["bus"].dtype == np.dtype(np.float32)
    assert matrix["car"].dtype == np.dtype(np.int16)
    np.testing.assert_array_equal(matrix["bus"].toarray(), [[0, 2], [3, 0]])
    np.testing.assert_array_equal(matrix["car"].toarray(), np.eye(2))


@pytest.mark.parametrize(
    "data, match",
    [
        pytest.param({"car": np.eye(2), "bus": sparse.csr_array(np.eye(2))}, "expected dense", id="dense_first"),
        pytest.param({"bus": sparse.csr_array(np.eye(2)), "car": np.eye(2)}, "expected sparse", id="sparse_first"),
    ],
)
def test_from_dict_rejects_mixed_dense_and_sparse_values(data, match):
    with pytest.raises(ValueError, match=match):
        from_dict(data, index=[20, 10])


def test_from_dict_rejects_an_empty_mapping():
    with pytest.raises(ValueError, match="at least one matrix"):
        from_dict({}, index=[20, 10])


def test_from_dict_rejects_fields_that_do_not_match_the_index_length():
    with pytest.raises(ValueError, match="bad shape"):
        from_dict({"car": np.ones((2, 3))}, index=[20, 10])


def test_from_df_infers_the_sorted_union_of_origin_and_destination_ids(od_records, sparse_output):
    matrix = from_df(od_records, "origin", "destination", sparse=sparse_output)

    np.testing.assert_array_equal(matrix.index, [10, 20])
    assert tuple(matrix) == ("car", "bus")
    np.testing.assert_array_equal(dense(matrix["car"]), [[0, 3], [6, 0]])
    np.testing.assert_array_equal(dense(matrix["bus"]), [[0, 0], [6, 0]])


def test_from_df_uses_supplied_zone_order_and_includes_unused_zones(od_records, sparse_output):
    matrix = from_df(
        od_records,
        "origin",
        "destination",
        index=[30, 20, 10],
        index_name="zones",
        dtype=np.int32,
        sparse=sparse_output,
    )

    assert isinstance(matrix, AequilibraeSparseMatrix if sparse_output else AequilibraeMatrix)
    assert matrix.index_name == "zones"
    np.testing.assert_array_equal(matrix.index, [30, 20, 10])
    assert matrix["car"].dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(dense(matrix["car"]), [[0, 0, 0], [0, 0, 6], [0, 3, 0]])


def test_from_df_selects_value_columns_in_requested_order(od_records, sparse_output):
    records = od_records.assign(note=["first", "second", "third"])
    matrix = from_df(records, "origin", "destination", subset=["bus", "car"], sparse=sparse_output)

    assert tuple(matrix) == ("bus", "car")
    np.testing.assert_array_equal(dense(matrix["bus"]), [[0, 0], [6, 0]])
    np.testing.assert_array_equal(dense(matrix["car"]), [[0, 3], [6, 0]])


def test_from_df_applies_fill_value_only_to_pairs_without_records(od_records):
    matrix = from_df(od_records, "origin", "destination", subset=["car"], fill_value=-1, dtype=np.int16)

    np.testing.assert_array_equal(matrix["car"], [[-1, 3], [6, -1]])
    assert matrix["car"].dtype == np.dtype(np.int16)


def test_sparse_from_df_rejects_nonzero_fill_value(od_records):
    with pytest.raises(ValueError, match="sparse matrices require fill_value=0"):
        from_df(od_records, "origin", "destination", fill_value=1, sparse=True)


def test_sparse_from_df_removes_zero_entries_after_duplicate_cancellation():
    records = pd.DataFrame({"origin": [20, 20, 10], "destination": [10, 10, 20], "car": [2, -2, 0]})
    matrix = from_df(records, "origin", "destination", sparse=True)

    assert isinstance(matrix["car"], sparse.csr_array)
    assert matrix["car"].nnz == 0
    assert matrix.shape == (2, 2)


@pytest.mark.parametrize("index", [None, [20, 10]], ids=["infer_no_zones", "explicit_zones"])
def test_from_df_accepts_empty_records(sparse_output, index):
    records = pd.DataFrame(
        {"origin": pd.Series(dtype=int), "destination": pd.Series(dtype=int), "car": pd.Series(dtype=float)}
    )
    matrix = from_df(records, "origin", "destination", index=index, dtype=np.float32, sparse=sparse_output)
    count = 0 if index is None else len(index)

    assert tuple(matrix) == ("car",)
    assert matrix.shape == (count, count)
    assert matrix["car"].dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(dense(matrix["car"]), np.zeros((count, count)))


def test_from_df_empty_subset_retains_inferred_index_without_creating_fields(od_records, sparse_output):
    matrix = from_df(od_records, "origin", "destination", subset=[], sparse=sparse_output)

    assert len(matrix) == 0
    np.testing.assert_array_equal(matrix.index, [10, 20])


@pytest.mark.parametrize("origin, destination", [(30, 10), (10, 30)], ids=["unknown_origin", "unknown_destination"])
def test_from_df_rejects_records_outside_the_supplied_index(sparse_output, origin, destination):
    records = pd.DataFrame({"origin": [origin], "destination": [destination], "car": [2]})
    with pytest.raises(ValueError, match="not in the index"):
        from_df(records, "origin", "destination", index=[10, 20], sparse=sparse_output)


@pytest.mark.parametrize("column", ["origin", "destination"])
def test_from_df_rejects_non_integer_zone_columns(column):
    records = pd.DataFrame({"origin": [10], "destination": [20], "car": [2]})
    records[column] = [10.5]
    with pytest.raises(TypeError, match="zone IDs must be integers"):
        from_df(records, "origin", "destination")


@pytest.mark.parametrize("name, matrix_type", [("dense", AequilibraeMatrix), ("sparse", AequilibraeSparseMatrix)])
def test_from_file_infers_storage_type_and_the_only_zone_mapping(mixed_file, name, matrix_type):
    matrix = from_file(mixed_file, subset=[name])

    assert isinstance(matrix, matrix_type)
    assert tuple(matrix) == (name,)
    assert matrix.index_name == "zones"
    np.testing.assert_array_equal(matrix.index, [20, 10])
    assert matrix.metadata["year"] == 2026


def test_from_file_selects_matrices_by_metadata(mixed_file):
    matrix = from_file(mixed_file, subset=[{"purpose": "work"}])

    assert tuple(matrix) == ("dense",)
    np.testing.assert_array_equal(matrix["dense"], [[0, 2], [3, 0]])
    assert matrix.matrix_metadata("dense")["purpose"] == "work"


@pytest.mark.parametrize("name", ["dense", "sparse"])
def test_from_file_loads_independent_values_index_and_metadata(mixed_file, name):
    matrix = from_file(mixed_file, subset=[name])
    original = dense(matrix[name]).copy()
    matrix[name][0, 1] = 99
    matrix.index = [30, 40]
    matrix.metadata["year"] = 2030
    matrix.matrix_metadata(name)["purpose"] = "changed"

    with from_file(mixed_file, store=True) as store:
        np.testing.assert_array_equal(dense(store[name]), original)
        np.testing.assert_array_equal(store.index, [20, 10])
        assert store.metadata["year"] == 2026
        assert store.matrix_metadata(name)["purpose"] != "changed"


def test_from_file_rejects_a_mixed_storage_selection(mixed_file):
    with pytest.raises(ValueError, match="Mismatched matrices"):
        from_file(mixed_file)


@pytest.mark.parametrize("sparse_required", [False, True], ids=["require_dense", "require_sparse"])
def test_from_file_storage_requirement_validates_instead_of_filtering(mixed_file, sparse_required):
    with pytest.raises(ValueError, match="Mismatched matrices"):
        from_file(mixed_file, sparse=sparse_required)


@pytest.mark.parametrize("name, sparse_required", [("dense", False), ("sparse", True)])
def test_from_file_accepts_a_selection_matching_the_required_storage(mixed_file, name, sparse_required):
    matrix = from_file(mixed_file, subset=[name], sparse=sparse_required)

    assert tuple(matrix) == (name,)
    assert isinstance(matrix, AequilibraeSparseMatrix if sparse_required else AequilibraeMatrix)


@pytest.mark.parametrize("sparse_required", [None, False, True])
def test_from_file_empty_selection_returns_an_empty_dense_collection(mixed_file, sparse_required):
    matrix = from_file(mixed_file, subset=[], sparse=sparse_required)

    assert isinstance(matrix, AequilibraeMatrix)
    assert len(matrix) == 0
    np.testing.assert_array_equal(matrix.index, [20, 10])


@pytest.mark.parametrize("sparse_required", [False, True])
def test_from_file_store_exposes_both_storage_types_regardless_of_sparse_option(mixed_file, sparse_required):
    with from_file(mixed_file, store=True, sparse=sparse_required) as store:
        assert isinstance(store, MatrixStore)
        assert set(store) == {"dense", "sparse"}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        pytest.param({"mode": "r"}, "in-memory matrix cannot have a mode", id="mode_for_memory"),
        pytest.param({"store": True, "subset": []}, "store=True and subset", id="subset_for_store"),
        pytest.param({"store": True, "mode": "w"}, "stores only support modes", id="creation_mode_for_store"),
    ],
)
def test_from_file_rejects_invalid_options_before_opening_a_file(tmp_path, kwargs, match):
    path = tmp_path / "missing.omx"
    with pytest.raises(ValueError, match=match):
        from_file(path, **kwargs)
    assert not path.exists()


def test_from_file_prefers_main_index_when_several_mappings_exist(mixed_file):
    with h5py.File(mixed_file, "r+") as file:
        file["lookup"].create_dataset("main_index", data=[10, 20])
    matrix = from_file(mixed_file, subset=["dense"])

    assert matrix.index_name == "main_index"
    np.testing.assert_array_equal(matrix.index, [10, 20])


def test_from_file_requires_an_explicit_mapping_when_nondefault_mappings_are_ambiguous(mixed_file):
    with h5py.File(mixed_file, "r+") as file:
        file["lookup"].create_dataset("other", data=[30, 40])
    with pytest.raises(ValueError, match="Multiple zone indices"):
        from_file(mixed_file, subset=["dense"])
    matrix = from_file(mixed_file, index_name="other", subset=["dense"])

    assert matrix.index_name == "other"
    np.testing.assert_array_equal(matrix.index, [30, 40])


def test_from_file_rejects_a_file_without_zone_mappings(tmp_path):
    path = tmp_path / "no_index.omx"
    with omx.open_file(path, mode="x") as file:
        file.create_matrix("demand", obj=np.ones((2, 2)))
    with pytest.raises(ValueError, match="No zone index found"):
        from_file(path)


@pytest.mark.parametrize(
    "reader, name", [(AequilibraeMatrix.from_omx, "dense"), (AequilibraeSparseMatrix.from_omx, "sparse")]
)
def test_from_omx_reads_only_its_own_storage_group(mixed_file, reader, name):
    matrix = reader(mixed_file, index_name="zones")

    assert tuple(matrix) == (name,)
    assert matrix.metadata["year"] == 2026


@pytest.mark.parametrize(
    "reader, wrong_name", [(AequilibraeMatrix.from_omx, "sparse"), (AequilibraeSparseMatrix.from_omx, "dense")]
)
def test_from_omx_reports_names_that_exist_only_in_the_other_storage_group(mixed_file, reader, wrong_name):
    with pytest.raises(KeyError, match="other storage group"):
        reader(mixed_file, index_name="zones", subset=[wrong_name])


@pytest.mark.parametrize(
    "reader", [AequilibraeMatrix.from_omx, AequilibraeSparseMatrix.from_omx, from_file], ids=["dense", "sparse", "auto"]
)
def test_file_readers_reject_a_missing_explicit_mapping(mixed_file, reader):
    with pytest.raises(ValueError, match="index not found"):
        reader(mixed_file, index_name="missing")


@pytest.mark.parametrize("mode", [None, "r"], ids=["default", "explicit_read_only"])
def test_from_file_store_opens_read_only_and_rejects_index_changes(mixed_file, mode):
    with from_file(mixed_file, store=True, mode=mode) as store:
        assert store.omx.mode == "r"
        with pytest.raises(PermissionError, match="read-only"):
            store.index = [30, 40]
        np.testing.assert_array_equal(store.index, [20, 10])


def test_matrix_store_rejects_a_supplied_index_in_read_only_mode(mixed_file):
    with pytest.raises(ValueError, match="cannot supply an index"):
        MatrixStore(mixed_file, index=[30, 40], index_name="zones")


def test_from_file_can_open_a_store_for_persistent_updates(mixed_file):
    with from_file(mixed_file, store=True, mode="r+") as store:
        store["dense"] = np.array([[0, 12], [13, 0]], dtype=np.float32)
    np.testing.assert_array_equal(from_file(mixed_file, subset=["dense"])["dense"], [[0, 12], [13, 0]])


@pytest.mark.parametrize("name", ["dense", "sparse"])
@pytest.mark.parametrize("mode", ["r", "r+"])
def test_store_reads_return_independent_arrays_even_when_the_store_is_writable(mixed_file, name, mode):
    with from_file(mixed_file, store=True, mode=mode) as store:
        original = dense(store[name]).copy()
        returned = store[name]
        returned[0, 1] = 99

        np.testing.assert_array_equal(dense(store[name]), original)


def test_read_only_store_rejects_a_representative_dense_write(mixed_file):
    with from_file(mixed_file, store=True) as store:
        with pytest.raises(OSError):
            store["dense"] = np.zeros((2, 2), dtype=np.float32)
    np.testing.assert_array_equal(from_file(mixed_file, subset=["dense"])["dense"], [[0, 2], [3, 0]])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        pytest.param({"index_name": "missing"}, "index not found", id="missing_mapping"),
        pytest.param({"index_name": "zones", "index": [1, 2, 3], "mode": "r+"}, "shape", id="wrong_index_length"),
    ],
)
def test_failed_store_construction_closes_its_file(mixed_file, monkeypatch, kwargs, match):
    opened = []
    original_open = omx.open_file

    def record_open(*args, **options):
        file = original_open(*args, **options)
        opened.append(file)
        return file

    monkeypatch.setattr(omx, "open_file", record_open)
    with pytest.raises(ValueError, match=match):
        MatrixStore(mixed_file, **kwargs)

    assert len(opened) == 1
    assert not opened[0].id.valid
