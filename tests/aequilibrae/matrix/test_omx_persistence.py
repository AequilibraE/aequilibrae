"""Our OMX update, preservation and temporary-object cleanup contracts."""

import h5py
import numpy as np
import openmatrix as omx
import pytest
from scipy import sparse

from aequilibrae.matrix import AequilibraeMatrix, AequilibraeSparseMatrix, MatrixStore, from_file
from aequilibrae.matrix.aequilibrae_matrix import OMX_SPECIAL_ATTRS


@pytest.fixture(params=[np.asarray, sparse.coo_array], ids=["dense", "sparse"])
def as_matrix_values(request):
    return request.param


@pytest.fixture
def matrix(as_matrix_values):
    matrix_type = AequilibraeSparseMatrix if as_matrix_values is sparse.coo_array else AequilibraeMatrix
    result = matrix_type(index=[20, 10], index_name="zones")
    result["demand"] = as_matrix_values(np.array([[0, 2], [3, 0]], dtype=np.float64))
    result["untouched"] = as_matrix_values(np.array([[4, 0], [0, 5]], dtype=np.int16))
    result.metadata.update(year=2026, source_only="source metadata")
    result.matrix_metadata("demand").update(purpose="work", source_only="source attribute")
    result.matrix_metadata("untouched")["purpose"] = "unchanged"
    return result


@pytest.fixture
def destination_path(tmp_path, matrix):
    path = tmp_path / "destination.omx"
    matrix.save_as_omx(path)
    return path


@pytest.fixture(params=["memory", "store"])
def source(request, matrix, tmp_path):
    if request.param == "memory":
        yield matrix
    else:
        path = tmp_path / "source.omx"
        matrix.save_as_omx(path)
        with MatrixStore(path, index_name="zones", mode="r+") as store:
            yield store


@pytest.fixture
def narrow_index_path(tmp_path):
    path = tmp_path / "narrow_index.omx"
    with omx.open_file(path, mode="x", shape=(2, 2)) as file:
        index = file.lookup.create_dataset("zones", data=np.array([20, 10], dtype=np.uint8))
        index.attrs["description"] = "Zone mapping"
    return path


def dense(value):
    return value.toarray() if sparse.issparse(value) else value


def assert_matrix_names_on_disk(path, expected):
    with h5py.File(path, "r") as file:
        assert set(file["data"]) | set(file.get("sparse", {})) == expected


def assert_original_demand_survives(path):
    with MatrixStore(path, index_name="zones") as loaded:
        np.testing.assert_array_equal(dense(loaded["demand"]), [[0, 2], [3, 0]])
        assert loaded["demand"].dtype == np.dtype(np.float64)
        assert loaded.matrix_metadata("demand")["purpose"] == "work"
        np.testing.assert_array_equal(dense(loaded["untouched"]), [[4, 0], [0, 5]])
    assert_matrix_names_on_disk(path, {"demand", "untouched"})


def test_subset_export_from_memory_or_store_updates_only_selected_matrices(tmp_path, source, as_matrix_values):
    path = tmp_path / "destination.omx"
    source.save_as_omx(path)
    other_values = np.array([[6, 0], [0, 7]])
    with MatrixStore(path, index_name="zones", mode="r+") as destination:
        destination["other_storage"] = (
            other_values if as_matrix_values is sparse.coo_array else sparse.csr_array(other_values)
        )
    source["demand"] = as_matrix_values(np.array([[0, 12], [13, 0]], dtype=np.float64))
    source["untouched"] = as_matrix_values(np.full((2, 2), -99))
    source.save_as_omx(path, mode="r+", subset=["demand"])

    with MatrixStore(path, index_name="zones") as loaded:
        assert set(loaded) == {"demand", "untouched", "other_storage"}
        np.testing.assert_array_equal(dense(loaded["demand"]), [[0, 12], [13, 0]])
        np.testing.assert_array_equal(dense(loaded["untouched"]), [[4, 0], [0, 5]])
        np.testing.assert_array_equal(dense(loaded["other_storage"]), other_values)


def test_save_writes_each_metadata_query_match_once(tmp_path, matrix, monkeypatch):
    path = tmp_path / "selected.omx"
    written = []
    original_write = matrix._write_omx_matrix

    def record_write(file, name):
        written.append(name)
        original_write(file, name)

    monkeypatch.setattr(matrix, "_write_omx_matrix", record_write)
    matrix.save_as_omx(path, subset=[{"purpose": "work"}, "demand", {"purpose": "work"}])

    assert written == ["demand"]
    assert set(from_file(path)) == {"demand"}


@pytest.mark.parametrize("dtype", [np.float64, np.int32], ids=["same_dtype", "new_dtype"])
def test_matrix_attributes_are_merged_when_values_are_updated(destination_path, matrix, as_matrix_values, dtype):
    with MatrixStore(destination_path, index_name="zones", mode="r+") as destination:
        destination.matrix_metadata("demand").update(purpose="old purpose", destination_only="keep attribute")
    matrix["demand"] = as_matrix_values(np.array([[0, 12], [13, 0]], dtype=dtype))
    matrix.matrix_metadata("demand").update(purpose="commute", TITLE="Matrix title")
    matrix.save_as_omx(destination_path, mode="r+", subset=["demand"])

    loaded = from_file(destination_path)
    assert dict(loaded.matrix_metadata("demand")) == {
        "purpose": "commute",
        "TITLE": "Matrix title",
        "source_only": "source attribute",
        "destination_only": "keep attribute",
    }
    assert loaded["demand"].dtype == np.dtype(dtype)
    np.testing.assert_array_equal(dense(loaded["demand"]), [[0, 12], [13, 0]])
    assert loaded.matrix_metadata("untouched")["purpose"] == "unchanged"


def test_file_attributes_are_merged_when_saving_an_update(destination_path, matrix):
    with MatrixStore(destination_path, index_name="zones", mode="r+") as destination:
        destination.metadata["destination_only"] = "keep metadata"
    matrix.metadata["year"] = 2030
    matrix.save_as_omx(destination_path, mode="r+")

    assert dict(from_file(destination_path).metadata) == {
        "year": 2030,
        "source_only": "source metadata",
        "destination_only": "keep metadata",
    }


def test_save_relabels_index_without_replacing_its_dtype_or_attributes(destination_path, matrix):
    with h5py.File(destination_path, "r+") as file:
        # An existing mapping may use a narrower dtype and carry its own attributes.
        del file["lookup/zones"]
        index = file["lookup"].create_dataset("zones", data=np.array([20, 10], dtype=np.uint16))
        index.attrs["description"] = "keep index attribute"
    matrix.index = [30, 40]
    matrix.save_as_omx(destination_path, mode="r+")

    with h5py.File(destination_path, "r") as file:
        np.testing.assert_array_equal(file["lookup/zones"], [30, 40])
        assert file["lookup/zones"].dtype == np.dtype(np.uint16)
        assert file["lookup/zones"].attrs["description"] == "keep index attribute"


@pytest.mark.parametrize("subset", [[], [{"purpose": "missing"}]], ids=["empty", "no_match"])
def test_empty_selection_creates_index_and_file_metadata_without_matrices(tmp_path, matrix, subset):
    path = tmp_path / "destination.omx"
    matrix.save_as_omx(path, mode="a", subset=subset)

    with MatrixStore(path, index_name="zones") as loaded:
        assert len(loaded) == 0
        np.testing.assert_array_equal(loaded.index, [20, 10])
        assert loaded.metadata["year"] == 2026
        np.testing.assert_array_equal(loaded.omx.attrs["SHAPE"], [2, 2])


@pytest.mark.parametrize("subset", [[], [{"purpose": "missing"}]], ids=["empty", "no_match"])
def test_empty_selection_updates_index_and_metadata_but_not_matrices(
    destination_path, matrix, as_matrix_values, subset
):
    matrix.index = [30, 40]
    matrix.metadata["year"] = 2030
    matrix["demand"] = as_matrix_values(np.full((2, 2), -99))
    matrix.save_as_omx(destination_path, mode="a", subset=subset)

    with MatrixStore(destination_path, index_name="zones") as loaded:
        assert set(loaded) == {"demand", "untouched"}
        np.testing.assert_array_equal(loaded.index, [30, 40])
        assert loaded.metadata["year"] == 2030
        np.testing.assert_array_equal(dense(loaded["demand"]), [[0, 2], [3, 0]])
        np.testing.assert_array_equal(dense(loaded["untouched"]), [[4, 0], [0, 5]])


def test_save_ignores_file_attributes_managed_by_omx(tmp_path, source):
    path = tmp_path / "destination.omx"
    source.save_as_omx(path, title="Destination title")
    with h5py.File(path, "r") as file:
        original = {name: file.attrs[name] for name in OMX_SPECIAL_ATTRS}
    source.metadata.update(dict.fromkeys(OMX_SPECIAL_ATTRS, "must not overwrite"))
    source.metadata["year"] = 2030
    source.save_as_omx(path, mode="r+")

    with h5py.File(path, "r") as file:
        for name, expected in original.items():
            np.testing.assert_array_equal(file.attrs[name], expected)
        assert file.attrs["year"] == 2030
    assert not from_file(path).metadata.keys() & OMX_SPECIAL_ATTRS


def test_sparse_omx_uses_the_csr_extension_layout(tmp_path):
    path = tmp_path / "sparse.omx"
    matrix = AequilibraeSparseMatrix(index=[20, 10])
    matrix["demand"] = sparse.coo_array([[0, 2], [3, 0]])
    matrix.save_as_omx(path)

    with h5py.File(path, "r") as file:
        group = file["sparse/demand"]
        assert "demand" not in file["data"]
        assert group.attrs["FORMAT"] == "csr"
        assert set(group) == {"row_index", "col_index", "data"}
        np.testing.assert_array_equal(group["row_index"], [0, 1, 2])
        np.testing.assert_array_equal(group["col_index"], [1, 0])
        np.testing.assert_array_equal(group["data"], [2, 3])
        np.testing.assert_array_equal(file.attrs["SHAPE"], [2, 2])


def test_sparse_save_ignores_user_format_and_load_hides_the_storage_marker(tmp_path):
    path = tmp_path / "sparse.omx"
    matrix = AequilibraeSparseMatrix(index=[20, 10])
    matrix["demand"] = sparse.coo_array([[0, 2], [3, 0]])
    matrix.matrix_metadata("demand")["FORMAT"] = "user format"
    matrix.save_as_omx(path)
    loaded = from_file(path)

    assert "FORMAT" not in loaded.matrix_metadata("demand")
    np.testing.assert_array_equal(loaded["demand"].toarray(), [[0, 2], [3, 0]])


def test_dense_matrix_metadata_keeps_the_user_format_attribute(tmp_path):
    path = tmp_path / "dense.omx"
    matrix = AequilibraeMatrix(index=[20, 10])
    matrix["demand"] = [[0, 2], [3, 0]]
    matrix.matrix_metadata("demand")["FORMAT"] = "user format"
    matrix.save_as_omx(path)

    assert from_file(path).matrix_metadata("demand")["FORMAT"] == "user format"


def test_same_dtype_dense_update_keeps_dataset_and_compression(tmp_path):
    path = tmp_path / "destination.omx"
    with MatrixStore(
        path, index=[20, 10], mode="x", filters={"complib": "gzip", "complevel": 9, "shuffle": True}
    ) as store:
        store["demand"] = np.array([[0, 2], [3, 0]], dtype=np.int16)
        store.matrix_metadata("demand")["purpose"] = "work"
        original = store.omx.data["demand"]
        chunks = original.chunks
        store["demand"] = np.array([[0, 12], [13, 0]], dtype=np.int16)
        replacement = store.omx.data["demand"]

        assert replacement.id == original.id
        assert replacement.compression == "gzip"
        assert replacement.compression_opts == 9
        assert replacement.shuffle
        assert replacement.chunks == chunks
        assert replacement.dtype == np.dtype(np.int16)
        assert replacement.attrs["purpose"] == "work"
        np.testing.assert_array_equal(original, [[0, 12], [13, 0]])


def test_dense_dtype_change_replaces_dataset_but_keeps_attributes(tmp_path):
    path = tmp_path / "destination.omx"
    with MatrixStore(path, index=[20, 10], mode="x") as store:
        store["demand"] = np.array([[0, 2], [3, 0]], dtype=np.int16)
        store.matrix_metadata("demand")["purpose"] = "work"
        original = store.omx.data["demand"]
        store["demand"] = np.array([[0, 12.5], [13.5, 0]], dtype=np.float64)
        replacement = store.omx.data["demand"]

        assert replacement.id != original.id
        assert replacement.attrs["purpose"] == "work"
        assert replacement.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(replacement, [[0, 12.5], [13.5, 0]])


def test_sparse_replacement_keeps_attributes_and_regenerates_its_format_marker(tmp_path):
    path = tmp_path / "destination.omx"
    with MatrixStore(path, index=[20, 10], mode="x") as store:
        store["demand"] = sparse.coo_array(np.array([[0, 2], [3, 0]], dtype=np.int16))
        store.matrix_metadata("demand")["purpose"] = "work"
        original = store.omx["/"]["sparse/demand"]
        # The storage marker must be regenerated rather than inherited from old attributes.
        original.attrs["FORMAT"] = "old format"
        store["demand"] = sparse.coo_array(np.array([[0, 12.5], [13.5, 0]], dtype=np.float64))
        replacement = store.omx["/"]["sparse/demand"]

        assert replacement.id != original.id
        assert replacement.attrs["purpose"] == "work"
        assert replacement.attrs["FORMAT"] == "csr"
        assert store["demand"].dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(store["demand"].toarray(), [[0, 12.5], [13.5, 0]])


@pytest.mark.parametrize("mode", ["a", "r+"])
def test_update_rejects_a_destination_with_a_different_shape(tmp_path, matrix, as_matrix_values, mode):
    path = tmp_path / "destination.omx"
    with MatrixStore(path, index=[20, 10, 30], index_name="zones", mode="x") as destination:
        destination["demand"] = as_matrix_values(np.ones((3, 3)))
    before = path.read_bytes()

    # These are preflight failures: the existing file must remain byte-for-byte unchanged.
    with pytest.raises(ValueError, match="destination matrix shape does not match"):
        matrix.save_as_omx(path, mode=mode)
    assert path.read_bytes() == before


@pytest.mark.parametrize("mode", ["a", "r+"])
def test_update_rejects_a_destination_mapping_with_a_different_length(destination_path, matrix, mode):
    with h5py.File(destination_path, "r+") as file:
        del file["lookup/zones"]
        file["lookup"].create_dataset("zones", data=[20, 10, 30])
    before = destination_path.read_bytes()

    with pytest.raises(ValueError, match="index shape may not change"):
        matrix.save_as_omx(destination_path, mode=mode)
    assert destination_path.read_bytes() == before


@pytest.mark.parametrize("mode", ["a", "r+"])
def test_update_rejects_a_lossy_zone_index_cast(destination_path, matrix, mode):
    with h5py.File(destination_path, "r+") as file:
        del file["lookup/zones"]
        file["lookup"].create_dataset("zones", data=np.array([20, 10], dtype=np.uint8))
    matrix.index = [20, 300]
    before = destination_path.read_bytes()

    with pytest.raises(ValueError):
        matrix.save_as_omx(destination_path, mode=mode)
    assert destination_path.read_bytes() == before


@pytest.mark.parametrize("mode", ["a", "r+"])
def test_storage_type_collision_is_rejected_before_writing_any_updates(tmp_path, source, as_matrix_values, mode):
    path = tmp_path / "destination.omx"
    with MatrixStore(path, index=[20, 10], index_name="zones", mode="x") as destination:
        destination["demand"] = (
            np.ones((2, 2)) if as_matrix_values is sparse.coo_array else sparse.csr_array(np.ones((2, 2)))
        )
    source["new"] = as_matrix_values(np.ones((2, 2)))
    source.index = [30, 40]
    source.metadata["year"] = 2030
    before = path.read_bytes()

    with pytest.raises(ValueError, match="storage type"):
        source.save_as_omx(path, mode=mode)
    assert path.read_bytes() == before


def test_shape_argument_is_validated_before_overwriting_an_existing_file(destination_path, matrix):
    before = destination_path.read_bytes()
    with pytest.raises(ValueError, match="destination matrix shape does not match"):
        matrix.save_as_omx(destination_path, mode="w", shape=(3, 3))
    assert destination_path.read_bytes() == before


def test_store_index_update_preserves_dataset_dtype_and_attributes(narrow_index_path):
    with MatrixStore(narrow_index_path, index_name="zones", mode="r+") as store:
        original = store.omx.lookup["zones"]
        store.index = [30, 40]

        assert store.omx.lookup["zones"].id == original.id
        assert original.dtype == np.dtype(np.uint8)
        assert original.attrs["description"] == "Zone mapping"
        np.testing.assert_array_equal(original, [30, 40])


def test_failed_store_index_cast_keeps_memory_and_disk_indices_unchanged(narrow_index_path):
    with MatrixStore(narrow_index_path, index_name="zones", mode="r+") as store:
        with pytest.raises(ValueError):
            store.index = [20, 300]

        np.testing.assert_array_equal(store.index, [20, 10])
        np.testing.assert_array_equal(store.omx.lookup["zones"], [20, 10])


@pytest.mark.parametrize("mode", [None, "x", "w-"], ids=["default", "exclusive", "exclusive_alias"])
def test_exclusive_save_refuses_existing_file_without_modifying_it(destination_path, matrix, mode):
    before = destination_path.read_bytes()
    kwargs = {} if mode is None else {"mode": mode}
    with pytest.raises(FileExistsError):
        matrix.save_as_omx(destination_path, **kwargs)
    assert destination_path.read_bytes() == before


def test_overwrite_replaces_existing_file_contents(tmp_path, matrix):
    path = tmp_path / "destination.omx"
    old = AequilibraeMatrix(index=[1, 2, 3], index_name="old_index")
    old["obsolete"] = np.ones((3, 3))
    old.metadata["obsolete"] = "old metadata"
    old.save_as_omx(path)
    matrix.save_as_omx(path, mode="w")

    loaded = from_file(path)
    assert set(loaded) == {"demand", "untouched"}
    np.testing.assert_array_equal(loaded.index, matrix.index)
    assert loaded.index_name == "zones"
    assert "obsolete" not in loaded.metadata
    for name in matrix:
        np.testing.assert_array_equal(dense(loaded[name]), dense(matrix[name]))
    with h5py.File(path, "r") as file:
        assert set(file["lookup"]) == {"zones"}


@pytest.mark.parametrize("mode", ["a", "w", "x", "w-"])
def test_save_modes_can_create_missing_file(tmp_path, matrix, mode):
    path = tmp_path / "destination.omx"
    matrix.save_as_omx(path, mode=mode)

    assert from_file(path) == matrix


def test_update_mode_requires_an_existing_file(tmp_path, matrix):
    path = tmp_path / "missing.omx"
    with pytest.raises(FileNotFoundError):
        matrix.save_as_omx(path, mode="r+")
    assert not path.exists()


@pytest.mark.parametrize("mode", ["r", "invalid"])
def test_unsupported_save_mode_is_rejected_before_creating_a_file(tmp_path, matrix, mode):
    path = tmp_path / "destination.omx"
    with pytest.raises(ValueError, match="save_as_omx only supports modes"):
        matrix.save_as_omx(path, mode=mode)
    assert not path.exists()


@pytest.fixture
def fail_after_temporary_matrix_created(monkeypatch, destination_path, as_matrix_values):
    # Create the destination before enabling failures. Fail before the old matrix
    # is deleted, not during the final storage swap.
    attempted = []
    if as_matrix_values is sparse.coo_array:
        original_create = h5py.Group.create_dataset

        def fail_after_creation(group, name, *args, **kwargs):
            result = original_create(group, name, *args, **kwargs)
            if group.name.rsplit("/", 1)[-1].startswith("__aequilibrae_") and name == "data":
                attempted.append(group.name)
                raise OSError("injected matrix write failure")
            return result

        monkeypatch.setattr(h5py.Group, "create_dataset", fail_after_creation)
    else:
        original_create = omx.File.create_matrix

        def fail_after_creation(file, name, *args, **kwargs):
            result = original_create(file, name, *args, **kwargs)
            if name.startswith("__aequilibrae_"):
                attempted.append(name)
                raise OSError("injected matrix write failure")
            return result

        monkeypatch.setattr(omx.File, "create_matrix", fail_after_creation)
    return attempted


def test_failed_save_preserves_original_matrix_and_removes_temporary_nodes(
    destination_path, matrix, as_matrix_values, fail_after_temporary_matrix_created
):
    matrix["demand"] = as_matrix_values(np.array([[0, 12], [13, 0]], dtype=np.int32))
    with pytest.raises(OSError, match="injected matrix write failure"):
        matrix.save_as_omx(destination_path, mode="r+", subset=["demand"])

    assert len(fail_after_temporary_matrix_created) == 1
    assert_original_demand_survives(destination_path)


def test_failed_store_assignment_preserves_original_matrix_and_removes_temporary_nodes(
    destination_path, as_matrix_values, fail_after_temporary_matrix_created
):
    with MatrixStore(destination_path, index_name="zones", mode="r+") as destination:
        with pytest.raises(OSError, match="injected matrix write failure"):
            destination["demand"] = as_matrix_values(np.array([[0, 12], [13, 0]], dtype=np.int32))

    assert len(fail_after_temporary_matrix_created) == 1
    assert_original_demand_survives(destination_path)


def test_failed_save_of_new_matrix_leaves_no_partial_matrix(
    destination_path, matrix, fail_after_temporary_matrix_created
):
    with MatrixStore(destination_path, index_name="zones", mode="r+") as destination:
        del destination["demand"]
    with pytest.raises(OSError, match="injected matrix write failure"):
        matrix.save_as_omx(destination_path, mode="r+", subset=["demand"])

    assert len(fail_after_temporary_matrix_created) == 1
    assert_matrix_names_on_disk(destination_path, {"untouched"})


def test_failed_store_assignment_of_new_matrix_leaves_no_partial_matrix(
    destination_path, as_matrix_values, fail_after_temporary_matrix_created
):
    with MatrixStore(destination_path, index_name="zones", mode="r+") as destination:
        del destination["demand"]
        with pytest.raises(OSError, match="injected matrix write failure"):
            destination["demand"] = as_matrix_values(np.ones((2, 2)))

    assert len(fail_after_temporary_matrix_created) == 1
    assert_matrix_names_on_disk(destination_path, {"untouched"})
