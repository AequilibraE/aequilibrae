from collections.abc import MutableMapping
from contextlib import ExitStack

import numpy as np
import openmatrix as omx
import pandas as pd
import pytest
import scipy.sparse as sparse

from aequilibrae.matrix import (
    AequilibraeMatrix,
    AequilibraeSparseMatrix,
    MatrixStore,
    from_df,
    from_dict,
    from_file,
    from_scipy,
)


@pytest.fixture
def index():
    return np.array([309, 0, 101])


@pytest.fixture
def data():
    return {
        "car": np.array([[0, 12.5, 8], [10, 0, 6], [7, 9, 0]]),
        "bus": np.array([[0, 1.5, np.nan], [2, 0, 3], [4, 5, 0]]),
    }


@pytest.fixture
def metadata():
    return {"description": "Test demand", "year": 2026}


@pytest.fixture
def manage():
    with ExitStack() as stack:

        def register(matrix):
            if isinstance(matrix, MatrixStore):
                stack.callback(matrix.close)
            return matrix

        yield register


@pytest.fixture(params=[AequilibraeMatrix, MatrixStore], ids=["memory", "store"])
def matrix_factory(request, tmp_path, manage):
    def create(index, index_name="zones"):
        if request.param is MatrixStore:
            return manage(MatrixStore(tmp_path / "matrix.omx", index, index_name, mode="w"))

        return AequilibraeMatrix(index, index_name)

    return create


@pytest.fixture
def matrix(matrix_factory, index, data, metadata):
    matrix = matrix_factory(index)
    matrix.update(data)
    matrix.metadata.update(metadata)

    return matrix


@pytest.fixture
def matrix_pair():
    data = {"car": [[0.0, 1.0], [2.0, 0.0]]}
    return from_dict(data, [10, 20]), from_dict(data, [10, 20])


@pytest.fixture
def memory_matrix(index, data):
    return from_dict(data, index, index_name="zones")


@pytest.fixture
def omx_path(tmp_path, index, data, metadata):
    # Create input independently of save_as_omx so loading is tested separately.
    path = tmp_path / "input.omx"
    with omx.open_file(path, "w") as file:
        file.create_mapping("zones", index)

        for name, values in data.items():
            file[name] = values
            file[name].attrs["mode"] = name
        file.attrs.update(metadata)

    return path


@pytest.fixture
def uint8_index_omx_path(tmp_path):
    path = tmp_path / "uint8_index.omx"
    with omx.open_file(path, "w") as file:
        # Use a small dtype to test casts that may change zone IDs.
        dataset = file.create_mapping("zones", np.array([10, 20], dtype=np.uint8))
        dataset.attrs["description"] = "Zone IDs"
    return path


def assert_matrix(matrix, index, data, index_name="zones", metadata=None):
    assert matrix.index_name == index_name
    assert matrix.shape == (len(index), len(index))
    np.testing.assert_array_equal(matrix.index, index)

    assert set(matrix) == set(data)
    assert len(matrix) == len(data)

    for name, expected in data.items():
        actual = matrix[name]
        np.testing.assert_array_equal(actual.toarray() if isinstance(actual, sparse.sparray) else actual, expected)

    if metadata is not None:
        for key, value in metadata.items():
            assert matrix.metadata[key] == value


def test_construction_copies_index(matrix_factory, index):
    supplied = index.copy()
    matrix = matrix_factory(supplied)
    supplied[0] = 999
    np.testing.assert_array_equal(matrix.index, index)


def test_index_is_read_only(matrix):
    with pytest.raises(ValueError):
        matrix.index[0] = 999


def test_assignment_copies_values(matrix):
    values = np.zeros(matrix.shape)
    matrix["car"] = values
    values[:] = 999
    np.testing.assert_array_equal(matrix["car"], np.zeros(matrix.shape))


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_assignment_uses_supplied_dtype(matrix, dtype):
    matrix["car"] = np.zeros(matrix.shape, dtype=dtype)
    assert matrix["car"].dtype == np.dtype(dtype)


@pytest.mark.parametrize("array_type", [np.asarray, list, tuple], ids=["array", "list", "tuple"])
def test_assignment_accepts_array_like(matrix, data, array_type):
    matrix["car"] = array_type(data["car"].tolist())
    np.testing.assert_array_equal(matrix["car"], data["car"])
    assert matrix["car"].dtype == data["car"].dtype


@pytest.mark.parametrize("dtype", [object, "U3", "S3", "M8[D]", "m8[D]", "V8"])
def test_non_numeric_matrix_dtypes_are_rejected(matrix, index, data, metadata, dtype):
    with pytest.raises(TypeError, match="dtype"):
        matrix["car"] = np.zeros(matrix.shape, dtype=dtype)
    assert_matrix(matrix, index, data, metadata=metadata)


@pytest.mark.parametrize("invalid_values", [1, [[1, 2], [3]]])
def test_invalid_assignment_preserves_values(matrix, index, data, metadata, invalid_values):
    with pytest.raises((TypeError, ValueError)):
        matrix["car"] = invalid_values
    assert_matrix(matrix, index, data, metadata=metadata)


@pytest.mark.parametrize("store", [False, True], ids=["memory", "store"])
def test_from_dict_accepts_array_like(tmp_path, manage, index, data, store):
    kwargs = {"path": tmp_path / "lists.omx"} if store else {}
    matrix = manage(from_dict({name: values.tolist() for name, values in data.items()}, index, store=store, **kwargs))
    assert_matrix(matrix, index, data, index_name="main_index")


def test_from_dict(index, data, metadata):
    matrix = from_dict(data, index, index_name="zones", metadata=metadata)
    assert isinstance(matrix, AequilibraeMatrix)
    assert_matrix(matrix, index, data, metadata=metadata)


def test_from_dict_store_persists_data(tmp_path, index, data, metadata):
    path = tmp_path / "demand.omx"
    with from_dict(data, index, index_name="zones", metadata=metadata, path=path, store=True) as store:
        assert isinstance(store, MatrixStore)
    assert_matrix(from_file(path, index_name="zones"), index, data, metadata=metadata)


def test_from_dict_copies_metadata(index):
    periods = np.array([1, 2, 3])
    matrix = from_dict({}, index, metadata={"periods": periods})
    periods[0] = 9
    np.testing.assert_array_equal(matrix.metadata["periods"], [1, 2, 3])


@pytest.mark.parametrize("store", [False, True], ids=["memory", "store"])
def test_from_dict_ignores_omx_attributes(tmp_path, manage, index, data, store):
    attributes = {"OMX_VERSION": "ignored", "OMX_CREATED_WITH": "ignored", "SHAPE": [99, 99], "TITLE": "ignored"}
    kwargs = {"path": tmp_path / "demand.omx"} if store else {}
    matrix = manage(from_dict(data, index, metadata=attributes, store=store, **kwargs))
    np.testing.assert_array_equal(matrix.metadata.get("SHAPE", matrix.shape), matrix.shape)
    assert matrix.metadata.get("OMX_VERSION") != "ignored"
    assert matrix.metadata.get("OMX_CREATED_WITH") != "ignored"
    assert "TITLE" not in matrix.metadata


@pytest.mark.parametrize("store", [False, True], ids=["memory", "store"])
def test_from_file(omx_path, manage, index, data, metadata, store):
    matrix = manage(from_file(omx_path, index_name="zones", store=store))
    assert isinstance(matrix, MatrixStore if store else AequilibraeMatrix)
    assert_matrix(matrix, index, data, metadata=metadata)


@pytest.mark.parametrize("store", [False, True], ids=["memory", "store"])
def test_from_file_infers_single_index(omx_path, manage, index, data, store):
    matrix = manage(from_file(omx_path, store=store))
    assert isinstance(matrix, MatrixStore if store else AequilibraeMatrix)
    assert_matrix(matrix, index, data, index_name="zones")


def test_from_file_prefers_main_index_when_multiple_exist(omx_path, index, data):
    with omx.open_file(omx_path, mode="a") as file:
        file.create_mapping("main_index", index + 1000)

    matrix = from_file(omx_path)
    assert_matrix(matrix, index + 1000, data, index_name="main_index")


def test_from_file_requires_index_when_multiple_nondefault_indices_exist(omx_path, index):
    with omx.open_file(omx_path, mode="a") as file:
        file.create_mapping("other_zones", index + 1000)

    with pytest.raises(ValueError, match="Multiple zone indices"):
        from_file(omx_path)
    assert from_file(omx_path, index_name="zones").index_name == "zones"


@pytest.mark.parametrize("mode, expected", [(None, "r"), ("r", "r"), ("r+", "r+")])
def test_from_file_store_modes(omx_path, mode, expected):
    with from_file(omx_path, index_name="zones", mode=mode, store=True) as store:
        assert store.omx.mode == expected


@pytest.mark.parametrize("mode", ["w", "a", "w-", "x", "invalid"])
def test_from_file_store_rejects_invalid_modes(omx_path, index, data, metadata, mode):
    with pytest.raises(ValueError, match="only support modes 'r' and 'r\\+'"):
        from_file(omx_path, index_name="zones", mode=mode, store=True)

    assert_matrix(from_file(omx_path, index_name="zones"), index, data, metadata=metadata)


@pytest.mark.parametrize("mode", ["r", "r+"])
def test_from_file_store_requires_existing_file(tmp_path, mode):
    path = tmp_path / "missing.omx"
    with pytest.raises(FileNotFoundError):
        from_file(path, mode=mode, store=True)
    assert not path.exists()


def test_from_file_memory_rejects_mode(tmp_path):
    with pytest.raises(ValueError, match="in-memory matrix cannot have a mode"):
        from_file(tmp_path / "missing.omx", mode="r")


def test_from_omx(omx_path, index, data, metadata):
    matrix = AequilibraeMatrix.from_omx(omx_path, index_name="zones")
    assert_matrix(matrix, index, data, metadata=metadata)


@pytest.mark.parametrize("subset", [["car"], []], ids=["one", "empty"])
def test_from_file_subset(omx_path, index, data, metadata, subset):
    matrix = from_file(omx_path, index_name="zones", subset=subset)
    assert_matrix(matrix, index, {name: data[name] for name in subset}, metadata=metadata)


def test_store_load_rejects_subset(omx_path):
    with pytest.raises(ValueError):
        from_file(omx_path, index_name="zones", store=True, subset=["car"])


def test_from_dict_store_requires_path(index, data):
    with pytest.raises(ValueError):
        from_dict(data, index, store=True)


def test_from_dict_memory_rejects_path(tmp_path, index, data):
    with pytest.raises(ValueError):
        from_dict(data, index, path=tmp_path / "demand.omx")


def test_from_dict_memory_rejects_mode(index, data):
    with pytest.raises(ValueError, match="in-memory matrix cannot have a mode"):
        from_dict(data, index, mode="x")


@pytest.mark.parametrize("mode", ["r", "invalid"])
def test_from_dict_store_rejects_invalid_modes(tmp_path, index, data, mode):
    path = tmp_path / "invalid.omx"
    with pytest.raises(ValueError, match="from_dict stores only support modes"):
        from_dict(data, index, path=path, mode=mode, store=True)
    assert not path.exists()


def test_from_dict_store_does_not_overwrite_by_default(omx_path, index, data, metadata):
    with pytest.raises(FileExistsError):
        from_dict(data, [809, 0, 801], path=omx_path, index_name="zones", store=True)

    assert_matrix(from_file(omx_path, index_name="zones"), index, data, metadata=metadata)


@pytest.mark.parametrize("mode", ["r+", "a", "w"])
@pytest.mark.parametrize("invalid_values", [np.zeros((2, 2)), np.zeros((3, 3), dtype=object)])
def test_from_dict_checks_values_before_changing_file(omx_path, index, data, metadata, mode, invalid_values):
    with pytest.raises((TypeError, ValueError)):
        from_dict({"car": invalid_values}, [809, 0, 801], path=omx_path, index_name="zones", mode=mode, store=True)
    assert_matrix(from_file(omx_path, index_name="zones"), index, data, metadata=metadata)


def test_from_dict_checks_names_before_creating_file(tmp_path, index, data):
    path = tmp_path / "invalid.omx"
    with pytest.raises(ValueError, match="not a path"):
        from_dict({"mode/car": data["car"]}, index, path=path, store=True)
    assert not path.exists()


def test_from_dict_closes_store_after_failed_metadata_write(tmp_path, index, data):
    path = tmp_path / "invalid_metadata.omx"
    with pytest.raises(TypeError) as error:
        from_dict(data, index, path=path, store=True, metadata={"bad": {"nested": "value"}})
    # The traceback still holds the failed call's local variables.
    assert error.traceback
    with omx.open_file(path, "r") as file:
        assert file.mode == "r"


def test_from_dict_store_allows_explicit_update(omx_path, index):
    values = np.ones((3, 3))
    with from_dict({"car": values}, index, path=omx_path, index_name="zones", mode="r+", store=True) as store:
        np.testing.assert_array_equal(store["car"], values)


@pytest.mark.parametrize("subset", [None, ["car"], []], ids=["all", "one", "empty"])
def test_omx_round_trip(matrix, tmp_path, index, data, metadata, subset):
    path = tmp_path / "export.omx"
    matrix.save_as_omx(path, subset=subset)

    loaded = from_file(path, index_name="zones")
    expected = data if subset is None else {name: data[name] for name in subset}

    assert_matrix(loaded, index, expected, metadata=metadata)
    assert_matrix(matrix, index, data, metadata=metadata)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.float32, np.float64, np.bool_, np.complex128])
def test_matrix_dtype_survives_save(matrix, tmp_path, dtype):
    values = np.arange(9).reshape(3, 3).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values += 1j
    matrix["car"] = values
    path = tmp_path / "dtype.omx"
    matrix.save_as_omx(path, subset=["car"])
    loaded = from_file(path, index_name="zones")
    assert loaded["car"].dtype == np.dtype(dtype)
    np.testing.assert_array_equal(loaded["car"], values)


def test_save_default_does_not_overwrite(memory_matrix, omx_path, index, data, metadata):
    with pytest.raises(FileExistsError):
        memory_matrix.save_as_omx(omx_path)
    assert_matrix(from_file(omx_path, index_name="zones"), index, data, metadata=metadata)


@pytest.mark.parametrize("mode", ["r+", "a"])
def test_save_updates_existing_values(memory_matrix, omx_path, mode):
    memory_matrix["car"] = np.full(memory_matrix.shape, 7.0)
    memory_matrix.save_as_omx(omx_path, mode=mode)
    np.testing.assert_array_equal(from_file(omx_path, index_name="zones")["car"], memory_matrix["car"])


@pytest.mark.parametrize("mode", ["x", "w-", "a", "w"])
def test_save_creates_new_file(memory_matrix, tmp_path, index, data, mode):
    path = tmp_path / "new.omx"
    memory_matrix.save_as_omx(path, mode=mode)
    assert_matrix(from_file(path, index_name="zones"), index, data)


@pytest.mark.parametrize("mode", ["x", "w-"])
def test_save_create_mode_rejects_existing_file(omx_path, index, data, metadata, mode):
    matrix = from_dict({}, index, index_name="zones")
    with pytest.raises(FileExistsError):
        matrix.save_as_omx(omx_path, mode=mode)
    assert_matrix(from_file(omx_path, index_name="zones"), index, data, metadata=metadata)


def test_save_update_mode_requires_existing_file(memory_matrix, tmp_path):
    path = tmp_path / "missing.omx"
    with pytest.raises(FileNotFoundError):
        memory_matrix.save_as_omx(path, mode="r+")
    assert not path.exists()


@pytest.mark.parametrize("mode", ["r", "invalid", None])
def test_save_rejects_invalid_modes(memory_matrix, tmp_path, mode):
    path = tmp_path / "invalid.omx"
    with pytest.raises(ValueError, match="only supports modes"):
        memory_matrix.save_as_omx(path, mode=mode)
    assert not path.exists()


def test_save_write_mode_replaces_entire_file(omx_path):
    matrix = from_dict({"new": np.ones((2, 2))}, [10, 20], metadata={"description": "New file"})
    matrix.save_as_omx(omx_path, "w")
    loaded = from_file(omx_path)
    assert_matrix(loaded, [10, 20], {"new": np.ones((2, 2))}, index_name="main_index")
    assert loaded.metadata == {"description": "New file"}
    with omx.open_file(omx_path, "r") as file:
        assert file.list_mappings() == ["main_index"]


@pytest.mark.parametrize("mode", ["r+", "a"])
def test_save_updates_existing_index(uint8_index_omx_path, mode):
    matrix = from_dict({"car": np.ones((2, 2))}, [30, 40], index_name="zones")
    matrix.save_as_omx(uint8_index_omx_path, mode=mode)
    with omx.open_file(uint8_index_omx_path, "r") as file:
        dataset = file.lookup["zones"]
        assert dataset.dtype == np.dtype("uint8")
        assert dataset.attrs["description"] == "Zone IDs"
        np.testing.assert_array_equal(dataset[:], [30, 40])
        np.testing.assert_array_equal(file["car"][:], matrix["car"])


def test_save_rejects_lossy_index_cast_before_writing(uint8_index_omx_path):
    matrix = from_dict({"car": np.ones((2, 2))}, [100, 300], index_name="zones")
    with pytest.raises(ValueError, match="same_value"):
        matrix.save_as_omx(uint8_index_omx_path, mode="r+", title="Changed title")

    with omx.open_file(uint8_index_omx_path, "r") as file:
        dataset = file.lookup["zones"]
        assert dataset.dtype == np.dtype("uint8")
        assert dataset.attrs["description"] == "Zone IDs"
        np.testing.assert_array_equal(dataset[:], [10, 20])
        assert file.list_matrices() == []
        assert "TITLE" not in file.attrs


def test_save_update_preserves_matrix_attributes(matrix, tmp_path, index, data, metadata):
    path = tmp_path / "attributes.omx"
    matrix.matrix_metadata("car").update({"mode": "car", "units": "trips"})
    matrix.matrix_metadata("bus")["mode"] = "bus"
    matrix.save_as_omx(path)
    with omx.open_file(path, "a") as file:
        file["car"].attrs["periods"] = [1, 2, 3]

    replacement = np.full(matrix.shape, 7.0)
    matrix["car"] = replacement
    matrix.save_as_omx(path, mode="r+", subset=["car"])

    assert_matrix(from_file(path, index_name="zones"), index, {**data, "car": replacement}, metadata=metadata)
    with omx.open_file(path, "r") as file:
        attrs = file["car"].attrs
        assert attrs["mode"] == "car"
        assert attrs["units"] == "trips"
        np.testing.assert_array_equal(attrs["periods"], [1, 2, 3])
        assert file["bus"].attrs["mode"] == "bus"


def test_save_subset_keeps_unselected_matrices(memory_matrix, omx_path, data):
    memory_matrix["car"] = np.zeros(memory_matrix.shape)
    memory_matrix.save_as_omx(omx_path, mode="r+", subset=["car"])
    loaded = from_file(omx_path, index_name="zones")
    np.testing.assert_array_equal(loaded["car"], memory_matrix["car"])
    np.testing.assert_array_equal(loaded["bus"], data["bus"])


@pytest.mark.parametrize(
    "subset, names",
    [
        ([{"mode": "car"}], {"car"}),
        ([{"mode": "car", "year": 2030}], {"car"}),
        ([{"mode": "car", "year": 2029}], set()),
        ([{"missing": "car"}], set()),
        ([{"description": "Test demand"}], set()),
        ([{}], {"car", "bus"}),
        (["bus", {"mode": "car"}, "car"], {"car", "bus"}),
        ([{"mode": "car"}, {"mode": "bus"}], {"car", "bus"}),
        ([{"mode": "car"}, {"year": 2030}, "car"], {"car"}),
        ([{"periods": [1, 2]}], {"car"}),
        ([{"value": np.nan}], {"car"}),
        ([{"values": [1.0, np.nan]}], {"car"}),
    ],
)
def test_save_attribute_queries(matrix, tmp_path, index, data, metadata, subset, names):
    matrix.matrix_metadata("car").update(
        {"mode": "car", "year": 2030, "periods": [1, 2], "value": np.nan, "values": [1.0, np.nan]}
    )
    matrix.matrix_metadata("bus")["mode"] = "bus"
    path = tmp_path / "query.omx"

    matrix.save_as_omx(path, subset=subset)

    loaded = from_file(path, index_name="zones")
    assert_matrix(loaded, index, {name: data[name] for name in names}, metadata=metadata)
    for name in names:
        for key, value in matrix.matrix_metadata(name).items():
            np.testing.assert_array_equal(loaded.matrix_metadata(name)[key], value)
    assert_matrix(matrix, index, data, metadata=metadata)


@pytest.mark.parametrize("mode", ["r+", "a"])
def test_save_attribute_query_updates_source_matches(matrix, omx_path, index, data, metadata, mode):
    matrix.matrix_metadata("car")["mode"] = "source"
    matrix["car"] = np.zeros(matrix.shape)
    matrix["bus"] = np.ones(matrix.shape)
    matrix.index = index + 1000
    matrix.metadata["description"] = "Updated demand"

    matrix.save_as_omx(omx_path, mode=mode, subset=[{"mode": "source"}])

    loaded = from_file(omx_path, index_name="zones")
    assert_matrix(
        loaded,
        index + 1000,
        {"car": np.zeros(matrix.shape), "bus": data["bus"]},
        metadata={**metadata, "description": "Updated demand"},
    )
    assert loaded.matrix_metadata("car")["mode"] == "source"
    assert loaded.matrix_metadata("bus")["mode"] == "bus"


@pytest.mark.parametrize("mode", ["r+", "a"])
def test_save_unmatched_query_still_updates_index_and_metadata(matrix, omx_path, index, data, mode):
    matrix.index = index + 1000
    matrix.metadata["description"] = "Updated demand"
    matrix["car"] = np.zeros(matrix.shape)

    matrix.save_as_omx(omx_path, mode=mode, subset=[{"missing": "car"}])

    loaded = from_file(omx_path, index_name="zones")
    assert_matrix(loaded, index + 1000, data, metadata={"description": "Updated demand"})


@pytest.mark.parametrize(
    "entry, error, message",
    [
        ("missing", KeyError, "missing"),
        ("bad/path", ValueError, "not a path"),
        (None, TypeError, "matrix names or attribute queries"),
        (17, TypeError, "matrix names or attribute queries"),
        (["car"], TypeError, "matrix names or attribute queries"),
    ],
)
def test_save_rejects_invalid_query_subset_before_creating_file(matrix, tmp_path, entry, error, message):
    path = tmp_path / "invalid.omx"
    with pytest.raises(error, match=message):
        matrix.save_as_omx(path, subset=[{}, entry])
    assert not path.exists()


@pytest.mark.parametrize("mode", ["w", "r+", "a"])
@pytest.mark.parametrize("entry, error", [("missing", KeyError), (None, TypeError)])
def test_save_rejects_invalid_query_subset_before_changing_file(matrix, omx_path, mode, entry, error):
    original = omx_path.read_bytes()
    with pytest.raises(error):
        matrix.save_as_omx(omx_path, mode=mode, subset=[{}, entry])
    assert omx_path.read_bytes() == original


def test_save_attribute_query_reads_each_selected_matrix_once(matrix, tmp_path, monkeypatch):
    matrix.matrix_metadata("car")["mode"] = "car"
    reads = []
    original_getitem = type(matrix).__getitem__

    def track_reads(self, key):
        reads.append(key)
        return original_getitem(self, key)

    monkeypatch.setattr(type(matrix), "__getitem__", track_reads)
    matrix.save_as_omx(tmp_path / "selected.omx", subset=[{"mode": "car"}, "car", {"mode": "car"}])
    assert reads == ["car"]


def test_read_only_store_save_attribute_query(omx_path, tmp_path, index, data, metadata):
    path = tmp_path / "selected.omx"
    with from_file(omx_path, index_name="zones", store=True) as store:
        store.save_as_omx(path, subset=[{"mode": "car"}])

    loaded = from_file(path, index_name="zones")
    assert_matrix(loaded, index, {"car": data["car"]}, metadata=metadata)
    assert loaded.matrix_metadata("car")["mode"] == "car"


def test_save_empty_subset_still_writes_index(memory_matrix, omx_path, index):
    memory_matrix.index = index + 1000
    memory_matrix.save_as_omx(omx_path, mode="r+", subset=[])
    np.testing.assert_array_equal(from_file(omx_path, index_name="zones").index, index + 1000)


def test_save_merges_file_metadata(memory_matrix, omx_path, metadata):
    memory_matrix.metadata.update({"description": "Changed demand", "units": "trips"})
    memory_matrix.save_as_omx(omx_path, mode="r+", subset=[])
    assert from_file(omx_path, index_name="zones").metadata == {
        **metadata,
        "description": "Changed demand",
        "units": "trips",
    }


@pytest.mark.parametrize(
    "value",
    [2030, "Morning demand", [1, 2, 3], ["car", "bus"], np.nan, [1.0, np.nan]],
    ids=["number", "text", "array", "text-array", "nan", "array-with-nan"],
)
def test_save_accepts_unchanged_metadata(tmp_path, value):
    path = tmp_path / "metadata.omx"
    matrix = from_dict({"car": np.ones((2, 2))}, [10, 20], metadata={"value": value})
    matrix.save_as_omx(path)
    matrix["bus"] = np.zeros((2, 2))
    matrix.save_as_omx(path, mode="r+", subset=["bus"])
    loaded = from_file(path)
    assert set(loaded) == {"car", "bus"}
    np.testing.assert_array_equal(loaded.metadata["value"], value)


@pytest.mark.parametrize("replacement", [[1, 3], [1], [1, np.nan]])
def test_save_updates_metadata_arrays(tmp_path, replacement):
    path = tmp_path / "metadata.omx"
    matrix = from_dict({"car": np.ones((2, 2))}, [10, 20], metadata={"values": [1, 2]})
    matrix.save_as_omx(path)
    matrix.metadata["values"] = replacement
    matrix["bus"] = np.zeros((2, 2))
    matrix.save_as_omx(path, mode="r+", subset=["bus"])
    np.testing.assert_array_equal(from_file(path).metadata["values"], replacement)


def test_save_writes_supplied_metadata_dtype(tmp_path):
    path = tmp_path / "metadata_dtype.omx"
    matrix = from_dict({}, [10, 20], metadata={"year": np.int16(2030)})
    matrix.save_as_omx(path)
    matrix.metadata["year"] = np.int32(2030)
    matrix.save_as_omx(path, mode="r+")
    with omx.open_file(path, "r") as file:
        assert file.attrs["year"].dtype == np.dtype("int32")


@pytest.mark.parametrize("mode", ["r+", "a"])
def test_save_rejects_destination_shape_before_writing(memory_matrix, tmp_path, mode):
    path = tmp_path / "different_shape.omx"
    original = from_dict({"other": np.ones((2, 2))}, [10, 20], index_name="other")
    original.save_as_omx(path)

    with pytest.raises(ValueError, match="destination matrix shape"):
        memory_matrix.save_as_omx(path, mode=mode)
    assert_matrix(from_file(path, index_name="other"), [10, 20], {"other": np.ones((2, 2))}, index_name="other")
    with omx.open_file(path, "r") as file:
        assert "zones" not in file.list_mappings()


def test_save_rejects_shape_option_before_creating_file(memory_matrix, tmp_path):
    path = tmp_path / "invalid.omx"
    with pytest.raises(ValueError, match="destination matrix shape"):
        memory_matrix.save_as_omx(path, shape=(2, 2))
    assert not path.exists()


@pytest.mark.parametrize("subset", [["car"], [], ["bus", "car"]], ids=["one", "empty", "all"])
def test_subset_selects_names(matrix, index, data, subset):
    selected = matrix.subset(subset)
    assert isinstance(selected, AequilibraeMatrix)
    assert_matrix(selected, index, {name: data[name] for name in subset})


def test_subset_copies_values(matrix, data):
    selected = matrix.subset(["car"])
    selected["car"][:] = 42
    np.testing.assert_array_equal(matrix["car"], data["car"])


def test_subset_preserves_dtype(matrix):
    matrix["car"] = np.zeros(matrix.shape, dtype=np.float32)
    assert matrix.subset(["car"])["car"].dtype == np.dtype("float32")


def test_subset_copies_index(matrix, index):
    selected = matrix.subset(["car"])
    assert not np.shares_memory(selected.index, matrix.index)
    selected.index = index + 1000
    np.testing.assert_array_equal(matrix.index, index)


def test_subset_copies_file_metadata(matrix, metadata):
    matrix.metadata["periods"] = np.array([1, 2, 3])
    selected = matrix.subset(["car"])
    assert selected.metadata["description"] == metadata["description"]
    np.testing.assert_array_equal(selected.metadata["periods"], [1, 2, 3])
    assert not np.shares_memory(selected.metadata["periods"], matrix.metadata["periods"])
    selected.metadata["description"] = "Selected demand"
    assert matrix.metadata["description"] == metadata["description"]


def test_subset_ignores_omx_file_attributes(matrix):
    matrix.metadata["TITLE"] = "Source title"
    selected = matrix.subset(["car"])
    assert not {"OMX_VERSION", "OMX_CREATED_WITH", "SHAPE", "TITLE"} & selected.metadata.keys()


def test_store_subset_survives_close(omx_path, index, data, metadata):
    with from_file(omx_path, index_name="zones", store=True) as store:
        selected = store.subset(["car"])
    assert_matrix(selected, index, {"car": data["car"]}, metadata=metadata)
    assert selected.matrix_metadata("car")["mode"] == "car"


@pytest.mark.parametrize(
    "subset, names",
    [
        ([{"mode": "car"}], {"car"}),
        ([{"mode": "car", "year": 2030}], {"car"}),
        ([{"mode": "car", "year": 2029}], set()),
        ([{"missing": "car"}], set()),
        ([{}], {"car", "bus"}),
        (["bus", {"mode": "car"}, "car"], {"car", "bus"}),
        ([{"periods": [1, 2]}], {"car"}),
        ([{"value": np.nan}], {"car"}),
    ],
)
def test_subset_attribute_queries(matrix, subset, names):
    matrix.matrix_metadata("car").update({"mode": "car", "year": 2030, "periods": [1, 2], "value": np.nan})
    selected = matrix.subset(subset)
    assert set(selected) == names


def test_from_file_selects_by_attributes(omx_path, data):
    loaded = from_file(omx_path, index_name="zones", subset=[{"mode": "car"}])
    assert set(loaded) == {"car"}
    np.testing.assert_array_equal(loaded["car"], data["car"])


def test_save_includes_matrix_attributes(matrix, tmp_path):
    path = tmp_path / "attributes.omx"
    matrix.matrix_metadata("car").update({"mode": "car", "units": "trips"})
    matrix.save_as_omx(path)
    with omx.open_file(path, "r") as file:
        assert dict(file["car"].attrs) == {"mode": "car", "units": "trips"}


def test_replacing_matrix_keeps_metadata(matrix):
    matrix.matrix_metadata("car")["mode"] = "car"
    matrix["car"] = np.ones(matrix.shape)
    assert dict(matrix.matrix_metadata("car")) == {"mode": "car"}


def test_deleting_matrix_removes_metadata(matrix):
    matrix.matrix_metadata("car")["mode"] = "car"
    del matrix["car"]
    with pytest.raises(KeyError):
        matrix.matrix_metadata("car")
    matrix["car"] = np.zeros(matrix.shape)
    assert dict(matrix.matrix_metadata("car")) == {}


def test_subset_copies_matrix_metadata(matrix):
    matrix.matrix_metadata("car").update({"mode": "car", "periods": np.array([1, 2, 3]), "TITLE": "Car demand"})
    selected = matrix.subset(["car"])
    assert selected.matrix_metadata("car")["mode"] == "car"
    assert selected.matrix_metadata("car")["TITLE"] == "Car demand"
    np.testing.assert_array_equal(selected.matrix_metadata("car")["periods"], [1, 2, 3])

    assert not np.shares_memory(selected.matrix_metadata("car")["periods"], matrix.matrix_metadata("car")["periods"])
    selected.matrix_metadata("car")["mode"] = "updated"
    selected.matrix_metadata("car")["periods"] = np.array([9, 2, 3])
    assert matrix.matrix_metadata("car")["mode"] == "car"
    np.testing.assert_array_equal(matrix.matrix_metadata("car")["periods"], [1, 2, 3])


@pytest.mark.parametrize("name", [None, "car"], ids=["file", "matrix"])
def test_metadata_values_can_be_replaced(matrix, name):
    attributes = matrix.metadata if name is None else matrix.matrix_metadata(name)
    attributes["periods"] = [1, 2, 3]
    attributes["periods"] = [9, 2]
    np.testing.assert_array_equal(attributes["periods"], [9, 2])


def test_matrix_attributes_can_be_deleted(matrix):
    attributes = matrix.matrix_metadata("car")
    attributes["units"] = "trips"
    del attributes["units"]
    assert "units" not in matrix.matrix_metadata("car")


def test_read_only_store_rejects_matrix_metadata_edits(omx_path):
    with from_file(omx_path, index_name="zones", store=True) as store:
        with pytest.raises(OSError):
            store.matrix_metadata("car")["description"] = "Updated demand"
        assert "description" not in store.matrix_metadata("car")


def test_save_matrix_metadata_precedence(matrix, tmp_path, index):
    path = tmp_path / "precedence.omx"
    destination = from_dict({"car": np.zeros(matrix.shape)}, index, index_name="zones")
    destination.matrix_metadata("car").update({"mode": "destination", "year": 2029, "units": "trips"})
    destination.save_as_omx(path)

    matrix.matrix_metadata("car").update({"mode": "source", "year": 2030, "description": "Source demand"})
    matrix.save_as_omx(path, mode="r+", subset=["car"])

    with omx.open_file(path, "r") as file:
        assert dict(file["car"].attrs) == {
            "mode": "source",
            "year": 2030,
            "units": "trips",
            "description": "Source demand",
        }
    assert matrix.matrix_metadata("car")["mode"] == "source"


def test_mapping_lists_names(matrix, data):
    assert isinstance(matrix, MutableMapping)
    assert set(matrix) == set(matrix.keys()) == set(data)
    assert len(matrix) == len(data)
    assert "car" in matrix
    assert "missing" not in matrix


def test_mapping_items(matrix, data):
    for name, values in matrix.items():
        np.testing.assert_array_equal(values, data[name])


def test_mapping_values(matrix, data):
    for name, values in zip(matrix, matrix.values(), strict=True):
        np.testing.assert_array_equal(values, data[name])


def test_mapping_update(matrix):
    values = np.zeros(matrix.shape)
    matrix.update({"new": values})
    np.testing.assert_array_equal(matrix["new"], values)


def test_mapping_pop(matrix, data):
    np.testing.assert_array_equal(matrix.pop("car"), data["car"])
    assert "car" not in matrix


def test_mapping_delete(matrix):
    del matrix["bus"]
    assert set(matrix) == {"car"}


def test_matrix_equals_itself(memory_matrix):
    assert memory_matrix == memory_matrix


def test_matrix_equality(matrix_pair):
    left, right = matrix_pair
    assert left == right
    assert not left != right


def test_equality_across_backends(omx_path):
    memory = from_file(omx_path, index_name="zones")
    with from_file(omx_path, index_name="zones", store=True) as left:
        with from_file(omx_path, index_name="zones", store=True) as right:
            assert left == right
            assert left == memory
            assert memory == left
            memory.imat["car", 0, 0] = 42
            assert left != memory
            assert memory != left


def test_equality_matches_stored_metadata(memory_matrix, tmp_path):
    memory_matrix.metadata["periods"] = [1.0, np.nan]
    memory_matrix.matrix_metadata("car").update({"periods": [1.0, np.nan], "modes": ["car", "bus"]})
    path = tmp_path / "metadata.omx"
    memory_matrix.save_as_omx(path)
    with from_file(path, index_name="zones", store=True) as store:
        assert memory_matrix == store


def test_equality_checks_names(matrix_pair):
    left, right = matrix_pair
    right["bus"] = right.pop("car")
    assert left != right


@pytest.mark.parametrize("value", [0.0, np.nan], ids=["number", "nan"])
def test_equality_checks_values(matrix_pair, value):
    left, right = matrix_pair
    right.imat["car", 0, 1] = value
    assert left != right


def test_equality_treats_matching_nans_as_equal(matrix_pair):
    left, right = matrix_pair
    left.imat["car", 0, 1] = np.nan
    right.imat["car", 0, 1] = np.nan
    assert left == right


def test_equality_compares_values_not_dtypes(matrix_pair):
    left, right = matrix_pair
    right["car"] = right["car"].astype(np.int32)
    assert left == right


@pytest.mark.parametrize("zones", [[20, 10], [10, 30]], ids=["order", "labels"])
def test_equality_checks_zone_index(matrix_pair, zones):
    left, right = matrix_pair
    right.index = zones
    assert left != right


def test_equality_checks_index_name(matrix_pair):
    left, _ = matrix_pair
    right = from_dict(dict(left), left.index, index_name="other")
    assert left != right


@pytest.mark.parametrize(
    "value",
    [2030, "Morning demand", ["car", "bus"], np.nan, [1.0, np.nan], np.array([1 + 2j, complex(np.nan, 0)])],
    ids=["integer", "text", "text-array", "nan", "array-with-nan", "complex-array"],
)
def test_equality_accepts_matching_metadata(matrix_pair, value):
    left, right = matrix_pair
    left.metadata["value"] = value
    right.metadata["value"] = value
    assert left == right


@pytest.mark.parametrize("name", [None, "car"], ids=["file", "matrix"])
def test_equality_checks_metadata_values(matrix_pair, name):
    left, right = matrix_pair
    left_attrs = left.metadata if name is None else left.matrix_metadata(name)
    right_attrs = right.metadata if name is None else right.matrix_metadata(name)
    left_attrs["periods"] = [1, 2]
    right_attrs["periods"] = [2, 1]
    assert left != right


@pytest.mark.parametrize("name", [None, "car"], ids=["file", "matrix"])
def test_equality_checks_metadata_keys(matrix_pair, name):
    left, right = matrix_pair
    attrs = right.metadata if name is None else right.matrix_metadata(name)
    attrs["extra"] = 1
    assert left != right


def test_equality_ignores_omx_file_attributes(matrix_pair):
    left, right = matrix_pair
    right.metadata.update(
        {"OMX_VERSION": "ignored", "OMX_CREATED_WITH": "ignored", "SHAPE": [99, 99], "TITLE": "ignored"}
    )
    assert left == right


def test_equality_ignores_matrix_insertion_order(memory_matrix):
    other = memory_matrix.subset(list(memory_matrix)[::-1])
    assert list(memory_matrix) != list(other)
    assert memory_matrix == other


@pytest.mark.parametrize("period, equal", [(1.0, True), (2.0, False)], ids=["equal", "different"])
def test_equality_compares_nested_metadata(matrix_pair, period, equal):
    left, right = matrix_pair
    left.metadata["details"] = {"periods": np.array([1.0, np.nan])}
    right.metadata["details"] = {"periods": [period, np.nan]}
    assert (left == right) is equal


@pytest.mark.parametrize("zones", [[10, 20], []], ids=["zones", "no-zones"])
def test_empty_matrices_compare_equal(zones):
    assert AequilibraeMatrix(zones) == AequilibraeMatrix(zones)


def test_empty_matrices_compare_zone_indices():
    assert AequilibraeMatrix([10, 20]) != AequilibraeMatrix([])


def test_equality_does_not_compare_with_plain_mappings(memory_matrix):
    other = dict(memory_matrix)
    assert memory_matrix != other
    assert other != memory_matrix


def test_setdefault_returns_memory_view(memory_matrix):
    result = memory_matrix.setdefault("new", np.zeros(memory_matrix.shape))
    result[0, 0] = 42
    assert memory_matrix.imat["new", 0, 0] == 42


def test_setdefault_returns_store_copy(omx_path):
    with from_file(omx_path, index_name="zones", store=True, mode="r+") as store:
        result = store.setdefault("new", np.zeros(store.shape))
        result[0, 0] = 42
        assert store.imat["new", 0, 0] == 0


@pytest.mark.parametrize("default", [None, np.zeros((3, 3))], ids=["none", "array"])
def test_setdefault_keeps_existing_values_and_metadata(matrix, data, default):
    matrix.matrix_metadata("car")["units"] = "trips"
    np.testing.assert_array_equal(matrix.setdefault("car", default), data["car"])
    assert matrix.matrix_metadata("car")["units"] == "trips"


def test_setdefault_requires_values_for_missing_name(memory_matrix):
    with pytest.raises(TypeError):
        memory_matrix.setdefault("new")
    assert "new" not in memory_matrix


def test_clear_removes_matrices_and_their_attributes(matrix, index, metadata):
    matrix.matrix_metadata("car")["units"] = "trips"
    matrix.clear()
    assert_matrix(matrix, index, {}, metadata=metadata)
    with pytest.raises(KeyError):
        matrix.matrix_metadata("car")
    matrix["car"] = np.zeros(matrix.shape)
    assert dict(matrix.matrix_metadata("car")) == {}


def test_clear_empty_matrix(matrix_factory, index):
    matrix = matrix_factory(index)
    matrix.clear()
    assert len(matrix) == 0


def test_store_clear_does_not_read_values(omx_path, monkeypatch):
    def unexpected_read(*args):
        pytest.fail("clear() must not read matrix values")

    with from_file(omx_path, index_name="zones", store=True, mode="r+") as store:
        monkeypatch.setattr(MatrixStore, "__getitem__", unexpected_read)
        store.clear()
        assert len(store) == 0


def test_read_missing_matrix(matrix):
    with pytest.raises(KeyError):
        matrix["missing"]


def test_delete_missing_matrix(matrix):
    with pytest.raises(KeyError):
        del matrix["missing"]


def test_metadata_requires_existing_matrix(matrix):
    with pytest.raises(KeyError):
        matrix.matrix_metadata("missing")


@pytest.mark.parametrize("accessor", ["mat", "imat"])
def test_indexed_read_requires_existing_matrix(matrix, accessor):
    with pytest.raises(KeyError):
        getattr(matrix, accessor)["missing", 0, 0]


@pytest.mark.parametrize("accessor", ["mat", "imat"])
def test_indexed_write_requires_existing_matrix(matrix, accessor):
    with pytest.raises(KeyError):
        getattr(matrix, accessor)["missing", 0, 0] = 42


@pytest.mark.parametrize(
    "selection, values",
    [
        pytest.param((0, 1), 42, id="cell"),
        pytest.param((slice(None), 1), [11, 12, 13], id="column"),
        pytest.param((slice(0, 2), slice(0, 2)), [[1, 2], [3, 4]], id="block"),
        pytest.param(([0, 2], slice(None)), 42, id="fancy-rows"),
        pytest.param((np.eye(3, dtype=bool),), 42, id="boolean-mask"),
    ],
)
def test_indexed_reads_and_writes(matrix, index, data, metadata, selection, values):
    np.testing.assert_array_equal(matrix.imat["car", *selection], data["car"][selection])
    expected = data["car"].copy()
    expected[selection] = values
    matrix.imat["car", *selection] = values
    assert_matrix(matrix, index, {**data, "car": expected}, metadata=metadata)


def test_whole_matrix_augmented_assignment(matrix, data):
    matrix["car"] *= 2
    np.testing.assert_array_equal(matrix["car"], data["car"] * 2)


@pytest.mark.parametrize("rows", [slice(None), [0, 2]], ids=["slice", "fancy"])
def test_indexed_augmented_assignment(matrix, data, rows):
    expected = data["car"].copy()
    expected[rows, 2] *= 3
    matrix.imat["car", rows, 2] *= 3
    np.testing.assert_array_equal(matrix["car"], expected)


@pytest.mark.parametrize("accessor", ["imat", "mat"])
def test_indexer_whole_matrix(matrix, data, accessor):
    indexer = getattr(matrix, accessor)
    np.testing.assert_array_equal(indexer["car"], matrix["car"])
    np.testing.assert_array_equal(indexer["car",], matrix["car"])
    indexer["car",] = data["car"] * 2
    np.testing.assert_array_equal(matrix["car"], data["car"] * 2)
    indexer["new"] = data["bus"]
    np.testing.assert_array_equal(matrix["new"], data["bus"])


@pytest.mark.parametrize("key", [0, ["car"], {"mode": "car"}, ("car",), ("car", 0, 1, 2)])
def test_plain_indexing_requires_string(matrix, data, key):
    with pytest.raises(TypeError, match="string"):
        matrix[key]
    with pytest.raises(TypeError, match="string"):
        matrix[key] = data["car"]
    np.testing.assert_array_equal(matrix["car"], data["car"])


@pytest.mark.parametrize("key", ["", ".", "..", "mode/car", "/lookup/zones", "mode\\car", "car\0bus"])
def test_matrix_names_reject_paths(matrix, key, tmp_path):
    with pytest.raises(ValueError, match="not a path"):
        matrix[key] = np.zeros(matrix.shape)

    with pytest.raises(ValueError, match="not a path"):
        matrix[key]

    with pytest.raises(ValueError, match="not a path"):
        matrix[key] = np.zeros(matrix.shape)

    with pytest.raises(ValueError, match="not a path"):
        del matrix[key]

    with pytest.raises(ValueError, match="not a path"):
        assert key in matrix

    with pytest.raises(ValueError, match="not a path"):
        matrix.matrix_metadata(key)

    with pytest.raises(ValueError, match="not a path"):
        matrix.subset([key])

    path = tmp_path / "invalid.omx"
    with pytest.raises(ValueError, match="not a path"):
        matrix.save_as_omx(path, subset=[key])
    assert not path.exists()

    with pytest.raises(ValueError, match="not a path"):
        matrix.mat[key, :, :]

    with pytest.raises(ValueError, match="not a path"):
        matrix.imat[key, :, :]

    with pytest.raises(ValueError, match="not a path"):
        matrix.mat[key, :, :] = 42

    with pytest.raises(ValueError, match="not a path"):
        matrix.imat[key, :, :] = 42


@pytest.mark.parametrize("name", ["", ".", "..", "nested/zones", "/outside", "nested\\zones", "zones\0suffix"])
def test_index_names_reject_paths(index, name):
    with pytest.raises(ValueError, match="index name"):
        AequilibraeMatrix(index, index_name=name)


@pytest.mark.parametrize("name", [1, None])
def test_index_name_must_be_a_string(index, name):
    with pytest.raises(TypeError, match="index name"):
        AequilibraeMatrix(index, index_name=name)


def test_invalid_index_name_does_not_create_store(tmp_path, index):
    path = tmp_path / "invalid.omx"
    with pytest.raises(ValueError, match="index name"):
        MatrixStore(path, index, index_name="nested/zones", mode="x")
    assert not path.exists()


def test_invalid_index_name_does_not_overwrite_file(omx_path, index, data, metadata):
    with pytest.raises(ValueError, match="index name"):
        MatrixStore(omx_path, index, index_name="nested/zones", mode="w")
    assert_matrix(from_file(omx_path, index_name="zones"), index, data, metadata=metadata)


def test_from_dict_checks_index_name_before_creating_file(tmp_path, index, data):
    path = tmp_path / "invalid.omx"
    with pytest.raises(ValueError, match="index name"):
        from_dict(data, index, index_name="nested/zones", path=path, store=True)
    assert not path.exists()


@pytest.mark.parametrize("store", [False, True], ids=["memory", "store"])
def test_from_file_rejects_index_paths(omx_path, store):
    with pytest.raises(ValueError, match="index name"):
        from_file(omx_path, index_name="/lookup/zones", store=store)


def test_from_file_rejects_path_in_subset(omx_path):
    with pytest.raises(ValueError, match="not a path"):
        from_file(omx_path, index_name="zones", subset=["/lookup/zones"])


@pytest.mark.parametrize(
    "key",
    [(), (0, 1, 2, "car"), 0, ["car"], {"mode": "car"}, (0, ["car"]), (0, 1, {"mode": "car"})],
)
def test_indexer_rejects_bad_arguments(memory_matrix, key):
    # Both indexers use the same argument handling.
    with pytest.raises(TypeError):
        memory_matrix.imat[key]
    with pytest.raises(TypeError):
        memory_matrix.imat[key] = 42


@pytest.mark.parametrize(
    "labels, positions",
    [
        pytest.param((309,), (0,), id="row"),
        pytest.param((309, 0), (0, 1), id="cell"),
        pytest.param((slice(None), 101), (slice(None), 2), id="column"),
        pytest.param((np.array([309, 101]),), ([0, 2],), id="array"),
        pytest.param(([],), ([],), id="empty"),
    ],
)
def test_mat_translates_zones(memory_matrix, data, labels, positions):
    np.testing.assert_array_equal(memory_matrix.mat["car", *labels], data["car"][positions])
    expected = data["car"].copy()
    expected[positions] = 42
    memory_matrix.mat["car", *labels] = 42
    np.testing.assert_array_equal(memory_matrix["car"], expected)


@pytest.mark.parametrize(
    "labels, positions",
    [
        pytest.param(slice(309, 0), slice(0, 2), id="inclusive-stop"),
        pytest.param(slice(309, 101, 2), slice(0, 3, 2), id="step"),
        pytest.param(slice(None, 0), slice(None, 2), id="open-start"),
        pytest.param(slice(0, None), slice(1, None), id="open-stop"),
        pytest.param(slice(101, 309, -1), slice(2, None, -1), id="reverse-to-first"),
        pytest.param(slice(101, 0, -1), slice(2, 0, -1), id="reverse-stop"),
    ],
)
def test_mat_inclusive_slices(memory_matrix, data, labels, positions):
    np.testing.assert_array_equal(memory_matrix.mat["car", labels], data["car"][positions])
    expected = data["car"].copy()
    expected[positions] = 42
    memory_matrix.mat["car", labels] = 42
    np.testing.assert_array_equal(memory_matrix["car"], expected)


@pytest.mark.parametrize(
    "key",
    [("car", 999), ("car", slice(None), [309, 999]), ("car", slice(999, 101)), ("car", slice(309, 999))],
)
def test_mat_unknown_zone(memory_matrix, data, key):
    with pytest.raises(KeyError):
        memory_matrix.mat[key]
    with pytest.raises(KeyError):
        memory_matrix.mat[key] = 42
    np.testing.assert_array_equal(memory_matrix["car"], data["car"])


@pytest.mark.parametrize("selection", [Ellipsis, None, [True, False, True], np.eye(3, dtype=bool)])
def test_mat_passes_special_selections_unchanged(memory_matrix, data, selection):
    np.testing.assert_array_equal(memory_matrix.mat["car", selection], data["car"][selection])
    expected = data["car"].copy()
    expected[selection] = 42
    memory_matrix.mat["car", selection] = 42
    np.testing.assert_array_equal(memory_matrix["car"], expected)


@pytest.mark.parametrize("accessor", ["imat", "mat"])
def test_indexer_paired_selections(memory_matrix, data, accessor):
    indexer = getattr(memory_matrix, accessor)
    rows, cols = ([309, 0, 101], [0, 101, 309]) if accessor == "mat" else ([0, 1, 2], [1, 2, 0])
    np.testing.assert_array_equal(indexer["car", rows, cols], data["car"][[0, 1, 2], [1, 2, 0]])

    indexer["car", rows, cols] = [11, 12, 13]
    expected = data["car"].copy()
    expected[[0, 1, 2], [1, 2, 0]] = [11, 12, 13]
    np.testing.assert_array_equal(memory_matrix["car"], expected)


def test_mat_uses_relabelled_index(matrix, data):
    indexer = matrix.mat
    matrix.index = [1001, 1002, 1003]
    assert indexer["car", 1001, 1002] == data["car"][0, 1]
    with pytest.raises(KeyError):
        indexer["car", 309]


def test_mat_large_zone_ids():
    zones = [0, 2**64 - 1]
    matrix = from_dict({"car": [[1, 2], [3, 4]]}, index=np.array(zones, dtype=np.uint64))
    assert matrix.mat["car", zones[0], zones[1]] == 2
    np.testing.assert_array_equal(matrix.mat["car", zones], matrix["car"])
    np.testing.assert_array_equal(matrix.mat["car", np.array(zones, dtype=np.uint64)], matrix["car"])


def test_store_omx_access(omx_path, data):
    with from_file(omx_path, index_name="zones", store=True) as store:
        assert isinstance(store.omx, omx.File)
        assert set(store.omx.list_matrices()) == set(data)
        np.testing.assert_array_equal(np.asarray(store.omx["car"]), data["car"])


def test_indexed_write_keeps_matrix_attributes(matrix):
    matrix.matrix_metadata("car")["units"] = "trips"
    matrix.imat["car", :, 1] = 7
    assert matrix.matrix_metadata("car")["units"] == "trips"


def test_augmented_assignment_keeps_matrix_attributes(matrix):
    matrix.matrix_metadata("car")["units"] = "trips"
    matrix["car"] *= 2
    assert matrix.matrix_metadata("car")["units"] == "trips"


def test_store_matrix_attributes_survive_close(omx_path):
    with from_file(omx_path, index_name="zones", mode="r+", store=True) as store:
        store.matrix_metadata("car")["units"] = "trips"
    with from_file(omx_path, index_name="zones", store=True) as store:
        assert dict(store.matrix_metadata("car")) == {"mode": "car", "units": "trips"}


def test_failed_store_replacement_with_invalid_filters_keeps_matrix(omx_path, data):
    with MatrixStore(omx_path, index_name="zones", mode="r+", filters={"complib": "invalid"}) as store:
        with pytest.raises(ValueError, match="filter"):
            store["car"] = np.zeros(store.shape, dtype=np.int32)
        assert set(store) == set(data)
        np.testing.assert_array_equal(store["car"], data["car"])
        assert store.matrix_metadata("car")["mode"] == "car"


def test_store_same_dtype_write_keeps_dataset_and_compression(tmp_path):
    path = tmp_path / "compressed.omx"
    with MatrixStore(path, [10, 20], mode="x", filters={"complib": "zlib", "complevel": 1}) as store:
        store["car"] = np.ones((2, 2))
        dataset = store.omx.data["car"]
        attributes = store.matrix_metadata("car")
        attributes["units"] = "trips"
        store["car"] = np.zeros((2, 2))
        assert store.omx.data["car"].id == dataset.id
        assert dataset.compression == "gzip"
        assert dataset.compression_opts == 1
        assert attributes["units"] == "trips"
        np.testing.assert_array_equal(dataset[:], np.zeros((2, 2)))


def test_store_dtype_replacement_does_not_truncate_values(omx_path):
    with from_file(omx_path, index_name="zones", store=True, mode="r+") as store:
        store["car"] = np.ones(store.shape, dtype=np.int32)
        store["car"] = np.full(store.shape, 1.5)
        assert store["car"].dtype == np.dtype(float)
        np.testing.assert_array_equal(store["car"], np.full(store.shape, 1.5))
        assert store.matrix_metadata("car")["mode"] == "car"
        assert set(store) == {"car", "bus"}


@pytest.mark.parametrize("store", [False, True], ids=["memory", "store"])
def test_loading_rejects_non_numeric_matrices(tmp_path, store):
    path = tmp_path / "text.omx"
    with omx.open_file(path, "x") as file:
        file.create_mapping("main_index", [10, 20])
        file["car"] = np.full((2, 2), b"car")
    with pytest.raises(TypeError, match="dtype"):
        from_file(path, store=store)


@pytest.mark.parametrize("dtype", [np.float64, np.int32])
def test_failed_save_matrix_metadata_keeps_values(omx_path, data, dtype):
    matrix = from_file(omx_path, index_name="zones", subset=["car"])
    matrix["car"] = np.zeros(matrix.shape, dtype=dtype)
    matrix.matrix_metadata("car")["bad"] = {"nested": "value"}
    with pytest.raises(TypeError):
        matrix.save_as_omx(omx_path, mode="r+")
    loaded = from_file(omx_path, index_name="zones")
    assert set(loaded) == set(data)
    np.testing.assert_array_equal(loaded["car"], data["car"])
    assert loaded.matrix_metadata("car")["mode"] == "car"


def test_read_only_store_rejects_index_replacement(omx_path, index):
    with from_file(omx_path, index_name="zones", store=True) as store:
        with pytest.raises(PermissionError, match="read-only"):
            store.index = [809, 0, 801]
        np.testing.assert_array_equal(store.index, index)
        np.testing.assert_array_equal(store.omx.lookup["zones"][:], index)


def test_open_writable_store_preserves_index(uint8_index_omx_path):
    with MatrixStore(uint8_index_omx_path, index_name="zones", mode="r+") as store:
        dataset = store.omx.lookup["zones"]
        assert dataset.dtype == np.dtype("uint8")
        assert dataset.attrs["description"] == "Zone IDs"
        np.testing.assert_array_equal(dataset[:], [10, 20])
        np.testing.assert_array_equal(store.index, [10, 20])


def test_store_index_replacement_preserves_dtype_and_attributes(uint8_index_omx_path):
    with MatrixStore(uint8_index_omx_path, index_name="zones", mode="r+") as store:
        store.index = [100, 200]
        dataset = store.omx.lookup["zones"]
        assert dataset.dtype == np.dtype("uint8")
        assert dataset.attrs["description"] == "Zone IDs"
        np.testing.assert_array_equal(dataset[:], [100, 200])
        np.testing.assert_array_equal(store.index, [100, 200])


def test_failed_store_index_cast_leaves_index_unchanged(uint8_index_omx_path):
    with MatrixStore(uint8_index_omx_path, index_name="zones", mode="r+") as store:
        with pytest.raises(ValueError, match="same_value"):
            store.index = [100, 300]
        np.testing.assert_array_equal(store.index, [10, 20])
        np.testing.assert_array_equal(store.omx.lookup["zones"][:], [10, 20])


def test_store_constructor_writes_supplied_index(uint8_index_omx_path):
    with MatrixStore(uint8_index_omx_path, index=[100, 200], index_name="zones", mode="r+") as store:
        dataset = store.omx.lookup["zones"]
        assert dataset.dtype == np.dtype("uint8")
        assert dataset.attrs["description"] == "Zone IDs"
        np.testing.assert_array_equal(dataset[:], [100, 200])
        np.testing.assert_array_equal(store.index, [100, 200])


def test_failed_store_constructor_index_cast_leaves_file_unchanged(uint8_index_omx_path):
    with pytest.raises(ValueError, match="same_value"):
        MatrixStore(uint8_index_omx_path, index=[100, 300], index_name="zones", mode="r+")

    with omx.open_file(uint8_index_omx_path, "r") as file:
        np.testing.assert_array_equal(file.lookup["zones"][:], [10, 20])


def test_failed_store_construction_has_no_cleanup_error(tmp_path, recwarn):
    with pytest.raises(FileNotFoundError):
        MatrixStore(tmp_path / "missing.omx")
    assert not recwarn


def test_store_matrix_write_survives_reopening(omx_path, data):
    total = data["car"] + data["bus"]
    with from_file(omx_path, index_name="zones", mode="r+", store=True) as store:
        store["total"] = total
    with from_file(omx_path, index_name="zones", store=True) as store:
        np.testing.assert_array_equal(store["total"], total)


def test_store_file_metadata_survives_reopening(omx_path):
    with from_file(omx_path, index_name="zones", mode="r+", store=True) as store:
        store.metadata["description"] = "Updated demand"
    with from_file(omx_path, index_name="zones", store=True) as store:
        assert store.metadata["description"] == "Updated demand"


def test_store_cell_edit_survives_reopening(omx_path):
    with from_file(omx_path, index_name="zones", mode="r+", store=True) as store:
        store.imat["car", 0, 1] = 42
    with from_file(omx_path, index_name="zones", store=True) as store:
        assert store.imat["car", 0, 1] == 42


def test_store_deletion_survives_reopening(omx_path):
    with from_file(omx_path, index_name="zones", mode="r+", store=True) as store:
        del store["bus"]
    with from_file(omx_path, index_name="zones", store=True) as store:
        assert set(store) == {"car"}


@pytest.mark.parametrize("mode", ["r", "r+"])
def test_store_rejects_rectangular_file(tmp_path, mode):
    path = tmp_path / "rectangular.omx"
    with omx.open_file(path, "w") as file:
        file["car"] = np.zeros((2, 3))
        file.create_mapping("zones", [10, 20])

    with pytest.raises(ValueError, match="matrix shape does not match index"):
        MatrixStore(path, index_name="zones", mode=mode)

    with omx.open_file(path, "r") as file:
        assert file["car"].shape == (2, 3)
        np.testing.assert_array_equal(file.lookup["zones"][:], [10, 20])


def test_read_only_store_rejects_supplied_index(omx_path, index):
    with pytest.raises(ValueError, match="cannot supply an index.*read-only"):
        MatrixStore(omx_path, index=index, index_name="zones", mode="r")

    with omx.open_file(omx_path, "r") as file:
        np.testing.assert_array_equal(file.lookup["zones"][:], index)


@pytest.mark.parametrize("new_index", [[10, 20], [10, 20, 30, 40]], ids=["shorter", "longer"])
def test_store_rejects_supplied_index_length(omx_path, index, new_index):
    with pytest.raises(ValueError, match="matrix shape does not match index"):
        MatrixStore(omx_path, index=new_index, index_name="zones", mode="r+")

    with omx.open_file(omx_path, "r") as file:
        np.testing.assert_array_equal(file.lookup["zones"][:], index)


def test_store_rejects_stored_index_length(omx_path):
    with omx.open_file(omx_path, "a") as file:
        # Use h5py directly because OMX rejects mappings with the wrong length.
        file.lookup.create_dataset("wrong_length", data=[10, 20])

    with pytest.raises(ValueError, match="matrix shape does not match index"):
        MatrixStore(omx_path, index_name="wrong_length")


@pytest.mark.parametrize("shape", [(2, 2), (3, 2), (3, 3, 1)])
def test_matrix_shape_validation(matrix, index, data, metadata, shape):
    with pytest.raises(ValueError):
        matrix["car"] = np.zeros(shape)
    assert_matrix(matrix, index, data, metadata=metadata)


@pytest.mark.parametrize(
    "invalid_index, error",
    [
        pytest.param([[1, 2]], ValueError, id="dimensions"),
        pytest.param([1, 1], ValueError, id="duplicates"),
        pytest.param([-1, 2], ValueError, id="negative"),
        pytest.param([1.0, 2.0], TypeError, id="float"),
        pytest.param(["1", "2"], TypeError, id="string"),
        pytest.param([False, True], TypeError, id="bool-list"),
        pytest.param([True, 2], TypeError, id="mixed-bool"),
        pytest.param(np.array([False, True]), TypeError, id="bool-array"),
        pytest.param(np.array([1, False], dtype=object), TypeError, id="object-bool"),
        pytest.param([0, 2**64], ValueError, id="too-large"),
        pytest.param([0, np.nan], TypeError, id="nan"),
        pytest.param([0, None], TypeError, id="missing"),
    ],
)
def test_index_validation(invalid_index, error):
    with pytest.raises(error):
        AequilibraeMatrix(invalid_index)


def test_store_validates_supplied_zone_ids(tmp_path):
    with pytest.raises(TypeError, match="zone IDs"):
        MatrixStore(tmp_path / "invalid.omx", [False, True], mode="x")


@pytest.mark.parametrize(
    "zones",
    [
        [0, 2**64 - 1],
        (0, 2**64 - 1),
        np.array([0, 2**64 - 1], dtype=np.uint64),
        np.array([0, 2**64 - 1], dtype=object),
    ],
    ids=["list", "tuple", "uint64", "object"],
)
def test_index_accepts_full_uint64_range(zones):
    matrix = AequilibraeMatrix(zones)
    assert matrix.index.dtype == np.dtype("uint64")
    assert matrix.index.tolist() == [0, 2**64 - 1]


def test_index_replacement_accepts_full_uint64_range(matrix):
    matrix.index = [0, 2**64 - 1, 2**63]
    assert matrix.index.tolist() == [0, 2**64 - 1, 2**63]
    assert matrix.mat["car", 0, 2**64 - 1] == 12.5


@pytest.mark.parametrize("store", [False, True], ids=["memory", "store"])
def test_large_zone_ids_survive_save(tmp_path, manage, store):
    path = tmp_path / "large_zones.omx"
    zones = [0, 2**64 - 1, 2**63]
    matrix = from_dict({"car": np.zeros((3, 3))}, zones)
    matrix.save_as_omx(path)
    loaded = manage(from_file(path, store=store))
    assert loaded.index.dtype == np.dtype("uint64")
    assert loaded.index.tolist() == zones


@pytest.mark.parametrize("invalid_index", [[1, 1, 1], [True, 2, 3], [1.0, 2.0, 3.0], [0, 1, 2**64]])
def test_invalid_index_replacement_keeps_zone_ids(matrix, index, invalid_index):
    with pytest.raises((TypeError, ValueError)):
        matrix.index = invalid_index
    np.testing.assert_array_equal(matrix.index, index)
    assert matrix.mat["car", 309, 0] == 12.5


def test_invalid_store_index_leaves_file_unchanged(omx_path, index):
    with from_file(omx_path, index_name="zones", store=True, mode="r+") as store:
        with pytest.raises(TypeError, match="zone IDs"):
            store.index = [True, 2, 3]
    np.testing.assert_array_equal(from_file(omx_path, index_name="zones").index, index)


@pytest.mark.parametrize("new_index", [[309, 0], [309, 0, 101, 205]], ids=["shorter", "longer"])
def test_index_size_is_fixed(matrix, index, data, metadata, new_index):
    with pytest.raises(ValueError):
        matrix.index = new_index
    assert_matrix(matrix, index, data, metadata=metadata)


def test_from_df_infers_sorted_zone_ids():
    df = pd.DataFrame({"origin": [309, 101], "destination": [101, 205], "car": [2, 3]})
    matrix = from_df(df, "origin", "destination")
    np.testing.assert_array_equal(matrix.index, [101, 205, 309])


def test_from_df_sums_duplicate_pairs():
    df = pd.DataFrame({"origin": [10, 10], "destination": [20, 20], "car": [2, 3], "bus": [1, 3]})
    matrix = from_df(df, "origin", "destination")
    np.testing.assert_array_equal(matrix["car"], [[0, 5], [0, 0]])
    np.testing.assert_array_equal(matrix["bus"], [[0, 4], [0, 0]])


def test_from_df_default_dtype_is_float():
    df = pd.DataFrame({"origin": [10], "destination": [20], "car": [2]})
    assert from_df(df, "origin", "destination")["car"].dtype == np.dtype(float)


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_from_df_uses_supplied_dtype(dtype):
    df = pd.DataFrame({"origin": [10], "destination": [20], "car": [2]})
    assert from_df(df, "origin", "destination", dtype=dtype)["car"].dtype == np.dtype(dtype)


def test_from_df_preserves_supplied_zone_order_and_extra_zones():
    df = pd.DataFrame({"origin": [10], "destination": [20], "car": [2]})
    matrix = from_df(df, "origin", "destination", index=[20, 10, 30], index_name="zones")
    assert_matrix(matrix, [20, 10, 30], {"car": [[0, 0, 0], [2, 0, 0], [0, 0, 0]]})


def test_from_df_selects_columns():
    df = pd.DataFrame({"origin": [10], "destination": [20], "car": [2], "bus": [3]})
    matrix = from_df(df, "origin", "destination", subset=["car"])
    assert set(matrix) == {"car"}


@pytest.mark.parametrize("fill_value", [0, 7, np.nan], ids=["zero", "nonzero", "nan"])
def test_from_df_fills_missing_pairs(fill_value):
    df = pd.DataFrame({"origin": [10], "destination": [20], "car": [2]})
    matrix = from_df(df, "origin", "destination", fill_value=fill_value)
    np.testing.assert_array_equal(matrix["car"], [[fill_value, 2], [fill_value, fill_value]])


def test_from_df_requires_known_zones():
    df = pd.DataFrame({"origin": [101], "destination": [309], "car": [7]})
    with pytest.raises(ValueError):
        from_df(df, "origin", "destination", index=[101, 205])


@pytest.mark.parametrize("index", [None, [0, 1]], ids=["inferred", "supplied"])
@pytest.mark.parametrize("column", ["origin", "destination"])
def test_from_df_checks_zone_types_with_or_without_index(index, column):
    df = pd.DataFrame({"origin": [0, 1], "destination": [1, 0], "car": [2, 3]})
    df[column] = [0.0, 1.0]
    with pytest.raises(TypeError, match="zone IDs"):
        from_df(df, "origin", "destination", index=index)


@pytest.mark.parametrize(
    "index, expected_index",
    [(None, [0, 2**64 - 1]), ([2**64 - 1, 0], [2**64 - 1, 0])],
    ids=["inferred", "supplied"],
)
def test_from_df_keeps_large_zone_ids_exact(index, expected_index):
    df = pd.DataFrame({"origin": [0], "destination": np.array([2**64 - 1], dtype=np.uint64), "car": [2]})
    matrix = from_df(df, "origin", "destination", index=index)
    assert matrix.index.tolist() == expected_index
    assert matrix.mat["car", 0, 2**64 - 1] == 2


def test_from_df_accepts_nullable_integer_ids():
    df = pd.DataFrame({"origin": pd.Series([0, 1], dtype="Int64"), "destination": [1, 0], "car": [2, 3]})
    matrix = from_df(df, "origin", "destination")
    np.testing.assert_array_equal(matrix["car"], [[0, 2], [3, 0]])


@pytest.mark.parametrize(
    "array_type", [sparse.coo_array, sparse.csr_array, sparse.csc_array], ids=["coo", "csr", "csc"]
)
def test_from_scipy(index, data, array_type):
    matrix = from_scipy({name: array_type(values) for name, values in data.items()}, index, index_name="zones")
    assert isinstance(matrix, AequilibraeSparseMatrix)
    assert_matrix(matrix, index, data)
    assert all(isinstance(value, array_type) for value in matrix.values())


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_from_scipy_preserves_dtype(index, dtype):
    values = np.arange(9, dtype=dtype).reshape(3, 3)
    matrix = from_scipy({"car": sparse.csr_array(values)}, index)
    assert matrix["car"].dtype == np.dtype(dtype)
    np.testing.assert_array_equal(matrix["car"].toarray(), values)
