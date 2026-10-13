from collections.abc import MutableMapping
from contextlib import ExitStack

import numpy as np
import openmatrix as omx
import pandas as pd
import pytest
import scipy.sparse as sparse

from aequilibrae.matrix import AequilibraEMatrix, MatrixStore, from_df, from_dict, from_file, from_scipy


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


@pytest.fixture(params=[AequilibraEMatrix, MatrixStore], ids=["memory", "store"])
def matrix_factory(request, tmp_path, manage):
    def create(index, index_name="zones"):
        if request.param is MatrixStore:
            return manage(MatrixStore(tmp_path / "matrix.omx", index, index_name, mode="w"))

        return AequilibraEMatrix(index, index_name)

    return create


@pytest.fixture
def matrix(matrix_factory, index, data, metadata):
    matrix = matrix_factory(index)
    matrix.update(data)
    matrix.metadata.update(metadata)

    return matrix


@pytest.fixture
def memory_matrix(index, data):
    return from_dict(data, index)


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
        np.testing.assert_array_equal(matrix[name], expected)

    if metadata is not None:
        for key, value in metadata.items():
            assert matrix.metadata[key] == value


@pytest.mark.parametrize("index_type", [np.asarray, list, tuple], ids=["array", "list", "tuple"])
def test_construction_copies_index(matrix_factory, index, index_type):
    supplied_index = index_type(index.tolist())
    matrix = matrix_factory(supplied_index)

    if not isinstance(supplied_index, tuple):
        supplied_index[0] = 999

    assert_matrix(matrix, index, {})
    with pytest.raises(ValueError):
        matrix.index[0] = 999


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_assignment_copies_values_and_preserves_dtype(matrix, dtype):
    values = np.arange(9, dtype=dtype).reshape(3, 3)
    expected = values.copy()

    matrix["car"] = values
    values[:] = 999

    assert matrix["car"].dtype == np.dtype(dtype)
    np.testing.assert_array_equal(matrix["car"], expected)


@pytest.mark.parametrize("array_type", [np.asarray, list, tuple], ids=["array", "list", "tuple"])
def test_assignment_accepts_array_like(matrix, data, array_type):
    matrix["car"] = array_type(data["car"].tolist())
    np.testing.assert_array_equal(matrix["car"], data["car"])
    assert matrix["car"].dtype == data["car"].dtype

    matrix.iloc[0, 1, "car"] = 2.5
    assert matrix.iloc[0, 1, "car"] == 2.5
    matrix["car"] *= 1.1
    assert matrix.iloc[0, 1, "car"] == pytest.approx(2.75)


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


@pytest.mark.parametrize("store", [False, True], ids=["memory", "store"])
def test_from_dict(tmp_path, manage, index, data, metadata, store):
    kwargs = {"path": tmp_path / "demand.omx"} if store else {}
    periods = np.array([1, 2, 3])
    supplied_metadata = {
        **metadata,
        "periods": periods,
        "OMX_VERSION": "ignored",
        "OMX_CREATED_WITH": "ignored",
        "SHAPE": [99, 99],
        "TITLE": "ignored",
    }
    matrix = manage(from_dict(data, index, index_name="zones", metadata=supplied_metadata, store=store, **kwargs))
    assert isinstance(matrix, MatrixStore if store else AequilibraEMatrix)
    periods[0] = 9

    if store:
        matrix.close()
        matrix = manage(from_file(kwargs["path"], index_name="zones", store=True))

    assert_matrix(matrix, index, data, metadata=metadata)
    np.testing.assert_array_equal(matrix.metadata["periods"], [1, 2, 3])
    np.testing.assert_array_equal(matrix.metadata.get("SHAPE", matrix.shape), matrix.shape)
    assert matrix.metadata.get("OMX_VERSION") != "ignored"
    assert matrix.metadata.get("OMX_CREATED_WITH") != "ignored"
    assert "TITLE" not in matrix.metadata


@pytest.mark.parametrize("store", [False, True], ids=["memory", "store"])
def test_from_file(omx_path, manage, index, data, metadata, store):
    matrix = manage(from_file(omx_path, index_name="zones", store=store))
    assert isinstance(matrix, MatrixStore if store else AequilibraEMatrix)
    assert_matrix(matrix, index, data, metadata=metadata)


@pytest.mark.parametrize("mode", [None, "r", "r+"])
def test_from_file_store_modes(omx_path, mode):
    with from_file(omx_path, index_name="zones", mode=mode, store=True) as store:
        assert store.omx.mode == ("r" if mode is None else mode)


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


@pytest.mark.parametrize("mode", ["r", "r+", "w"])
def test_from_file_memory_rejects_mode(tmp_path, mode):
    path = tmp_path / "missing.omx"
    with pytest.raises(ValueError, match="in-memory matrix cannot have a mode"):
        from_file(path, mode=mode)
    assert not path.exists()


def test_from_omx(omx_path, index, data, metadata):
    matrix = AequilibraEMatrix.from_omx(omx_path, index_name="zones")
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


@pytest.mark.parametrize("mode", ["r+", "x"])
def test_from_dict_memory_rejects_mode(index, data, mode):
    with pytest.raises(ValueError, match="in-memory matrix cannot have a mode"):
        from_dict(data, index, mode=mode)


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


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_matrix_dtype_survives_subset_and_save(matrix, tmp_path, dtype):
    values = np.arange(9, dtype=dtype).reshape(3, 3)
    matrix["car"] = values
    selected = matrix.subset(["car"])
    assert selected["car"].dtype == np.dtype(dtype)

    path = tmp_path / "dtype.omx"
    selected.save_as_omx(path)
    loaded = from_file(path, index_name="zones")
    assert loaded["car"].dtype == np.dtype(dtype)
    np.testing.assert_array_equal(loaded["car"], values)


def test_save_requires_explicit_overwrite(matrix, tmp_path, index, data, metadata):
    path = tmp_path / "export.omx"
    matrix.save_as_omx(path)
    replacement = np.full(matrix.shape, 7.0)
    matrix["car"] = replacement
    matrix.metadata["description"] = "Updated demand"

    with pytest.raises(ValueError):
        matrix.save_as_omx(path)
    assert_matrix(from_file(path, index_name="zones"), index, data, metadata=metadata)

    matrix.save_as_omx(path, overwrite=True)
    assert_matrix(
        from_file(path, index_name="zones"),
        index,
        {**data, "car": replacement},
        metadata={**metadata, "description": "Updated demand"},
    )


@pytest.mark.parametrize("overwrite", [False, True])
def test_save_reuses_matching_index(uint8_index_omx_path, overwrite):
    values = np.ones((2, 2))
    matrix = from_dict({"car": values}, [10, 20], index_name="zones")
    matrix.save_as_omx(uint8_index_omx_path, overwrite=overwrite)

    with omx.open_file(uint8_index_omx_path, "r") as file:
        dataset = file.lookup["zones"]
        assert dataset.dtype == np.dtype("uint8")
        assert dataset.attrs["description"] == "Zone IDs"
        np.testing.assert_array_equal(dataset[:], [10, 20])
        np.testing.assert_array_equal(file["car"][:], values)


def test_save_conflicting_index_requires_overwrite(uint8_index_omx_path):
    matrix = from_dict({"car": np.ones((2, 2))}, [30, 40], index_name="zones")
    with pytest.raises(ValueError, match="index 'zones' already exists with different values"):
        matrix.save_as_omx(uint8_index_omx_path)

    with omx.open_file(uint8_index_omx_path, "r") as file:
        np.testing.assert_array_equal(file.lookup["zones"][:], [10, 20])
        assert file.list_matrices() == []

    matrix.save_as_omx(uint8_index_omx_path, overwrite=True)
    with omx.open_file(uint8_index_omx_path, "r") as file:
        dataset = file.lookup["zones"]
        assert dataset.dtype == np.dtype("uint8")
        assert dataset.attrs["description"] == "Zone IDs"
        np.testing.assert_array_equal(dataset[:], [30, 40])
        np.testing.assert_array_equal(file["car"][:], matrix["car"])


def test_save_rejects_lossy_index_cast_before_writing(uint8_index_omx_path):
    matrix = from_dict({"car": np.ones((2, 2))}, [100, 300], index_name="zones")
    with pytest.raises(ValueError, match="same_value"):
        matrix.save_as_omx(uint8_index_omx_path, overwrite=True, title="Changed title")

    with omx.open_file(uint8_index_omx_path, "r") as file:
        dataset = file.lookup["zones"]
        assert dataset.dtype == np.dtype("uint8")
        assert dataset.attrs["description"] == "Zone IDs"
        np.testing.assert_array_equal(dataset[:], [10, 20])
        assert file.list_matrices() == []
        assert "TITLE" not in file.attrs


def test_save_overwrite_preserves_matrix_attributes(matrix, tmp_path, index, data, metadata):
    path = tmp_path / "attributes.omx"
    matrix.matrix_metadata("car").update({"mode": "car", "units": "trips"})
    matrix.matrix_metadata("bus")["mode"] = "bus"
    matrix.save_as_omx(path)
    with omx.open_file(path, "a") as file:
        file["car"].attrs["periods"] = [1, 2, 3]

    replacement = np.full(matrix.shape, 7.0)
    matrix["car"] = replacement
    matrix.save_as_omx(path, subset=["car"], overwrite=True)

    assert_matrix(from_file(path, index_name="zones"), index, {**data, "car": replacement}, metadata=metadata)
    with omx.open_file(path, "r") as file:
        attrs = file["car"].attrs
        assert attrs["mode"] == "car"
        assert attrs["units"] == "trips"
        np.testing.assert_array_equal(attrs["periods"], [1, 2, 3])
        assert file["bus"].attrs["mode"] == "bus"


def test_save_file_metadata_requires_overwrite(matrix, tmp_path, index, data, metadata):
    path = tmp_path / "export.omx"
    matrix.save_as_omx(path)
    addition = from_dict(
        {"total": np.zeros(matrix.shape)}, index, index_name="other", metadata={"description": "Changed demand"}
    )

    addition.matrix_metadata("total")["mode"] = "all"
    with pytest.raises(ValueError, match="attributes .* already exist"):
        addition.save_as_omx(path, title="Changed title")
    assert_matrix(from_file(path, index_name="zones"), index, data, metadata=metadata)
    with omx.open_file(path, "r") as file:
        assert "other" not in file.list_mappings()
        assert "total" not in file.list_matrices()
        assert file.attrs.get("TITLE", "") != "Changed title"

    addition.save_as_omx(path, overwrite=True)
    assert from_file(path, index_name="other").metadata["description"] == "Changed demand"


@pytest.mark.parametrize("overwrite", [False, True])
def test_save_rejects_destination_shape_before_writing(matrix, tmp_path, overwrite):
    path = tmp_path / "different_shape.omx"
    original = from_dict({"other": np.ones((2, 2))}, [10, 20], index_name="other")
    original.save_as_omx(path)

    with pytest.raises(ValueError, match="destination matrix shape"):
        matrix.save_as_omx(path, overwrite=overwrite)
    assert_matrix(from_file(path, index_name="other"), [10, 20], {"other": np.ones((2, 2))}, index_name="other")
    with omx.open_file(path, "r") as file:
        assert "zones" not in file.list_mappings()


def test_save_rejects_shape_option_before_creating_file(matrix, tmp_path):
    path = tmp_path / "invalid.omx"
    with pytest.raises(ValueError, match="destination matrix shape"):
        matrix.save_as_omx(path, shape=(2, 2))
    assert not path.exists()


@pytest.mark.parametrize("subset", [["car"], [], ["bus", "car"]], ids=["one", "empty", "all"])
def test_subset_is_independent(matrix, index, data, metadata, subset):
    matrix.metadata["periods"] = np.array([1, 2, 3])
    matrix.metadata["TITLE"] = "Source title"
    selected = matrix.subset(subset)
    assert isinstance(selected, AequilibraEMatrix)
    assert not {"OMX_VERSION", "OMX_CREATED_WITH", "SHAPE", "TITLE"} & selected.metadata.keys()
    assert_matrix(selected, index, {name: data[name] for name in subset}, metadata=metadata)

    selected.metadata["description"] = "Selected demand"
    selected.metadata["periods"][0] = 9
    selected.index = [809, 0, 801]
    if subset:
        selected.iloc[0, 1, subset[0]] = 42
    assert_matrix(matrix, index, data, metadata=metadata)
    np.testing.assert_array_equal(matrix.metadata["periods"], [1, 2, 3])


def test_attribute_selection_and_store_subset_survives_close(omx_path, index, data, metadata):
    query = {"mode": "car"}
    loaded = from_file(omx_path, index_name="zones", subset=[query])
    assert_matrix(loaded, index, {"car": data["car"]}, metadata=metadata)
    assert loaded.matrix_metadata("car")["mode"] == "car"

    with from_file(omx_path, index_name="zones", store=True) as store:
        selected = store.subset([query])
        np.testing.assert_array_equal(selected.iloc[:, 1, "car"], data["car"][:, 1])
        assert len(store.subset([{"mode": "missing"}])) == 0

    assert isinstance(selected, AequilibraEMatrix)
    assert_matrix(selected, index, {"car": data["car"]}, metadata=metadata)
    assert selected.matrix_metadata("car")["mode"] == "car"


def test_export_matrix_attributes(matrix, tmp_path, index, data):
    path = tmp_path / "attributes.omx"
    matrix.matrix_metadata("car")["mode"] = "car"
    matrix.matrix_metadata("bus")["mode"] = "bus"
    matrix.save_as_omx(path)
    selected = from_file(path, index_name="zones", subset=[{"mode": "car"}])
    assert_matrix(selected, index, {"car": data["car"]})
    assert selected.matrix_metadata("car")["mode"] == "car"

    selected.matrix_metadata("car")["units"] = "trips"
    output = tmp_path / "selected.omx"
    selected.save_as_omx(output)
    with omx.open_file(output, "r") as file:
        assert file.list_matrices() == ["car"]
        assert dict(file["car"].attrs) == {"mode": "car", "units": "trips"}


def test_matrix_metadata_follows_matrix_lifetime(matrix):
    attributes = matrix.matrix_metadata("car")
    attributes.update({"mode": "car", "units": "trips"})
    assert dict(matrix.matrix_metadata("car")) == {"mode": "car", "units": "trips"}
    del attributes["units"]
    matrix["car"] = np.ones(matrix.shape)
    assert dict(matrix.matrix_metadata("car")) == {"mode": "car"}

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

    selected.matrix_metadata("car")["mode"] = "updated"
    selected.matrix_metadata("car")["periods"][0] = 9
    assert matrix.matrix_metadata("car")["mode"] == "car"
    np.testing.assert_array_equal(matrix.matrix_metadata("car")["periods"], [1, 2, 3])


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
    matrix.save_as_omx(path, subset=["car"], overwrite=True)

    with omx.open_file(path, "r") as file:
        assert dict(file["car"].attrs) == {
            "mode": "source",
            "year": 2030,
            "units": "trips",
            "description": "Source demand",
        }
    assert matrix.matrix_metadata("car")["mode"] == "source"


def test_mapping_operations(matrix, data):
    assert isinstance(matrix, MutableMapping)
    assert set(matrix.keys()) == set(data)
    for name, values in matrix.items():
        np.testing.assert_array_equal(values, data[name])

    total = data["car"] + data["bus"]
    matrix.update({"total": total})
    assert "total" in matrix
    assert len(matrix) == 3
    np.testing.assert_array_equal(matrix["total"], total)

    replacement = np.full(matrix.shape, 7.0)
    matrix["car"] = replacement
    np.testing.assert_array_equal(matrix["car"], replacement)
    np.testing.assert_array_equal(matrix.pop("total"), total)
    del matrix["bus"]
    assert set(matrix) == {"car"}
    assert len(matrix) == 1


def test_missing_matrix(matrix):
    with pytest.raises(KeyError):
        matrix["missing"]
    for indexer in (matrix.loc, matrix.iloc):
        with pytest.raises(KeyError):
            indexer[0, 0, "missing"]
        with pytest.raises(KeyError):
            indexer[0, 0, "missing"] = 42
    with pytest.raises(KeyError):
        del matrix["missing"]
    with pytest.raises(KeyError):
        matrix.matrix_metadata("missing")


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
    np.testing.assert_array_equal(matrix.iloc[*selection, "car"], data["car"][selection])
    expected = data["car"].copy()
    expected[selection] = values
    matrix.iloc[*selection, "car"] = values
    assert_matrix(matrix, index, {**data, "car": expected}, metadata=metadata)


def test_augmented_assignment(matrix, data):
    expected = data["car"].copy()

    expected *= 2
    expected[:, 2] *= 3
    expected[[0, 2], :] += 4

    matrix["car"] *= 2
    matrix.iloc[:, 2, "car"] *= 3
    matrix.iloc[[0, 2], :, "car"] += 4

    np.testing.assert_array_equal(matrix["car"], expected)
    np.testing.assert_array_equal(matrix["bus"], data["bus"])


@pytest.mark.parametrize(
    "selection, shares_memory",
    [
        pytest.param((), True, id="whole-matrix"),
        pytest.param((slice(0, 2), slice(None)), True, id="basic-slice"),
        pytest.param(([0, 2], slice(None)), False, id="fancy-rows"),
    ],
)
def test_mutating_read_results(matrix, data, selection, shares_memory):
    result = matrix.iloc[*selection, "car"]
    result[:] = 42
    expected = data["car"].copy()

    if isinstance(matrix, AequilibraEMatrix) and shares_memory:
        expected[selection] = 42

    np.testing.assert_array_equal(matrix["car"], expected)


@pytest.mark.parametrize("accessor", ["loc", "iloc"])
def test_indexer_whole_matrix(matrix, data, accessor):
    indexer = getattr(matrix, accessor)
    np.testing.assert_array_equal(indexer["car"], matrix["car"])
    np.testing.assert_array_equal(indexer["car",], matrix["car"])
    indexer["car",] = data["car"] * 2
    np.testing.assert_array_equal(matrix["car"], data["car"] * 2)
    indexer["new"] = data["bus"]
    np.testing.assert_array_equal(matrix["new"], data["bus"])


@pytest.mark.parametrize("key", [0, ["car"], {"mode": "car"}, ("car",), ("car", 0, 1)])
def test_plain_indexing_requires_string(matrix, data, key):
    with pytest.raises(TypeError, match="string"):
        matrix[key]
    with pytest.raises(TypeError, match="string"):
        matrix[key] = data["car"]
    np.testing.assert_array_equal(matrix["car"], data["car"])


@pytest.mark.parametrize(
    "key",
    [(), (0, 1, 2, "car"), 0, ["car"], {"mode": "car"}, (0, ["car"]), (0, 1, {"mode": "car"})],
)
def test_indexer_rejects_bad_arguments(memory_matrix, key):
    # Both indexers use the same argument handling.
    with pytest.raises(TypeError):
        memory_matrix.iloc[key]
    with pytest.raises(TypeError):
        memory_matrix.iloc[key] = 42


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
def test_loc_translates_zones(memory_matrix, data, labels, positions):
    np.testing.assert_array_equal(memory_matrix.loc[*labels, "car"], data["car"][positions])
    expected = data["car"].copy()
    expected[positions] = 42
    memory_matrix.loc[*labels, "car"] = 42
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
def test_loc_inclusive_slices(memory_matrix, data, labels, positions):
    np.testing.assert_array_equal(memory_matrix.loc[labels, "car"], data["car"][positions])
    expected = data["car"].copy()
    expected[positions] = 42
    memory_matrix.loc[labels, "car"] = 42
    np.testing.assert_array_equal(memory_matrix["car"], expected)


@pytest.mark.parametrize(
    "key",
    [(999, "car"), (slice(None), [309, 999], "car"), (slice(999, 101), "car"), (slice(309, 999), "car")],
)
def test_loc_unknown_zone(memory_matrix, data, key):
    with pytest.raises(KeyError):
        memory_matrix.loc[key]
    with pytest.raises(KeyError):
        memory_matrix.loc[key] = 42
    np.testing.assert_array_equal(memory_matrix["car"], data["car"])


@pytest.mark.parametrize("selection", [Ellipsis, None, [True, False, True], np.eye(3, dtype=bool)])
def test_loc_passes_special_selections_unchanged(memory_matrix, data, selection):
    np.testing.assert_array_equal(memory_matrix.loc[selection, "car"], data["car"][selection])
    expected = data["car"].copy()
    expected[selection] = 42
    memory_matrix.loc[selection, "car"] = 42
    np.testing.assert_array_equal(memory_matrix["car"], expected)


@pytest.mark.parametrize("accessor", ["loc", "iloc"])
def test_indexer_paired_selections(matrix, data, accessor):
    indexer = getattr(matrix, accessor)
    rows, cols = ([309, 0, 101], [0, 101, 309]) if accessor == "loc" else ([0, 1, 2], [1, 2, 0])
    if isinstance(matrix, MatrixStore):
        # h5py does not support two index arrays. Do not work around that limit.
        with pytest.raises(TypeError):
            indexer[rows, cols, "car"]
        with pytest.raises(TypeError):
            indexer[rows, cols, "car"] = [11, 12, 13]
        np.testing.assert_array_equal(matrix["car"], data["car"])
    else:
        np.testing.assert_array_equal(indexer[rows, cols, "car"], data["car"][[0, 1, 2], [1, 2, 0]])
        indexer[rows, cols, "car"] = [11, 12, 13]
        expected = data["car"].copy()
        expected[[0, 1, 2], [1, 2, 0]] = [11, 12, 13]
        np.testing.assert_array_equal(matrix["car"], expected)


@pytest.mark.parametrize("labels", [slice(101, 309, -1), [101, 309], [309, 309]])
def test_loc_keeps_store_limits(omx_path, labels):
    # h5py rejects reverse slices, unsorted positions and repeated positions.
    with from_file(omx_path, index_name="zones", mode="r+", store=True) as store:
        with pytest.raises((TypeError, ValueError)):
            store.loc[labels, "car"]
        with pytest.raises((TypeError, ValueError)):
            store.loc[labels, "car"] = 42


def test_loc_uses_relabelled_index(matrix, data):
    loc = matrix.loc
    matrix.index = [1001, 1002, 1003]
    assert loc[1001, 1002, "car"] == data["car"][0, 1]
    with pytest.raises(KeyError):
        loc[309, "car"]
    with pytest.raises(ValueError):
        matrix.index = [1, 1, 1]
    loc[1001, 1002, "car"] += 5
    assert matrix.iloc[0, 1, "car"] == data["car"][0, 1] + 5


def test_loc_large_zone_ids():
    zones = [0, 2**64 - 1]
    matrix = from_dict({"car": [[1, 2], [3, 4]]}, index=np.array(zones, dtype=np.uint64))
    assert matrix.loc[zones[0], zones[1], "car"] == 2
    np.testing.assert_array_equal(matrix.loc[zones, "car"], matrix["car"])
    np.testing.assert_array_equal(matrix.loc[np.array(zones, dtype=np.uint64), "car"], matrix["car"])


def test_store_omx_access(omx_path, data):
    with from_file(omx_path, index_name="zones", store=True) as store:
        assert isinstance(store.omx, omx.File)
        assert set(store.omx.list_matrices()) == set(data)
        np.testing.assert_array_equal(np.asarray(store.omx["car"]), data["car"])


@pytest.mark.parametrize("operation", ["replace", "scale", "slice"])
def test_store_updates_preserve_matrix_attributes(omx_path, data, operation):
    expected = data["car"].copy()
    with from_file(omx_path, index_name="zones", mode="r+", store=True) as store:
        store.matrix_metadata("car")["units"] = "trips"
        if operation == "replace":
            expected[:] = 7
            store["car"] = expected
        elif operation == "scale":
            expected *= 2
            store["car"] *= 2
        else:
            expected[:, 1] = 7
            store.iloc[:, 1, "car"] = 7
        assert dict(store.matrix_metadata("car")) == {"mode": "car", "units": "trips"}
        np.testing.assert_array_equal(store.subset([{"mode": "car"}])["car"], expected)

    with from_file(omx_path, index_name="zones", store=True) as store:
        assert dict(store.matrix_metadata("car")) == {"mode": "car", "units": "trips"}
        np.testing.assert_array_equal(store["car"], expected)


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


def test_store_edits_survive_reopening(omx_path, data, metadata):
    car = data["car"].copy()
    car[0, 1] = 42
    car[:, 2] = [1, 2, 3]
    car *= 2
    total = car + data["bus"]
    new_index = [809, 0, 801]
    expected_metadata = {**metadata, "description": "Updated demand"}

    with from_file(omx_path, index_name="zones", mode="r+", store=True) as store:
        store.iloc[0, 1, "car"] = 42
        store.iloc[:, 2, "car"] = [1, 2, 3]
        store["car"] *= 2
        store["total"] = store["car"] + store["bus"]
        del store["bus"]
        store.index = new_index
        store.metadata.update(expected_metadata)

    with from_file(omx_path, index_name="zones", store=True) as reopened:
        assert_matrix(reopened, new_index, {"car": car, "total": total}, metadata=expected_metadata)


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
    "invalid_index",
    [[[1, 2]], [1, 1], [-1, 2], [1.0, 2.0], ["1", "2"]],
    ids=["dimensions", "duplicates", "negative", "float", "string"],
)
def test_index_validation(matrix_factory, invalid_index):
    with pytest.raises((TypeError, ValueError)):
        matrix_factory(invalid_index)


@pytest.mark.parametrize("new_index", [[309, 0], [309, 0, 101, 205]], ids=["shorter", "longer"])
def test_index_size_is_fixed(matrix, index, data, metadata, new_index):
    with pytest.raises(ValueError):
        matrix.index = new_index
    assert_matrix(matrix, index, data, metadata=metadata)


def test_from_df_infers_zones_and_sums_duplicate_trips():
    df = pd.DataFrame({"origin": [309, 309, 101], "destination": [101, 101, 205], "car": [2, 3, 7], "bus": [1, 4, 2]})
    matrix = from_df(df, "origin", "destination", index_name="zones")
    expected = {
        "car": np.array([[0, 7, 0], [0, 0, 0], [5, 0, 0]]),
        "bus": np.array([[0, 2, 0], [0, 0, 0], [5, 0, 0]]),
    }
    assert_matrix(matrix, [101, 205, 309], expected)
    assert matrix["car"].dtype == np.dtype(float)


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_from_df_uses_supplied_dtype(dtype):
    df = pd.DataFrame({"origin": [10, 10], "destination": [20, 20], "car": [2, 3]})
    matrix = from_df(df, "origin", "destination", fill_value=7, dtype=dtype)
    assert matrix["car"].dtype == np.dtype(dtype)
    np.testing.assert_array_equal(matrix["car"], [[7, 5], [7, 7]])


def test_from_df_preserves_supplied_zones_subset_and_fill_value():
    df = pd.DataFrame({"origin": [309, 309, 101], "destination": [101, 101, 205], "car": [2, 3, 7], "bus": [1, 4, 2]})
    index = [309, 205, 101, 0]
    matrix = from_df(df, "origin", "destination", ["car"], index=index, index_name="zones", fill_value=np.nan)
    expected = np.full((4, 4), np.nan)
    expected[0, 2] = 5
    expected[2, 1] = 7
    assert_matrix(matrix, index, {"car": expected})


def test_from_df_requires_known_zones():
    df = pd.DataFrame({"origin": [101], "destination": [309], "car": [7]})
    with pytest.raises(ValueError):
        from_df(df, "origin", "destination", index=[101, 205])


@pytest.mark.parametrize(
    "array_type", [sparse.coo_array, sparse.csr_array, sparse.csc_array], ids=["coo", "csr", "csc"]
)
def test_from_scipy(index, data, array_type):
    matrix = from_scipy({name: array_type(values) for name, values in data.items()}, index, index_name="zones")
    assert isinstance(matrix, AequilibraEMatrix)
    assert_matrix(matrix, index, data)


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_from_scipy_preserves_dtype(index, dtype):
    values = np.arange(9, dtype=dtype).reshape(3, 3)
    matrix = from_scipy({"car": sparse.csr_array(values)}, index)
    assert matrix["car"].dtype == np.dtype(dtype)
    np.testing.assert_array_equal(matrix["car"], values)
