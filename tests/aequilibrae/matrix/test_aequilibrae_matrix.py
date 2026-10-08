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


def test_assignment_copies_and_converts_values(matrix):
    values = np.arange(9, dtype=np.int32).reshape(3, 3)
    expected = values.copy()

    matrix["car"] = values
    values[:] = 999

    assert matrix["car"].dtype == np.dtype("float64")
    np.testing.assert_array_equal(matrix["car"], expected)


@pytest.mark.parametrize("array_type", [np.asarray, list, tuple], ids=["array", "list", "tuple"])
def test_assignment_accepts_array_like(matrix, data, array_type):
    matrix["car"] = array_type(data["car"].tolist())
    np.testing.assert_array_equal(matrix["car"], data["car"])
    assert matrix["car"].dtype == np.dtype("float64")

    matrix["car", 0, 1] = 2.5
    assert matrix["car", 0, 1] == 2.5
    matrix["car"] *= 1.1
    assert matrix["car", 0, 1] == pytest.approx(2.75)


@pytest.mark.parametrize("invalid_values", [1, [[1, 2], [3]], [["bad"] * 3] * 3])
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
    matrix = manage(from_dict(data, index, index_name="zones", metadata=metadata, store=store, **kwargs))
    assert isinstance(matrix, MatrixStore if store else AequilibraEMatrix)

    if store:
        matrix.close()
        matrix = manage(from_file(kwargs["path"], index_name="zones", store=True))

    assert_matrix(matrix, index, data, metadata=metadata)


@pytest.mark.parametrize("store", [False, True], ids=["memory", "store"])
def test_from_file(omx_path, manage, index, data, metadata, store):
    matrix = manage(from_file(omx_path, index_name="zones", store=store))
    assert isinstance(matrix, MatrixStore if store else AequilibraEMatrix)
    assert_matrix(matrix, index, data, metadata=metadata)


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


@pytest.mark.parametrize("subset", [None, ["car"], []], ids=["all", "one", "empty"])
def test_omx_round_trip(matrix, tmp_path, index, data, metadata, subset):
    path = tmp_path / "export.omx"
    matrix.save_as_omx(path, subset=subset)

    loaded = from_file(path, index_name="zones")
    expected = data if subset is None else {name: data[name] for name in subset}

    assert_matrix(loaded, index, expected, metadata=metadata)
    assert_matrix(matrix, index, data, metadata=metadata)


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


@pytest.mark.parametrize("matrix_metadata", [None, {"car": {"mode": "updated", "year": 2026}}])
def test_save_overwrite_preserves_matrix_attributes(matrix, tmp_path, index, data, metadata, matrix_metadata):
    path = tmp_path / "attributes.omx"
    matrix.save_as_omx(path, matrix_metadata={"car": {"mode": "car", "units": "trips"}, "bus": {"mode": "bus"}})
    with omx.open_file(path, "a") as file:
        file["car"].attrs["periods"] = [1, 2, 3]

    replacement = np.full(matrix.shape, 7.0)
    matrix["car"] = replacement
    matrix.save_as_omx(path, subset=["car"], matrix_metadata=matrix_metadata, overwrite=True)

    assert_matrix(from_file(path, index_name="zones"), index, {**data, "car": replacement}, metadata=metadata)
    with omx.open_file(path, "r") as file:
        attrs = file["car"].attrs
        assert attrs["mode"] == ("car" if matrix_metadata is None else "updated")
        assert attrs["units"] == "trips"
        np.testing.assert_array_equal(attrs["periods"], [1, 2, 3])
        if matrix_metadata is not None:
            assert attrs["year"] == 2026
        assert file["bus"].attrs["mode"] == "bus"


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("subset", [None, ["car"], []], ids=["all", "one", "empty"])
def test_save_rejects_metadata_outside_subset(matrix, tmp_path, subset, overwrite):
    path = tmp_path / "invalid.omx"
    name = "missing" if subset is None else "bus"
    with pytest.raises(ValueError, match="outside the save subset"):
        matrix.save_as_omx(path, subset=subset, matrix_metadata={name: {"mode": name}}, overwrite=overwrite)
    assert not path.exists()


def test_save_rejects_metadata_for_existing_matrix_outside_subset(matrix, tmp_path, index, data, metadata):
    path = tmp_path / "export.omx"
    matrix.save_as_omx(path)

    with pytest.raises(ValueError, match="outside the save subset"):
        matrix.save_as_omx(path, subset=["car"], matrix_metadata={"bus": {"mode": "bus"}}, overwrite=True)
    assert_matrix(from_file(path, index_name="zones"), index, data, metadata=metadata)
    with omx.open_file(path, "r") as file:
        assert "mode" not in file["bus"].attrs


@pytest.mark.parametrize("matrix_metadata", [None, {"total": {"mode": "all"}}], ids=["no-attributes", "attributes"])
def test_save_file_metadata_requires_overwrite(matrix, tmp_path, index, data, metadata, matrix_metadata):
    path = tmp_path / "export.omx"
    matrix.save_as_omx(path)
    addition = from_dict(
        {"total": np.zeros(matrix.shape)}, index, index_name="other", metadata={"description": "Changed demand"}
    )

    with pytest.raises(ValueError, match="attributes .* already exist"):
        addition.save_as_omx(path, matrix_metadata=matrix_metadata, title="Changed title")
    assert_matrix(from_file(path, index_name="zones"), index, data, metadata=metadata)
    with omx.open_file(path, "r") as file:
        assert "other" not in file.list_mappings()
        assert "total" not in file.list_matrices()
        assert file.attrs.get("TITLE", "") != "Changed title"

    addition.save_as_omx(path, matrix_metadata=matrix_metadata, overwrite=True)
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
    selected = matrix.subset(subset)
    assert isinstance(selected, AequilibraEMatrix)
    assert_matrix(selected, index, {name: data[name] for name in subset}, metadata=metadata)

    selected.metadata["description"] = "Selected demand"
    selected.index = [809, 0, 801]
    if subset:
        selected[subset[0], 0, 1] = 42
    assert_matrix(matrix, index, data, metadata=metadata)


def test_attribute_selection_and_store_subset_survives_close(omx_path, index, data, metadata):
    query = {"mode": "car"}
    loaded = from_file(omx_path, index_name="zones", subset=[query])
    assert_matrix(loaded, index, {"car": data["car"]}, metadata=metadata)

    with from_file(omx_path, index_name="zones", store=True) as store:
        selected = store.subset([query])
        columns = store[query, :, 1]

        assert isinstance(columns, list)
        assert len(columns) == 1
        np.testing.assert_array_equal(columns[0], data["car"][:, 1])
        assert store[{"mode": "missing"}] == []

    assert isinstance(selected, AequilibraEMatrix)
    assert_matrix(selected, index, {"car": data["car"]}, metadata=metadata)


def test_export_matrix_attributes(matrix, tmp_path, index, data):
    path = tmp_path / "attributes.omx"
    matrix.save_as_omx(path, matrix_metadata={"car": {"mode": "car"}, "bus": {"mode": "bus"}})
    selected = from_file(path, index_name="zones", subset=[{"mode": "car"}])
    assert_matrix(selected, index, {"car": data["car"]})


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
    with pytest.raises(KeyError):
        matrix["missing", 0, 1]
    with pytest.raises(KeyError):
        del matrix["missing"]


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
    np.testing.assert_array_equal(matrix["car", *selection], data["car"][selection])
    expected = data["car"].copy()
    expected[selection] = values
    matrix["car", *selection] = values
    assert_matrix(matrix, index, {**data, "car": expected}, metadata=metadata)


def test_augmented_assignment(matrix, data):
    expected = data["car"].copy()

    expected *= 2
    expected[:, 2] *= 3
    expected[[0, 2], :] += 4

    matrix["car"] *= 2
    matrix["car", :, 2] *= 3
    matrix["car", [0, 2], :] += 4

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
    result = matrix["car", *selection]
    result[:] = 42
    expected = data["car"].copy()

    if isinstance(matrix, AequilibraEMatrix) and shares_memory:
        expected[selection] = 42

    np.testing.assert_array_equal(matrix["car"], expected)


def test_store_omx_access(omx_path, data):
    with from_file(omx_path, index_name="zones", store=True) as store:
        assert isinstance(store.omx, omx.File)
        assert set(store.omx.list_matrices()) == set(data)
        np.testing.assert_array_equal(np.asarray(store.omx["car"]), data["car"])


@pytest.mark.parametrize("operation", ["replace", "scale", "slice"])
def test_store_updates_preserve_matrix_attributes(omx_path, data, operation):
    expected = data["car"].copy()
    with from_file(omx_path, index_name="zones", mode="r+", store=True) as store:
        if operation == "replace":
            expected[:] = 7
            store["car"] = expected
        elif operation == "scale":
            expected *= 2
            store["car"] *= 2
        else:
            expected[:, 1] = 7
            store["car", :, 1] = 7
        assert store.omx["car"].attrs["mode"] == "car"
        np.testing.assert_array_equal(store[{"mode": "car"}][0], expected)

    with from_file(omx_path, index_name="zones", store=True) as store:
        assert store.omx["car"].attrs["mode"] == "car"
        np.testing.assert_array_equal(store["car"], expected)


def test_read_only_store_rejects_index_replacement(omx_path, index):
    with from_file(omx_path, index_name="zones", store=True) as store:
        with pytest.raises(PermissionError, match="read-only"):
            store.index = [809, 0, 801]
        np.testing.assert_array_equal(store.index, index)
        np.testing.assert_array_equal(store.omx.lookup["zones"][:], index)


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
        store["car", 0, 1] = 42
        store["car", :, 2] = [1, 2, 3]
        store["car"] *= 2
        store["total"] = store["car"] + store["bus"]
        del store["bus"]
        store.index = new_index
        store.metadata.update(expected_metadata)

    with from_file(omx_path, index_name="zones", store=True) as reopened:
        assert_matrix(reopened, new_index, {"car": car, "total": total}, metadata=expected_metadata)


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
