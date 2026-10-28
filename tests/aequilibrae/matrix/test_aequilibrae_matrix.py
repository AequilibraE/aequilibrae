import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from aequilibrae.matrix import AequilibraeMatrix, AequilibraeSparseMatrix, MatrixStore, from_df, from_dict, from_file


@pytest.fixture
def index():
    return np.array([309, 0, 101])


@pytest.fixture
def demand():
    return np.array([[0.0, 12.5, 8], [10, 0, 6], [7, 9, 0]])


def test_matrix_mapping_indexing_and_metadata(index, demand):
    matrix = AequilibraeMatrix(index=index)
    matrix["car"] = demand
    matrix.metadata["year"] = 2026
    matrix.matrix_metadata("car")["purpose"] = "commute"

    np.testing.assert_array_equal(matrix["car"], demand)
    np.testing.assert_array_equal(matrix.mat["car", 309, 0], 12.5)
    assert matrix["car"].shape == (3, 3)
    assert matrix.totals()["car"] == pytest.approx(demand.sum())
    assert matrix.metadata["year"] == 2026
    assert matrix.matrix_metadata("car")["purpose"] == "commute"


def test_indexed_edits_and_subset_are_independent(index, demand):
    matrix = from_dict({"car": demand, "bus": demand * 2}, index)
    matrix.mat["car", 309, 0] = 42
    assert matrix["car"][0, 1] == 42

    selected = matrix.subset(["car"])
    selected["car"][0, 1] = 7
    assert matrix["car"][0, 1] == 42
    assert list(selected) == ["car"]


def test_omx_round_trip_preserves_values_index_and_metadata(tmp_path, index, demand):
    matrix = from_dict({"car": demand}, index)
    matrix.metadata["year"] = 2026
    matrix.matrix_metadata("car")["purpose"] = "commute"
    path = tmp_path / "demand.omx"
    matrix.save_as_omx(path)

    loaded = from_file(path)
    np.testing.assert_array_equal(loaded["car"], demand)
    np.testing.assert_array_equal(loaded.index, index)
    assert loaded.metadata["year"] == 2026
    assert loaded.matrix_metadata("car")["purpose"] == "commute"


def test_store_updates_persist_and_subset_loads(tmp_path, index, demand):
    path = tmp_path / "demand.omx"
    with MatrixStore(path, index, mode="x") as store:
        store["car"] = demand
        store["bus"] = demand * 2
        store.mat["car", 309, 0] = 42

    with from_file(path, store=True) as store:
        selected = store.subset(["car"])
        assert list(selected) == ["car"]
        assert selected.imat["car", 0, 1] == 42


def test_from_df_builds_demand_from_zone_labels():
    records = pd.DataFrame({"origin": [20, 10, 20], "destination": [10, 20, 10], "trips": [2, 3, 4]})
    matrix = from_df(records, "origin", "destination", index=[20, 10])

    np.testing.assert_array_equal(matrix["trips"], [[0, 6], [3, 0]])
    np.testing.assert_array_equal(matrix.index, [20, 10])


def test_sparse_matrix_round_trip_and_zone_indexing(tmp_path, index):
    matrix = AequilibraeSparseMatrix(index)
    matrix["car"] = sp.csr_array(([5.0, 2.0], ([0, 2], [1, 0])), shape=(3, 3))
    assert matrix.imat["car", 0, 1] == 5
    assert matrix.mat["car", 309, 0] == 5

    path = tmp_path / "sparse.omx"
    matrix.save_as_omx(path)
    loaded = from_file(path)
    assert sp.issparse(loaded["car"])
    assert loaded.imat["car", 2, 0] == 2
    np.testing.assert_array_equal(loaded.index, index)
