import string
from math import floor
from random import choice

import numpy as np
import openmatrix as omx
import pandas as pd
import pytest

from aequilibrae.matrix import AequilibraeMatrix, MatrixStore
from aequilibrae.project.data.matrices import Matrices
from aequilibrae.utils.db_utils import NestedTransactionManager


def test_list_available_matrices(sioux_falls_example):
    mat = sioux_falls_example.scenario.matrices
    assert mat.get_available_matrices_and_metadata() == {
        "skims.omx": {"procedure": None, "procedure_id": None, "timestamp": None, "description": None},
        "demand_mc.omx": {"procedure": None, "procedure_id": None, "timestamp": None, "description": None},
        "demand.omx": {"procedure": None, "procedure_id": None, "timestamp": None, "description": None},
    }

    df = pd.DataFrame(columns=["procedure", "procedure_id", "timestamp", "description"])
    df.index.name = "file_name"
    df.loc["demand.omx", :] = [None] * 4
    df.loc["demand_mc.omx", :] = [None] * 4
    df.loc["skims.omx", :] = [None] * 4

    pd.testing.assert_frame_equal(mat.list(), df)

    demand_matrix = mat.get_matrix("demand.omx", mode="r+")

    demand_matrix.metadata.update(
        {
            "procedure": "testing",
            "procedure_id": 2,
            "timestamp": "now",
            "description": "Finding out if this works!",
        }
    )

    assert mat.get_available_matrices_and_metadata() == {
        "skims.omx": {"procedure": None, "procedure_id": None, "timestamp": None, "description": None},
        "demand_mc.omx": {"procedure": None, "procedure_id": None, "timestamp": None, "description": None},
        "demand.omx": {
            "procedure": "testing",
            "procedure_id": 2,
            "timestamp": "now",
            "description": "Finding out if this works!",
        },
    }


def test_get_matrix(sioux_falls_example):
    matrices = sioux_falls_example.matrices
    with pytest.raises(ValueError, match="matrices folder does not have a matrix file named 'omxq.omx'"):
        _ = matrices.get_matrix("omxq.omx")
    mat = matrices.get_matrix("demand.omx")
    assert isinstance(mat, MatrixStore)
    assert floor(sum(values.sum() for values in mat.values())) == 360600, "Matrix loaded incorrectly"
    mat.close()

    selected = matrices.get_matrix("demand.omx", subset=["matrix"])
    assert isinstance(selected, AequilibraeMatrix)
    assert floor(sum(values.sum() for values in selected.values())) == 360600, "Matrix subset loaded incorrectly"


def randomword(length):
    allowed_characters = string.ascii_letters + '_123456789@!()[]{};:"    -'
    val = "".join(choice(allowed_characters) for _ in range(length))
    if val[0] == "_" or val[-1] == "_":
        return randomword(length)
    return val


def mat_count(sioux_falls_example, should_have: int, error_message: str):
    cnt = len(sioux_falls_example.scenario.matrices.get_available_matrices_and_metadata())
    # with sioux_falls_example.db_connection as conn:
    #     cnt = conn.execute("Select count(*) from Matrices;").fetchone()[0]
    assert cnt == should_have, error_message


def test_get_record(sioux_falls_example):
    matrices = sioux_falls_example.matrices
    # rec = matrices.get_matrix("demand_mc.omx")
    file_name = "demand_mc.omx"
    assert matrices.cores_on_disk(file_name) == 3, "record populated wrong. Number of cores"
    with omx.open_file(matrices.folder / file_name, mode="r") as matrix:
        assert matrix.attrs.get("description") is None, "record populated wrong. Description"


@pytest.fixture
def matrix_table(tmp_path):
    manager = NestedTransactionManager(":memory:")
    manager._connection.execute(
        """CREATE TABLE matrices (
            name TEXT NOT NULL PRIMARY KEY,
            file_name TEXT NOT NULL UNIQUE,
            cores INTEGER NOT NULL DEFAULT 1,
            procedure TEXT,
            procedure_id TEXT,
            timestamp DATETIME DEFAULT current_timestamp,
            description TEXT
        )"""
    )
    table = Matrices(tmp_path)
    yield table
    manager.close()


@pytest.fixture(params=[AequilibraeMatrix, MatrixStore], ids=["memory", "store"])
def source_matrix(request, tmp_path):
    if request.param is MatrixStore:
        matrix = MatrixStore(tmp_path / "source.omx", index=[10, 20], mode="x")
    else:
        matrix = AequilibraeMatrix(index=[10, 20])
    matrix["demand"] = [[0, 1], [2, 0]]
    matrix.metadata["description"] = "Future demand"
    matrix.matrix_metadata("demand")["purpose"] = "work"
    yield matrix
    if isinstance(matrix, MatrixStore):
        matrix.close()


def test_save_matrix_persists_metadata_without_changing_source(matrix_table, source_matrix):
    original_metadata = dict(source_matrix.metadata)
    metadata = {
        "procedure": "Test export",
        "procedure_id": "export-id",
        "timestamp": "2026-01-01 12:00:00",
        "description": "Exported demand",
    }

    matrix_table.save_matrix("exported.omx", source_matrix, **metadata)

    assert matrix_table.get_available_matrices_and_metadata()["exported.omx"] == metadata
    assert set(source_matrix.metadata) == set(original_metadata)
    for key, value in original_metadata.items():
        np.testing.assert_equal(source_matrix.metadata[key], value)
    with matrix_table.get_matrix("exported.omx") as exported:
        np.testing.assert_array_equal(exported["demand"], source_matrix["demand"])
        assert exported.matrix_metadata("demand")["purpose"] == "work"


def test_save_matrix_retains_omitted_metadata(matrix_table, source_matrix):
    original_metadata = dict(source_matrix.metadata)

    matrix_table.save_matrix("exported.omx", source_matrix, procedure="Test export")

    metadata = matrix_table.get_available_matrices_and_metadata()["exported.omx"]
    assert metadata == {
        "procedure": "Test export",
        "procedure_id": None,
        "timestamp": None,
        "description": "Future demand",
    }
    assert set(source_matrix.metadata) == set(original_metadata)
    for key, value in original_metadata.items():
        np.testing.assert_equal(source_matrix.metadata[key], value)


def test_save_matrix_accepts_read_only_store(matrix_table, tmp_path):
    source = AequilibraeMatrix(index=[10, 20])
    source["demand"] = [[0, 1], [2, 0]]
    source.save_as_omx(tmp_path / "read_only.omx")

    with MatrixStore(tmp_path / "read_only.omx", mode="r") as store:
        matrix_table.save_matrix("exported.omx", store, description="Exported demand")
        assert "description" not in store.metadata

    assert matrix_table.get_available_matrices_and_metadata()["exported.omx"]["description"] == "Exported demand"


def test_get_matrix_uses_requested_index(matrix_table):
    matrix = AequilibraeMatrix(index=[30, 10], index_name="zones")
    matrix["demand"] = [[0, 3], [2, 0]]
    matrix_table.save_matrix("demand.omx", matrix)
    inferred = matrix_table.get_matrix("demand.omx")
    assert isinstance(inferred, MatrixStore)
    assert inferred.index_name == "zones"
    inferred.close()

    selected = matrix_table.get_matrix("demand.omx", subset=["demand"])
    assert isinstance(selected, AequilibraeMatrix)

    np.testing.assert_array_equal(selected["demand"], matrix["demand"])


def test_get_matrix_requires_index_when_multiple_nondefault_indices_exist(matrix_table):
    matrix = AequilibraeMatrix(index=[10, 20], index_name="zones")
    matrix["demand"] = [[0, 1], [2, 0]]
    matrix_table.save_matrix("demand.omx", matrix)
    with omx.open_file(matrix_table.folder / "demand.omx", mode="a") as file:
        file.lookup.create_dataset("other_zones", data=[10, 20])

    with pytest.raises(ValueError, match="Multiple zone indices"):
        matrix_table.get_matrix("demand.omx")
    np.testing.assert_array_equal(
        matrix_table.get_matrix("demand.omx", index_name="zones", subset=["demand"])["demand"], matrix["demand"]
    )


@pytest.mark.parametrize("extension", ["", ".aem", ".csv", ".OMX", ".Omx"])
def test_create_rejects_other_formats(matrix_table, source_matrix, extension):
    with pytest.raises(ValueError, match="Only OMX"):
        matrix_table.save_matrix(f"demand{extension}", source_matrix)
    assert f"demand{extension}" not in matrix_table.get_available_matrices_and_metadata()


def test_create_does_not_overwrite_existing_file(matrix_table, source_matrix):
    path = matrix_table.folder / "demand.omx"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        matrix_table.save_matrix(path.name, source_matrix)
    assert path.read_bytes() == b"existing"


def test_create_removes_partial_export(matrix_table, source_matrix, monkeypatch):
    def fail(path):
        path.write_bytes(b"partial")
        raise OSError("export failed")

    monkeypatch.setattr(source_matrix, "save_as_omx", fail)
    with pytest.raises(OSError, match="export failed"):
        matrix_table.save_matrix("demand.omx", source_matrix)
    assert not (matrix_table.folder / "demand.omx").exists()
    assert not list(matrix_table.folder.glob(".demand.*.tmp.omx"))
    assert "demand" not in matrix_table.get_available_matrices_and_metadata()


@pytest.mark.parametrize("subset", [None, ["demand"]])
def test_get_matrix_only_opens_requested_file(matrix_table, source_matrix, monkeypatch, subset):
    matrix_table.save_matrix("valid.omx", source_matrix)
    (matrix_table.folder / "damaged.omx").write_bytes(b"not an OMX file")

    def unexpected_catalogue_scan():
        pytest.fail("Loading one matrix must not scan other OMX files")

    monkeypatch.setattr(matrix_table, "get_available_matrices_and_metadata", unexpected_catalogue_scan)
    assert matrix_table.file_exists("valid.omx")
    assert not matrix_table.file_exists("missing.omx")
    loaded = matrix_table.get_matrix("valid.omx", subset=subset)
    try:
        np.testing.assert_array_equal(loaded["demand"], source_matrix["demand"])
    finally:
        if isinstance(loaded, MatrixStore):
            loaded.close()


@pytest.mark.parametrize("extension", [".omx", ".data", ""])
@pytest.mark.parametrize("subset", [None, ["demand"]])
def test_get_matrix_logs_unreadable_file_with_context(matrix_table, caplog, subset, extension):
    path = matrix_table.folder / f"damaged{extension}"
    path.write_bytes(b"not an OMX file")

    with pytest.raises(OSError, match=f"Could not load OMX matrix file .*{path.name}") as caught:
        matrix_table.get_matrix(path.name, subset=subset)

    assert isinstance(caught.value.__cause__, OSError)
    assert str(path) in caplog.text
    assert any(record.levelname == "ERROR" for record in caplog.records)


def test_matrix_metadata_logs_unreadable_file_with_context(matrix_table, caplog):
    path = matrix_table.folder / "damaged.omx"
    path.write_bytes(b"not an OMX file")
    with pytest.raises(OSError, match="Could not read metadata from OMX matrix file .*damaged.omx"):
        matrix_table.get_available_matrices_and_metadata()
    assert str(path) in caplog.text


@pytest.mark.parametrize("extension", [".OMX", ".Omx"])
def test_saving_and_discovery_require_exact_omx_suffix(matrix_table, source_matrix, extension):
    path = matrix_table.folder / f"demand{extension}"
    path.write_bytes(b"existing")

    with pytest.raises(ValueError, match="Only OMX"):
        matrix_table.save_matrix(path.name, source_matrix)

    assert matrix_table.file_exists(path.name)
    assert path.name not in matrix_table.get_available_matrices_and_metadata()
    assert path.read_bytes() == b"existing"
