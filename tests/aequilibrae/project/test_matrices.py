import string
from math import floor
from os.path import join
from random import choice, randint
from shutil import copyfile

import pytest

from aequilibrae.matrix import AequilibraeMatrix, MatrixStore
from aequilibrae.project.data.matrices import Matrices
from aequilibrae.utils.db_utils import NestedTransactionManager


def randomword(length):
    allowed_characters = string.ascii_letters + '_123456789@!()[]{};:"    -'
    val = "".join(choice(allowed_characters) for _ in range(length))
    if val[0] == "_" or val[-1] == "_":
        return randomword(length)
    return val


def mat_count(sioux_falls_example, should_have: int, error_message: str):
    with sioux_falls_example.db_connection as conn:
        cnt = conn.execute("Select count(*) from Matrices;").fetchone()[0]
    assert cnt == should_have, error_message


def test_set_record(sioux_falls_example):
    matrices = sioux_falls_example.matrices
    rec = matrices.get("demand_omx")
    with pytest.raises(ValueError, match="matrix of name"):
        matrices.register_matrix("demand_omx", rec.file_name)
    with pytest.raises(ValueError, match="file name"):
        matrices.register_matrix("another_matrix", rec.file_name)
    assert rec.file_name == "demand.omx"
    assert rec.cores == 1, "Setting a file that exists did not correct the number of cores"


def test_clear_database(sioux_falls_test):
    matrices = sioux_falls_test.matrices
    mat_count(sioux_falls_test, 3, "The test data started wrong")
    matrices.clear_database()
    mat_count(sioux_falls_test, 2, "Did not clear the database appropriately")


def test_update_database(sioux_falls_test):
    matrices = sioux_falls_test.matrices
    mat_count(sioux_falls_test, 3, "The test data started wrong")
    matrices.update_database()
    mat_count(sioux_falls_test, 4, "Did not add to the database appropriately")
    rec = matrices.get("omx")
    existing = join(matrices.folder, rec.file_name)
    new_name = "test_name.omx"
    new_name1 = "test_name1.omx"
    copyfile(existing, join(matrices.folder, new_name))
    matrices.register_matrix("test_name1", new_name)
    copyfile(existing, join(matrices.folder, new_name1))
    matrices.update_database()


def test_get_matrix(sioux_falls_example):
    matrices = sioux_falls_example.matrices
    with pytest.raises(ValueError, match="matrices has no record with name='omxq'"):
        _ = matrices.get_matrix("omxq")
    mat = matrices.get_matrix("demand_omx", index_name="taz")
    assert floor(sum(values.sum() for values in mat.values())) == 360600, "Matrix loaded incorrectly"


def test_get_record(sioux_falls_example):
    matrices = sioux_falls_example.matrices
    rec = matrices.get("demand_mc")
    assert rec.cores == 3, "record populated wrong. Number of cores"
    assert rec.description is None, "record populated wrong. Description"


def test_record_update_cores(sioux_falls_test):
    matrices = sioux_falls_test.matrices
    matrices.update("omx", cores=2)
    assert matrices.get("omx").cores == 2, "Cores update did not work"


def test_save_record(sioux_falls_example):
    matrices = sioux_falls_example.matrices
    text = randomword(randint(30, 100))
    matrices.update("demand_mc", description=text)
    with sioux_falls_example.db_connection as conn:
        cnt = conn.execute('select description from matrices where name="demand_mc";').fetchone()[0]
    assert text == cnt, "Saving matrix record description failed"


def test_delete(sioux_falls_example):
    matrices = sioux_falls_example.matrices
    matrices.delete_matrix("demand_omx")
    with sioux_falls_example.db_connection as conn:
        cnt = conn.execute('select count(*) from matrices where name="demand_omx";').fetchone()[0]
    assert cnt == 0, "Deleting matrix record failed"
    with pytest.raises(ValueError, match="matrices has no record with name='demand_omx'"):
        matrices.get("demand_omx")


def test_list(sioux_falls_example):
    matrices = sioux_falls_example.matrices
    df = matrices.list()
    with sioux_falls_example.db_connection as conn:
        cnt = conn.execute("select count(*) from Matrices").fetchone()[0]
    assert df.shape[0] == cnt, "Returned the wrong number of matrices in the database"
    assert df[df.status == "file missing"].shape[0] == 0, "Wrong # of records for missing matrix files"


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
    table = Matrices(manager, tmp_path)
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


def test_create_exports_and_registers_a_matrix(matrix_table, source_matrix):
    record = matrix_table.create("future_demand", "future_demand.omx", source_matrix)

    assert record == matrix_table.get("future_demand")
    assert record.cores == 1
    assert (matrix_table.folder / record.file_name).is_file()
    assert matrix_table.get_matrix(record.name) == source_matrix


def test_create_empty_matrix(matrix_table, source_matrix):
    source_matrix.clear()
    record = matrix_table.create("empty", "empty.omx", source_matrix)
    assert record.cores == 0
    assert matrix_table.get_matrix(record.name) == source_matrix


def test_get_matrix_uses_requested_index(matrix_table):
    matrix = AequilibraeMatrix(index=[30, 10], index_name="zones")
    matrix["demand"] = [[0, 3], [2, 0]]
    matrix_table.create("demand", "demand.omx", matrix)
    assert matrix_table.get_matrix("demand", index_name="zones") == matrix
    with pytest.raises(ValueError, match="main_index"):
        matrix_table.get_matrix("demand")


@pytest.mark.parametrize("extension", [".aem", ".csv"])
def test_create_rejects_other_formats(matrix_table, source_matrix, extension):
    with pytest.raises(ValueError, match="Only OMX"):
        matrix_table.create("demand", f"demand{extension}", source_matrix)
    assert "demand" not in matrix_table


def test_register_rejects_other_formats(matrix_table):
    path = matrix_table.folder / "demand.aem"
    path.write_bytes(b"old matrix")
    with pytest.raises(ValueError, match="Only OMX"):
        matrix_table.register_matrix("demand", path.name)


def test_register_counts_matrices_without_an_index(matrix_table, no_index_omx):
    record = matrix_table.register_matrix("no_index", no_index_omx.name)
    assert record.cores > 0


def test_update_database_only_registers_omx_files(matrix_table):
    matrix = AequilibraeMatrix(index=[1])
    matrix["demand"] = [[2]]
    matrix.save_as_omx(matrix_table.folder / "demand.OMX")
    (matrix_table.folder / "old.aem").write_bytes(b"old matrix")
    (matrix_table.folder / "notes.txt").write_text("notes")
    (matrix_table.folder / "directory.omx").mkdir()

    matrix_table.update_database()
    assert [record.file_name for record in matrix_table] == ["demand.OMX"]


def test_create_does_not_overwrite_existing_file(matrix_table, source_matrix):
    path = matrix_table.folder / "demand.omx"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        matrix_table.create("demand", path.name, source_matrix)
    assert path.read_bytes() == b"existing"
    assert "demand" not in matrix_table


def test_create_removes_partial_export(matrix_table, source_matrix, monkeypatch):
    def fail(path):
        path.write_bytes(b"partial")
        raise OSError("export failed")

    monkeypatch.setattr(source_matrix, "save_as_omx", fail)
    with pytest.raises(OSError, match="export failed"):
        matrix_table.create("demand", "demand.omx", source_matrix)
    assert not (matrix_table.folder / "demand.omx").exists()
    assert not list(matrix_table.folder.glob(".demand.*.tmp.omx"))
    assert "demand" not in matrix_table


def test_create_removes_export_when_registration_fails(matrix_table, source_matrix, monkeypatch):
    def fail(**kwargs):
        raise ValueError("insert failed")

    monkeypatch.setattr(matrix_table, "insert", fail)
    with pytest.raises(ValueError, match="insert failed"):
        matrix_table.create("demand", "demand.omx", source_matrix)
    assert not (matrix_table.folder / "demand.omx").exists()
    assert "demand" not in matrix_table


def test_generic_and_resource_deletes_are_distinct(matrix_table, omx_example):
    target = matrix_table.folder / "existing.omx"
    copyfile(omx_example, target)
    record = matrix_table.register_matrix("existing", target.name)

    matrix_table.delete(record.name)
    assert record.name not in matrix_table
    assert target.is_file()

    matrix_table.register_matrix(record.name, target.name)
    matrix_table.delete_matrix(record.name)
    assert record.name not in matrix_table
    assert not target.exists()


def test_sync_reconciles_matrix_files_and_metadata(matrix_table, omx_example):
    missing = matrix_table.folder / "missing.omx"
    copyfile(omx_example, missing)
    matrix_table.register_matrix("missing", missing.name)
    missing.unlink()

    orphan = matrix_table.folder / "orphan.omx"
    copyfile(omx_example, orphan)
    matrix_table.sync()

    assert "missing" not in matrix_table
    assert any(record.file_name == orphan.name for record in matrix_table)
    assert matrix_table.list().status.eq("").all()
