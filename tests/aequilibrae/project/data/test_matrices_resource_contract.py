import sqlite3

from aequilibrae.project.data.matrices import Matrices
from aequilibrae.utils.db_utils import ConnectionClosure


def test_generic_matrix_crud_is_metadata_only(tmp_path):
    closure = ConnectionClosure(sqlite3.connect(":memory:"))
    manager = closure.db_connection
    manager.connection.execute(
        "CREATE TABLE matrices (name TEXT PRIMARY KEY, file_name TEXT UNIQUE NOT NULL, "
        "cores INTEGER, procedure TEXT, procedure_id TEXT, timestamp TEXT, description TEXT)"
    )
    path = tmp_path / "demand.aem"
    path.write_bytes(b"resource")
    matrices = Matrices(manager, tmp_path)
    try:
        matrices.insert(name="demand", file_name=path.name, cores=1)
        matrices.update("demand", description="metadata only", file_name="renamed.aem")
        assert path.exists()
        matrices.delete("demand")
        assert path.exists()
    finally:
        closure.close()


def test_delete_matrix_removes_metadata_and_file(tmp_path):
    closure = ConnectionClosure(sqlite3.connect(":memory:"))
    manager = closure.db_connection
    manager.connection.execute(
        "CREATE TABLE matrices (name TEXT PRIMARY KEY, file_name TEXT UNIQUE NOT NULL, "
        "cores INTEGER, procedure TEXT, procedure_id TEXT, timestamp TEXT, description TEXT)"
    )
    path = tmp_path / "demand.aem"
    path.write_bytes(b"resource")
    matrices = Matrices(manager, tmp_path)
    try:
        matrices.insert(name="demand", file_name=path.name, cores=1)
        matrices.delete_matrix("demand")
        assert "demand" not in matrices
        assert not path.exists()
    finally:
        closure.close()
