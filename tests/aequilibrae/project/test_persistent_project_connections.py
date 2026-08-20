import sqlite3

import pytest

from aequilibrae import Project


def test_project_owns_named_persistent_connections(tmp_path):
    project = Project.new(tmp_path / "model")
    try:
        manager = project.scenario.connections["project"]
        assert manager is project.network.links._transactions
        assert set(project.scenario.connections) == {"project", "results", "transit"}
        assert all(
            project.scenario.connections[name].execute("PRAGMA foreign_keys").fetchone() == (1,)
            for name in project.scenario.connections
        )

        with project.db_connection as connection:
            assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
        project.network.modes.insert(mode_id="x", mode_name="Test", description="", pce=1)
        assert project.network.modes.get("x").mode_name == "Test"
    finally:
        project.shutdown()


def test_project_transaction_enters_every_manager_and_rolls_back(tmp_path):
    with Project.from_path(_new_project(tmp_path)) as project:
        with pytest.raises(ValueError):
            with project.transaction() as connections:
                assert set(connections) == {"project", "results", "transit"}
                assert all(project.scenario.connections[name].depth == 1 for name in project.scenario.connections)
                project.network.modes.insert(mode_id="x", mode_name="Test", description="", pce=1)
                assert project.network.modes.get("x").mode_name == "Test"
                raise ValueError
        assert "x" not in project.network.modes


def test_static_upgrade_owns_closed_connections(tmp_path):
    path = _new_project(tmp_path)
    Project.upgrade(path)
    with Project.from_path(path) as project:
        with project.db_connection as connection:
            assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)


def test_shutdown_is_idempotent(tmp_path):
    """Calling shutdown / close more than once is a no-op."""
    project = Project.new(tmp_path / "model")
    project.shutdown()
    project.shutdown()   # no-op
    project.close()      # alias — also no-op


def test_shutdown_closes_sqlite_connections(tmp_path):
    """After shutdown, SQLite operations raise ProgrammingError (closed database)."""
    project = Project.new(tmp_path / "model2")
    project.shutdown()

    with pytest.raises(sqlite3.ProgrammingError):
        # The underlying connection is closed; any SQL operation must fail.
        project.network.modes.get("c")


def test_open_does_not_create_optional_databases(tmp_path):
    path = _new_project(tmp_path)
    results = path / "results_database.sqlite"
    transit = path / "public_transport.sqlite"
    results.unlink()
    transit.unlink()

    before = {p.name for p in path.iterdir()}
    with Project.from_path(path) as project:
        assert set(project.scenario.connections) == {"project"}
        with pytest.raises(RuntimeError, match="no results database"):
            _ = project.results
        with pytest.raises(RuntimeError, match="no transit database"):
            _ = project.transit

    assert {p.name for p in path.iterdir()} == before
    assert not results.exists()
    assert not transit.exists()


def _new_project(tmp_path):
    path = tmp_path / "model"
    Project.new(path).shutdown()
    return path
