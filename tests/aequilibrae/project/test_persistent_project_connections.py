import sqlite3

import pytest

from aequilibrae import Project


def test_project_owns_persistent_connections(tmp_path):
    project = Project.new(tmp_path / "model")
    try:
        project_connection = project.scenario.connections.db_connection
        assert project_connection is project.network.links._transaction_manager
        assert project.scenario.connections.has_results_connection
        assert project.scenario.connections.has_transit_connection
        for connection in (
            project_connection,
            project.scenario.connections.results_connection,
            project.scenario.connections.transit_connection,
        ):
            assert connection.connection.execute("PRAGMA foreign_keys").fetchone() == (1,)

        with project.db_connection as connection:
            assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
        project.network.modes.insert(mode_id="x", mode_name="Test", description="", pce=1)
        assert project.network.modes.get("x").mode_name == "Test"
    finally:
        project.shutdown()


def test_project_transaction_enters_every_manager_and_rolls_back(tmp_path):
    with Project.from_path(_new_project(tmp_path)) as project:
        with pytest.raises(ValueError):
            with project.transaction():
                assert project.scenario.connections.db_connection.depth == 1
                assert project.scenario.connections.results_connection.depth == 1
                assert project.scenario.connections.transit_connection.depth == 1
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
    project = Project.new(tmp_path / "model")
    project.shutdown()
    project.shutdown()
    project.close()


def test_shutdown_closes_sqlite_connections(tmp_path):
    project = Project.new(tmp_path / "model2")
    project.shutdown()

    with pytest.raises(sqlite3.ProgrammingError):
        project.network.modes.get("c")


def test_open_does_not_create_optional_databases(tmp_path):
    path = _new_project(tmp_path)
    results = path / "results_database.sqlite"
    transit = path / "public_transport.sqlite"
    results.unlink()
    transit.unlink()

    before = {p.name for p in path.iterdir()}
    with Project.from_path(path) as project:
        assert not project.scenario.connections.has_results_connection
        assert not project.scenario.connections.has_transit_connection
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
