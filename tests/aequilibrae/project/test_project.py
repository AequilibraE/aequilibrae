import sqlite3

import pytest

import aequilibrae.project.scenario as scenario_module
from aequilibrae.project.project import Project
from aequilibrae.utils.db_utils import read_and_close


def test_opening_wrong_folder(tmp_path):
    not_a_project = str(tmp_path)
    with pytest.raises(FileNotFoundError):
        Project.from_path(not_a_project)


@pytest.mark.parametrize(
    "table, exp_column",
    [
        ("links", "distance"),
        ("nodes", "is_centroid"),
    ],
)
def test_table_creation(table: str, exp_column: str, empty_project):
    with read_and_close(empty_project.path_to_file) as conn:
        fields = {x[1] for x in conn.execute(f"PRAGMA table_info({table});").fetchall()}

    assert exp_column in fields, f"Table {table.upper()} was not created correctly"


def test_close_makes_project_unusable(empty_project, tmp_path):
    """After shutdown the project's SQLite connections are closed."""
    # Verify it works before close
    with empty_project.db_connection as conn:
        count = conn.execute("SELECT count(*) FROM links").fetchone()[0]
    assert count == 0

    empty_project.close()

    # Second close is a no-op (no error)
    empty_project.close()


def test_context_manager_closes_project(tmp_path):
    """Project contexts close their project on exit."""
    with Project.new(tmp_path / "ctx_proj") as project:
        with project.db_connection as conn:
            links = conn.execute("SELECT count(*) FROM links").fetchone()[0]
    assert links == 0

    with pytest.raises(sqlite3.ProgrammingError):
        project.network.modes.get("c")


def test_failed_scenario_construction_closes_connections(tmp_path, monkeypatch):
    """A gateway-construction error must not leak Scenario.create's closure."""
    path = tmp_path / "project"
    Project.new(path).shutdown()
    captured_closure = None
    original_open = scenario_module.ConnectionClosure.open

    def capture_closure(openers):
        nonlocal captured_closure
        captured_closure = original_open(openers)
        return captured_closure

    def fail_network(*args, **kwargs):
        raise RuntimeError("gateway construction failed")

    monkeypatch.setattr(scenario_module.ConnectionClosure, "open", capture_closure)
    monkeypatch.setattr(scenario_module, "Network", fail_network)

    with pytest.raises(RuntimeError, match="gateway construction failed"):
        scenario_module.Scenario.create("root", path)

    with pytest.raises(sqlite3.ProgrammingError):
        captured_closure["project"].execute("SELECT 1")
