import logging
import sqlite3
from pathlib import Path

import pytest

import aequilibrae
from aequilibrae import Project
from aequilibrae.utils.model_run_utils import import_file_as_module

MIGRATIONS = Path(aequilibrae.__file__).parent / "project" / "database_specification" / "network" / "migrations"


def _migration(file_name: str):
    return import_file_as_module(MIGRATIONS / file_name, f"test_scenarios_table_{Path(file_name).stem}", force=True)


def _execute(database: Path, *statements: str) -> list:
    conn = sqlite3.connect(database)
    try:
        with conn:
            return [conn.execute(statement).fetchall() for statement in statements]
    finally:
        conn.close()


def _scenario_about(database: Path) -> tuple:
    (rows,) = _execute(database, "SELECT infoname, infovalue FROM about")
    about = dict(rows)
    return about["scenario_name"], about["scenario_description"]


@pytest.fixture
def legacy_project(tmp_path) -> Path:
    """A project as migration 011 left it, with its scenarios registered in the root's scenarios table."""
    path = tmp_path / "legacy"
    with Project.create(path) as project:
        project.create_empty_scenario("empty")
        project.clone_scenario("clone")

    # Scenario databases used to carry the about records of the database they were cloned from
    for scenario in ("empty", "clone"):
        _execute(
            path / "scenarios" / scenario / "project_database.sqlite",
            "UPDATE about SET infovalue='Example' WHERE infoname='scenario_name'",
            "UPDATE about SET infovalue='OSM Jan/2021' WHERE infoname='scenario_description'",
        )

    conn = sqlite3.connect(path / "project_database.sqlite")
    try:
        with conn:
            _migration("002_add_scenario_table.py").migrate(project_conn=conn)
            _migration("005_fix_default_period_delete_trigger.py").migrate(project_conn=conn)
            conn.execute(
                """CREATE TRIGGER aequilibrae_root_scenario_update BEFORE UPDATE ON scenarios
                WHEN new.scenario_name = 'root'
                BEGIN
                    SELECT RAISE(ABORT,'Cannot update root scenario');
                END"""
            )
            conn.executemany(
                "INSERT INTO scenarios (scenario_name, description) VALUES (?, ?)",
                [("empty", "An empty network"), ("clone", None), ("deleted", "Its folder no longer exists")],
            )
            conn.execute("UPDATE about SET infovalue='Example' WHERE infoname='scenario_name'")
            conn.execute("UPDATE about SET infovalue='OSM Jan/2021' WHERE infoname='scenario_description'")
            conn.execute("UPDATE migrations SET status='MISSING' WHERE id=12")
    finally:
        conn.close()
    return path


def test_migration_moves_scenarios_into_each_about_table(legacy_project, caplog):
    """Test that migration 012 writes each scenario's name and description to its about table and drops the table."""
    root_description = _migration("002_add_scenario_table.py").ROOT_DESCRIPTION

    with caplog.at_level(logging.WARNING), Project(legacy_project) as project:
        project.upgrade()
        scenarios = project.list_scenarios()

    assert _scenario_about(legacy_project / "project_database.sqlite") == ("root", root_description)
    assert _scenario_about(legacy_project / "scenarios" / "empty" / "project_database.sqlite") == (
        "empty",
        "An empty network",
    )
    assert _scenario_about(legacy_project / "scenarios" / "clone" / "project_database.sqlite") == ("clone", None)
    assert scenarios["scenario_name"].tolist() == ["root", "clone", "empty"]
    assert scenarios["description"].fillna("missing").tolist() == [root_description, "missing", "An empty network"]
    assert "Dropping scenario 'deleted'" in caplog.text

    leftovers, documentation, status = _execute(
        legacy_project / "project_database.sqlite",
        "SELECT type, name FROM sqlite_master WHERE tbl_name='scenarios'",
        "SELECT attribute FROM attributes_documentation WHERE name_table='scenarios'",
        "SELECT status FROM migrations WHERE id=12",
    )
    assert leftovers == []
    assert documentation == []
    assert status == [("APPLIED",)]


def test_migration_adds_missing_about_records():
    """Test that migration 012 adds the scenario records to an about table that does not have them."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE about (infoname TEXT UNIQUE NOT NULL, infovalue TEXT)")
        conn.execute("CREATE TABLE attributes_documentation (name_table TEXT, attribute TEXT, description TEXT)")
        migration_002 = _migration("002_add_scenario_table.py")
        migration_002.migrate(project_conn=conn)
        _migration("012_move_scenarios_table_to_about_table.py").migrate(project_conn=conn)

        about = dict(conn.execute("SELECT infoname, infovalue FROM about").fetchall())
        assert about == {"scenario_name": "root", "scenario_description": migration_002.ROOT_DESCRIPTION}
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='scenarios'").fetchone() is None
    finally:
        conn.close()


def test_migration_leaves_databases_without_a_scenarios_table_unchanged():
    """Test that migration 012 changes nothing in a scenario database, which has no scenarios table."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE about (infoname TEXT UNIQUE NOT NULL, infovalue TEXT)")
        conn.execute("INSERT INTO about (infoname, infovalue) VALUES ('scenario_name', 'child')")
        _migration("012_move_scenarios_table_to_about_table.py").migrate(project_conn=conn)

        assert conn.execute("SELECT infoname, infovalue FROM about").fetchall() == [("scenario_name", "child")]
    finally:
        conn.close()
