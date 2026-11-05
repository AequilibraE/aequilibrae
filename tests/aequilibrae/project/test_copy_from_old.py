"""Destination-driven project database imports, without opening the old project."""

import importlib
import logging
import shutil
import sqlite3
from contextlib import closing
from graphlib import CycleError

import pytest

from aequilibrae.project import Project
from aequilibrae.project.tools import copy_from_old
from aequilibrae.project.tools.cli import cli

backup_module = importlib.import_module("aequilibrae.project.tools.backup")


def make_database(folder, sql):
    folder.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(folder / "project_database.sqlite")) as conn:
        conn.executescript(sql)
    return folder


def query(folder, sql):
    with closing(sqlite3.connect(folder / "project_database.sqlite")) as conn:
        return conn.execute(sql).fetchall()


@pytest.fixture
def source(tmp_path):
    return make_database(
        tmp_path / "old #project's",
        'CREATE TABLE payload (id INTEGER PRIMARY KEY, "odd""column" BLOB);'
        'INSERT INTO payload VALUES (1, X\'00FF\'), (2, X\'12\');',
    )


@pytest.fixture
def destination(tmp_path):
    return make_database(
        tmp_path / "new #project's", 'CREATE TABLE payload (id INTEGER PRIMARY KEY, "odd""column" BLOB);'
    )


def test_import_uses_existing_schema_and_leaves_source_unchanged(source, destination):
    original = (source / "project_database.sqlite").read_bytes()
    schema = query(destination, "SELECT * FROM sqlite_master")
    assert copy_from_old(source, destination) == destination.resolve()
    assert query(destination, "SELECT * FROM payload") == [(1, b"\x00\xff"), (2, b"\x12")]
    assert query(destination, "SELECT * FROM sqlite_master") == schema
    assert (source / "project_database.sqlite").read_bytes() == original


def test_logs_custom_tables_columns_and_destination_only_tables(source, destination, caplog):
    with closing(sqlite3.connect(source / "project_database.sqlite")) as conn:
        conn.executescript(
            "CREATE TABLE custom (value INTEGER); INSERT INTO custom VALUES (42);"
            "CREATE TABLE source_only (value TEXT);"
            "ALTER TABLE payload ADD COLUMN obsolete TEXT;"
        )
    with closing(sqlite3.connect(destination / "project_database.sqlite")) as conn:
        conn.executescript(
            "CREATE TABLE custom (value INTEGER, new_column TEXT DEFAULT 'new');"
            "CREATE TABLE destination_only (value TEXT);"
            "INSERT INTO destination_only VALUES ('Keep me');"
        )
    caplog.set_level(logging.INFO, logger=backup_module.__name__)
    copy_from_old(source, destination)
    assert query(destination, "SELECT * FROM custom") == [(42, "new")]
    assert query(destination, "SELECT * FROM destination_only") == [("Keep me",)]
    assert not query(destination, "SELECT 1 FROM sqlite_master WHERE name='source_only'")
    assert "Copying table custom" in caplog.text
    assert "matching columns: ['value']" in caplog.text
    assert "Using destination default for custom.new_column: 'new'" in caplog.text
    assert "Omitting column payload.obsolete" in caplog.text
    assert "Omitting table source_only" in caplog.text
    assert "Leaving destination table destination_only untouched" in caplog.text


@pytest.mark.parametrize("default, expected", [("'standard'", "standard"), ("(2 + 3)", 5), ("NULL", None)])
def test_missing_columns_use_explicit_defaults(source, destination, default, expected):
    with closing(sqlite3.connect(destination / "project_database.sqlite")) as conn:
        conn.execute(f"ALTER TABLE payload ADD COLUMN new_column DEFAULT {default}")
    copy_from_old(source, destination)
    assert query(destination, "SELECT new_column FROM payload") == [(expected,), (expected,)]


def test_all_destination_columns_can_use_defaults(tmp_path):
    source = make_database(tmp_path / "old", "CREATE TABLE custom (old_field); INSERT INTO custom VALUES (1), (2);")
    destination = make_database(tmp_path / "new", "CREATE TABLE custom (new_field TEXT DEFAULT 'default');")
    copy_from_old(source, destination)
    assert query(destination, "SELECT * FROM custom") == [("default",), ("default",)]


@pytest.mark.parametrize("empty_source", [False, True])
@pytest.mark.parametrize("declaration", ["TEXT", "TEXT NOT NULL", "INTEGER PRIMARY KEY"])
def test_missing_column_without_default_aborts(source, destination, empty_source, declaration, caplog):
    with closing(sqlite3.connect(destination / "project_database.sqlite")) as conn:
        conn.executescript(f"CREATE TABLE custom (new_field {declaration});")
    with closing(sqlite3.connect(source / "project_database.sqlite")) as conn:
        conn.executescript(
            "CREATE TABLE custom (old_field);" + ("" if empty_source else "INSERT INTO custom VALUES (1);")
        )
    with pytest.raises(ValueError, match="missing columns without defaults.*new_field"):
        copy_from_old(source, destination)
    assert query(destination, "SELECT * FROM payload") == []
    assert "source column new_field is missing and has no destination default" in caplog.text


@pytest.mark.parametrize("kind", ["VIRTUAL", "STORED"])
def test_generated_columns_are_recalculated(tmp_path, kind, caplog):
    source = make_database(tmp_path / "old", "CREATE TABLE custom (value INTEGER); INSERT INTO custom VALUES (5);")
    destination = make_database(
        tmp_path / "new", f"CREATE TABLE custom (value INTEGER, doubled GENERATED ALWAYS AS (value * 2) {kind});"
    )
    caplog.set_level(logging.INFO, logger=backup_module.__name__)
    copy_from_old(source, destination)
    assert query(destination, "SELECT * FROM custom") == [(5, 10)]
    assert "Not copying column custom.doubled: generated or hidden" in caplog.text


def test_constraint_failure_rolls_back_prior_tables(tmp_path):
    source = make_database(
        tmp_path / "old", "CREATE TABLE a (id INTEGER PRIMARY KEY); INSERT INTO a VALUES (2);"
        "CREATE TABLE z (value INTEGER); INSERT INTO z VALUES (-1);"
    )
    destination = make_database(
        tmp_path / "new", "CREATE TABLE a (id INTEGER PRIMARY KEY); INSERT INTO a VALUES (1);"
        "CREATE TABLE z (value INTEGER CHECK(value > 0));"
    )
    with pytest.raises(sqlite3.IntegrityError, match="CHECK") as error:
        copy_from_old(source, destination)
    assert "While copying table 'z'" in str(error.value.__notes__)
    assert query(destination, "SELECT * FROM a") == [(1,)]
    assert query(destination, "SELECT * FROM z") == []


def test_custom_foreign_keys_determine_copy_order(tmp_path):
    sql = "CREATE TABLE a_child (id INTEGER REFERENCES z_parent(id)); CREATE TABLE z_parent (id INTEGER PRIMARY KEY);"
    source = make_database(tmp_path / "old", sql + "INSERT INTO z_parent VALUES (1); INSERT INTO a_child VALUES (1);")
    destination = make_database(tmp_path / "new", sql)
    copy_from_old(source, destination)
    assert query(destination, "SELECT * FROM a_child") == [(1,)]
    assert query(destination, "PRAGMA foreign_key_check") == []


def test_foreign_keys_remain_enabled(tmp_path):
    sql = "CREATE TABLE a_child (id INTEGER REFERENCES z_parent(id)); CREATE TABLE z_parent (id INTEGER PRIMARY KEY);"
    source = make_database(tmp_path / "old", sql + "INSERT INTO a_child VALUES (99);")
    destination = make_database(tmp_path / "new", sql)
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        copy_from_old(source, destination)
    assert query(destination, "SELECT * FROM a_child") == []


def test_cycles_abort_before_inserting(tmp_path):
    sql = "CREATE TABLE a (id PRIMARY KEY, b REFERENCES b(id)); CREATE TABLE b (id PRIMARY KEY, a REFERENCES a(id));"
    source = make_database(tmp_path / "old", sql)
    destination = make_database(tmp_path / "new", sql)
    with pytest.raises(CycleError):
        copy_from_old(source, destination)


def test_infrastructure_and_migrations_are_not_copied(tmp_path, caplog):
    sql = (
        "CREATE TABLE payload (id INTEGER PRIMARY KEY AUTOINCREMENT);"
        "CREATE TABLE migrations (status TEXT);"
        "CREATE TABLE spatialite_history (value TEXT);"
    )
    source = make_database(
        tmp_path / "old", sql + "INSERT INTO migrations VALUES ('old'); INSERT INTO spatialite_history VALUES ('old');"
        "INSERT INTO payload VALUES (1000); DELETE FROM payload;"
    )
    destination = make_database(
        tmp_path / "new", sql + "INSERT INTO migrations VALUES ('new'); INSERT INTO spatialite_history VALUES ('new');"
    )
    caplog.set_level(logging.INFO, logger=backup_module.__name__)
    copy_from_old(source, destination)
    assert query(destination, "SELECT * FROM migrations") == [("new",)]
    assert query(destination, "SELECT * FROM spatialite_history") == [("new",)]
    assert query(destination, "SELECT seq FROM sqlite_sequence WHERE name='payload'") in ([], [(0,)])
    for table in ("sqlite_sequence", "migrations", "spatialite_history"):
        assert f"Not copying table {table}: SQLite/SpatiaLite infrastructure or migration records" in caplog.text


def test_only_root_project_database_is_imported(source, destination):
    for folder in (source, destination):
        for filename in ("public_transport.sqlite", "results_database.sqlite", "parameters.yml", "aequilibrae.log"):
            (folder / filename).write_text(str(folder))
        for name in ("matrices", "run", "scenarios/future"):
            path = folder / name
            path.mkdir(parents=True)
            (path / "project_database.sqlite").write_text(str(folder))
    untouched = {
        path: path.read_bytes() for path in destination.rglob("*")
        if path.is_file() and path != destination / "project_database.sqlite"
    }
    copy_from_old(source, destination)
    assert all(path.read_bytes() == contents for path, contents in untouched.items())
    assert {path for path in destination.rglob("*") if path.is_file()} == set(untouched) | {
        destination / "project_database.sqlite"
    }


def test_import_includes_only_committed_wal_data(source, destination):
    with closing(sqlite3.connect(source / "project_database.sqlite")) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute('INSERT INTO payload VALUES (3, X\'AB\')')
        conn.commit()
        conn.execute('INSERT INTO payload VALUES (4, X\'CD\')')
        copy_from_old(source, destination)
        assert query(destination, "SELECT id FROM payload ORDER BY id") == [(1,), (2,), (3,)]


@pytest.mark.parametrize("missing", ["source", "destination"])
def test_missing_database_is_not_created(source, destination, missing):
    folder = source if missing == "source" else destination
    (folder / "project_database.sqlite").unlink()
    with pytest.raises(FileNotFoundError):
        copy_from_old(source, destination)
    assert not (folder / "project_database.sqlite").exists()


def test_missing_destination_directory_is_not_created(source, tmp_path):
    destination = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        copy_from_old(source, destination)
    assert not destination.exists()


def test_same_database_is_rejected(source, tmp_path):
    with pytest.raises(ValueError, match="must be different"):
        copy_from_old(source, source)
    alias = tmp_path / "alias"
    alias.mkdir()
    (alias / "project_database.sqlite").hardlink_to(source / "project_database.sqlite")
    with pytest.raises(ValueError, match="must be different"):
        copy_from_old(source, alias)


@pytest.mark.parametrize("linked", ["source", "destination"])
def test_symlink_database_is_rejected(source, destination, tmp_path, linked):
    folder = source if linked == "source" else destination
    database = folder / "project_database.sqlite"
    actual = tmp_path / "actual.sqlite"
    database.rename(actual)
    database.symlink_to(actual)
    with pytest.raises(ValueError, match="Symbolic links"):
        copy_from_old(source, destination)


@pytest.fixture
def spatial_source(tmp_path):
    from aequilibrae.project.project_creation import create_base_tables
    from aequilibrae.reference_files import spatialite_database
    from aequilibrae.utils.spatialite_utils import connect_spatialite

    source = tmp_path / "spatial_old"
    source.mkdir()
    shutil.copyfile(spatialite_database, source / "project_database.sqlite")
    with closing(connect_spatialite(source / "project_database.sqlite")) as conn, conn:
        create_base_tables(conn, "network")
        conn.execute("INSERT INTO modes (mode_name, mode_id) VALUES ('Test mode', 'q')")
        conn.execute("INSERT INTO link_types (link_type, link_type_id) VALUES ('arterial', 'a')")
        conn.execute(
            "INSERT INTO nodes (node_id, geometry) VALUES (1, MakePoint(0, 0, 4326)), (2, MakePoint(1, 1, 4326))"
        )
        conn.execute(
            "INSERT INTO links (link_id, a_node, b_node, modes, link_type, geometry) "
            "VALUES (1, 1, 2, 'q', 'arterial', GeomFromText('LINESTRING(0 0, 1 1)', 4326))"
        )
        conn.execute(
            "INSERT INTO turn_restrictions (restriction_id, from_node, via_node, to_node, modes) "
            "VALUES (11, 1, 2, 1, 'q')"
        )
        conn.execute("UPDATE migrations SET status='APPLIED'")
        conn.execute("UPDATE scenarios SET description='Imported root' WHERE scenario_name='root'")
    return source


@pytest.fixture
def spatial_destination(tmp_path):
    destination = tmp_path / "spatial_new"
    project = Project()
    project.new(destination)
    project.close()
    return destination


def test_spatial_import_replaces_seeds_and_preserves_triggers(spatial_source, spatial_destination):
    from aequilibrae.utils.spatialite_utils import connect_spatialite

    original = (spatial_source / "project_database.sqlite").read_bytes()
    schema = query(spatial_destination, "SELECT * FROM sqlite_master")
    copy_from_old(spatial_source, spatial_destination)
    assert (spatial_source / "project_database.sqlite").read_bytes() == original
    assert query(spatial_destination, "SELECT * FROM sqlite_master") == schema
    assert query(spatial_destination, "SELECT description FROM scenarios WHERE scenario_name='root'") == [
        ("Imported root",)
    ]
    assert query(spatial_destination, "SELECT DISTINCT status FROM migrations") == [("SKIPPED",)]
    assert query(spatial_destination, "SELECT node_id, modes FROM nodes ORDER BY node_id") == [(1, "q"), (2, "q")]
    assert query(spatial_destination, "SELECT geometry FROM nodes ORDER BY node_id") == query(
        spatial_source, "SELECT geometry FROM nodes ORDER BY node_id"
    )
    with closing(connect_spatialite(spatial_destination / "project_database.sqlite")) as conn:
        assert conn.execute("SELECT CheckSpatialIndex('nodes', 'geometry')").fetchone() == (1,)
        assert conn.execute("SELECT CheckSpatialIndex('links', 'geometry')").fetchone() == (1,)
        assert conn.execute("SELECT distance = GeodesicLength(geometry) FROM links").fetchone() == (1,)
        assert conn.execute("SELECT geometry IS NOT NULL FROM turn_restrictions").fetchone() == (1,)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_initial_rows_without_source_conflicts_remain(spatial_source, spatial_destination):
    with closing(sqlite3.connect(spatial_source / "project_database.sqlite")) as conn, conn:
        conn.execute("DELETE FROM modes WHERE mode_id IN ('b', 'w')")
        conn.execute("UPDATE modes SET pce=2 WHERE mode_id='c'")
    copy_from_old(spatial_source, spatial_destination)
    assert query(spatial_destination, "SELECT mode_id FROM modes ORDER BY mode_id") == [
        ("b",), ("c",), ("q",), ("t",), ("w",)
    ]
    assert query(spatial_destination, "SELECT pce FROM modes WHERE mode_id='c'") == [(2,)]


def test_import_accepts_trigger_generated_nodes(spatial_source, spatial_destination):
    from aequilibrae.utils.spatialite_utils import connect_spatialite

    with closing(connect_spatialite(spatial_source / "project_database.sqlite")) as conn, conn:
        conn.execute("UPDATE links SET b_node=999, geometry=GeomFromText('LINESTRING(0 0, 2 2)', 4326)")
    copy_from_old(spatial_source, spatial_destination)
    assert query(spatial_destination, "SELECT a_node, b_node FROM links") == [(1, 3)]
    assert query(spatial_destination, "SELECT COUNT(*) FROM nodes") == [(3,)]
    with closing(connect_spatialite(spatial_destination / "project_database.sqlite")) as conn:
        assert conn.execute("SELECT AsText(geometry) FROM nodes WHERE node_id=3").fetchone() == ("POINT(2 2)",)
        assert conn.execute("SELECT CheckSpatialIndex('nodes', 'geometry')").fetchone() == (1,)


@pytest.mark.parametrize("table", ["links", "turn_restrictions"])
def test_import_rejects_unknown_modes(spatial_source, spatial_destination, table):
    with closing(sqlite3.connect(spatial_source / "project_database.sqlite")) as conn, conn:
        conn.execute(f"UPDATE {table} SET modes='x'")
    with pytest.raises(sqlite3.IntegrityError, match="[Mm]ode codes"):
        copy_from_old(spatial_source, spatial_destination)
    assert query(spatial_destination, "SELECT * FROM nodes") == []
    assert query(spatial_destination, "SELECT description FROM scenarios WHERE scenario_name='root'") != [
        ("Imported root",)
    ]


def test_import_rejects_duplicate_node_geometry(spatial_source, spatial_destination):
    from aequilibrae.utils.spatialite_utils import connect_spatialite

    with closing(connect_spatialite(spatial_source / "project_database.sqlite")) as conn, conn:
        conn.execute("INSERT INTO nodes (node_id, geometry) VALUES (3, MakePoint(0, 0, 4326))")
    with pytest.raises(sqlite3.IntegrityError, match="Cannot create on-top of other node"):
        copy_from_old(spatial_source, spatial_destination)


def test_table_order_combines_foreign_keys_and_trigger_dependencies():
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript(
            "CREATE TABLE modes (id PRIMARY KEY);"
            "CREATE TABLE link_types (id PRIMARY KEY);"
            "CREATE TABLE nodes (id PRIMARY KEY);"
            "CREATE TABLE links (kind REFERENCES link_types(id));"
            "CREATE TABLE turn_restrictions (node REFERENCES nodes(id));"
        )
        order = backup_module._table_copy_order(conn, ["turn_restrictions", "links", "nodes", "link_types", "modes"])
        for parent in ("nodes", "link_types", "modes"):
            assert order.index(parent) < order.index("links")
        assert order.index("links") < order.index("turn_restrictions")


def test_cli_create_and_import(spatial_source, tmp_path, capsys):
    destination = tmp_path / "created"
    cli(["-p", str(destination), "create"])
    assert (destination / "project_database.sqlite").is_file()
    cli(["-p", str(destination), "copy-from-old", str(spatial_source)])
    assert query(destination, "SELECT node_id FROM nodes ORDER BY node_id") == [(1,), (2,)]
    assert str(destination) in capsys.readouterr().out


def test_cli_create_refuses_existing_directory(tmp_path):
    with pytest.raises(FileExistsError):
        cli(["-p", str(tmp_path), "create"])


def test_cli_create_rejects_extra_arguments_before_creating(tmp_path):
    destination = tmp_path / "not_created"
    with pytest.raises(SystemExit):
        cli(["-p", str(destination), "create", "--unknown"])
    assert not destination.exists()


def test_cli_import_dispatch_without_opening_project(tmp_path, monkeypatch, capsys):
    calls = []

    def operation(source, destination):
        calls.append((source, destination))
        return destination

    monkeypatch.setattr(backup_module, "copy_from_old", operation)
    source, target = str(tmp_path / "source"), str(tmp_path / "target")
    cli(["-p", target, "copy-from-old", source])
    assert calls == [(source, target)]
    assert target in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli(["-p", target, "copy-from-old", source, "--unknown"])
    assert len(calls) == 1
