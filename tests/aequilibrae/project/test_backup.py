"""Backup coverage, including SQLite-only sources that cannot be opened as Projects."""

import importlib
import shutil
import sqlite3
from contextlib import closing
from graphlib import CycleError

import pytest

from aequilibrae.project.tools import backup, copy_to_new
from aequilibrae.project.tools.cli import cli

backup_module = importlib.import_module("aequilibrae.project.tools.backup")
DATABASES = ("project_database.sqlite", "public_transport.sqlite", "results_database.sqlite")


def make_database(path, value=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE payload (id INTEGER PRIMARY KEY, value BLOB)")
        conn.execute("INSERT INTO payload VALUES (?, ?)", (value, b"\x00\xff"))
        conn.execute("CREATE INDEX payload_value ON payload(value)")
        conn.execute("CREATE VIEW payload_view AS SELECT * FROM payload")


def query(path, sql):
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(sql).fetchall()


@pytest.fixture
def plain_project(tmp_path):
    source = tmp_path / "source"
    for folder in (source, source / "scenarios" / "future"):
        for name in DATABASES:
            make_database(folder / name)
        (folder / "parameters.yml").write_text("run: {}\n")
        (folder / "aequilibrae.log").write_text("A project log\n")
        (folder / "matrices").mkdir()
        (folder / "matrices" / "demand.omx").write_bytes(b"matrix contents")
        (folder / "run" / "nested").mkdir(parents=True)
        (folder / "run" / "nested" / "helper.py").write_text("VALUE = 1\n")
        (folder / "run" / "__pycache__").mkdir()
        (folder / "run" / "__pycache__" / "helper.pyc").write_bytes(b"cache")
        (folder / "unrelated.txt").write_text("Not an AequilibraE file")
        make_database(folder / "unrelated.sqlite")
    return source


def test_backup_preserves_databases_and_only_project_files(plain_project, tmp_path):
    target = backup(plain_project, tmp_path / "backup")
    for relative in (".", "scenarios/future"):
        source = plain_project / relative
        copied = target / relative
        for name in DATABASES:
            assert query(source / name, "SELECT * FROM sqlite_master") == query(
                copied / name, "SELECT * FROM sqlite_master"
            )
            assert query(copied / name, "SELECT * FROM payload") == [(1, b"\x00\xff")]
        assert (copied / "parameters.yml").read_bytes() == (source / "parameters.yml").read_bytes()
        assert (copied / "aequilibrae.log").read_bytes() == (source / "aequilibrae.log").read_bytes()
        assert (copied / "matrices" / "demand.omx").read_bytes() == b"matrix contents"
        assert (copied / "run" / "nested" / "helper.py").is_file()
        assert not (copied / "run" / "__pycache__").exists()
        assert not (copied / "unrelated.txt").exists()
        assert not (copied / "unrelated.sqlite").exists()


def test_cli_backup(plain_project, tmp_path, capsys):
    target = tmp_path / "cli-backup"
    cli(["-p", str(plain_project), "backup", str(target)])
    assert query(target / "project_database.sqlite", "SELECT id FROM payload") == [(1,)]
    assert str(target) in capsys.readouterr().out


def test_backup_includes_committed_wal(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    database = source / "project_database.sqlite"
    with closing(sqlite3.connect(database)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("CREATE TABLE data (value INTEGER)")
        conn.execute("INSERT INTO data VALUES (42)")
        conn.commit()
        conn.execute("INSERT INTO data VALUES (99)")
        target = backup(source, tmp_path / "backup")
        assert query(target / database.name, "SELECT * FROM data") == [(42,)]
        assert not (target / "project_database.sqlite-wal").exists()


@pytest.mark.parametrize("operation", [backup, copy_to_new])
def test_path_safety(operation, plain_project, tmp_path):
    with pytest.raises(FileExistsError):
        operation(plain_project, plain_project)
    with pytest.raises(ValueError, match="outside"):
        operation(plain_project, plain_project / "nested")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        operation(plain_project, existing)
    with pytest.raises(FileNotFoundError):
        operation(tmp_path / "missing", tmp_path / "backup")
    assert not (tmp_path / "backup").exists()


def test_failure_leaves_output_and_does_not_copy_files(plain_project, tmp_path, monkeypatch):
    original = backup_module._backup_database

    def fail_transit(source, destination):
        if source.name == "public_transport.sqlite":
            raise sqlite3.DatabaseError("deliberate failure")
        original(source, destination)

    monkeypatch.setattr(backup_module, "_backup_database", fail_transit)
    target = tmp_path / "partial"
    with pytest.raises(sqlite3.DatabaseError, match="deliberate failure"):
        backup(plain_project, target)
    assert (target / "project_database.sqlite").is_file()
    assert not (target / "parameters.yml").exists()
    assert not (target / "matrices").exists()


def make_spatial_database(path, schema, scenario=False):
    from aequilibrae.project.project_creation import create_base_tables
    from aequilibrae.reference_files import spatialite_database
    from aequilibrae.utils.spatialite_utils import connect_spatialite

    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(spatialite_database, path)
    with closing(connect_spatialite(path)) as conn, conn:
        create_base_tables(conn, schema)
        if scenario:
            conn.execute("DROP TABLE scenarios")
        if schema == "network":
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
        else:
            conn.execute("INSERT INTO agencies (agency_id, agency) VALUES (7, 'Test agency')")
            conn.execute(
                "INSERT INTO stops (stop_id, stop, agency_id, geometry) "
                "VALUES ('1', 'Stop 1', 7, MakePoint(0, 0, 4326)), ('2', 'Stop 2', 7, MakePoint(1, 1, 4326))"
            )
            conn.execute(
                "INSERT INTO routes (pattern_id, route_id, route, agency_id, route_type) VALUES (8, 8, 'Route', 7, 3)"
            )
            conn.execute(
                "INSERT INTO route_links (transit_link, pattern_id, seq, from_stop, to_stop, distance, geometry) "
                "VALUES (9, 8, 0, 1, 2, 1, GeomFromText('LINESTRING(0 0, 1 1)', 4326))"
            )
            conn.execute("INSERT INTO trips (trip_id, dir, pattern_id) VALUES (10, 0, 8)")
            conn.execute("INSERT INTO trips_schedule (trip_id, seq, arrival, departure) VALUES (10, 0, 0, 60)")


@pytest.fixture
def spatial_project(tmp_path):
    source = tmp_path / "source"
    for folder in (source, source / "scenarios" / "future"):
        make_spatial_database(folder / "project_database.sqlite", "network", scenario=folder != source)
        make_spatial_database(folder / "public_transport.sqlite", "transit")
        with closing(sqlite3.connect(folder / "project_database.sqlite")) as conn, conn:
            conn.execute("ALTER TABLE modes ADD COLUMN custom_column TEXT")
            conn.execute("CREATE TABLE custom_table (value INTEGER)")
            conn.execute("INSERT INTO custom_table VALUES (123)")
            conn.execute("UPDATE migrations SET status='APPLIED'")
            conn.execute(
                "INSERT INTO results (table_name, procedure, procedure_id, procedure_report) "
                "VALUES ('registered', 'test', 'id', 'null')"
            )
        with closing(sqlite3.connect(folder / "results_database.sqlite")) as conn, conn:
            conn.execute('CREATE TABLE registered ("odd""column" INTEGER, volume REAL)')
            conn.execute("INSERT INTO registered VALUES (1, 12.5)")
            conn.execute('CREATE INDEX result_index ON registered ("odd""column")')
            conn.execute("CREATE TABLE unregistered (value INTEGER)")
        (folder / "parameters.yml").write_text("run: {}\n")
    with closing(sqlite3.connect(source / "project_database.sqlite")) as conn, conn:
        conn.execute("INSERT INTO scenarios VALUES ('future', 'Future scenario')")
    return source


def test_copy_to_new_rebuilds_all_scenarios(spatial_project, tmp_path, caplog):
    from aequilibrae.utils.spatialite_utils import connect_spatialite

    originals = {path: path.read_bytes() for path in spatial_project.rglob("*.sqlite")}
    target = copy_to_new(spatial_project, tmp_path / "new")
    assert all(path.read_bytes() == data for path, data in originals.items())
    for relative in (".", "scenarios/future"):
        original = spatial_project / relative
        copied = target / relative
        project_db = copied / "project_database.sqlite"
        assert query(project_db, "SELECT node_id, geometry FROM nodes ORDER BY node_id") == query(
            original / "project_database.sqlite", "SELECT node_id, geometry FROM nodes ORDER BY node_id"
        )
        assert query(project_db, "SELECT link_id, a_node, b_node, geometry FROM links") == query(
            original / "project_database.sqlite", "SELECT link_id, a_node, b_node, geometry FROM links"
        )
        with closing(connect_spatialite(project_db)) as conn:
            assert conn.execute("SELECT CheckSpatialIndex('nodes', 'geometry')").fetchone() == (1,)
            assert conn.execute("SELECT CheckSpatialIndex('links', 'geometry')").fetchone() == (1,)
            assert conn.execute("SELECT distance = GeodesicLength(geometry) FROM links").fetchone() == (1,)
            assert conn.execute("SELECT geometry IS NOT NULL FROM turn_restrictions").fetchone() == (1,)
        assert query(project_db, "SELECT node_id, modes FROM nodes ORDER BY node_id") == [(1, "q"), (2, "q")]
        assert query(original / "project_database.sqlite", "SELECT distance FROM links") == [(None,)]
        assert query(project_db, "SELECT DISTINCT status FROM migrations") == [("SKIPPED",)]
        assert not query(project_db, "SELECT 1 FROM sqlite_master WHERE name='custom_table'")
        assert "custom_column" not in [row[1] for row in query(project_db, "PRAGMA table_info(modes)")]
        assert query(project_db, "SELECT name FROM sqlite_master WHERE name LIKE 'aequilibrae_%'")
        transit_db = copied / "public_transport.sqlite"
        assert query(transit_db, "SELECT agency_id, agency FROM agencies") == [(7, "Test agency")]
        assert query(transit_db, "SELECT trip_id, departure FROM trips_schedule") == [(10, 60)]
        with closing(connect_spatialite(transit_db)) as conn:
            assert conn.execute("SELECT distance = GeodesicLength(geometry) FROM route_links").fetchone() == (1,)
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert query(copied / "results_database.sqlite", "SELECT * FROM registered") == [(1, 12.5)]
        assert query(copied / "results_database.sqlite", "SELECT name FROM sqlite_master WHERE type='index'") == [
            ("result_index",)
        ]
        assert not query(copied / "results_database.sqlite", "SELECT 1 FROM sqlite_master WHERE name='unregistered'")
        assert (copied / "parameters.yml").is_file()
    assert query(target / "project_database.sqlite", "SELECT scenario_name FROM scenarios ORDER BY scenario_name") == [
        ("future",),
        ("root",),
    ]
    assert not query(
        target / "scenarios/future/project_database.sqlite", "SELECT 1 FROM sqlite_master WHERE name='scenarios'"
    )
    assert "custom_column" in caplog.text
    assert "custom_table" in caplog.text
    assert "unregistered" in caplog.text
    assert query(spatial_project / "project_database.sqlite", "SELECT * FROM custom_table") == [(123,)]
    assert query(spatial_project / "project_database.sqlite", "SELECT DISTINCT status FROM migrations") == [
        ("APPLIED",)
    ]


def test_copy_to_new_uses_new_defaults(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with closing(sqlite3.connect(source / "project_database.sqlite")) as conn, conn:
        conn.execute("CREATE TABLE modes (mode_name TEXT, mode_id TEXT)")
        conn.execute("INSERT INTO modes VALUES ('car', 'c')")
    target = copy_to_new(source, tmp_path / "new")
    assert query(target / "project_database.sqlite", "SELECT mode_id, pce FROM modes") == [("c", 1.0)]
    assert query(target / "project_database.sqlite", "SELECT scenario_name FROM scenarios") == [("root",)]
    assert not (target / "public_transport.sqlite").exists()
    assert not (target / "results_database.sqlite").exists()


def test_copy_to_new_stops_on_constraint_failure(tmp_path, caplog):
    source = tmp_path / "source"
    source.mkdir()
    with closing(sqlite3.connect(source / "project_database.sqlite")) as conn, conn:
        conn.execute("CREATE TABLE nodes (node_id INTEGER, is_centroid INTEGER)")
        conn.execute("INSERT INTO nodes VALUES (1, 7)")
    (source / "parameters.yml").write_text("run: {}\n")
    target = tmp_path / "partial"
    with pytest.raises(sqlite3.IntegrityError):
        copy_to_new(source, target)
    assert (target / "project_database.sqlite").is_file()
    assert not (target / "parameters.yml").exists()
    assert "partial output" in caplog.text
    assert query(source / "project_database.sqlite", "SELECT * FROM nodes") == [(1, 7)]


def test_copy_to_new_checks_foreign_keys(spatial_project, tmp_path):
    with closing(sqlite3.connect(spatial_project / "project_database.sqlite")) as conn, conn:
        conn.execute("INSERT INTO transit_graph_configs (period_id, config) VALUES (99, '{}')")
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY|Foreign-key"):
        copy_to_new(spatial_project, tmp_path / "new")


def test_copy_to_new_accepts_trigger_generated_nodes(spatial_project, tmp_path):
    from aequilibrae.utils.spatialite_utils import connect_spatialite

    with closing(connect_spatialite(spatial_project / "project_database.sqlite")) as conn, conn:
        conn.execute("UPDATE links SET b_node=999, geometry=GeomFromText('LINESTRING(0 0, 2 2)', 4326)")
    target = copy_to_new(spatial_project, tmp_path / "new")
    assert query(target / "project_database.sqlite", "SELECT a_node, b_node FROM links") == [(1, 3)]
    assert query(target / "project_database.sqlite", "SELECT COUNT(*) FROM nodes") == [(3,)]
    with closing(connect_spatialite(target / "project_database.sqlite")) as conn:
        assert conn.execute("SELECT AsText(geometry) FROM nodes WHERE node_id=3").fetchone() == ("POINT(2 2)",)
        assert conn.execute("SELECT CheckSpatialIndex('nodes', 'geometry')").fetchone() == (1,)
    assert query(spatial_project / "project_database.sqlite", "SELECT b_node FROM links") == [(999,)]


@pytest.mark.parametrize("table", ["links", "turn_restrictions"])
def test_copy_to_new_rejects_unknown_modes(spatial_project, tmp_path, table):
    with closing(sqlite3.connect(spatial_project / "project_database.sqlite")) as conn, conn:
        conn.execute(f"UPDATE {table} SET modes='x'")
    target = tmp_path / "new"
    with pytest.raises(sqlite3.IntegrityError, match="[Mm]ode codes"):
        copy_to_new(spatial_project, target)
    assert target.is_dir()
    assert not (target / "parameters.yml").exists()


def test_copy_to_new_rejects_duplicate_node_geometry(spatial_project, tmp_path):
    from aequilibrae.utils.spatialite_utils import connect_spatialite

    with closing(connect_spatialite(spatial_project / "project_database.sqlite")) as conn, conn:
        conn.execute("INSERT INTO nodes (node_id, geometry) VALUES (3, MakePoint(0, 0, 4326))")
    with pytest.raises(sqlite3.IntegrityError, match="Cannot create on-top of other node"):
        copy_to_new(spatial_project, tmp_path / "new")


@pytest.mark.parametrize("relative", [".", "scenarios/future"])
def test_missing_results_database_rejected_only_by_copy_to_new(spatial_project, tmp_path, relative):
    (spatial_project / relative / "results_database.sqlite").unlink()
    target = tmp_path / "new"
    with pytest.raises(FileNotFoundError, match="Results database missing"):
        copy_to_new(spatial_project, target)
    assert target.is_dir()
    assert not (target / "parameters.yml").exists()
    copied = backup(spatial_project, tmp_path / "backup")
    assert not (copied / relative / "results_database.sqlite").exists()
    assert query(copied / relative / "project_database.sqlite", "SELECT table_name FROM results") == [("registered",)]


@pytest.mark.parametrize("remove_directory", [False, True])
def test_missing_scenario_rejected_only_by_copy_to_new(spatial_project, tmp_path, remove_directory):
    scenario = spatial_project / "scenarios" / "future"
    if remove_directory:
        shutil.rmtree(scenario)
    else:
        (scenario / "project_database.sqlite").unlink()
    target = tmp_path / "new"
    with pytest.raises(FileNotFoundError, match="Registered scenario databases missing.*future"):
        copy_to_new(spatial_project, target)
    assert target.is_dir()
    assert not (target / "parameters.yml").exists()
    copied = backup(spatial_project, tmp_path / "backup")
    assert not (copied / "scenarios" / "future").exists()
    assert query(copied / "project_database.sqlite", "SELECT scenario_name FROM scenarios WHERE scenario_name='future'")


def test_table_order_combines_foreign_keys_and_trigger_dependencies():
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript(
            "CREATE TABLE modes (id PRIMARY KEY);"
            "CREATE TABLE link_types (id PRIMARY KEY);"
            "CREATE TABLE nodes (id PRIMARY KEY);"
            "CREATE TABLE links (kind REFERENCES link_types(id));"
            "CREATE TABLE turn_restrictions (node REFERENCES nodes(id));"
        )
        order = backup_module._table_copy_order(
            conn, ["turn_restrictions", "links", "nodes", "link_types", "modes"], "network"
        )
        for parent in ("nodes", "link_types", "modes"):
            assert order.index(parent) < order.index("links")
        assert order.index("links") < order.index("turn_restrictions")


def test_transit_table_order(spatial_project):
    tables = [
        "trips_schedule",
        "route_links",
        "pattern_mapping",
        "links",
        "trips",
        "routes",
        "stops",
        "agencies",
        "nodes",
        "node_types",
        "trigger_settings",
        "modes",
        "link_types",
    ]
    with closing(sqlite3.connect(spatial_project / "public_transport.sqlite")) as conn:
        order = backup_module._table_copy_order(conn, tables, "transit")
    for parent, child in (
        ("agencies", "stops"),
        ("agencies", "routes"),
        ("stops", "route_links"),
        ("routes", "route_links"),
        ("routes", "pattern_mapping"),
        ("routes", "trips"),
        ("trips", "trips_schedule"),
        ("node_types", "nodes"),
        ("nodes", "links"),
        ("modes", "links"),
        ("link_types", "links"),
        ("stops", "links"),
        ("trigger_settings", "links"),
    ):
        assert order.index(parent) < order.index(child)


def test_table_order_rejects_cycles():
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript("CREATE TABLE a (id PRIMARY KEY, b REFERENCES b(id)); CREATE TABLE b (a REFERENCES a(id));")
        with pytest.raises(CycleError):
            backup_module._table_copy_order(conn, ["a", "b"], "network")


def test_generated_result_columns(spatial_project, tmp_path):
    with closing(sqlite3.connect(spatial_project / "results_database.sqlite")) as conn, conn:
        conn.execute("ALTER TABLE registered ADD COLUMN doubled REAL GENERATED ALWAYS AS (volume * 2)")
    target = copy_to_new(spatial_project, tmp_path / "new")
    assert query(target / "results_database.sqlite", "SELECT doubled FROM registered") == [(25.0,)]


def test_missing_registered_result_stops_copy(spatial_project, tmp_path):
    with closing(sqlite3.connect(spatial_project / "results_database.sqlite")) as conn, conn:
        conn.execute("DROP TABLE registered")
    with pytest.raises(ValueError, match="Registered result tables missing"):
        copy_to_new(spatial_project, tmp_path / "new")


@pytest.mark.parametrize("command,function", [("backup", "backup"), ("copy-to-new", "copy_to_new")])
def test_cli_dispatch_without_opening_project(command, function, tmp_path, monkeypatch, capsys):
    calls = []

    def operation(source, destination):
        calls.append((source, destination))
        return destination

    monkeypatch.setattr(backup_module, function, operation)
    source, target = str(tmp_path / "source"), str(tmp_path / "target")
    cli(["-p", source, command, target])
    assert calls == [(source, target)]
    assert target in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli(["-p", source, command, target, "--unknown"])
    assert len(calls) == 1
