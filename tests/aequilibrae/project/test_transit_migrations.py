"""Phase 1 upgrades: saved transit graphs are dropped, the default period spans the service day, every scenario is
migrated, and the older transit migrations no longer read the live schema files."""

import json
import logging
import re
import sqlite3
from pathlib import Path

import pytest

from aequilibrae import Project
from aequilibrae.project.tools import MigrationManager, MigrationStatus
from aequilibrae.transit import Transit
from aequilibrae.utils.db_utils import commit_and_close
from aequilibrae.utils.model_run_utils import import_file_as_module

SPECIFICATION = Path(__file__).parents[3] / "src" / "aequilibrae" / "project" / "database_specification"
TRANSIT_MIGRATIONS = SPECIFICATION / "transit" / "migrations"
NETWORK_MIGRATIONS = SPECIFICATION / "network" / "migrations"

OLD_CONFIG = {"period_id": 1, "projected_crs": "EPSG:3857", "with_walking_edges": False, "connector_method": "x"}


def migration_module(path: Path):
    return import_file_as_module(path, f"test_{path.parent.parent.name}_{path.stem}", force=True)


def columns(conn, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def forget_migration(conn, migration_id: int):
    conn.execute("DELETE FROM migrations WHERE id = ?", [migration_id])


def status(path: Path, migration_id: int) -> str:
    with commit_and_close(path, spatial=False) as conn:
        return conn.execute("SELECT status FROM migrations WHERE id = ?", [migration_id]).fetchone()[0]


def add_pre_phase1_graph(transit_path: Path, project_path: Path):
    """Turn a transit database into one with a graph saved before migration 007, with head-to-tail links."""
    old_links = migration_module(TRANSIT_MIGRATIONS / "005_repair_foreign_keys.py").LINKS_SQL
    with commit_and_close(transit_path, spatial=True) as conn:
        conn.execute("SELECT DropTable(NULL, 'links')")
        for sql in old_links.split("--#"):
            conn.execute(sql)
        conn.executemany(
            "INSERT INTO links (link_id, period_id, a_node, b_node, direction, modes, trav_time, freq, geometry) "
            "VALUES (?, 1, ?, ?, 1, 't', ?, ?, GeomFromText('LINESTRING(0 0, 1 1)', 4326))",
            [(1, 2, 1, 495.0, 1e20), (2, 3, 2, 30.0, 0.004)],
        )
        conn.executemany(
            "INSERT INTO nodes (node_id, period_id, node_type, taz_id, geometry) "
            "VALUES (?, 1, ?, ?, GeomFromText('POINT(0 0)', 4326))",
            [(1, "od", 1), (2, "stop", -1), (3, "boarding", -1)],
        )
        forget_migration(conn, 7)

    with commit_and_close(project_path, spatial=True) as conn:
        conn.execute("INSERT INTO transit_graph_configs (period_id, config) VALUES (1, ?)", [json.dumps(OLD_CONFIG)])


def restore_one_day_default_period(project_path: Path):
    """Put back the default period and protecting trigger that projects had before migration 010."""
    with commit_and_close(project_path, spatial=False) as conn:
        conn.execute("DROP TRIGGER aequilibrae_default_period_update")
        conn.execute(
            "UPDATE periods SET period_end = 86400, period_description = 'Default time period, whole day' "
            "WHERE period_id = 1"
        )
        conn.execute(
            """CREATE TRIGGER aequilibrae_default_period_update BEFORE UPDATE ON periods
            WHEN new.period_id = 1
            BEGIN
                SELECT RAISE(ABORT,'Cannot update default period');
            END"""
        )
        forget_migration(conn, 10)


def test_upgrade_drops_saved_graphs_and_logs_their_configs(coquimbo_example, caplog):
    """Test that migration 007 empties saved graphs, logs their configs and recreates links with physical columns."""
    base = coquimbo_example.project_base_path
    add_pre_phase1_graph(base / "public_transport.sqlite", base / "project_database.sqlite")
    with coquimbo_example.transit_connection as conn:
        assert "trav_time" in columns(conn, "links")
        assert conn.execute("SELECT COUNT(*) FROM links").fetchone()[0] == 2

    with caplog.at_level(logging.WARNING):
        coquimbo_example.upgrade()

    with coquimbo_example.transit_connection as conn:
        assert conn.execute("SELECT COUNT(*) FROM links").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 0
        link_columns = columns(conn, "links")
        indices = {row[1] for row in conn.execute("PRAGMA index_list(links)")}
    for column in ["segment_id", "run_time", "dwell_time", "walk_distance", "walk_time", "headway", "trip_count"]:
        assert column in link_columns
    assert {"line_freq", "capacity"} <= set(link_columns)
    assert not {"trav_time", "freq", "distance", "o_line_id", "d_line_id", "transfer_id"} & set(link_columns)
    assert {"idx_link", "idx_period_links", "idx_links_a_node_b_node"} <= indices

    with coquimbo_example.db_connection as conn:
        assert conn.execute("SELECT COUNT(*) FROM transit_graph_configs").fetchone()[0] == 0

    dropped = [r for r in caplog.records if "Dropped the saved transit graph of period 1" in r.getMessage()]
    assert len(dropped) == 1
    # Logged under the aequilibrae logger, which writes the project's log file
    assert dropped[0].name.startswith("aequilibrae.")
    logged = [r.getMessage() for r in dropped]
    assert json.loads(re.search(r"configuration was: (\{.*\})", logged[0]).group(1)) == OLD_CONFIG
    assert status(base / "public_transport.sqlite", 7) == MigrationStatus.APPLIED.name


def test_upgrade_without_saved_graphs_leaves_supply_alone(coquimbo_example):
    """Test that migration 007 on a project without saved graphs keeps its routes, stops and trips."""
    base = coquimbo_example.project_base_path
    tables = ["routes", "stops", "trips", "trips_schedule", "route_links", "pattern_mapping"]
    with coquimbo_example.transit_connection as conn:
        before = {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables}
        forget_migration(conn, 7)

    coquimbo_example.upgrade()

    with coquimbo_example.transit_connection as conn:
        after = {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables}
    assert after == before
    assert status(base / "public_transport.sqlite", 7) == MigrationStatus.APPLIED.name


def test_load_skips_configs_without_saved_graphs(coquimbo_example):
    """Test that Transit.load ignores a graph configuration whose nodes and links are gone."""
    with coquimbo_example.db_connection as conn:
        conn.execute("INSERT INTO transit_graph_configs (period_id, config) VALUES (1, ?)", [json.dumps(OLD_CONFIG)])

    transit = Transit(coquimbo_example)
    transit.load()
    assert transit.graphs == {}


def test_upgrade_extends_the_default_period_to_the_service_day(coquimbo_example):
    """Test that migration 010 makes the default period 00:00 to 48:00 and keeps it protected from updates."""
    project_path = coquimbo_example.project_base_path / "project_database.sqlite"
    restore_one_day_default_period(project_path)

    coquimbo_example.upgrade()

    default = coquimbo_example.network.periods.get(1)
    assert (default.period_start, default.period_end) == (0, 172800)
    assert default.period_description == "Default time period, whole service day"
    assert status(project_path, 10) == MigrationStatus.APPLIED.name
    with pytest.raises(sqlite3.IntegrityError, match="Cannot update default period"):
        coquimbo_example.network.periods.update(1, period_end=86400)


def test_new_projects_default_period_spans_the_service_day(empty_project):
    """Test that a new project's default period runs from 00:00 to 48:00."""
    default = empty_project.network.periods.get(1)
    assert (default.period_start, default.period_end) == (0, 172800)


def test_upgrade_migrates_every_scenario(scenario_example):
    """Test that upgrading with root active also migrates the databases of the other scenarios."""
    scenario_example.use_scenario("root")
    coquimbo = scenario_example.root_scenario.base_path / "scenarios" / "coquimbo"
    restore_one_day_default_period(coquimbo / "project_database.sqlite")
    add_pre_phase1_graph(coquimbo / "public_transport.sqlite", coquimbo / "project_database.sqlite")

    scenario_example.upgrade()

    assert status(coquimbo / "project_database.sqlite", 10) == MigrationStatus.APPLIED.name
    assert status(coquimbo / "public_transport.sqlite", 7) == MigrationStatus.APPLIED.name
    scenario_example.use_scenario("coquimbo")
    assert scenario_example.network.periods.get(1).period_end == 172800
    with scenario_example.transit_connection as conn:
        assert "run_time" in columns(conn, "links")
        assert conn.execute("SELECT COUNT(*) FROM links").fetchone()[0] == 0


@pytest.mark.parametrize("migration", ["002_support_saving_multiple_transit_periods.py", "005_repair_foreign_keys.py"])
def test_older_transit_migrations_do_not_read_the_schema_files(migration):
    """Test that migrations 002 and 005 carry their own SQL instead of reading tables/ and triggers/ at upgrade time."""
    source = (TRANSIT_MIGRATIONS / migration).read_text(encoding="utf-8")
    code = re.sub(r'r"""(.|\n)*?"""', "", source)  # the inlined SQL may mention anything
    code = re.sub(r"#.*", "", code)
    references = ["__file__", "open(", "read_text", ".sql", "run_queries_from_sql_file"]
    references += ["add_triggers", "remove_triggers"]
    for reference in references:
        assert reference not in code, reference


def test_foreign_key_repair_rebuilds_links_from_its_own_schema(coquimbo_example):
    """Test that migration 005 rebuilds links with the columns of its time, whatever the live links.sql says."""
    migrate = migration_module(TRANSIT_MIGRATIONS / "005_repair_foreign_keys.py").migrate
    with coquimbo_example.transit_connection as conn:
        migrate(project_conn=None, transit_conn=conn, results_conn=None)
        link_columns = columns(conn, "links")
    assert {"trav_time", "freq"} <= set(link_columns)
    assert "run_time" not in link_columns


def test_frozen_schemas_match_the_schema_before_phase_1():
    """Test that migrations 002 and 005 inline the same links schema, which has the costed trav_time and freq."""
    links_002 = migration_module(TRANSIT_MIGRATIONS / "002_support_saving_multiple_transit_periods.py").LINKS_SQL
    links_005 = migration_module(TRANSIT_MIGRATIONS / "005_repair_foreign_keys.py").LINKS_SQL
    assert links_002 == links_005
    assert "trav_time       NUMERIC  NOT NULL" in links_002


def test_new_migrations_are_registered():
    """Test that transit migration 007 and network migration 010 are part of their migration series."""
    transit = MigrationManager(MigrationManager.transit_migration_file).migrations
    network = MigrationManager(MigrationManager.network_migration_file).migrations
    assert transit[7].name == "physical_transit_graphs"
    assert network[10].name == "default_period_whole_service_day"
    assert max(transit) == 7
    assert max(network) == 10
    assert (NETWORK_MIGRATIONS / "010_default_period_whole_service_day.py").exists()


def test_projects_without_turn_restrictions_open_and_upgrade(empty_project):
    """Test that a project from before the turn_restrictions table, as v1.7.2 made them, opens and upgrades."""
    path = empty_project.project_base_path
    database = path / "project_database.sqlite"
    empty_project.close()
    with commit_and_close(database, spatial=True) as conn:
        for trigger in (
            "aequilibrae_mode_keep_if_in_use_on_turn_restrictions_updating",
            "aequilibrae_mode_keep_if_in_use_on_turn_restrictions_deleting",
            "aequilibrae_turn_restrictions_update_geometry_on_node_move",
        ):
            conn.execute(f"DROP TRIGGER {trigger}")
        conn.execute("SELECT DropTable(NULL, 'turn_restrictions')")
        forget_migration(conn, 9)

    project = Project.from_path(path)
    try:
        project.upgrade()
        assert status(database, 9) == MigrationStatus.APPLIED.name
        assert len(project.network.turn_restrictions) == 0
    finally:
        project.close()
