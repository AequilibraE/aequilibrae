import sqlite3
from pathlib import Path

import pytest

from aequilibrae.project.network.visum_sqlite_importer import (
    VISUM_SQLITE_CONNECTOR_EPSILON_MINUTES,
    discover_visum_sqlite,
    visum_sqlite_source_connectivity,
)


def _create_visum_sqlite(path: Path, *, invalid_crs: bool = False, omit_table: str | None = None) -> Path:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE NETWORK(PROJECTIONDEFINITION TEXT);
            CREATE TABLE NODE(NO INTEGER PRIMARY KEY, NAME TEXT, TYPENO INTEGER, XCOORD REAL, YCOORD REAL);
            CREATE TABLE ZONE(
                NO INTEGER PRIMARY KEY,
                NAME TEXT,
                TYPENO INTEGER,
                XCOORD REAL,
                YCOORD REAL,
                SURFACEID INTEGER
            );
            CREATE TABLE TSYS(CODE TEXT PRIMARY KEY, NAME TEXT, TYPE TEXT, PCU REAL);
            CREATE TABLE MODE(CODE TEXT PRIMARY KEY, NAME TEXT, TSYSSET TEXT);
            CREATE TABLE LINKTYPE(NO INTEGER PRIMARY KEY, NAME TEXT, TSYSSET TEXT, CAPPRT INTEGER, V0PRT REAL);
            CREATE TABLE LINK(
                NO INTEGER,
                FROMNODENO INTEGER,
                TONODENO INTEGER,
                NAME TEXT,
                TYPENO INTEGER,
                TSYSSET TEXT,
                LENGTH REAL,
                CAPPRT INTEGER,
                V0PRT REAL,
                LC TEXT
            );
            CREATE TABLE CONNECTOR(
                ZONENO INTEGER,
                NODENO INTEGER,
                DIRECTION TEXT,
                TYPENO INTEGER,
                TSYSSET TEXT,
                LENGTH REAL,
                "T0_TSYS(CAR)" INTEGER,
                "T0_TSYS(HGV)" INTEGER,
                "WEIGHT(PRT)" INTEGER
            );
            CREATE TABLE LINKPOLY(FROMNODENO INTEGER, TONODENO INTEGER, "INDEX" INTEGER, XCOORD REAL, YCOORD REAL);
            CREATE TABLE POINT(ID INTEGER PRIMARY KEY, XCOORD REAL, YCOORD REAL);
            CREATE TABLE EDGE(ID INTEGER PRIMARY KEY, FROMPOINTID INTEGER, TOPOINTID INTEGER);
            CREATE TABLE FACEITEM(FACEID INTEGER, "INDEX" INTEGER, EDGEID INTEGER, DIRECTION INTEGER);
            CREATE TABLE SURFACEITEM(SURFACEID INTEGER, FACEID INTEGER, ENCLAVE INTEGER);
            CREATE TABLE COUNTLOCATION(
                NO INTEGER PRIMARY KEY,
                LINKNO INTEGER,
                FROMNODENO INTEGER,
                TONODENO INTEGER,
                CAR_ORIG INTEGER,
                HVG_ORIG INTEGER,
                MOTOR_ORIG INTEGER,
                DTVW INTEGER,
                CARS_LEFT INTEGER
            );
            CREATE TABLE STOP(NO INTEGER PRIMARY KEY);
            """
        )
        crs = "not-a-crs" if invalid_crs else "EPSG:4326"
        conn.execute("INSERT INTO NETWORK VALUES(?)", (crs,))
        conn.executemany(
            "INSERT INTO TSYS VALUES(?, ?, ?, ?)",
            [
                ("CAR", "Car", "PrT", 1.0),
                ("HGV", "HGV", "PrT", 2.0),
                ("BUS", "Bus", "PuT", 1.0),
            ],
        )
        conn.executemany(
            "INSERT INTO MODE VALUES(?, ?, ?)",
            [("C", "Car", "CAR"), ("H", "HGV", "HGV"), ("PuT", "PuT", "BUS")],
        )
        conn.executemany(
            "INSERT INTO LINKTYPE VALUES(?, ?, ?, ?, ?)",
            [(10, "arterial", "CAR,HGV", 1200, 60.0), (20, "closed", "", 0, 0.0)],
        )
        conn.executemany(
            "INSERT INTO NODE VALUES(?, ?, ?, ?, ?)",
            [
                (1, "A", 1, 0.0, 0.0),
                (2, "B", 1, 0.01, 0.0),
                (3, "B duplicate", 1, 0.01, 0.0),
                (1001, "Regular node colliding with zone ID", 1, 0.03, 0.0),
            ],
        )
        conn.executemany(
            "INSERT INTO ZONE VALUES(?, ?, ?, ?, ?, ?)",
            [(1001, "Z1", 1, -0.01, 0.0, 1), (1002, "Z2", 1, 0.02, 0.0, None)],
        )
        conn.executemany(
            "INSERT INTO LINK VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (550000000, 1, 2, "L1", 10, "CAR,HGV", 1.0, 1200, 60.0, "ARTERIAL"),
                (550000000, 2, 1, "L1", 10, "CAR", 1.1, 1100, 55.0, "LOCAL"),
                (660000000, 2, 1001, "Closed", 20, "", 0.1, 0, 0.0, "closed"),
                (660000000, 1001, 2, "Closed", 20, "", 0.1, 0, 0.0, "closed"),
            ],
        )
        conn.execute("INSERT INTO LINKPOLY VALUES(1, 2, 1, 0.005, 0.001)")
        conn.executemany(
            'INSERT INTO CONNECTOR VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)',
            [
                (1001, 1, "O", 0, "CAR,HGV", 0.1, 60, 60, 100),
                (1001, 1, "D", 0, "CAR,HGV", 0.1, 120, 120, 100),
                (1002, 2, "O", 0, "CAR,HGV", 0.0, 0, 0, 100),
                (1002, 2, "D", 0, "CAR,HGV", 0.0, 0, 0, 100),
            ],
        )
        conn.executemany(
            "INSERT INTO POINT VALUES(?, ?, ?)",
            [(1, -0.02, -0.01), (2, 0.0, -0.01), (3, 0.0, 0.01), (4, -0.02, 0.01)],
        )
        conn.executemany("INSERT INTO EDGE VALUES(?, ?, ?)", [(1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 1)])
        conn.executemany(
            "INSERT INTO FACEITEM VALUES(?, ?, ?, ?)",
            [(1, 1, 1, 0), (1, 2, 2, 0), (1, 3, 3, 0), (1, 4, 4, 0)],
        )
        conn.execute("INSERT INTO SURFACEITEM VALUES(1, 1, 0)")
        conn.execute("INSERT INTO COUNTLOCATION VALUES(1, 550000000, 1, 2, 950, 120, 1070, 1300, 10)")
        if omit_table is not None:
            conn.execute(f'DROP TABLE "{omit_table}"')
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture
def visum_sqlite_file(tmp_path):
    return _create_visum_sqlite(tmp_path / "visum.sqlite3")


def test_discover_visum_sqlite_reports_tables(visum_sqlite_file):
    report = discover_visum_sqlite(visum_sqlite_file)

    assert not report.errors
    assert report.discovered_layers["node"] == "NODE"
    assert "STOP" in report.deferred_layers
    assert any(diag.code == "deferred-table" for diag in report.diagnostics)


def test_discover_visum_sqlite_rejects_missing_required_table(tmp_path):
    path = _create_visum_sqlite(tmp_path / "visum.sqlite3", omit_table="CONNECTOR")
    report = discover_visum_sqlite(path)

    assert {diag.code for diag in report.errors} == {"missing-table"}
    assert {diag.layer for diag in report.errors} == {"CONNECTOR"}


def test_create_from_visum_sqlite_imports_network(empty_project, visum_sqlite_file):
    report = empty_project.network.create_from_visum_sqlite(visum_sqlite_file)

    assert report.imported_counts == {"nodes": 4, "zones": 2, "links": 1, "connectors": 2}
    assert report.source_references["links"] == {550000000: 1}
    assert report.source_references["count_locations"] == [
        {
            "source_id": 1,
            "link_id": 1,
            "counts": {"DTVW": 1300, "HVG_ORIG": 120, "MOTOR_ORIG": 1070, "CAR_ORIG": 950},
        }
    ]
    assert any(diag.code == "sqlite-zero-connector-time" for diag in report.diagnostics)
    assert any(diag.code == "coincident-node-offset" for diag in report.diagnostics)
    assert any(diag.code == "node-id-remapped" for diag in report.diagnostics)

    with empty_project.db_connection as conn:
        assert conn.execute("select count(*) from nodes").fetchone()[0] == 6
        assert conn.execute("select count(*) from zones").fetchone()[0] == 2
        assert conn.execute("select count(*) from links").fetchone()[0] == 4
        assert conn.execute("select link_id from links order by link_id").fetchall() == [(1,), (2,), (3,), (4,)]
        assert conn.execute("select visum_link_no from links where link_id=1").fetchone()[0] == 550000000
        assert conn.execute("select direction, modes from links where link_id=1").fetchone() == (1, "ch")
        assert conn.execute("select direction, modes from links where link_id=2").fetchone() == (-1, "c")
        assert conn.execute("select travel_time_ab from links where link_id=1").fetchone()[0] == pytest.approx(1.0)
        assert conn.execute("select travel_time_ba from links where link_id=2").fetchone()[0] == pytest.approx(1.2)
        assert conn.execute("select visum_length_ab from links where link_id=1").fetchone()[0] == pytest.approx(1000.0)
        assert conn.execute("select count(*) from nodes where node_id=1001 and is_centroid=1").fetchone()[0] == 1
        assert conn.execute("select count(*) from nodes where visum_node_no=1001 and node_id<>1001").fetchone()[0] == 1
        assert conn.execute(
            "select min(travel_time_ab), min(travel_time_ba) from links where link_type='centroid_connector'"
        ).fetchone() == pytest.approx((VISUM_SQLITE_CONNECTOR_EPSILON_MINUTES, VISUM_SQLITE_CONNECTOR_EPSILON_MINUTES))
    with empty_project.db_connection_spatial as conn:
        assert conn.execute("select AsText(geometry) from links where link_id=1").fetchone()[0] == (
            "LINESTRING(0 0, 0.005 0.001, 0.01 0)"
        )


def test_visum_sqlite_graph_is_assignment_ready(empty_project, visum_sqlite_file):
    empty_project.network.create_from_visum_sqlite(visum_sqlite_file)

    empty_project.network.build_graphs(
        fields=["distance", "travel_time_ab", "travel_time_ba", "capacity_ab", "capacity_ba"],
        modes=["c"],
    )
    empty_project.network.set_time_field("travel_time")
    graph = empty_project.network.graphs["c"]

    assert "capacity" in graph.graph.columns
    assert not graph.graph.travel_time.isna().any()
    assert not graph.graph.capacity.isna().any()


def test_visum_sqlite_source_connectivity_matches_import(empty_project, visum_sqlite_file):
    source = visum_sqlite_source_connectivity(visum_sqlite_file)["c"]
    empty_project.network.create_from_visum_sqlite(visum_sqlite_file)
    imported = set()
    with empty_project.db_connection as conn:
        for a_node, b_node, direction, modes in conn.execute("select a_node, b_node, direction, modes from links"):
            if "c" not in modes:
                continue
            if direction in (0, 1):
                imported.add((a_node, b_node))
            if direction in (0, -1):
                imported.add((b_node, a_node))

    assert imported == source
    assert (2, 1001) not in imported
    assert (1001, 2) not in imported


def test_visum_sqlite_rejects_unparseable_crs(tmp_path, empty_project):
    path = _create_visum_sqlite(tmp_path / "visum.sqlite3", invalid_crs=True)

    with pytest.raises(ValueError, match="invalid-crs"):
        empty_project.network.create_from_visum_sqlite(path)
