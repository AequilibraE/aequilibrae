"""
.. _import_from_visum_sqlite:

Importing a network from VISUM SQLite
=====================================

This example creates a tiny VISUM-like SQLite export locally and imports it as
an AequilibraE private-traffic network.
"""

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from aequilibrae import Project
from aequilibrae.project.network.visum_sqlite_importer import visum_sqlite_source_connectivity


def create_visum_sqlite(path):
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE NETWORK(PROJECTIONDEFINITION TEXT);
            CREATE TABLE NODE(NO INTEGER PRIMARY KEY, NAME TEXT, TYPENO INTEGER, XCOORD REAL, YCOORD REAL);
            CREATE TABLE ZONE(NO INTEGER PRIMARY KEY, NAME TEXT, TYPENO INTEGER, XCOORD REAL, YCOORD REAL);
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
                "T0_TSYS(CAR)" REAL,
                "T0_TSYS(HGV)" REAL,
                WEIGHTPRT REAL
            );
            CREATE TABLE COUNTLOCATION(
                NO INTEGER PRIMARY KEY,
                LINKNO INTEGER,
                FROMNODENO INTEGER,
                TONODENO INTEGER,
                CAR_ORIG REAL
            );
            CREATE TABLE LINKPOLY(FROMNODENO INTEGER, TONODENO INTEGER, "INDEX" INTEGER, XCOORD REAL, YCOORD REAL);
            """
        )
        conn.execute("INSERT INTO NETWORK VALUES('EPSG:4326')")
        conn.executemany(
            "INSERT INTO TSYS VALUES(?, ?, ?, ?)",
            [("CAR", "Car", "PrT", 1.0), ("HGV", "HGV", "PrT", 2.0)],
        )
        conn.execute("INSERT INTO MODE VALUES('CAR', 'Car', 'CAR,HGV')")
        conn.execute("INSERT INTO LINKTYPE VALUES(1, 'arterial', 'CAR,HGV', 1200, 60)")
        conn.executemany(
            "INSERT INTO NODE VALUES(?, ?, ?, ?, ?)",
            [(1, "A", 1, 0.0, 0.0), (2, "B", 1, 0.01, 0.0)],
        )
        conn.executemany(
            "INSERT INTO ZONE VALUES(?, ?, ?, ?, ?)",
            [(1001, "Z1", 1, -0.01, 0.0), (1002, "Z2", 1, 0.02, 0.0)],
        )
        conn.executemany(
            "INSERT INTO LINK VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (100, 1, 2, "AB", 1, "CAR,HGV", 1.0, 1200, 60, "ARTERIAL"),
                (100, 2, 1, "BA", 1, "CAR", 1.1, 1100, 55, "ARTERIAL"),
            ],
        )
        conn.executemany(
            "INSERT INTO LINKPOLY VALUES(?, ?, ?, ?, ?)",
            [(1, 2, 1, 0.0, 0.0), (1, 2, 2, 0.005, 0.001), (1, 2, 3, 0.01, 0.0)],
        )
        conn.executemany(
            'INSERT INTO CONNECTOR VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)',
            [
                (1001, 1, "O", 1, "CAR,HGV", 0.1, 0.0, 0.0, 100),
                (1001, 1, "D", 1, "CAR,HGV", 0.1, 0.0, 0.0, 100),
                (1002, 2, "O", 1, "CAR,HGV", 0.1, 30.0, 30.0, 100),
                (1002, 2, "D", 1, "CAR,HGV", 0.1, 30.0, 30.0, 100),
            ],
        )
        conn.execute("INSERT INTO COUNTLOCATION VALUES(1, 100, 1, 2, 950)")
        conn.commit()
    finally:
        conn.close()


with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
    temp_dir = Path(temp_dir)
    sqlite_path = temp_dir / "visum.sqlite"
    create_visum_sqlite(sqlite_path)

    project = Project()
    project.new(temp_dir / "visum_project")

    report = project.network.create_from_visum_sqlite(sqlite_path)
    project.network.build_graphs(
        fields=["distance", "travel_time_ab", "travel_time_ba", "capacity_ab", "capacity_ba"], modes=["c"]
    )
    project.network.set_time_field("travel_time")

    source_graph = visum_sqlite_source_connectivity(sqlite_path)

    print(report.imported_counts)
    print({mode: len(arcs) for mode, arcs in source_graph.items()})

    project.close()
