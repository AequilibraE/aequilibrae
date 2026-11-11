# ruff: noqa: E501 (the inlined SQL is kept as it was written)
import json
import logging
import sqlite3
from typing import Optional

# Migrations are imported outside the package, so name the logger to reach the project log
logger = logging.getLogger("aequilibrae.project.migrations")

# The links schema as of this migration: edges from tail a_node to head b_node, with physical attributes and no costs
LINKS_SQL = r"""--@ The links table holds the edges of the saved Spiess & Florian transit graphs, one graph per
--@ **period_id**.
--@
--@ Each edge runs from its tail, **a_node**, to its head, **b_node**, which are entries in
--@ the node_id field of the nodes table for the same period.
--@
--@ Edges hold physical attributes only. A transit class turns them into the costs and frequencies it
--@ assigns with:
--@
--@ * **run_time** is the mean scheduled run time of on-board edges, in seconds
--@ * **dwell_time** is the mean scheduled dwell at the stop of dwell, boarding and alighting edges
--@ * **walk_distance** is the geodesic length of connectors and of walks between stops, in metres, and
--@   **walk_time** that length over the walking speed
--@ * **headway** is the scheduled headway between departures of a boarding edge's line, and
--@   **trip_count** the line's trips in the period
--@ * **line_freq** is the line's distinct departures over the period length on boarding edges, and
--@   infinite elsewhere
--@ * **capacity** is the capacity of all the period's trips on on-board edges
--@
--@ **segment_id** names the physical element of an edge: *pattern_id:seq* for line edges, the stop
--@ for connectors, and *from_stop:to_stop* for walks between stops.
--@
--@ The **link_type** corresponds to the *link_type* field from the *link_types* table.
--@
--@ The table is indexed on **link_id** and **period_id**, **a_node** and **b_node**.


CREATE TABLE if not exists links (ogc_fid         INTEGER PRIMARY KEY,
                                  link_id         INTEGER NOT NULL,
                                  period_id       INTEGER NOT NULL,
                                  a_node          INTEGER,
                                  b_node          INTEGER,
                                  direction       INTEGER NOT NULL DEFAULT 0,
                                  modes           TEXT    NOT NULL,
                                  link_type       TEXT    REFERENCES link_types(link_type) ON update RESTRICT ON delete RESTRICT,
                                  line_id         TEXT,
                                  stop_id         TEXT    REFERENCES stops(stop_id) ON update RESTRICT ON delete RESTRICT,
                                  line_seg_idx    INTEGER,
                                  segment_id      TEXT,
                                  run_time        NUMERIC NOT NULL DEFAULT 0,
                                  dwell_time      NUMERIC NOT NULL DEFAULT 0,
                                  walk_distance   NUMERIC NOT NULL DEFAULT 0,
                                  walk_time       NUMERIC NOT NULL DEFAULT 0,
                                  headway         NUMERIC NOT NULL DEFAULT 0,
                                  trip_count      INTEGER NOT NULL DEFAULT 0,
                                  line_freq       NUMERIC NOT NULL,
                                  capacity        NUMERIC NOT NULL DEFAULT 0,
                                  CHECK(TYPEOF(link_id) == 'integer')
                                  UNIQUE(link_id, period_id) ON CONFLICT ABORT
                                  CHECK(link_id > 0)
                                  CHECK(TYPEOF(a_node) == 'integer')
                                  CHECK(TYPEOF(b_node) == 'integer')
                                  CHECK(TYPEOF(direction) == 'integer')
                                  CHECK(LENGTH(modes)>0)
                                  CHECK(LENGTH(direction)==1));

--#
select AddGeometryColumn( 'links', 'geometry', 4326, 'LINESTRING', 'XY', 1);

--#
CREATE UNIQUE INDEX idx_link ON links (link_id, period_id);

--#
CREATE INDEX idx_period_links ON links (period_id);

--#
SELECT CreateSpatialIndex( 'links' , 'geometry' );

--#
CREATE INDEX idx_link_anode ON links (a_node);

--#
CREATE INDEX idx_link_bnode ON links (b_node);

--#
CREATE INDEX idx_link_modes ON links (modes);

--#
CREATE INDEX idx_link_link_type ON links (link_type);

--#
CREATE INDEX idx_links_a_node_b_node ON links (a_node, b_node);

--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','link_id', 'Unique link ID within the period');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','period_id', 'Period of the graph');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','a_node', 'tail node of the link');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','b_node', 'head node of the link');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','direction', 'Flow direction allowed on the link');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','modes', 'modes allowed on the link');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','link_type', 'Link type');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','line_id', 'ID of the line the link belongs to');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','stop_id', 'ID of the stop the link belongs to ');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','line_seg_idx', 'Line segment index');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','segment_id', 'Physical segment, stop or stop pair the link belongs to');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','run_time', 'Mean scheduled run time (s)');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','dwell_time', 'Mean scheduled dwell at the stop (s)');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','walk_distance', 'Geodesic walking distance (m)');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','walk_time', 'Walking distance over walking speed (s)');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','headway', 'Scheduled headway of the line (s)');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','trip_count', 'Trips of the line in the period');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','line_freq', 'Distinct departures over the period length (1/s)');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','capacity', 'Capacity of the period trips');
"""


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection],
    results_conn: Optional[sqlite3.Connection],
):
    """Drop saved transit graphs, whose links ran from head to tail and stored costed times.

    Their costs were never saved, so they cannot be converted. Each configuration is logged as JSON, so the graph can
    be rebuilt with ``TransitGraphBuilder(project, **config)``. Supply tables are left untouched.
    """
    if transit_conn is None:
        raise RuntimeError("Transit migration 007 requires a transit_conn connection")

    tables = _tables(transit_conn)
    if "links" not in tables:
        logger.info("Migration finished, transit schema does not contain saved graphs.")
        return

    periods = [row[0] for row in transit_conn.execute("SELECT DISTINCT period_id FROM links ORDER BY period_id")]

    configs = []
    if project_conn is not None and "transit_graph_configs" in _tables(project_conn):
        configs = project_conn.execute("SELECT period_id, config FROM transit_graph_configs ORDER BY period_id")
        configs = configs.fetchall()
        for period_id, config in configs:
            logger.warning(f"Dropped the saved transit graph of period {period_id}. Its configuration was: {config}")
        project_conn.execute("DELETE FROM transit_graph_configs")
    elif periods:
        logger.warning("Saved transit graphs were dropped, but their configurations could not be read")

    logged = {period_id for period_id, _ in configs}
    for period_id in periods:
        if period_id not in logged:
            logger.warning(f"Dropped the saved transit graph of period {period_id}, which had no configuration")

    transit_conn.execute("DELETE FROM nodes")
    transit_conn.execute("SELECT DropTable(NULL, 'links')")
    for sql in LINKS_SQL.split("--#"):
        transit_conn.execute(sql)

    logger.info(f"Dropped {len(periods)} saved transit graph(s): {json.dumps(periods)}")
