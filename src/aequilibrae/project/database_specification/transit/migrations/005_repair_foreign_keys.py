# ruff: noqa: E501, W291 (the inlined SQL is kept as it was written)
import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

# The schema as it stood when this migration was written. Later changes to the files in tables/ reach existing
# projects through their own migrations, so this one must not read them.

LINKS_SQL = r"""--@ The links table holds all the links available in the aequilibrae network model
--@ regardless of the modes allowed on it.
--@
--@ All information on the fields a_node and b_node correspond to a entries in
--@ the node_id field in the nodes table. They are automatically managed with
--@ triggers as the user edits the network, but they are not protected by manual
--@ editing, which would break the network if it were to happen.
--@
--@ The **modes** field is a concatenation of all the ids (mode_id) of the models allowed
--@ on each link, and map directly to the mode_id field in the **Modes** table. A mode
--@ can only be added to a link if it exists in the **Modes** table.
--@
--@ The **link_type** corresponds to the *link_type* field from the *link_types* table.
--@ As it is the case for modes, a link_type can only be assigned to a link if it exists
--@ in the **link_types** table.
--@
--@ The fields **length**, **node_a** and **node_b** are automatically
--@ updated by triggers based in the links' geometries and node positions. Link length
--@ is always measured in **meters**.
--@
--@ The table is indexed on **link_id** (its primary key), **node_a** and **node_b**.


CREATE TABLE if not exists links (ogc_fid         INTEGER PRIMARY KEY,
                                  link_id         INTEGER NOT NULL,
                                  period_id       INTEGER NOT NULL,
                                  a_node          INTEGER,
                                  b_node          INTEGER,
                                  direction       INTEGER NOT NULL DEFAULT 0,
                                  distance        NUMERIC,
                                  modes           TEXT    NOT NULL,
                                  link_type       TEXT    REFERENCES link_types(link_type) ON update RESTRICT ON delete RESTRICT,
                                  line_id         TEXT,
                                  stop_id         TEXT    REFERENCES stops(stop_id) ON update RESTRICT ON delete RESTRICT,
                                  line_seg_idx    INTEGER,
                                  trav_time       NUMERIC  NOT NULL,
                                  freq            NUMERIC  NOT NULL,
                                  o_line_id       TEXT,
                                  d_line_id       TEXT,
                                  transfer_id     TEXT,
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
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','link_id', 'Unique link ID');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','a_node', 'origin node for the link');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','b_node', 'destination node for the link');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','direction', 'Flow direction allowed on the link');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','distance', 'length of the link');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','modes', 'modes allowed on the link');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','link_type', 'Link type');
-- --#
--  INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','name', 'Name of the street/link');
-- --#
--  INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','speed_*', 'Directional speeds (if allowed)');
-- --#
--  INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','capacity_*', 'Directional link capacities (if allowed)');
-- --#
--  INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','travel_time_*', 'Directional free-flow travel time (if allowed)');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','line_id', 'ID of the line the link belongs to');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','stop_id', 'ID of the stop the link belongs to ');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','line_seg_idx', 'Line segment index');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','trav_time', 'Travel time');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','freq', 'Frequency of link traversal');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','*_line_id', 'Origin/Destination line ID for transfer links');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','transfer_id', 'Transfer link ID');
--#
-- INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','period_start', 'Graph start time');
-- --#
-- INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','period_end', 'Graph end period');
"""

FARE_RULES_SQL = r"""--@ The *fare_rules* table holds information about the fare values.
--@ This table information comes from the GTFS file *fare_rules.txt*.
--@ Given that this file is optional in GTFS, it can be empty.
--@ 
--@ The **fare_id** identifies a fare class
--@ 
--@ The **route_id** identifies a route associated with the fare class
--@ 
--@ The **origin** field identifies the transit fare zone for origin
--@ 
--@ The **destination** field identifies the transit fare zone for destination
--@ 
--@ The **contains** field identifies the zones that a rider will enter while using
--@ a given fare class.

create TABLE IF NOT EXISTS fare_rules (
	fare_id     INTEGER  NOT NULL,
	route_id    INTEGER,
	origin      TEXT,
	destination TEXT,
	contains    INTEGER,
	FOREIGN KEY(fare_id) REFERENCES fare_attributes(fare_id) deferrable initially deferred
);"""

PATTERN_MAPPING_SQL = r"""--@ The *pattern_mapping* table holds information on the stop pattern 
--@ for each route.
--@ 
--@ **pattern_id** is an unique pattern for the route
--@ 
--@ **seq** identifies the sequence of the stops for a trip
--@
--@ **link** identifies the *link_id* in the links table that corresponds to the
--@ pattern matching
--@ 
--@ **dir** indicates the direction of travel for a trip

CREATE TABLE IF NOT EXISTS pattern_mapping (
	pattern_id  INTEGER    NOT NULL,
	seq         INTEGER    NOT NULL,
	link        INTEGER    NOT NULL,
	dir         INTEGER    NOT NULL,
	PRIMARY KEY(pattern_id, "seq"),
	FOREIGN KEY(pattern_id) REFERENCES routes (pattern_id) deferrable initially deferred
);

--#
SELECT AddGeometryColumn( 'pattern_mapping', 'geometry', 4326, 'LINESTRING', 'XY');

--#
SELECT CreateSpatialIndex( 'pattern_mapping' , 'geometry' );
"""

SCHEMA = {"links": LINKS_SQL, "fare_rules": FARE_RULES_SQL, "pattern_mapping": PATTERN_MAPPING_SQL}


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})") if row[1] != "ogc_fid"]


def _rebuild_child_table(conn: sqlite3.Connection, table: str) -> None:
    """Recreate a leaf table after retaining its rows outside the schema."""
    columns = _columns(conn, table)
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    rows = conn.execute(f'SELECT {quoted_columns} FROM "{table}"').fetchall()

    conn.execute(f"SELECT DropTable(NULL, '{table}')")
    for sql in SCHEMA[table].split("--#"):
        conn.execute(sql)
    if rows:
        placeholders = ", ".join("?" for _ in columns)
        conn.executemany(f'INSERT INTO "{table}" ({quoted_columns}) VALUES ({placeholders})', rows)


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection],
    results_conn: Optional[sqlite3.Connection],
):
    """Repair invalid transit foreign keys without relaxing FK enforcement."""
    if transit_conn is None:
        raise RuntimeError("Transit migration 005 requires a transit_conn connection")

    existing_tables = {row[0] for row in transit_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required_tables = {"stops", "links", "fare_rules", "pattern_mapping"}
    if not required_tables <= existing_tables:
        logger.info("Migration finished, transit schema does not contain the affected tables.")
        return

    stop_columns = _columns(transit_conn, "stops")
    for column, declared_type in (("zone_id", "INTEGER"), ("transit_fare_zone", "TEXT")):
        if column not in stop_columns:
            transit_conn.execute(f"ALTER TABLE stops ADD COLUMN {column} {declared_type}")

    _rebuild_child_table(transit_conn, "links")
    _rebuild_child_table(transit_conn, "fare_rules")
    _rebuild_child_table(transit_conn, "pattern_mapping")

    logger.info("Repaired transit foreign-key definitions")
