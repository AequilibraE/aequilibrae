# ruff: noqa: E501, W291 (the inlined SQL is kept as it was written)
import logging
import sqlite3
from typing import Optional

from aequilibrae.project.project_creation import recreate_columns

logger = logging.getLogger(__name__)

# The schema and triggers as they stood when this migration was written. Later changes to the files in tables/ and
# triggers/ reach existing projects through their own migrations, so this one must not read them.

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

NODES_SQL = r"""--@ The *nodes* table holds all the network nodes available in AequilibraE transit model.
--@
--@ The **node_id** field is an identifier of the node.
--@
--@ The **is_centroid** field holds information if the node is a centroid
--@ of a network or not. Assumes values 0 or 1. Defaults to **0**.
--@
--@ The **stop_id** field indicates which stop this node belongs too. This field
--@ is TEXT as it might encode a street name or such.
--@
--@ The **line_id** field indicates which line this node belongs too. This field
--@ is TEXT as it might encode a street name or such.
--@
--@ The **line_seg_idx** field indexes the segment of line **line_id**. Zero based.
--@
--@ The **modes** field identifies all modes connected to the node.
--@
--@ The **link_type** field identifies all link types connected to the node.
--@
--@ The **node_type** field identifies the types of this node.
--@
--@ The **taz_id** field is an identifier for the transit assignment zone this node
--@ belongs to.
--@

CREATE TABLE if not exists nodes (ogc_fid      INTEGER PRIMARY KEY,
                                  node_id      INTEGER NOT NULL,
                                  period_id    INTEGER NOT NULL,
                                  is_centroid  INTEGER        NOT NULL DEFAULT 0,
                                  stop_id      TEXT,
                                  line_id      TEXT,
                                  line_seg_idx INTEGER,
                                  modes        TEXT,
                                  link_types   TEXT,
                                  node_type    TEXT,
                                  taz_id       INTEGER,
                                  CHECK(TYPEOF(taz_id) == 'integer')
                                  CHECK(TYPEOF(node_id) == 'integer')
                                  CHECK(TYPEOF(is_centroid) == 'integer')
                                  CHECK(is_centroid>=0)
                                  CHECK(is_centroid<=1));

--#
SELECT AddGeometryColumn( 'nodes', 'geometry', 4326, 'POINT', 'XY', 1);

--#
SELECT CreateSpatialIndex( 'nodes' , 'geometry' );

--#
CREATE INDEX idx_node ON nodes (node_id, period_id);

--#
CREATE INDEX idx_period_nodes ON nodes (period_id);

--#
CREATE INDEX idx_node_is_centroid ON nodes (is_centroid);

--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('nodes','node_id', 'Unique node ID');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('nodes','is_centroid', 'Flag identifying centroids');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('nodes','stop_id', 'ID of the Stop this node belongs to');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('nodes','line_id', 'ID of the Line this node belongs to');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('nodes','line_seg_idx', 'Index of the line segment this node belongs to');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('nodes','modes', 'Modes connected to the node');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('nodes','link_types', 'Link types connected to the node');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('nodes','node_type', 'Node types of this node');
--#
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('nodes','taz_id', 'Transit assignment zone id');
--#
-- INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('nodes','period_start', 'Graph start time');
-- --#
-- INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('nodes','period_end', 'Graph end time');
"""

TRANSIT_TRIGGERS_SQL = r"""create trigger aequilibrae_enforces_route_link_length_update after update of distance on route_links
begin
  update route_links set distance = GeodesicLength(new.geometry)
  where route_links.rowid = new.rowid;
end;

--#
create trigger aequilibrae_enforces_route_link_geo_update after update of geometry on route_links
begin
  update route_links set distance = GeodesicLength(new.geometry)
  where route_links.rowid = new.rowid;
end;

--#
create trigger aequilibrae_new_route_link after insert on route_links
begin
  update route_links set distance = GeodesicLength(new.geometry)
  where route_links.rowid = new.rowid;
end;"""

TRANSIT_TRIGGERS = [
    "aequilibrae_enforces_route_link_length_update",
    "aequilibrae_enforces_route_link_geo_update",
    "aequilibrae_new_route_link",
]


def migrate(
    *,
    project_conn: sqlite3.Connection,
    transit_conn: Optional[sqlite3.Connection],
    results_conn: Optional[sqlite3.Connection],
):
    if project_conn is None:
        raise RuntimeError("Transit migration 002 requires a project_conn connection")
    if transit_conn is None:
        raise RuntimeError("Transit migration 002 requires a transit_conn connection")

    logger.info("Beginning migration to support saving and loading multiple Transit graphs")

    try:
        period_ids = project_conn.execute("SELECT period_id FROM transit_graph_configs").fetchall()
    except sqlite3.OperationalError:
        logger.info("Migration finished, no 'transit_graph_configs' table found.")
        return

    existing_links = transit_conn.execute("SELECT link_id FROM links LIMIT 1").fetchone()

    if len(period_ids) > 1:
        raise ValueError(
            "more than one period_id found in 'transit_graph_configs' cannot migrate with multiple possible period_ids"
        )
    elif len(period_ids) == 0:
        if existing_links is not None:
            raise ValueError("no period_id found in 'transit_graph_configs' cannot migrate with without period_id")
        else:
            period_id = 1
    else:
        period_id = period_ids[0][0]

    sqls = [
        "DROP INDEX idx_link",
        "DROP INDEX idx_link_anode",
        "DROP INDEX idx_link_bnode",
        "DROP INDEX idx_link_modes",
        "DROP INDEX idx_link_link_type",
        "DROP INDEX idx_links_a_node_b_node",
        "SELECT RenameTable(NULL, 'links', '__old_links')",
        "DROP INDEX idx_node",
        "DROP INDEX idx_node_is_centroid",
        "SELECT RenameTable(NULL, 'nodes', '__old_nodes')",
        *LINKS_SQL.split("--#"),
        *NODES_SQL.split("--#"),
    ]

    for trigger in TRANSIT_TRIGGERS:
        transit_conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    try:
        for sql in sqls:
            transit_conn.execute(sql)

        for table in ["links", "nodes"]:
            columns = recreate_columns(transit_conn, table, f"__old_{table}")

            transit_conn.execute(
                f"INSERT INTO {table}({','.join(columns)},'period_id') "
                f"SELECT {','.join(columns)},{period_id} FROM __old_{table}"
            )
            transit_conn.execute(f"SELECT DropTable(NULL, '__old_{table}')")

    finally:
        for sql in TRANSIT_TRIGGERS_SQL.split("--#"):
            transit_conn.execute(sql)

    logger.info("Migration successful")
