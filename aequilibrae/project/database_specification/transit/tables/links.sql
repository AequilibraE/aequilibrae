--@ The links table holds all the links available in the aequilibrae network model
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
                                  stop_id         TEXT    REFERENCES stops(stop) ON update RESTRICT ON delete RESTRICT,
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
