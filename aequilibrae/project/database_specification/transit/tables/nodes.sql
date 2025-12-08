--@ The *nodes* table holds all the network nodes available in AequilibraE transit model.
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
INSERT OR REPLACE INTO 'attributes_documentation' (name_table, attribute, description) VALUES('nodes','line_seg_idx', 'Index of the line segement this node belongs to');
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
