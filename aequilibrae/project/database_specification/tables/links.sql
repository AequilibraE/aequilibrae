CREATE TABLE  if not exists links (ogc_fid         INTEGER PRIMARY KEY,
                                   link_id         INTEGER NOT NULL UNIQUE,
                                   a_node          INTEGER,
                                   b_node          INTEGER,
                                   direction       INTEGER NOT NULL DEFAULT 0,
                                   distance        NUMERIC,
                                   modes           TEXT    NOT NULL,
                                   link_type       TEXT    REFERENCES link_types(link_type) ON update RESTRICT ON delete RESTRICT,
                                   'name'          TEXT,
                                   speed_ab        NUMERIC,
                                   speed_ba        NUMERIC,
                                   travel_time_ab  NUMERIC,
                                   travel_time_ba  NUMERIC,
                                   capacity_ab     NUMERIC,
                                   capacity_ba     NUMERIC
                                 );

--#
select AddGeometryColumn( 'links', 'geometry', 4326, 'LINESTRING', 'XY', 1);

--#
CREATE UNIQUE INDEX idx_link ON links (link_id);

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
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','link_id', 'Unique link ID');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','a_node', 'origin node for the link');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','b_node', 'destination node for the link');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','direction', 'Flow direction allowed on the link');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','distance', 'length of the link');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','modes', 'modes allowed on the link');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','link_type', 'Link type');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','name', 'Name of the street/link');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','speed_*', 'Directional speeds (if allowed)');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','capacity_*', 'Directional link capacities (if allowed)');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','travel_time_*', 'Directional free-flow travel time (if allowed)');

