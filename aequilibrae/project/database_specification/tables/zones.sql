CREATE TABLE 'zones' (ogc_fid    INTEGER PRIMARY KEY,
                      zone_id    INTEGER UNIQUE NOT NULL,
                      area       NUMERIC,
                      "name"     TEXT,
                      population INTEGER,
                      employment INTEGER);

--#
SELECT AddGeometryColumn( 'zones', 'geometry', 4326, 'MULTIPOLYGON', 'XY', 1);
--#
CREATE UNIQUE INDEX idx_zone ON zones (zone_id);
--#
SELECT CreateSpatialIndex( 'zones' , 'geometry' );
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('zones','zone_id', 'Unique node ID');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('zones','area', 'Area of the zone in km2');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('zones','name', 'Name of the zone, if any');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('zones','population', "Zone's total population");
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('zones','employment', "Zone's total employment");
