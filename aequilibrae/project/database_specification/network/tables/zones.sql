--@ The *zones* table holds information on the Traffic Analysis Zones (TAZs) 
--@ in AequilibraE's model.
--@
--@ The **zone_id** field identifies the zone.
--@
--@ The **area** field corresponds to the area of the zone in **km2**.
--@ TAZs' area is automatically updated by triggers.
--@
--@ The **name** fields allows one to identity the zone using a name
--@ or any other description.

CREATE TABLE 'zones' (ogc_fid    INTEGER PRIMARY KEY,
                      zone_id    INTEGER UNIQUE NOT NULL,
                      area       NUMERIC,
                      "name"     TEXT);

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
