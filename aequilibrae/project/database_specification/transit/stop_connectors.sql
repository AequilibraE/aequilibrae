--@ At GTFS import we create stop-to-node (to the closest node)
--@ At connector_building step we would build the connection between stops close to each other AND
--@ the connections between the stop and the closest centroid
--@ conn_type: centroid-to-stop, stop-to-centroid, stop-to-stop
--@
--@
--@
--@

CREATE TABLE IF NOT EXISTS stop_connectors (
	id_from        INTEGER  NOT NULL,
	id_to          INTEGER  NOT NULL,
	conn_type      INTEGER  NOT NULL, -- Transfer, access_link, egress_link
	traversal_time INTEGER  NOT NULL,
	penalty_cost   INTEGER  NOT NULL);

SELECT AddGeometryColumn('stop_connectors', 'geometry', SRID_PARAMETER, 'LINESTRING', 'XY', 1);

SELECT CreateSpatialIndex('stop_connectors' , 'geometry');

CREATE INDEX IF NOT EXISTS stop_connectors_id_from ON stop_connectors (id_from);

CREATE INDEX IF NOT EXISTS stop_connectors_id_to ON stop_connectors (id_to);
