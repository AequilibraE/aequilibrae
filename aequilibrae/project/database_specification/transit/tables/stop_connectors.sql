CREATE TABLE IF NOT EXISTS stop_connectors (
	id_from        INTEGER  NOT NULL,
	id_to          INTEGER  NOT NULL,
	conn_type      INTEGER  NOT NULL, -- Transfer, access_link, egress_link
	traversal_time INTEGER  NOT NULL,
	penalty_cost   INTEGER  NOT NULL);

--#
SELECT AddGeometryColumn('stop_connectors', 'geometry', 4326, 'LINESTRING', 'XY', 1);

--#
SELECT CreateSpatialIndex('stop_connectors' , 'geometry');

--#
CREATE INDEX IF NOT EXISTS stop_connectors_id_from ON stop_connectors (id_from);

--#
CREATE INDEX IF NOT EXISTS stop_connectors_id_to ON stop_connectors (id_to);