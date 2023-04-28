--@ The *stops_connectors* table holds information on the connection of
--@ the GTFS network with the real network.
--@
--@ **id_from** identifies the network link the vehicle departs
--@
--@ **id_to** identifies the network link th vehicle is heading to
--@
--@ **conn_type** identifies the type of connection used to connect the links
--@
--@ **traversal_time** represents the time spent crossing the link
--@
--@ **penalty_cost** identifies the penalty in the connection


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