--@ The *routes* table holds information on the available transit routes for a
--@ specific day. This table information comes from the GTFS file *routes.txt*.
--@ You can find more information about it `here <https://developers.google.com/transit/gtfs/reference#routestxt>`_.
--@
--@ **pattern_id** is an unique pattern for the route
--@
--@ **route_id** identifies a route
--@
--@ **route** identifies the name of a route
--@ 
--@ **agency_id** identifies the agency for the specified route
--@ 
--@ **shortname** identifies the short name of a route
--@ 
--@ **longname** identifies the long name of a route
--@ 
--@ **description** provides useful description of a route
--@ 
--@ **route_type** indicates the type of transporation used on a route
--@ 
--@ **seated_capacity** indicates the seated capacity of a route
--@ 
--@ **total_capacity** indicates the total capacity of a route

CREATE TABLE IF NOT EXISTS routes (
	pattern_id      INTEGER  NOT NULL PRIMARY KEY AUTOINCREMENT,
	route_id        INTEGER  NOT NULL,
	route           TEXT     NOT NULL,
	agency_id       INTEGER  NOT NULL,
	shortname       TEXT,
	longname        TEXT,
	description     TEXT,
	route_type      INTEGER  NOT NULL,
	seated_capacity INTEGER,
	total_capacity  INTEGER,
	FOREIGN KEY(agency_id) REFERENCES agencies(agency_id) deferrable initially deferred
);

--#
select AddGeometryColumn( 'routes', 'geometry', 4326, 'MULTILINESTRING', 'XY');

--#
select CreateSpatialIndex( 'routes' , 'geometry' );