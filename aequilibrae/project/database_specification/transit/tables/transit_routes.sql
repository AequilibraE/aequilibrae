--@ The transit routes correspond to the routes table in GTFS feeds, but this
--@ table includes only those routes for which that are active services for the
--@ for which services have been imported. Descriptive information, as well as
--@ capacity is included in this table, although the latter can be overwritten
--@ if information is provided in the routes or trips tables.
--@
--@ The routes can be traced back to the agency directly through the encoding of
--@ their trip_id, as explained in the documentation for the agencies
--@ table.

CREATE TABLE IF NOT EXISTS routes (
	route_id        INTEGER  NOT NULL,
	pattern_id      INTEGER  NOT NULL PRIMARY KEY AUTOINCREMENT,
	route	        TEXT     NOT NULL,
	agency_id	    INTEGER  NOT NULL,
	shortname       TEXT,
	longname	    TEXT,
	description     TEXT,
	route_type      INTEGER  NOT NULL,
	seated_capacity	INTEGER,
	total_capacity	INTEGER,
	FOREIGN KEY(agency_id) REFERENCES agencies(agency_id) deferrable initially deferred
);

select AddGeometryColumn( 'routes', 'geometry', SRID_PARAMETER, 'LINESTRING', 'XY');

select CreateSpatialIndex( 'routes' , 'geometry' );