CREATE TABLE IF NOT EXISTS routes (
	route_id        INTEGER  NOT NULL,
	pattern_id      INTEGER  NOT NULL PRIMARY KEY,
	pattern			TEXT,
	route	        TEXT     NOT NULL,
	agency_id	    INTEGER  NOT NULL,
	shortname       TEXT,
	longname	    TEXT,
	description     TEXT,
	route_type      INTEGER  NOT NULL,
	seated_capacity	INTEGER,
	design_capacity INTEGER,
	total_capacity	INTEGER,
	number_of_cars  INTEGER,
	FOREIGN KEY(agency_id) REFERENCES agencies(agency_id) deferrable initially deferred
);

--#
select AddGeometryColumn( 'routes', 'geometry', 4326, 'MULTILINESTRING', 'XY');

--#
select CreateSpatialIndex( 'routes' , 'geometry' );