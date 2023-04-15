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