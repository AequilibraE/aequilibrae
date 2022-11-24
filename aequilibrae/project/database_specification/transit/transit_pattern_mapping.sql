--@ Information pertaining to GTFS map-matching is held on this table if the
--@ feed has been map-matched during the import process. This map-matching is
--@ required for running transit in traffic, but it is not required by the
--@ transit assignment per se.

CREATE TABLE IF NOT EXISTS pattern_mapping (
	pattern_id	INTEGER NOT NULL,
	seq         INTEGER NOT NULL,
	link	    INTEGER NOT NULL,
	dir	        INTEGER NOT NULL,
	PRIMARY KEY(pattern_id,"index"),
	FOREIGN KEY(pattern_id) REFERENCES routes(pattern_id) deferrable initially deferred,
	FOREIGN KEY(link) REFERENCES Link(link) deferrable initially deferred
);

SELECT AddGeometryColumn( 'pattern_mapping', 'geometry', SRID_PARAMETER, 'LINESTRING', 'XY');

SELECT CreateSpatialIndex( 'pattern_mapping' , 'geometry' );
