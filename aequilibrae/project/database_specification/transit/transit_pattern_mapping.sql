--@ Information pertaining to GTFS map-matching is held on this table if the
--@ feed has been map-matched during the import process. This map-matching is
--@ required for running transit in traffic, but it is not required by the
--@ transit assignment per se.

CREATE TABLE IF NOT EXISTS "transit_pattern_mapping" (
	pattern_id	INTEGER NOT NULL,
	"index"	    INTEGER NOT NULL,
	link	    INTEGER NOT NULL,
	dir	        INTEGER NOT NULL,
	stop_id	    INTEGER,
	offset	    REAL,
	PRIMARY KEY(pattern_id,"index"),
	FOREIGN KEY(pattern_id) REFERENCES transit_routes(pattern_id) deferrable initially deferred,
	FOREIGN KEY(stop_id) REFERENCES transit_stops(stop_id) deferrable initially deferred,
	FOREIGN KEY(link) REFERENCES Link(link) deferrable initially deferred
);

CREATE INDEX IF NOT EXISTS transit_pattern_mapping_stop_id ON transit_pattern_mapping (stop_id);

SELECT AddGeometryColumn( 'transit_pattern_mapping', 'geo', SRID_PARAMETER, 'LINESTRING', 'XY');

SELECT CreateSpatialIndex( 'transit_pattern_mapping' , 'geo' );
