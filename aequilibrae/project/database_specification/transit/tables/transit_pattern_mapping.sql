--@ Information pertaining to GTFS map-matching is held on this table if the
--@ feed has been map-matched during the import process. This map-matching is
--@ required for running transit in traffic, but it is not required by the
--@ transit assignment per se.

CREATE TABLE IF NOT EXISTS pattern_mapping (
	pattern_id	INTEGER NOT NULL,
	seq         INTEGER NOT NULL,
	link	    INTEGER NOT NULL,
	dir	        INTEGER NOT NULL,
	stop_id		INTEGER,
	"offset"    REAL,
	-- PRIMARY KEY(pattern_id, "seq"),
	FOREIGN KEY (stop_id) REFERENCES stops(stop_id),
	FOREIGN KEY(pattern_id) REFERENCES routes(pattern_id) deferrable initially deferred
	-- FOREIGN KEY(link) REFERENCES routes_links(link) deferrable initially deferred
);

--#
SELECT AddGeometryColumn( 'pattern_mapping', 'geometry', 4326, 'LINESTRING', 'XY');

--#
SELECT CreateSpatialIndex( 'pattern_mapping' , 'geometry' );
