CREATE TABLE IF NOT EXISTS pattern_mapping (
	pattern_id  INTEGER    NOT NULL,
	seq         INTEGER    NOT NULL,
	link        INTEGER    NOT NULL,
	dir         INTEGER    NOT NULL,
	PRIMARY KEY(pattern_id, "seq"),
	FOREIGN KEY(pattern_id) REFERENCES routes (pattern_id) deferrable initially deferred,
	FOREIGN KEY(link) REFERENCES route_links (link) deferrable initially deferred
);

--#
SELECT AddGeometryColumn( 'pattern_mapping', 'geometry', 4326, 'LINESTRING', 'XY');

--#
SELECT CreateSpatialIndex( 'pattern_mapping' , 'geometry' );
