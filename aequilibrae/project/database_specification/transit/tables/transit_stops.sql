CREATE TABLE IF NOT EXISTS stops (
	stop_id	          INTEGER PRIMARY KEY AUTOINCREMENT ,
	stop	          TEXT    NOT NULL ,
	agency_id         INTEGER NOT NULL,
	link	          INTEGER,
	dir	              INTEGER,
	"offset"		  INTEGER,
	setback			  INTEGER,
	X			  	  REAL,
	Y			  	  REAL,
	Z			  	  REAL,
	name	          TEXT,
	parent_station    TEXT,
	description	      TEXT,
	street	          TEXT,
	taz				  INTEGER,
	zone			  TEXT,
	has_parking       INTEGER,
	route_type        INTEGER NOT NULL DEFAULT -1,
	moved_by_matching INTEGER,
	FOREIGN KEY(agency_id) REFERENCES agencies(agency_id)
);

--#
create INDEX IF NOT EXISTS stops_stop_id ON stops (stop_id);

--#
select AddGeometryColumn( 'stops', 'geometry', 4326, 'POINT', 'XY', 1);

--#
select CreateSpatialIndex( 'stops' , 'geometry' );
