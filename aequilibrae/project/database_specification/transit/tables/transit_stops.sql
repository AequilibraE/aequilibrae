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
	zone			  TEXT,
	has_parking       INTEGER,
	moved_by_matching INTEGER,
	fare_zone_id   	  INTEGER,
	route_type        INTEGER NOT NULL DEFAULT -1,
	FOREIGN KEY(agency_id) REFERENCES agencies(agency_id),
	FOREIGN KEY("fare_zone_id") REFERENCES fare_zones("fare_zone_id")
);

--#
create INDEX IF NOT EXISTS stops_stop_id ON stops (stop_id);

--#
select AddGeometryColumn( 'stops', 'geometry', 4326, 'POINT', 'XY', 1);

--#
select CreateSpatialIndex( 'stops' , 'geometry' );
