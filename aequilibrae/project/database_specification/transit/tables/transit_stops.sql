--@ Lists all the transit stops in the model on which at least one transit trip
--@ stops during the day. It  lists the agency it is associated to, as well as
--@ the closest network link to it
--@
--@ Additionally to transit stops, this table also holds nodes in the network
--@ associated with the active networks (walk and bike), more specifically the
--@ nodes created in the network to allow transit stops to be linked in what was
--@ previously the middle of a network link.

--@ If a node had to be moved during the GTFS map-matching (probably due to a
--@ too-sparse of a network), then the moved_by_matching field will contain the
--@ straight-line distance the stop was moved.


CREATE TABLE IF NOT EXISTS stops (
	stop_id	          INTEGER PRIMARY KEY AUTOINCREMENT ,
	stop	          TEXT    NOT NULL ,
	agency_id         INTEGER NOT NULL,
	link	          INTEGER,
	dir	              INTEGER,
	name	          TEXT,
	parent_station    TEXT,
	description	      TEXT,
	street	          TEXT,
	fare_zone_id   INTEGER,
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
