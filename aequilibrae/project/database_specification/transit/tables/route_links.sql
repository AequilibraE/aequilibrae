--@ The *route_links* table holds information on the links of a route.
--@ 
--@ **transit_link** identifies the GTFS transit links for the route
--@ 
--@ **pattern_id** is an unique pattern for the route
--@ 
--@ **seq** identifies the sequence of the stops for a trip
--@ 
--@ **from_stop** identifies the stop the vehicle is departing
--@ 
--@ **to_stop** identifies the next stop the vehicle is going to arrive
--@ 
--@ **distance** identifies the distance (in meters) the vehicle travel
--@ between the stops

CREATE TABLE IF NOT EXISTS route_links (
	transit_link            INTEGER    NOT NULL,
	pattern_id              INTEGER    NOT NULL,
	seq  	                INTEGER    NOT NULL,
	from_stop               INTEGER    NOT NULL,
	to_stop                 INTEGER    NOT NULL,
	distance                INTEGER    NOT NULL,
	FOREIGN KEY(pattern_id) REFERENCES "routes"(pattern_id) deferrable initially deferred,
	FOREIGN KEY(from_stop)  REFERENCES "stops"(stop_id) deferrable initially deferred
	FOREIGN KEY(to_stop)    REFERENCES "stops"(stop_id) deferrable initially deferred
);

--#
create UNIQUE INDEX IF NOT EXISTS route_links_stop_id ON route_links (pattern_id, transit_link);

--#
select AddGeometryColumn( 'route_links', 'geometry', 4326, 'LINESTRING', 'XY');

--#
select CreateSpatialIndex( 'route_links' , 'geometry' );