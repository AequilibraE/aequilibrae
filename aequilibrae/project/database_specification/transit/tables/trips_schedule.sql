--@ The *trips_schedule* table holds information on the sequence of stops
--@ of a trip.
--@ 
--@ **trip_id** is an unique identifier of a trip
--@ 
--@ **seq** identifies the sequence of the stops for a trip
--@ 
--@ **arrival** identifies the arrival time at the stop
--@ 
--@ **departure** identifies the departure time at the stop

CREATE TABLE IF NOT EXISTS trips_schedule (
	trip_id   INTEGER  NOT NULL,
	seq       INTEGER  NOT NULL,
	arrival   INTEGER  NOT NULL,
	departure INTEGER  NOT NULL,
	PRIMARY KEY(trip_id,"seq"),
	FOREIGN KEY(trip_id) REFERENCES trips(trip_id) deferrable initially deferred
);
