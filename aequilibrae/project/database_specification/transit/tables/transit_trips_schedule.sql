--@ For each transit trip, this table lists the arrival and departure at each
--@ stop, listed in order by their index in the sequence available in
--@ the transit pattern links table
--@
--@ The transit trip ID can be traced back to the pattern, route and agency
--@ directly through the encoding of their trip_id, as explained in the
--@ documentation for the agencies table.
--@
--
--@ The time_source field indicates whether the stop timing came from the GTFS
--@ feed (**0**) or from any pre-processing (i.e. stop_time_de-duplication) (**1**).

CREATE TABLE IF NOT EXISTS trips_schedule (
	trip_id	  INTEGER  NOT NULL,
	seq  	  INTEGER  NOT NULL,
	arrival	  INTEGER  NOT NULL,
	departure INTEGER  NOT NULL,
	PRIMARY KEY(trip_id,"seq"),
	FOREIGN KEY(trip_id) REFERENCES trips(trip_id) deferrable initially deferred
);
