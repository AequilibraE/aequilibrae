CREATE TABLE IF NOT EXISTS trips_schedule (
	trip_id	  INTEGER  NOT NULL,
	seq  	  INTEGER  NOT NULL,
	arrival	  INTEGER  NOT NULL,
	departure INTEGER  NOT NULL,
	time_source INTEGER,
	PRIMARY KEY(trip_id,"seq"),
	FOREIGN KEY(trip_id) REFERENCES trips(trip_id) deferrable initially deferred
);
