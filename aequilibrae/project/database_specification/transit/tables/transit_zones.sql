CREATE TABLE IF NOT EXISTS fare_zones (
	fare_zone_id	INTEGER PRIMARY KEY,
	transit_zone	TEXT    NOT NULL,
	agency_id	    INTEGER NOT NULL,
	FOREIGN KEY(agency_id) REFERENCES agencies(agency_id) deferrable initially deferred
);