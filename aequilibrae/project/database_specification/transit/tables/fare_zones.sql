--@ The *zones* tables holds information on the fare transit zones and
--@ the TAZs they are in.
--@ 
--@ **fare_zone_id** identifies the fare zone for a stop
--@ 
--@ **transit_zone** identifies the TAZ for a fare zone
--@ 
--@ **agency_id** identifies the agency fot the specified route

CREATE TABLE IF NOT EXISTS fare_zones (
	fare_zone_id    INTEGER  PRIMARY KEY,
	transit_zone    TEXT     NOT NULL,
	agency_id       INTEGER  NOT NULL,
	FOREIGN KEY(agency_id) REFERENCES agencies(agency_id) deferrable initially deferred
);