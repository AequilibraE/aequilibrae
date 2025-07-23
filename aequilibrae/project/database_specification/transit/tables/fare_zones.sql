--@ The *fare_zones* table hold information on the transit fare zones and
--@ the transit agencies that operate in it.
--@
--@ **fare_zone_id** identifies the fare zone
--@ 
--@ **transit_zone** identifies the transit fare zones
--@ 
--@ **agency_id** identifies the agency/agencies for the specified fare zone

CREATE TABLE IF NOT EXISTS fare_zones (
	fare_zone_id    INTEGER  PRIMARY KEY,
	transit_zone    TEXT,
	agency_id       INTEGER,
	FOREIGN KEY(agency_id) REFERENCES agencies(agency_id) deferrable initially deferred
);