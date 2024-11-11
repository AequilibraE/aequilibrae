--@ The *fare_zones* table hold information on the transit fare zones and
--@ the transit agencies that operate in it.
--@
--@ **transit_fare_zone** identifies the transit fare zones
--@ 
--@ **agency_id** identifies the agency/agencies for the specified fare zone

CREATE TABLE IF NOT EXISTS fare_zones (
	transit_fare_zone    TEXT    NOT NULL,
	agency_id            INTEGER NOT NULL,
	FOREIGN KEY(agency_id) REFERENCES agencies(agency_id) deferrable initially deferred
);