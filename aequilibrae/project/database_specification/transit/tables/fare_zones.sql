--@ The *zones* tables holds information on the fare transit zones and
--@ the TAZs they are in.
--@ 
--@ **fare_zone_id** identifies the fare zone for a stop
--@ 
--@ **transit_zone** identifies the TAZ for a fare zone

CREATE TABLE IF NOT EXISTS fare_zones (
	fare_zone_id    INTEGER  NOT NULL,
	transit_zone    TEXT     NOT NULL
);