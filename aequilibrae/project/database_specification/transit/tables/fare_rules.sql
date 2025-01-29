--@ The *fare_rules* table holds information about the fare values.
--@ This table information comes from the GTFS file *fare_rules.txt*.
--@ Given that this file is optional in GTFS, it can be empty.
--@ 
--@ The **fare_id** identifies a fare class
--@ 
--@ The **route_id** identifies a route associated with the fare class
--@ 
--@ The **origin** field identifies the transit fare zone for origin
--@ 
--@ The **destination** field identifies the transit fare zone for destination
--@ 
--@ The **contains** field identifies the zones that a rider will enter while using
--@ a given fare class.

create TABLE IF NOT EXISTS fare_rules (
	fare_id     INTEGER  NOT NULL,
	route_id    INTEGER,
	origin      TEXT,
	destination TEXT,
	contains    INTEGER,
	FOREIGN KEY(fare_id) REFERENCES fare_attributes(fare_id) deferrable initially deferred,
	FOREIGN KEY(route_id) REFERENCES routes(route_id) deferrable initially deferred
);