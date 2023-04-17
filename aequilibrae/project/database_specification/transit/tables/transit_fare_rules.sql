--@ The *fare_rules* table holds information about the fare values.
--@ This table information comes from the GTFS file *fare_rules.txt*.
--@ Given that this file is optional in GTFS, it can be empty.
--@ 
--@ The **fare_id** identifies a fare class
--@ 
--@ The **route_id** identifies a route associated with the fare class.
--@ 
--@ The **origin** field identifies the origin zone
--@ 
--@ The **destination** field identifies the destination zone
--@ 
--@ The **contains** field identifies the zones that a rider will enter while using
--@ a given fare class.

create TABLE IF NOT EXISTS fare_rules (
	fare_id     INTEGER  NOT NULL,
	route_id    INTEGER,
	origin      INTEGER,
	destination INTEGER,
	contains    INTEGER,
	FOREIGN KEY(fare_id) REFERENCES fare_attributes(fare_id) deferrable initially deferred,
	FOREIGN KEY(route_id) REFERENCES routes(route_id) deferrable initially deferred,
	FOREIGN KEY(destination) REFERENCES fare_zones(fare_zone_id) deferrable initially deferred,
	FOREIGN KEY(origin) REFERENCES fare_zones(fare_zone_id) deferrable initially deferred,
	FOREIGN KEY(contains) REFERENCES fare_zones(fare_zone_id) deferrable initially deferred
);