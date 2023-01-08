create TABLE IF NOT EXISTS fare_rules (
	fare_id	    INTEGER NOT NULL,
	route_id	INTEGER,
	origin	    INTEGER,
	destination	INTEGER,
	contains    INTEGER,
	FOREIGN KEY(fare_id) REFERENCES fare_attributes(fare_id) deferrable initially deferred,
	FOREIGN KEY(route_id) REFERENCES routes(route_id) deferrable initially deferred,
	FOREIGN KEY(destination) REFERENCES fare_zones(fare_zone_id) deferrable initially deferred,
	FOREIGN KEY(origin) REFERENCES fare_zones(fare_zone_id) deferrable initially deferred,
	FOREIGN KEY(contains) REFERENCES fare_zones(fare_zone_id) deferrable initially deferred
);