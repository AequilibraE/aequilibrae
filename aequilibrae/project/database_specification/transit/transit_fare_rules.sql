--@ Transit fares can be associated to routes, in cases where routes have flat
--@ fares regardless of boarding and alighting stops. The other possible
--@ association is through the existing of fare zones, where trips are charged
--@ based on the combination of the transit zones for boarding and alighting.
--@
--@ This table includes both associations, thus records would have either the
--@ route_id or origin/destination fields as null.

create TABLE IF NOT EXISTS "transit_fare_rules" (
	fare_id	    INTEGER NOT NULL,
	route_id	INTEGER,
	origin	    INTEGER,
	destination	INTEGER,
	contains    INTEGER,
	FOREIGN KEY(fare_id) REFERENCES transit_fare_attributes(fare_id) deferrable initially deferred,
	FOREIGN KEY(route_id) REFERENCES transit_routes(route_id) deferrable initially deferred,
	FOREIGN KEY(destination) REFERENCES transit_zones(transit_zone_id) deferrable initially deferred,
	FOREIGN KEY(origin) REFERENCES transit_zones(transit_zone_id) deferrable initially deferred,
	FOREIGN KEY(contains) REFERENCES transit_zones(transit_zone_id) deferrable initially deferred
);
