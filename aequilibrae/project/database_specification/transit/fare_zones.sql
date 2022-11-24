--@ Transit fare zones, when applicable, are listed in this table.
--@
--@ No geometry is provided, but the information of transit zone is also
--@ available on stops whenever fares are zone based for the agency in question.
--@

CREATE TABLE IF NOT EXISTS fare_zones (
	fare_zone_id	INTEGER NOT NULL PRIMARY KEY,
	transit_zone	TEXT    NOT NULL,
	agency_id	    INTEGER NOT NULL,
	FOREIGN KEY(agency_id) REFERENCES agencies(agency_id) deferrable initially deferred
);