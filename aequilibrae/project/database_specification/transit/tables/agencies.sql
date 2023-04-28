--@ The *agencies* table holds information about the Public Transport
--@ agencies within the GTFS data. This table information comes from
--@ GTFS file *agency.txt*.
--@ You can check out more information `here <https://developers.google.com/transit/gtfs/reference#agencytxt>`_.
--@ 
--@ **agency_id** identifies the agency for the specified route
--@
--@ **agency** contains the fuill name of the transit agency
--@
--@ **feed_date** idicates the date for which the GTFS feed is being imported
--@
--@ **service_date** indicates the date for the indicate route scheduling
--@
--@ **description_field** provides useful description of a transit agency

create TABLE IF NOT EXISTS agencies (
	agency_id     INTEGER  NOT NULL PRIMARY KEY AUTOINCREMENT,
	agency        TEXT     NOT NULL,
	feed_date     TEXT,
	service_date  TEXT,
	description   TEXT
);

--#
create UNIQUE INDEX IF NOT EXISTS transit_operators_id ON agencies (agency_id);