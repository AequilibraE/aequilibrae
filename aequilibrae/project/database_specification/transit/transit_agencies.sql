--@ The agencies table holds information on all the transit agencies
--@ for which there are transit services included in the model.
--@ information for these agencies include the GTFS feed (feed_date) and
--@ operation day (service_date) for which services were imported into the
--@ model.
--@
--@ Encoding of ids for transit agencies, routes, patterns and trips follows a
--@ strict encoding that allow one to trace back each element to its parent
--@ (Agency->Route->Pattern->Trip).
--@ This encoding follows the following pattern: AARRRPPTTT.
--@

create TABLE IF NOT EXISTS agencies (
	agency_id	  INTEGER NOT NULL  PRIMARY KEY AUTOINCREMENT,
	agency	      TEXT    NOT NULL,
	feed_date	  TEXT,
	service_date  TEXT,
	description   TEXT
);

create UNIQUE INDEX IF NOT EXISTS transit_operators_id ON agencies (agency_id);