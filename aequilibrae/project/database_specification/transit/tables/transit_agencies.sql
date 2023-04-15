create TABLE IF NOT EXISTS agencies (
	agency_id     INTEGER  NOT NULL PRIMARY KEY AUTOINCREMENT,
	agency        TEXT     NOT NULL,
	feed_date     TEXT,
	service_date  TEXT,
	description   TEXT
);

--#
create UNIQUE INDEX IF NOT EXISTS transit_operators_id ON agencies (agency_id);