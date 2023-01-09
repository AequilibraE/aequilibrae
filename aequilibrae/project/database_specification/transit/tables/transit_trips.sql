CREATE TABLE IF NOT EXISTS trips (
	trip_id         INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	trip            TEXT,
	dir             INTEGER NOT NULL,
	pattern_id      INTEGER NOT NULL,
	FOREIGN KEY(pattern_id) REFERENCES routes(pattern_id) deferrable initially deferred
);