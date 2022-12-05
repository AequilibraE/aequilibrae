--@ The trips table holds the complete list of all transit services
--@ operating, for which one can override the information on capacity
--@ coming from route/pattern. There is also information on whether the
--@ vehicle is articulated or if it has multiple cars (applicable to rail), but
--@ both of these fields default to 0 in case of regular bus services.
--@
--@ The transit trip ID can be traced back to the pattern, route and agency
--@ directly through the encoding of their trip_id, as explained in the
--@ documentation for the agencies table.
--@

CREATE TABLE IF NOT EXISTS trips (
	trip_id         INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	trip            TEXT,
	dir	            INTEGER NOT NULL,
	pattern_id      INTEGER NOT NULL,
	FOREIGN KEY(pattern_id) REFERENCES routes(pattern_id) deferrable initially deferred
);