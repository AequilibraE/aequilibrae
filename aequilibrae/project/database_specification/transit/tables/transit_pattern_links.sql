--@ The route_links table holds the information on the sequence of
--@ of transit links that are traversed by each transit pattern
--@

CREATE TABLE IF NOT EXISTS route_links (
	pattern_id              INTEGER    NOT NULL,
	seq  	                INTEGER    NOT NULL,
	transit_link            INTEGER    NOT NULL,
	from_stop               INTEGER    NOT NULL,
	to_stop                 INTEGER    NOT NULL,
	distance                INTEGER    NOT NULL,
	FOREIGN KEY(pattern_id) REFERENCES "routes"(pattern_id) deferrable initially deferred,
	FOREIGN KEY(from_stop)  REFERENCES "stops"(stop_id) deferrable initially deferred
	FOREIGN KEY(to_stop)    REFERENCES "stops"(stop_id) deferrable initially deferred
);

--#
create UNIQUE INDEX IF NOT EXISTS route_links_stop_id ON route_links (pattern_id, transit_link);