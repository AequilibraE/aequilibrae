--@ The Transit_Pattern_Links table holds the information on the sequence of
--@ of transit links that are traversed by each transit pattern
--@

CREATE TABLE IF NOT EXISTS transit_pattern_links (
	pattern_id              INTEGER    NOT NULL,
	"index"	                INTEGER    NOT NULL,
	transit_link            INTEGER    NOT NULL,
	FOREIGN KEY(pattern_id) REFERENCES "transit_routes"(pattern_id) deferrable initially deferred,
	FOREIGN KEY(transit_link) REFERENCES "transit_links"(transit_link) deferrable initially deferred
);

create UNIQUE INDEX IF NOT EXISTS transit_pattern_links_stop_id ON Transit_Pattern_Links (pattern_id, transit_link);