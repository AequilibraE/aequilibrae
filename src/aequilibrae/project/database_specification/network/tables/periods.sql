--@ The periods table holds the time periods and their period_id. Times count seconds from the start
--@ of the GTFS service day, so they may exceed 86400, and periods never wrap around midnight. The
--@ default entry with id 1 is the whole service day, from 00:00:00 to 48:00:00.
--#
CREATE TABLE if not exists periods (period_id               INTEGER UNIQUE NOT NULL,
                                    period_start            INTEGER NOT NULL,
                                    period_end              INTEGER NOT NULL,
                                    period_description      TEXT
                                    CHECK(TYPEOF(period_id) == 'integer')
                                    CHECK(TYPEOF(period_start) == 'integer')
                                    CHECK(TYPEOF(period_end) == 'integer'));

--#
INSERT INTO periods (period_id, period_start, period_end, period_description) VALUES(1, 0, 172800, 'Default time period, whole service day');

--@ Attributes follow
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('periods','period_id', 'ID of the time period');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('periods','period_start', 'Start of the time period');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('periods','period_end', 'End of the time period');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('periods','period_description', 'Optional description of the time period');
