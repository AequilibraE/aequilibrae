CREATE TABLE if not exists modes (mode_name   VARCHAR UNIQUE NOT NULL,
                                  mode_id     VARCHAR UNIQUE NOT NULL       PRIMARY KEY,
                                  description VARCHAR,
                                  pce         NUMERIC        NOT NULL DEFAULT 1.0,
                                  vot         NUMERIC        NOT NULL DEFAULT 0);

--#
INSERT INTO 'modes' (mode_name, mode_id, description) VALUES('car', 'c', 'All motorized vehicles');
--#
INSERT INTO 'modes' (mode_name, mode_id, description) VALUES('transit', 't', 'Public transport vehicles');
--#
INSERT INTO 'modes' (mode_name, mode_id, description) VALUES('walk', 'w', 'Walking links');
--#
INSERT INTO 'modes' (mode_name, mode_id, description) VALUES('bicycle', 'b', 'Biking links');

--@ Attributes follow
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('modes','mode_name', 'The more descriptive name of the mode (e.g. Bicycle)');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('modes','mode_id', 'Single letter identifying the mode. E.g. b, for Bicycle');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('modes','description', 'Description of the same. E.g. Bicycles used to be human-powered two-wheeled vehicles');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('modes','pce', 'Passenger-Car equivalent for assignment');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('modes','vot', 'Value-of-Time for traffic assignment of class');