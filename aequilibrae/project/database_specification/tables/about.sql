--@ THIS TABLE REQUIRES FURTHER DOCUMENTATION
--@
--@

CREATE TABLE  if not exists about (infoname  TEXT UNIQUE NOT NULL,
                                   infovalue TEXT
                                  );
--#
INSERT INTO 'about' (infoname) VALUES('model_name');

--#
INSERT INTO 'about' (infoname) VALUES('region');

--#
INSERT INTO 'about' (infoname) VALUES('description');

--#
INSERT INTO 'about' (infoname) VALUES('author');

--#
INSERT INTO 'about' (infoname) VALUES('year');

--#
INSERT INTO 'about' (infoname) VALUES('scenario_description');

--#
INSERT INTO 'about' (infoname) VALUES('model_version');

--#
INSERT INTO 'about' (infoname) VALUES('project_id');

--#
INSERT INTO 'about' (infoname) VALUES('aequilibrae_version');

--#
INSERT INTO 'about' (infoname) VALUES('projection');

--#
INSERT INTO 'about' (infoname) VALUES('driving_side');

--#
INSERT INTO 'about' (infoname) VALUES('license');

--#
INSERT INTO 'about' (infoname) VALUES('scenario_name');
