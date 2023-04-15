create TABLE if not exists matrices (name          TEXT     NOT NULL PRIMARY KEY,
                                     file_name     TEXT     NOT NULL UNIQUE,
                                     cores         INTEGER  NOT NULL DEFAULT 1,
                                     procedure     TEXT,
                                     procedure_id  TEXT,
                                     timestamp     DATETIME DEFAULT current_timestamp,
                                     description   TEXT);


--#
CREATE INDEX name_matrices ON matrices (name);