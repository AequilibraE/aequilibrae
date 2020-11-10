create TABLE 'matrices' (name          TEXT     NOT NULL  PRIMARY KEY,
                         file_name     TEXT     NOT NULL UNIQUE,
                         procedure     TEXT     NOT NULL,
                         procedure_id  TEXT     NOT NULL,
                         timestamp     DATETIME DEFAULT current_timestamp,
                         description   TEXT);


#
CREATE INDEX name_matrices ON matrices (name);