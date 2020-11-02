create TABLE 'matrices' (matrix_file      TEXT     NOT NULL PRIMARY KEY,
                         procedure        TEXT     NOT NULL,
                         procedure_id     TEXT     NOT NULL UNIQUE,
                         timestamp        DATETIME DEFAULT current_timestamp,
                         description      TEXT);
