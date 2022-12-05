create TABLE if not exists results (table_name       TEXT     NOT NULL PRIMARY KEY,
                                    procedure        TEXT     NOT NULL,
                                    procedure_id     TEXT     NOT NULL UNIQUE,
                                    procedure_report TEXT     NOT NULL,
                                    timestamp        DATETIME DEFAULT current_timestamp,
                                    description      TEXT);
