--@ The *results* table holds the metadata for results stored in
--@ *results_database.sqlite*.
--@
--@ The **scenario** field describes which scenario this record belongs to.
--@
--@ The **year** field describes which year of the scenario was used.
--@
--@ The **table_name** field presents the actual name of the result
--@ table in *results_database.sqlite*.
--@
--@ The **procedure** field holds the name the the procedure that generated
--@ the result (e.g.: Traffic Assignment).
--@
--@ The **procedure_id** field holds an unique UUID identifier for this procedure,
--@ which is created at runtime.
--@
--@ The **procedure_report** field holds the output of the complete procedure report,
--@ encoded as JSON. Versions 1.5 to 1.7 and 2.0.0rc1 encoded that JSON text a second
--@ time, storing a JSON string whose value is the report.
--@
--@ The **timestamp** field holds the information when the procedure was executed.
--@
--@ The **description** field holds the user-provided description of the result.


create TABLE if not exists results (scenario         TEXT,
                                    year             TEXT,
                                    table_name       TEXT     NOT NULL PRIMARY KEY,
                                    reference_table  TEXT,
                                    procedure        TEXT     NOT NULL,
                                    procedure_id     TEXT     NOT NULL,
                                    procedure_report TEXT     NOT NULL,
                                    timestamp        DATETIME DEFAULT current_timestamp,
                                    description      TEXT);
