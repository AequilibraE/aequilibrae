*periods* table structure
-------------------------

The periods table holds the time periods and their period_id. Default entry with id 1 is the entire day.
Attributes follow

.. csv-table:: 
   :header: "Field", "Type", "NULL allowed", "Default Value"
   :widths:    30,     20,         20,          20

   period_id,INTEGER,NO,
   period_start,INTEGER,NO,
   period_end,INTEGER,NO,
   period_description,TEXT,YES,


(* - Primary key)



The SQL statement for table and index creation is below.


::

   CREATE TABLE if not exists periods (period_id               INTEGER UNIQUE NOT NULL,
                                       period_start            INTEGER NOT NULL,
                                       period_end              INTEGER NOT NULL,
                                       period_description      TEXT
                                       CHECK(TYPEOF(period_id) == 'integer')
                                       CHECK(TYPEOF(period_start) == 'integer')
                                       CHECK(TYPEOF(period_end) == 'integer'));
   
   INSERT INTO periods (period_id, period_start, period_end, period_description) VALUES(1, 0, 86400, 'Default time period, whole day');
   
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('periods','period_id', 'ID of the time period');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('periods','period_start', 'Start of the time period');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('periods','period_end', 'End of the time period');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('periods','period_description', 'Optional description of the time period');
