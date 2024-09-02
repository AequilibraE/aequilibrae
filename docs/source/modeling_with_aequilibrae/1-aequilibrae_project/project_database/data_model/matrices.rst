*matrices* table structure
--------------------------

The *matrices* table holds infromation about all matrices that exists in the
project *matrix* folder.

The **name** field presents the name of the table.

The **file_name** field holds the file name.

The **cores** field holds the information on the number of cores used.

The **procedure** field holds the name the the procedure that generated
the result (e.g.: Traffic Assignment).

The **procedure_id** field holds an unique alpha-numeric identifier for
this prodecure.

The **timestamp** field holds the information when the procedure was executed.

The **description** field holds the user-provided description of the result.

.. csv-table:: 
   :header: "Field", "Type", "NULL allowed", "Default Value"
   :widths:    30,     20,         20,          20

   name*,TEXT,NO,
   file_name,TEXT,NO,
   cores,INTEGER,NO,1
   procedure,TEXT,YES,
   procedure_id,TEXT,YES,
   timestamp,DATETIME,YES,current_timestamp
   description,TEXT,YES,


(* - Primary key)



The SQL statement for table and index creation is below.


::

   
   
   create TABLE if not exists matrices (name          TEXT     NOT NULL PRIMARY KEY,
                                        file_name     TEXT     NOT NULL UNIQUE,
                                        cores         INTEGER  NOT NULL DEFAULT 1,
                                        procedure     TEXT,
                                        procedure_id  TEXT,
                                        timestamp     DATETIME DEFAULT current_timestamp,
                                        description   TEXT);
   
   
   CREATE INDEX name_matrices ON matrices (name);
