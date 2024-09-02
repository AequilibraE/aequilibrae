*about* table structure
-----------------------

The *about* table holds information about the AequilibraE model
currently developed.

The **infoname** field holds the name of information being added

The **infovalue** field holds the information to add

.. csv-table:: 
   :header: "Field", "Type", "NULL allowed", "Default Value"
   :widths:    30,     20,         20,          20

   infoname,TEXT,NO,
   infovalue,TEXT,YES,


(* - Primary key)



The SQL statement for table and index creation is below.


::

   
   
   CREATE TABLE  if not exists about (infoname  TEXT UNIQUE NOT NULL,
                                      infovalue TEXT
                                     );
   INSERT INTO 'about' (infoname) VALUES('model_name');
   
   INSERT INTO 'about' (infoname) VALUES('region');
   
   INSERT INTO 'about' (infoname) VALUES('description');
   
   INSERT INTO 'about' (infoname) VALUES('author');
   
   INSERT INTO 'about' (infoname) VALUES('year');
   
   INSERT INTO 'about' (infoname) VALUES('scenario_description');
   
   INSERT INTO 'about' (infoname) VALUES('model_version');
   
   INSERT INTO 'about' (infoname) VALUES('project_id');
   
   INSERT INTO 'about' (infoname) VALUES('aequilibrae_version');
   
   INSERT INTO 'about' (infoname) VALUES('projection');
   
   INSERT INTO 'about' (infoname) VALUES('driving_side');
   
   INSERT INTO 'about' (infoname) VALUES('license');
   
   INSERT INTO 'about' (infoname) VALUES('scenario_name');
