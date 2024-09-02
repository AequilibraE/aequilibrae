*modes* table structure
-----------------------

The *modes* table holds the information on all the modes available in
the model's network.

The **mode_name** field contains the descriptive name of the field.

The **mode_id** field contains a single letter that identifies the mode.

The **description** field holds the description of the mode.

The **pce** field holds information on Passenger-Car equivalent
for assignment. Defaults to **1.0**.

The **vot** field holds information on Value-of-Time for traffic
assignment. Defaults to **0.0**.

The **ppv** field holds information on average persons per vehicle.
Defaults to **1.0**. **ppv** can assume value 0 for non-travel uses.
Attributes follow

.. csv-table:: 
   :header: "Field", "Type", "NULL allowed", "Default Value"
   :widths:    30,     20,         20,          20

   mode_name,VARCHAR,NO,
   mode_id*,VARCHAR,NO,
   description,VARCHAR,YES,
   pce,NUMERIC,NO,1.0
   vot,NUMERIC,NO,0
   ppv,NUMERIC,NO,1.0


(* - Primary key)



The SQL statement for table and index creation is below.


::

   
   
   CREATE TABLE if not exists modes (mode_name   VARCHAR UNIQUE NOT NULL,
                                     mode_id     VARCHAR UNIQUE NOT NULL       PRIMARY KEY,
                                     description VARCHAR,
                                     pce         NUMERIC        NOT NULL DEFAULT 1.0,
                                     vot         NUMERIC        NOT NULL DEFAULT 0,
                                     ppv         NUMERIC        NOT NULL DEFAULT 1.0
                                     CHECK(LENGTH(mode_id)==1));
   
   INSERT INTO 'modes' (mode_name, mode_id, description) VALUES('car', 'c', 'All motorized vehicles');
   INSERT INTO 'modes' (mode_name, mode_id, description) VALUES('transit', 't', 'Public transport vehicles');
   INSERT INTO 'modes' (mode_name, mode_id, description) VALUES('walk', 'w', 'Walking links');
   INSERT INTO 'modes' (mode_name, mode_id, description) VALUES('bicycle', 'b', 'Biking links');
   
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('modes','mode_name', 'The more descriptive name of the mode (e.g. Bicycle)');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('modes','mode_id', 'Single letter identifying the mode. E.g. b, for Bicycle');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('modes','description', 'Description of the same. E.g. Bicycles used to be human-powered two-wheeled vehicles');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('modes','pce', 'Passenger-Car equivalent for assignment');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('modes','vot', 'Value-of-Time for traffic assignment of class');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('modes','ppv', 'Average persons per vehicle. (0 for non-travel uses)');
