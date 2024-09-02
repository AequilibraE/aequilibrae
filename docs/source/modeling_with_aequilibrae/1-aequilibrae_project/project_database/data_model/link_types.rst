*link_types* table structure
----------------------------

The *link_types* table holds information about the available
link types in the network.

The **link_type** field corresponds to the link type, and it is the
table's primary key

The **link_type_id** field presents the identification of the link type

The **description** field holds the description of the link type

The **lanes** field presents the number or lanes for the link type

The **lane_capacity** field presents the number of lanes for the link type

The **speed** field holds information about the speed in the link type
Attributes follow

.. csv-table:: 
   :header: "Field", "Type", "NULL allowed", "Default Value"
   :widths:    30,     20,         20,          20

   link_type*,VARCHAR,NO,
   link_type_id,VARCHAR,NO,
   description,VARCHAR,YES,
   lanes,NUMERIC,YES,
   lane_capacity,NUMERIC,YES,
   speed,NUMERIC,YES,


(* - Primary key)



The SQL statement for table and index creation is below.


::

   
   
   CREATE TABLE  if not exists link_types (link_type     VARCHAR UNIQUE NOT NULL PRIMARY KEY,
                                           link_type_id  VARCHAR UNIQUE NOT NULL,
                                           description   VARCHAR,
                                           lanes         NUMERIC,
                                           lane_capacity NUMERIC,
                                           speed         NUMERIC
                                           CHECK(LENGTH(link_type_id) == 1));
   
   INSERT INTO 'link_types' (link_type, link_type_id, description, lanes, lane_capacity) VALUES('centroid_connector', 'z', 'VIRTUAL centroid connectors only', 10, 10000);
   
   INSERT INTO 'link_types' (link_type, link_type_id, description, lanes, lane_capacity) VALUES('default', 'y', 'Default general link type', 2, 900);
   
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('link_types','link_type', 'Link type name. E.g. arterial, or connector');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('link_types','link_type_id', 'Single letter identifying the mode. E.g. a, for arterial');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('link_types','description', 'Description of the same. E.g. Arterials are streets like AequilibraE Avenue');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('link_types','lanes', 'Default number of lanes in each direction. E.g. 2');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('link_types','lane_capacity', 'Default vehicle capacity per lane. E.g.  900');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('link_types','speed', 'Free flow velocity in m/s');
