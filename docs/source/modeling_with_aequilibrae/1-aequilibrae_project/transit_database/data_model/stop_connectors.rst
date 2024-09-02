*stop_connectors* table structure
---------------------------------

The *stops_connectors* table holds information on the connection of
the GTFS network with the real network.

**id_from** identifies the network link the vehicle departs

**id_to** identifies the network link th vehicle is heading to

**conn_type** identifies the type of connection used to connect the links

**traversal_time** represents the time spent crossing the link

**penalty_cost** identifies the penalty in the connection

.. csv-table:: 
   :header: "Field", "Type", "NULL allowed", "Default Value"
   :widths:    30,     20,         20,          20

   id_from,INTEGER,NO,
   id_to,INTEGER,NO,
   conn_type,INTEGER,NO,
   traversal_time,INTEGER,NO,
   penalty_cost,INTEGER,NO,
   geometry,LINESTRING,NO,''


(* - Primary key)



The SQL statement for table and index creation is below.


::

   
   
   CREATE TABLE IF NOT EXISTS stop_connectors (
   	id_from        INTEGER  NOT NULL,
   	id_to          INTEGER  NOT NULL,
   	traversal_time INTEGER  NOT NULL,
   	penalty_cost   INTEGER  NOT NULL);
   
   SELECT AddGeometryColumn('stop_connectors', 'geometry', 4326, 'LINESTRING', 'XY', 1);
   
   SELECT CreateSpatialIndex('stop_connectors' , 'geometry');
   
   CREATE INDEX IF NOT EXISTS stop_connectors_id_from ON stop_connectors (id_from);
   
   CREATE INDEX IF NOT EXISTS stop_connectors_id_to ON stop_connectors (id_to);
