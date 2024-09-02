*nodes* table structure
-----------------------

The *nodes* table holds all the network nodes available in AequilibraE model.

The **node_id** field is an identifier of the node.

The **is_centroid** field holds information if the node is a centroid
of a network or not. Assumes values 0 or 1. Defaults to **0**.

The **modes** field identifies all modes connected to the node.

The **link_types** field identifies all link types connected
to the node.

.. csv-table:: 
   :header: "Field", "Type", "NULL allowed", "Default Value"
   :widths:    30,     20,         20,          20

   ogc_fid*,INTEGER,YES,
   node_id,INTEGER,NO,
   is_centroid,INTEGER,NO,0
   modes,TEXT,YES,
   link_types,TEXT,YES,
   geometry,POINT,NO,''


(* - Primary key)



The SQL statement for table and index creation is below.


::

   
   
   CREATE TABLE if not exists nodes (ogc_fid     INTEGER PRIMARY KEY,
                                     node_id     INTEGER UNIQUE NOT NULL,
                                     is_centroid INTEGER        NOT NULL DEFAULT 0,
                                     modes       TEXT,
                                     link_types  TEXT
                                     CHECK(TYPEOF(node_id) == 'integer')
                                     CHECK(TYPEOF(is_centroid) == 'integer')
                                     CHECK(is_centroid>=0)
                                     CHECK(is_centroid<=1));
   
   SELECT AddGeometryColumn( 'nodes', 'geometry', 4326, 'POINT', 'XY', 1);
   
   SELECT CreateSpatialIndex( 'nodes' , 'geometry' );
   
   CREATE INDEX idx_node ON nodes (node_id);
   
   CREATE INDEX idx_node_is_centroid ON nodes (is_centroid);
   
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('nodes','node_id', 'Unique node ID');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('nodes','is_centroid', 'Flag identifying centroids');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('nodes','modes', 'Modes connected to the node');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('nodes','link_types', 'Link types connected to the node');
