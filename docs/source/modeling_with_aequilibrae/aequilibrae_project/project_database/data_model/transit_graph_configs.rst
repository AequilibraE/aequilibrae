*transit_graph_configs* table structure
---------------------------------------

The *transit_graph_configs* table holds configuration parameters for a TransitGraph of a particular `period_id`
Attributes follow

.. csv-table:: 
   :header: "Field", "Type", "NULL allowed", "Default Value"
   :widths:    30,     20,         20,          20

   period_id*,INTEGER,NO,
   config,TEXT,YES,


(* - Primary key)



The SQL statement for table and index creation is below.


::

   
   CREATE TABLE if not exists transit_graph_configs (period_id INTEGER UNIQUE NOT NULL PRIMARY KEY REFERENCES periods(period_id),
                                                     config    TEXT);
   
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('transit_graph_configs','period_id', 'The period this config is associated with.');
   INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('transit_graph_configs','mode_id', 'JSON string containing the configuration parameters.');
