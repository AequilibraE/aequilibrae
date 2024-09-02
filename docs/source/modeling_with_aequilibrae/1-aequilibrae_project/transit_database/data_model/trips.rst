*trips* table structure
-----------------------

The *trips* table holds information on trips for each route.
This table comes from the GTFS file *trips.txt*.
You can find more information about it `here <https://developers.google.com/transit/gtfs/reference#tripstxt>`_.

**trip_id** identifies a trip

**trip** identifies the trip to a rider

**dir** indicates the direction of travel for a trip

**pattern_id** is an unique pattern for the route

.. csv-table:: 
   :header: "Field", "Type", "NULL allowed", "Default Value"
   :widths:    30,     20,         20,          20

   trip_id*,INTEGER,NO,
   trip,TEXT,YES,
   dir,INTEGER,NO,
   pattern_id,INTEGER,NO,


(* - Primary key)



The SQL statement for table and index creation is below.


::

   
   CREATE TABLE IF NOT EXISTS trips (
   	trip_id         INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
   	trip            TEXT,
   	dir             INTEGER NOT NULL,
   	pattern_id      INTEGER NOT NULL,
   	FOREIGN KEY(pattern_id) REFERENCES routes(pattern_id) deferrable initially deferred
   );
