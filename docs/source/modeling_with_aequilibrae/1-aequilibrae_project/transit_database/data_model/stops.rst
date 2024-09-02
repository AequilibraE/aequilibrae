*stops* table structure
-----------------------

The *stops* table holds information on the stops where vehicles
pick up or drop off riders. This table information comes from
the GTFS file *stops.txt*. You can find more information about
it `here <https://developers.google.com/transit/gtfs/reference#stopstxt>`_.

**stop_id** is an unique identifier for a stop

**stop** idenfifies a stop, statio, or station entrance

**agency_id** identifies the agency fot the specified route

**link** identifies the *link_id* in the links table that corresponds to the
pattern matching

**dir** indicates the direction of travel for a trip

**name** identifies the name of a stop

**parent_station** defines hierarchy between different locations
defined in *stops.txt*.

**description** provides useful description of the stop location

**street** identifies the address of a stop

**fare_zone_id** identifies the fare zone for a stop

**transit_zone** identifies the TAZ for a fare zone

**route_type** indicates the type of transporation used on a route

.. csv-table:: 
   :header: "Field", "Type", "NULL allowed", "Default Value"
   :widths:    30,     20,         20,          20

   stop_id*,TEXT,YES,
   stop,TEXT,NO,
   agency_id,INTEGER,NO,
   link,INTEGER,YES,
   dir,INTEGER,YES,
   name,TEXT,YES,
   parent_station,TEXT,YES,
   description,TEXT,YES,
   street,TEXT,YES,
   fare_zone_id,INTEGER,YES,
   transit_zone,TEXT,YES,
   route_type,INTEGER,NO,-1
   geometry,POINT,NO,''


(* - Primary key)



The SQL statement for table and index creation is below.


::

   
   CREATE TABLE IF NOT EXISTS stops (
   	stop_id           TEXT     PRIMARY KEY,
   	stop              TEXT     NOT NULL ,
   	agency_id         INTEGER  NOT NULL,
   	link              INTEGER,
   	dir               INTEGER,
   	name              TEXT,
   	parent_station    TEXT,
   	description       TEXT,
   	street            TEXT,
   	fare_zone_id      INTEGER,
   	transit_zone      TEXT,
   	route_type        INTEGER  NOT NULL DEFAULT -1,
   	FOREIGN KEY(agency_id) REFERENCES agencies(agency_id),
   	FOREIGN KEY("fare_zone_id") REFERENCES fare_zones("fare_zone_id")
   );
   
   create INDEX IF NOT EXISTS stops_stop_id ON stops (stop_id);
   
   select AddGeometryColumn( 'stops', 'geometry', 4326, 'POINT', 'XY', 1);
   
   select CreateSpatialIndex( 'stops' , 'geometry' );
