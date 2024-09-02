*fare_attributes* table structure
---------------------------------

The *fare_attributes* table holds information about the fare values.
This table information comes from the GTFS file *fare_attributes.txt*.
Given that this file is optional in GTFS, it can be empty.
You can check out more information `here <https://developers.google.com/transit/gtfs/reference#fare_attributestxt>`_.

**fare_id** identifies a fare class

**fare** describes a fare class

**agency_id** identifies a relevant agency for a fare.

**price** especifies the fare price

**currency_code** especifies the currency used to pay the fare

**payment_method** indicates when the fare must be paid.

**transfer** indicates the number of transfers permitted on the fare

**transfer_duration** indicates the lenght of time in seconds before a
transfer expires.

.. csv-table:: 
   :header: "Field", "Type", "NULL allowed", "Default Value"
   :widths:    30,     20,         20,          20

   fare_id*,INTEGER,NO,
   fare,TEXT,NO,
   agency_id,INTEGER,NO,
   price,REAL,YES,
   currency,TEXT,YES,
   payment_method,INTEGER,YES,
   transfer,INTEGER,YES,
   transfer_duration,REAL,YES,


(* - Primary key)



The SQL statement for table and index creation is below.


::

   
   create TABLE IF NOT EXISTS fare_attributes (
   	fare_id           INTEGER  NOT NULL PRIMARY KEY AUTOINCREMENT,
   	fare              TEXT     NOT NULL,
   	agency_id         INTEGER  NOT NULL,
   	price             REAL,
   	currency          TEXT,
   	payment_method    INTEGER,
   	transfer          INTEGER,
   	transfer_duration REAL,
   	FOREIGN KEY(agency_id) REFERENCES agencies(agency_id) deferrable initially deferred
   );
   
   CREATE UNIQUE INDEX IF NOT EXISTS fare_transfer_uniqueness ON fare_attributes (fare_id, transfer);
