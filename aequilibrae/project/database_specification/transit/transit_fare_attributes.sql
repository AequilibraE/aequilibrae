--@ All transit fares for transit agencies in the model are included on this
--@ table. It includes the agency ID is applies to, as well as price and
--@ transfer criteria, which are crucial for proper consideration for trip
--@ routing.

create TABLE IF NOT EXISTS "transit_fare_attributes" (
	fare_id           INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	fare              TEXT    NOT NULL,
	agency_id	      INTEGER NOT NULL,
	price         	  REAL,
	currency	      TEXT,
	payment_method	  INTEGER,
	transfer	      INTEGER,
	transfer_duration REAL,
	FOREIGN KEY(agency_id) REFERENCES agencies(agency_id) deferrable initially deferred
);

CREATE UNIQUE INDEX IF NOT EXISTS fare_transfer_uniqueness ON transit_fare_attributes (fare_id, transfer);