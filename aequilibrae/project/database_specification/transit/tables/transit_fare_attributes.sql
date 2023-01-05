create TABLE IF NOT EXISTS fare_attributes (
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

--#
CREATE UNIQUE INDEX IF NOT EXISTS fare_transfer_uniqueness ON fare_attributes (fare_id, transfer);