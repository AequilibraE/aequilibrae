from sqlite3 import Connection
from aequilibrae.transit.constants import Constants, AGENCY_MULTIPLIER


class Fare:
    """Transit Fare

    * fare_id (:obj:`int`): ID of the fare as in the network model
    * fare (:obj:`str`): ID of the fare as in GTFS
    * agency (:obj:`str`): Corresponding agency as inputted during import
    * agency_id (:obj:`int`): ID of the corresponding agency as in the network model
    * price (:obj:`float`): As in GTFS
    * currency (:obj:`str`): As in GTFS
    * payment_method (:obj:`int`): As in GTFS
    * transfer (:obj:`int`): As in GTFS, with ``None`` for unlimited transfers
    * transfer_duration (:obj:`float`): As in GTFS, with ``None`` for transfers that do not expire"""

    def __init__(self, agency_id: int):
        self.fare = ""
        self.fare_id = -1
        self.agency = ""
        self.agency_id = agency_id
        self.price = -1
        self.currency = ""
        self.payment_method = 0
        self.transfer = 0
        self.transfer_duration = 0
        self.__get_fare_id()

    def populate(self, record: tuple, headers: list) -> None:
        """Adds fare information."""
        for key, value in zip(headers, record, strict=True):
            if key not in self.__dict__.keys():
                raise KeyError(f"{key} field in fare_attributes.txt is unknown field for that file on GTFS")

            # We convert fare_id into fare, as the GTFS header for it is not maintained in our database
            key = "fare" if key == "fare_id" else key
            key = "agency" if key == "agency_id" else key
            # Values are kept as published, as zero (e.g. a free fare) and empty (e.g. unlimited transfers) differ
            self.__dict__[key] = None if value == "" else value

    def save_to_database(self, conn: Connection) -> None:
        """Saves Fare attributes to the database"""

        data = [
            self.fare_id,
            self.fare,
            self.agency_id,
            self.price,
            self.currency,
            self.payment_method,
            self.transfer,
            self.transfer_duration,
        ]
        sql = """insert into fare_attributes (fare_id, fare, agency_id, price, currency, payment_method,
                                                     transfer, transfer_duration) VALUES (?, ?, ?, ?, ?, ?, ?, ?);"""
        conn.execute(sql, data)
        conn.commit()

    def __get_fare_id(self):
        c = Constants()
        self.fare_id = 1 + c.fares.get(self.agency_id, AGENCY_MULTIPLIER * self.agency_id)
        c.fares[self.agency_id] = self.fare_id
