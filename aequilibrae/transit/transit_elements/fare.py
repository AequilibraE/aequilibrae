from sqlite3 import Connection
from aequilibrae.transit.constants import Constants, AGENCY_MULTIPLIER


class Fare:
    """Transit Fare

    * fare_id (:obj:`int`): ID of the fare as in the network model
    * fare (:obj:`str`): ID of the fare as in GTFS
    * agency (:obj:`str`): Corresponding agency as inputed during import
    * agency_id (:obj:`int`): ID of the corresponding agency as in the network model
    * price (:obj:`int`): As in GTFS
    * currency (:obj:`str`): As in GTFS
    * payment_method (:obj:`int`): As in GTFS
    * transfer (:obj:`int`): As in GTFS
    * transfer_duration (:obj:`int`): As in GTFS"""

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
        for key, value in zip(headers, record):
            if key not in self.__dict__.keys():
                raise KeyError(f"{key} field in Routes.txt is unknown field for that file on GTFS")

            # We convert route_id into route, as the the GTFS header for it is not maintained in our database
            v = None if value in [0, ""] else value
            key = "fare" if key == "fare_id" else key
            key = "agency" if key == "agency_id" else key
            if key in ["payment_method", "transfer", "transfer_duration"]:
                v = v or self.__dict__[key]
            self.__dict__[key] = v

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
