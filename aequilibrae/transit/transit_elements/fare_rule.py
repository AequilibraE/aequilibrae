from sqlite3 import Connection

from aequilibrae.log import logger


class FareRule:
    """Transit Fare

    * fare_id (:obj:`int`): Fare Id to which this rule applies
    * fare (:obj:`str`): Name of the fare rule
    * route (:obj:`str`): Route ID as in GTFS to which this fare rule applies
    * route_id (:obj:`int`): Route ID as in network model to which this fare rule applies
    * origin (:obj:`int`): Transit zone ID for origin
    * destination (:obj:`int`): Transit zone ID for destination
    * contains (:obj:`str`): As in GTFS
    * agency_id (:obj:`int`): Agency ID as in the network model
    * origin_id (:obj:`int`): Origin zone ID as in the network model
    * destination_id (:obj:`int`): Destination Zone ID as in the network model"""

    def __init__(self):
        self.fare = ""
        self.fare_id = -1
        self.route = None
        self.route_id = None
        self.origin = -1
        self.destination = -1
        self.contains = None
        self.agency_id = -1
        self.origin_id = -1
        self.destination_id = -1

    def populate(self, record: tuple, headers: list) -> None:
        """Adds fare information."""
        for key, value in zip(headers, record):
            if key not in self.__dict__.keys():
                raise KeyError(f"{key} field in Routes.txt is unknown field for that file on GTFS")

            key = "fare" if key == "fare_id" else key
            key = "route" if key == "route_id" else key
            self.__dict__[key] = value

    def save_to_database(self, conn: Connection) -> None:
        """Saves Fare rules to the database"""
        self.contains = None if len(self.contains) == 0 else self.contains
        data = [self.fare_id, self.route_id, self.origin_id, self.destination_id, self.contains]
        if not self.__exists():
            logger.warning(f"Transit Fare rule with data {data} was not added")
            return

        sql = "insert into fare_rules (fare_id, route_id, origin, destination, contains) values (?, ?, ?, ?, ?);"
        conn.execute(sql, data)
        conn.commit()

    def __exists(self):
        return min(self.fare_id, self.route_id or 0, self.origin_id, self.destination_id) >= 0
