from sqlite3 import Connection

from shapely.geometry import LineString

from aequilibrae.transit.constants import Constants, TRIP_ID_MULTIPLIER
from aequilibrae.log import logger
from aequilibrae.transit.transit_elements.basic_element import BasicPTElement


class Trip(BasicPTElement):
    """Transit trips read from trips.txt

    * trip (:obj:`str`): Trip ID as read from the GTFS feed
    * route (:obj:`str`): Route ID as read from the GTFS feed
    * service_id (:obj:`str`): Service ID as read from the GTFS feed
    * trip_headsign (:obj:`str`): Trip headsign as read from the GTFS feed
    * trip_short_name (:obj:`str`): Trip short name as read from the GTFS feed
    * direction_id (:obj:`int`): Direction ID as read from the GTFS feed
    * block_id (:obj:`int`): Block ID as read from the GTFS feed
    * bikes_allowed (:obj:`int`): Bikes allowed flag as read from the GTFS feed
    * wheelchair_accessible (:obj:`int`): Wheelchair accessibility flag as read from the GTFS feed
    * shape_id (:obj:`str`): Shape ID as read from the GTFS feed

    * trip_id (:obj:`int`): Unique trip_id as it will go into the database
    * route_id (:obj:`int`): Unique Route ID as will be available in the routes table
    * pattern_id (:obj:`int`): Unique Pattern ID for this route/stop-pattern as it will go into the database
    * pattern_hash (:obj:`str`): Pattern ID derived from stops for this route/stop-pattern
    * arrivals (:obj:`List[int]`): Sequence of arrival at stops for this trip
    * departures (:obj:`List[int]`): Sequence of departures from stops for this trip
    * stops (:obj:`List[Stop]`): Sequence of stops for this trip
    * shape (:obj:`LineString`): Shape for this trip. Directly from shapes.txt or rebuilt from sequence of stops
    """

    def __init__(self):
        self.route_id = ""
        self.service_id = ""
        self.trip = ""
        self.trip_id = -1
        self.trip_headsign = ""
        self.trip_short_name = ""
        self.block_id = ""
        self.shape_id = ""
        self.direction_id = 0
        self.wheelchair_accessible = 0
        self.bikes_allowed = 0

        # Not from GTFS
        self.pattern_id = 0
        self.pattern_hash = ""
        self.arrivals = []
        self.departures = []
        self.stops = []
        self.shape = None  # type: LineString
        self._stop_based_shape = None  # type: LineString
        self.seated_capacity = None
        self.total_capacity = None
        self.source_time = []

    def _populate(self, record: tuple, headers: list) -> None:
        for key, value in zip(headers, record):
            if key not in self.__dict__.keys():
                raise KeyError(f"{key} field in Trips.txt is unknown field for that file on GTFS")
            key = "trip" if key == "trip_id" else key
            key = "route" if key == "route_id" else key
            self.__dict__[key] = value

    def save_to_database(self, conn: Connection, commit=True) -> None:
        """Saves trips to the database"""
        logger.debug(f"Saving {self.trip_id}/{self.trip} for pattern {self.pattern_id}")
        sql = """insert into trips (trip_id, trip, dir, pattern_id) values (?, ?, ?, ?);"""
        data = [self.trip_id, self.trip, int(self.direction_id), self.pattern_id]
        conn.execute(sql, data)

        sql = """insert into trips_schedule (trip_id, seq, arrival, departure)
                                            values (?, ?, ?, ?)"""
        data = []
        for i, (arr, dep) in enumerate(zip(self.arrivals, self.departures)):
            data.append([self.trip_id, i, arr, dep])
        conn.executemany(sql, data)
        if commit:
            conn.commit()

    def get_trip_id(self):
        c = Constants()
        self.trip_id = c.trips.get(self.pattern_id, self.pattern_id) + TRIP_ID_MULTIPLIER
        c.trips[self.pattern_id] = self.trip_id

    def __lt__(self, other):
        return self.departures[0] < other.departures[0]
