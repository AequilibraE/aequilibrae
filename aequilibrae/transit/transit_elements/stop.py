import dataclasses
from sqlite3 import Connection
from typing import Dict, Any, Optional

from shapely.geometry import Point

from aequilibrae.transit.constants import Constants, AGENCY_MULTIPLIER
from aequilibrae.transit.transit_elements.basic_element import BasicPTElement


@dataclasses.dataclass
class Stop(BasicPTElement):
    """Transit stop as read from the GTFS feed"""

    def __init__(self, agency_id: int, record: tuple, headers: list):
        self.stop_id = -1
        self.stop = ""
        self.stop_code = ""
        self.stop_name = ""
        self.stop_desc = ""
        self.stop_lat: float = None
        self.stop_lon: float = None
        self.stop_street = ""
        self.zone = ""
        self.zone_id = None
        self.stop_url = ""
        self.location_type = 0
        self.parent_station = ""
        self.stop_timezone = ""

        # Not part of GTFS
        self.taz = None
        self.agency = ""
        self.agency_id = agency_id
        self.link = None
        self.dir = None
        self.srid = -1
        self.geo: Optional[Point] = None
        self.route_type: Optional[int] = None
        self.___map_matching_id__: Dict[Any, Any] = dict()
        self.__moved_map_matching__ = 0

        for key, value in zip(headers, record):
            if key not in self.__dict__.keys():
                raise KeyError(f"{key} field in Stops.txt is unknown field for that file on GTFS")
            key = key if key != "stop_id" else "stop"
            key = key if key != "zone_id" else "zone"
            self.__dict__[key] = value

        if None not in [self.stop_lon, self.stop_lat]:
            self.geo = Point(self.stop_lon, self.stop_lat)
        if len(str(self.zone_id)) == 0:
            self.zone_id = None

    def save_to_database(self, conn: Connection, commit=True) -> None:
        """Saves Transit Stop to the database"""

        sql = """insert into stops (stop_id, stop, agency_id, link, dir, name,
                                    parent_station, description, street, fare_zone_id, transit_zone, route_type, geometry)
                 values (?,?,?,?,?,?,?,?,?,?,?,?, GeomFromWKB(?, ?));"""

        dt = self.data
        conn.execute(sql, dt)
        if commit:
            conn.commit()

    @property
    def data(self) -> list:
        return [
            self.stop_id,
            self.stop,
            self.agency_id,
            self.link,
            self.dir,
            self.stop_name,
            self.parent_station,
            self.stop_desc,
            self.stop_street,
            self.zone_id,
            self.taz,
            int(self.route_type),
            self.geo.wkb,
            self.srid,
        ]

    def get_node_id(self):
        c = Constants()

        val = 1 + c.stops.get(self.agency_id, AGENCY_MULTIPLIER * self.agency_id)
        c.stops[self.agency_id] = val
        self.stop_id = c.stops[self.agency_id]
