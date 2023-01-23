from sqlite3 import Connection
from shapely.geometry import MultiLineString
from aequilibrae.transit.constants import Constants, ROUTE_ID_MULTIPLIER, AGENCY_MULTIPLIER
from aequilibrae.transit.transit_elements.basic_element import BasicPTElement


class Route(BasicPTElement):
    """Transit route element to feed into Transit_routes

    * route_id (:obj:`str`): ID of this route, starting with the agency prefix ID
    * route_short_name (:obj:`str`): Short name as found in the GTFS feed
    * route_long_name (:obj:`str`): Long name as found in the GTFS feed
    * route_desc (:obj:`str`): Route description as found in the GTFS feed
    * route_type (:obj:`int`):  Route type (mode) for this transit link
    * route_url (:obj:`str`): Route URL as found in the GTFS feed
    * route_color (:obj:`str`): Route color for mapping as found in the GTFS feed
    * route_text_color (:obj:`str`): Route color (text) for mapping as found in the GTFS feed
    * route_sort_order (:obj:`int`): Route rendering order as found in the GTFS feed
    * agency_id (:obj:`str`): Agency ID
    * seated_capacity (:obj:`float`): Vehicle seated capacity for this route
    * total_capacity (:obj:`float`): Total vehicle capacity for this route"""

    def __init__(self, agency_id):
        self.route_id = -1
        self.route = ""
        self.route_short_name = ""
        self.route_long_name = ""
        self.route_desc = ""
        self.route_type = 0
        self.route_url = ""
        self.route_color = ""
        self.route_text_color = ""
        self.route_sort_order = 0
        self.agency_id = agency_id

        # Not part of GTFS
        self.pattern_id = 0
        self.pattern = ""
        self.seated_capacity = 0
        self.total_capacity = 0
        self.shape: MultiLineString
        self.srid = -1
        self.number_of_cars = 0
        self.__sql = """insert into routes (route_id, pattern_id, route, agency_id, shortname, longname, description,
                                            route_type, seated_capacity, total_capacity{})
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?{});"""
        self.sql = self.__sql
        self.__get_route_id()

    def populate(self, record: tuple, headers: list) -> None:
        for key, value in zip(headers, record):
            if key not in self.__dict__.keys():
                raise KeyError(f"{key} field in Routes.txt is unknown field for that file on GTFS")

            # We convert route_id into route, as the the GTFS header for it is not maintained in our database
            key = "route" if key == "route_id" else key
            self.__dict__[key] = value

    def save_to_database(self, conn: Connection, commit=True) -> None:
        """Saves route to the database"""

        data = self.data
        conn.execute(self.sql, data)
        if commit:
            conn.commit()

    @property
    def data(self):
        data = [
            self.route_id,
            self.pattern_id,
            self.route,
            self.agency_id,
            self.route_short_name,
            self.route_long_name,
            self.route_desc,
            int(self.route_type),
            self.seated_capacity,
            self.total_capacity,
        ]
        if self.shape is None:
            shape = ""
            geo_fld = ""
        else:
            geo_fld = ", geometry "
            shape = ", ST_Multi(GeomFromWKB(?, ?))"
            data.extend([self.shape.wkb, self.srid])
        self.sql = self.__sql.format(geo_fld, shape)
        return data

    def __get_route_id(self):
        c = Constants()
        val = c.routes.get(self.agency_id, AGENCY_MULTIPLIER * self.agency_id)
        c.routes[self.agency_id] = val + ROUTE_ID_MULTIPLIER
        self.route_id = c.routes[self.agency_id]
